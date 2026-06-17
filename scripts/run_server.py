"""Entry point for federated learning server."""

import mlflow
import torch
import logging
import os
import json
import shutil
from datetime import datetime, UTC
from pathlib import Path
from flwr.server import start_server, ServerConfig
from flwr.common import ndarrays_to_parameters

from src.server.strategy import WeightedFedAvg
from src.server.checkpoint_manager import CheckpointManager
from src.model.fraud_mlp import FraudMLP

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s %(levelname).1s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _archive_existing_dir(path: Path, label: str) -> Path | None:
    if not path.exists() or not any(path.iterdir()):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive_root = Path("outputs/archive")
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / f"{label}_{stamp}"
    shutil.move(str(path), str(target))
    path.mkdir(parents=True, exist_ok=True)
    return target


def _reset_monitoring_files() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in ("evaluation_history.json", "latest_metrics.json", "best_round.json"):
        path = results_dir / name
        if path.exists():
            path.unlink()


def _load_initial_parameters(ckpt: CheckpointManager):
    """Load initial parameters from the latest checkpoint, if available."""
    requested = os.environ.get("RESUME_CHECKPOINT")
    if requested:
        latest = Path(requested)
        if not latest.is_absolute():
            latest = ckpt.checkpoint_dir / latest
        if not latest.exists():
            logger.warning("RESUME missing=%s", latest)
            latest = None
    else:
        latest = ckpt.latest()
    if latest is None:
        logger.info("RESUME none")
        return None

    logger.info("RESUME ckpt=%s", latest.name)
    state = torch.load(latest, map_location="cpu")
    if not isinstance(state, dict):
        logger.warning("RESUME incompatible=%s reason=not_state_dict", latest.name)
        return None

    model = FraudMLP()
    current_state = model.state_dict()

    if set(current_state.keys()) != set(state.keys()):
        logger.warning(
            "RESUME incompatible=%s reason=key_mismatch",
            latest.name,
        )
        return None

    for k in current_state.keys():
        if state[k].shape != current_state[k].shape:
            logger.warning(
                "RESUME incompatible=%s reason=shape key=%s ckpt=%s model=%s",
                latest.name,
                k,
                tuple(state[k].shape),
                tuple(current_state[k].shape),
            )
            return None

    # ── Only serialize the keys that clients exchange — BN running stats
    # are client-local and must not be federated. This must match the
    # filter in FraudClient.get_parameters() / set_parameters() exactly.
    trainable_keys = [
        k for k in current_state.keys()
        if "running_mean" not in k
        and "running_var" not in k
        and "num_batches_tracked" not in k
    ]

    try:
        nds = [state[k].cpu().numpy() for k in trainable_keys]
    except Exception as exc:
        logger.warning("RESUME failed=%s err=%s", latest.name, exc)
        return None

    logger.info(
        "RESUME loaded=%s tensors=%s",
        latest.name,
        len(trainable_keys),
    )
    return ndarrays_to_parameters(nds)


def lr_schedule(server_round: int) -> dict:
    """
    Smooth learning rate decay schedule — per-round config sent to ALL clients.

    focal_alpha is per-client, so this returns a config with a sentinel;
    the strategy overrides it per-client in configure_fit().
    """
    r = max(int(server_round), 1)
    profile = os.environ.get("TRAINING_PROFILE", "ambitious").strip().lower()
    if profile == "post_target":
        if r <= 10:
            lr, epochs = 2e-4, 4
        elif r <= 40:
            lr, epochs = 1e-4, 4
        else:
            lr, epochs = 5e-5, 4
    elif profile == "ambitious":
        if r <= 5:
            lr, epochs = 1e-3, 3
        elif r <= 25:
            lr, epochs = 6e-4, 3
        elif r <= 60:
            lr, epochs = 2e-4, 4
        elif r <= 90:
            lr, epochs = 8e-5, 4
        else:
            lr, epochs = 4e-5, 4
    else:
        if r <= 5:
            lr, epochs = 1e-3, 2
        elif r <= 25:
            lr, epochs = 8e-4, 2
        elif r <= 60:
            lr, epochs = 4e-4, 3
        elif r <= 90:
            lr, epochs = 1e-4, 3
        else:
            lr, epochs = 5e-5, 3

    default_gamma = "1.5" if profile in {"ambitious", "post_target"} else "2.0"
    default_bce_mix = "0.25" if profile == "post_target" else "0.30"
    return {
        "lr": lr,
        "local_epochs": epochs,
        "weight_decay": float(os.environ.get("WEIGHT_DECAY", "1e-4")),
        "fedprox_mu": float(os.environ.get("FEDPROX_MU", "0.001")),
        "loss_mode": os.environ.get("LOSS_MODE", "hybrid"),
        "bce_mix": float(os.environ.get("BCE_MIX", default_bce_mix)),
        "focal_gamma": float(os.environ.get("FOCAL_GAMMA", default_gamma)),
        "local_lr_scheduler": os.environ.get("LOCAL_LR_SCHEDULER", "cosine"),
        "min_lr_ratio": float(os.environ.get("MIN_LR_RATIO", "0.15")),
    }


def _display_training_mode() -> str:
    profile = os.environ.get("TRAINING_PROFILE", "ambitious").strip().lower()
    if profile == "post_target":
        return "lower-loss"
    if profile == "ambitious":
        return "high-band"
    return profile


def main() -> None:
    """Start the federated learning server."""
    logger.info("SERVER start")
    mlflow.set_experiment("federated-fraud-detection")

    checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "outputs/checkpoints"))
    resume_from_checkpoint = _env_bool("RESUME_FROM_CHECKPOINT", default=False)
    fresh_run = _env_bool("FRESH_RUN", default=not resume_from_checkpoint)
    if fresh_run and not resume_from_checkpoint:
        archived = _archive_existing_dir(checkpoint_dir, "checkpoints")
        if archived:
            logger.info("ARCHIVE checkpoints=%s", archived)
        _reset_monitoring_files()
        logger.info("RESET monitoring")

    ckpt = CheckpointManager(checkpoint_dir)
    initial_parameters = _load_initial_parameters(ckpt) if resume_from_checkpoint else None

    strategy = WeightedFedAvg(
        checkpoint_manager=ckpt,
        on_fit_config_fn=lr_schedule,
        target_auprc=float(os.environ.get("TARGET_AUPRC", "0.70")),
        target_auroc=float(os.environ.get("TARGET_AUROC", "0.90")),
        target_f1=float(os.environ.get("TARGET_F1", "0.70")),
        keep_last_rounds=int(os.environ.get("KEEP_LAST_ROUNDS", "12")),
    )
    if initial_parameters is not None:
        try:
            setattr(strategy, "initial_parameters", initial_parameters)
            logger.info("RESUME parameters=loaded")
        except Exception:
            pass

    num_rounds = int(os.environ.get("NUM_ROUNDS", "140"))
    server_address = os.environ.get("SERVER_ADDRESS", "localhost:8080")
    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    training_mode = _display_training_mode()
    logger.info(
        "CONFIG addr=%s rounds=%s mode=%s fresh=%s resume=%s ckpt=%s target=%.2f/%.2f/%.2f high=%s/%s/%s floor=%s/%s/%s",
        server_address,
        num_rounds,
        training_mode,
        int(fresh_run),
        int(resume_from_checkpoint),
        checkpoint_dir,
        strategy.target_auprc,
        strategy.target_auroc,
        strategy.target_f1,
        os.environ.get("HIGH_TARGET_AUPRC", "0.85"),
        os.environ.get("HIGH_TARGET_AUROC", "0.95"),
        os.environ.get("HIGH_TARGET_F1", "0.80"),
        os.environ.get("CLIENT_FLOOR_AUPRC", "0.80"),
        os.environ.get("CLIENT_FLOOR_AUROC", "0.93"),
        os.environ.get("CLIENT_FLOOR_F1", "0.75"),
    )
    logger.info("WAIT clients")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "active_training_run.json").write_text(
        json.dumps(
            {
                "training_run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "resume_from_checkpoint": resume_from_checkpoint,
                "targets": {
                    "auprc": strategy.target_auprc,
                    "auroc": strategy.target_auroc,
                    "f1": strategy.target_f1,
                },
                "high_targets": {
                    "auprc": float(os.environ.get("HIGH_TARGET_AUPRC", "0.85")),
                    "auroc": float(os.environ.get("HIGH_TARGET_AUROC", "0.95")),
                    "f1": float(os.environ.get("HIGH_TARGET_F1", "0.80")),
                },
                "client_floors": {
                    "auprc": float(os.environ.get("CLIENT_FLOOR_AUPRC", "0.80")),
                    "auroc": float(os.environ.get("CLIENT_FLOOR_AUROC", "0.93")),
                    "f1": float(os.environ.get("CLIENT_FLOOR_F1", "0.75")),
                },
                "fedprox_mu": float(os.environ.get("FEDPROX_MU", "0.001")),
                "training_mode": training_mode,
                "loss_mode": os.environ.get("LOSS_MODE", "hybrid"),
                "bce_mix": float(os.environ.get("BCE_MIX", "0.30")),
                "focal_gamma": float(
                    os.environ.get(
                        "FOCAL_GAMMA",
                        "1.5"
                        if os.environ.get("TRAINING_PROFILE", "ambitious").strip().lower()
                        in {"ambitious", "post_target"}
                        else "2.0",
                    )
                ),
                "keep_last_rounds": strategy.keep_last_rounds,
            },
            indent=2,
        )
    )

    start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds, round_timeout=100000),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
