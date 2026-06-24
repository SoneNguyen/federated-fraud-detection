"""Server runtime assembly for Flower ServerApp and local tooling."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig

from scripts.run_paths import archive_flat_runtime_files
from scripts.run_paths import checkpoint_dir as default_checkpoint_dir
from scripts.run_paths import results_dir as default_results_dir
from src.model.fraud_mlp import FraudMLP
from src.server.checkpoint_manager import CheckpointManager
from src.server.strategy import WeightedFedAvg


logger = logging.getLogger(__name__)


def apply_run_config(run_config: Mapping[str, Any] | None) -> None:
    """Apply Flower run-config values to environment-style runtime settings."""

    if not run_config:
        return
    for key, value in run_config.items():
        env_key = key.upper().replace("-", "_")
        if value is not None:
            os.environ.setdefault(env_key, str(value))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    return min(max(value, min_value), max_value)


def client_sampling_config(num_clients: int) -> dict[str, int | float]:
    if num_clients >= 50:
        default_sample = min(64, max(40, math.ceil(num_clients * 0.40)))
        default_available = default_sample
    elif num_clients >= 10:
        default_sample = min(10, num_clients)
        default_available = max(num_clients - 1, 1)
    else:
        default_sample = num_clients
        default_available = num_clients

    min_fit = min(_env_int("MIN_FIT_CLIENTS", default_sample), num_clients)
    min_eval = min(
        _env_int("MIN_EVAL_CLIENTS", min(default_sample, num_clients)),
        num_clients,
    )
    min_available = min(
        _env_int("MIN_AVAILABLE_CLIENTS", max(default_available, min_fit, min_eval)),
        num_clients,
    )
    fraction_fit = _env_float("FRACTION_FIT", min(1.0, min_fit / max(num_clients, 1)))
    fraction_eval = _env_float(
        "FRACTION_EVALUATE",
        min(1.0, min_eval / max(num_clients, 1)),
    )
    return {
        "num_clients": num_clients,
        "min_fit": min_fit,
        "min_eval": min_eval,
        "min_available": min_available,
        "fraction_fit": fraction_fit,
        "fraction_eval": fraction_eval,
    }


def archive_existing_dir(path: Path, label: str) -> Path | None:
    if not path.exists() or not any(path.iterdir()):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archive_root = Path("outputs/archive")
    archive_root.mkdir(parents=True, exist_ok=True)
    run_label = path.name if path.name != label else ""
    target_name = f"{label}_{run_label}_{stamp}" if run_label else f"{label}_{stamp}"
    target = archive_root / target_name
    shutil.move(str(path), str(target))
    path.mkdir(parents=True, exist_ok=True)
    return target


def reset_monitoring_files(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "evaluation_history.json",
        "latest_metrics.json",
        "best_round.json",
        "training_summary.md",
    ):
        path = results_dir / name
        if path.exists():
            path.unlink()


def load_initial_parameters(ckpt: CheckpointManager):
    """Load initial parameters from the latest compatible checkpoint."""

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
        logger.warning("RESUME incompatible=%s reason=key_mismatch", latest.name)
        return None

    for key in current_state.keys():
        if state[key].shape != current_state[key].shape:
            logger.warning(
                "RESUME incompatible=%s reason=shape key=%s ckpt=%s model=%s",
                latest.name,
                key,
                tuple(state[key].shape),
                tuple(current_state[key].shape),
            )
            return None

    trainable_keys = [
        key
        for key in current_state.keys()
        if "running_mean" not in key
        and "running_var" not in key
        and "num_batches_tracked" not in key
    ]

    try:
        arrays = [state[key].cpu().numpy() for key in trainable_keys]
    except Exception as exc:
        logger.warning("RESUME failed=%s err=%s", latest.name, exc)
        return None

    logger.info("RESUME loaded=%s tensors=%s", latest.name, len(trainable_keys))
    return ndarrays_to_parameters(arrays)


def lr_schedule(server_round: int) -> dict[str, float | int | str]:
    """Smooth learning-rate decay schedule sent to selected clients."""

    round_no = max(int(server_round), 1)
    profile = os.environ.get("TRAINING_PROFILE", "ambitious").strip().lower()
    if profile == "post_target":
        if round_no <= 10:
            lr, epochs = 2e-4, 4
        elif round_no <= 40:
            lr, epochs = 1e-4, 4
        else:
            lr, epochs = 5e-5, 4
    elif profile in {"scalable", "scale", "large"}:
        if round_no <= 3:
            lr, epochs = 6e-4, 2
        elif round_no <= 20:
            lr, epochs = 4e-4, 2
        elif round_no <= 60:
            lr, epochs = 2e-4, 2
        elif round_no <= 100:
            lr, epochs = 8e-5, 3
        else:
            lr, epochs = 4e-5, 3
    elif profile == "ambitious":
        if round_no <= 5:
            lr, epochs = 1e-3, 3
        elif round_no <= 25:
            lr, epochs = 6e-4, 3
        elif round_no <= 60:
            lr, epochs = 2e-4, 4
        elif round_no <= 90:
            lr, epochs = 8e-5, 4
        else:
            lr, epochs = 4e-5, 4
    else:
        if round_no <= 5:
            lr, epochs = 1e-3, 2
        elif round_no <= 25:
            lr, epochs = 8e-4, 2
        elif round_no <= 60:
            lr, epochs = 4e-4, 3
        elif round_no <= 90:
            lr, epochs = 1e-4, 3
        else:
            lr, epochs = 5e-5, 3

    default_gamma = "1.5" if profile in {"ambitious", "post_target"} else "2.0"
    if profile in {"scalable", "scale", "large"}:
        default_gamma = "1.75"
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


def display_training_mode() -> str:
    profile = os.environ.get("TRAINING_PROFILE", "ambitious").strip().lower()
    if profile == "post_target":
        return "lower-loss"
    if profile == "ambitious":
        return "high-band"
    return profile


def build_server_components(
    run_config: Mapping[str, Any] | None = None,
) -> ServerAppComponents:
    """Build Flower ServerApp components for the current Flower runtime."""

    apply_run_config(run_config)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname).1s %(message)s",
        datefmt="%H:%M:%S",
    )
    mlflow.set_experiment("federated-fraud-detection")

    checkpoint_dir = default_checkpoint_dir()
    results_dir = default_results_dir()
    archived_flat = archive_flat_runtime_files()
    if archived_flat:
        logger.info("ARCHIVE flat_runtime_files=%s", len(archived_flat))
    os.environ["CHECKPOINT_DIR"] = str(checkpoint_dir)
    os.environ["RESULTS_DIR"] = str(results_dir)

    resume_from_checkpoint = _env_bool("RESUME_FROM_CHECKPOINT", default=False)
    fresh_run = _env_bool("FRESH_RUN", default=not resume_from_checkpoint)
    if fresh_run and not resume_from_checkpoint:
        archived = archive_existing_dir(checkpoint_dir, "checkpoints")
        if archived:
            logger.info("ARCHIVE checkpoints=%s", archived)
        archived_results = archive_existing_dir(results_dir, "results")
        if archived_results:
            logger.info("ARCHIVE results=%s", archived_results)
        reset_monitoring_files(results_dir)
        logger.info("RESET monitoring")

    ckpt = CheckpointManager(checkpoint_dir)
    initial_parameters = load_initial_parameters(ckpt) if resume_from_checkpoint else None

    num_clients = _env_int("NUM_CLIENTS", 3)
    if num_clients >= 50 and "TRAINING_PROFILE" not in os.environ:
        os.environ["TRAINING_PROFILE"] = "scalable"
    sampling = client_sampling_config(num_clients)

    strategy = WeightedFedAvg(
        checkpoint_manager=ckpt,
        on_fit_config_fn=lr_schedule,
        target_auprc=float(os.environ.get("TARGET_AUPRC", "0.70")),
        target_auroc=float(os.environ.get("TARGET_AUROC", "0.90")),
        target_f1=float(os.environ.get("TARGET_F1", "0.70")),
        keep_last_rounds=int(os.environ.get("KEEP_LAST_ROUNDS", "12")),
        fraction_fit=float(sampling["fraction_fit"]),
        fraction_evaluate=float(sampling["fraction_eval"]),
        min_fit_clients=int(sampling["min_fit"]),
        min_evaluate_clients=int(sampling["min_eval"]),
        min_available_clients=int(sampling["min_available"]),
    )
    if initial_parameters is not None:
        setattr(strategy, "initial_parameters", initial_parameters)
        logger.info("RESUME parameters=loaded")

    num_rounds = int(os.environ.get("NUM_ROUNDS", os.environ.get("NUM_SERVER_ROUNDS", "100")))
    run_id = datetime.now(UTC).strftime("run_%Y%m%d_%H%M%S")
    training_mode = display_training_mode()
    logger.info(
        "CONFIG rounds=%s clients=%s sample_fit=%s sample_eval=%s "
        "min_available=%s mode=%s fresh=%s resume=%s ckpt=%s results=%s",
        num_rounds,
        num_clients,
        int(sampling["min_fit"]),
        int(sampling["min_eval"]),
        int(sampling["min_available"]),
        training_mode,
        int(fresh_run),
        int(resume_from_checkpoint),
        checkpoint_dir,
        results_dir,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "active_training_run.json").write_text(
        json.dumps(
            {
                "training_run_id": run_id,
                "started_at": datetime.now(UTC).isoformat(),
                "num_clients": num_clients,
                "sampling": sampling,
                "checkpoint_dir": str(checkpoint_dir),
                "results_dir": str(results_dir),
                "resume_from_checkpoint": resume_from_checkpoint,
                "targets": {
                    "auprc": strategy.target_auprc,
                    "auroc": strategy.target_auroc,
                    "f1": strategy.target_f1,
                },
                "training_mode": training_mode,
                "loss_mode": os.environ.get("LOSS_MODE", "hybrid"),
                "keep_last_rounds": strategy.keep_last_rounds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds, round_timeout=300),
    )
