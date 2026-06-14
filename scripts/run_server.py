"""Entry point for federated learning server."""

import mlflow
import torch
import logging
import os
import json
from pathlib import Path
from flwr.server import start_server, ServerConfig
from flwr.common import ndarrays_to_parameters

from src.server.strategy import WeightedFedAvg
from src.server.checkpoint_manager import CheckpointManager
from src.model.fraud_mlp import FraudMLP
from src.model.fraud_mlp import FraudMLP, is_federated_param

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _load_initial_parameters(ckpt: CheckpointManager):
    """Load initial parameters from the latest checkpoint, if available."""
    latest = ckpt.latest()
    if latest is None:
        logger.info("No checkpoint found; starting with fresh parameters")
        return None

    logger.info(f"Loading checkpoint: {latest.name}")
    state = torch.load(latest, map_location="cpu")
    if not isinstance(state, dict):
        logger.warning(f"Incompatible checkpoint {latest.name}: expected a state_dict.")
        return None

    model = FraudMLP()
    current_state = model.state_dict()

    if set(current_state.keys()) != set(state.keys()):
        logger.warning(
            f"Incompatible checkpoint {latest.name}: key mismatch detected. "
            "Starting with fresh parameters."
        )
        return None

    for k in current_state.keys():
        if state[k].shape != current_state[k].shape:
            logger.warning(
                f"Incompatible checkpoint {latest.name}: shape mismatch on '{k}' "
                f"(checkpoint={tuple(state[k].shape)}, model={tuple(current_state[k].shape)}). "
                "Starting with fresh parameters."
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
        logger.warning(f"Failed to convert checkpoint {latest.name}: {exc}")
        return None

    logger.info(
        f"Loaded checkpoint: {len(trainable_keys)} trainable parameter tensors "
        f"({len(current_state) - len(trainable_keys)} BN buffers kept client-local)"
    )
    return ndarrays_to_parameters(nds)


def lr_schedule(server_round: int) -> dict:
    """
    Smooth learning rate decay schedule — per-round config sent to ALL clients.

    focal_alpha is per-client, so this returns a config with a sentinel;
    the strategy overrides it per-client in configure_fit().
    """
    r = max(int(server_round), 1)
    if r <= 5:
        lr, epochs = 2e-3, 5
    elif r <= 20:
        lr, epochs = 1e-3, 5
    elif r <= 35:
        lr, epochs = 5e-4, 5
    elif r <= 50:
        lr, epochs = 1e-4, 5
    elif r <= 70:
        lr, epochs = 5e-5, 8
    else:
        lr, epochs = 2e-5, 8
    return {"lr": lr, "local_epochs": epochs}


def main() -> None:
    """Start the federated learning server."""
    logger.info("Starting Flower Server")
    mlflow.set_experiment("federated-fraud-detection")
    logger.info("MLflow experiment: federated-fraud-detection")

    ckpt = CheckpointManager("outputs/checkpoints")
    initial_parameters = _load_initial_parameters(ckpt)

    strategy = WeightedFedAvg(checkpoint_manager=ckpt, on_fit_config_fn=lr_schedule)
    if initial_parameters is not None:
        try:
            setattr(strategy, "initial_parameters", initial_parameters)
            logger.info("Initial parameters attached to strategy")
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("Flower Server Configuration")
    logger.info("=" * 60)
    num_rounds = int(os.environ.get("NUM_ROUNDS", "80"))
    server_address = os.environ.get("SERVER_ADDRESS", "localhost:8080")
    logger.info(f"Server address: {server_address}")
    logger.info(f"Number of rounds: {num_rounds}")
    logger.info("Round timeout: 100000 seconds")
    logger.info("=" * 60)
    logger.info("Waiting for clients to connect...")

    start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds, round_timeout=100000),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
