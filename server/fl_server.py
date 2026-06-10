import mlflow
import torch
import logging
import os
from flwr.server import start_server
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from server.strategy import WeightedFedAvg
from server.checkpoint_manager import CheckpointManager
from client.model import FraudMLP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def _load_initial_parameters(ckpt: CheckpointManager):
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
    keys = list(current_state.keys())
    if set(keys) != set(state.keys()):
        logger.warning(
            f"Incompatible checkpoint {latest.name}: key mismatch detected. "
            "Starting with fresh parameters."
        )
        return None

    for k in keys:
        if state[k].shape != current_state[k].shape:
            logger.warning(
                f"Incompatible checkpoint {latest.name}: shape mismatch on '{k}' "
                f"(checkpoint={tuple(state[k].shape)}, model={tuple(current_state[k].shape)}). "
                "Starting with fresh parameters."
            )
            return None

    try:
        nds = [state[k].cpu().numpy() for k in keys]
    except Exception as exc:
        logger.warning(f"Failed to convert checkpoint {latest.name}: {exc}")
        return None
    logger.info(f"Successfully loaded checkpoint with {len(keys)} parameter groups")
    return ndarrays_to_parameters(nds)


def main() -> None:
    logger.info("Starting Flower Server")
    mlflow.set_experiment("federated-fraud-detection")
    logger.info("MLflow experiment: federated-fraud-detection")

    ckpt = CheckpointManager("checkpoints")
    initial_parameters = _load_initial_parameters(ckpt)

    strategy = WeightedFedAvg()
    # Attach initial parameters to the strategy if available (start_server
    # implementations may not accept an `initial_parameters` kwarg).
    if initial_parameters is not None:
        try:
            setattr(strategy, "initial_parameters", initial_parameters)
            logger.info("Initial parameters attached to strategy")
        except Exception:
            pass
    
    logger.info("=" * 60)
    logger.info("Flower Server Configuration")
    logger.info("=" * 60)
    num_rounds = int(os.environ.get("NUM_ROUNDS", "15"))
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