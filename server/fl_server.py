import mlflow
import torch
from flwr.server import start_server
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters

from server.strategy import WeightedFedAvg
from server.checkpoint_manager import CheckpointManager
from client.model import FraudMLP


def _load_initial_parameters(ckpt: CheckpointManager):
    latest = ckpt.latest()
    if latest is None:
        return None

    state = torch.load(latest, map_location="cpu")
    if not isinstance(state, dict):
        print(f"[WARN] Incompatible checkpoint {latest.name}: expected a state_dict.")
        return None

    model = FraudMLP()
    keys = list(model.state_dict().keys())
    if set(keys) != set(state.keys()):
        print(
            f"[WARN] Incompatible checkpoint {latest.name}: key mismatch detected. "
            "Starting with fresh parameters."
        )
        return None

    try:
        nds = [state[k].cpu().numpy() for k in keys]
    except Exception as exc:
        print(f"[WARN] Failed to convert checkpoint {latest.name}: {exc}")
        return None
    return ndarrays_to_parameters(nds)


def main() -> None:
    mlflow.set_experiment("federated-fraud-detection")

    ckpt = CheckpointManager("checkpoints")
    initial_parameters = _load_initial_parameters(ckpt)

    strategy = WeightedFedAvg()
    # Attach initial parameters to the strategy if available (start_server
    # implementations may not accept an `initial_parameters` kwarg).
    if initial_parameters is not None:
        try:
            setattr(strategy, "initial_parameters", initial_parameters)
        except Exception:
            pass
    print("Flower server running")

    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=10, round_timeout=100000),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()