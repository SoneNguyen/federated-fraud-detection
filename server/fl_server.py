import mlflow
from flwr.server import start_server
from flwr.server.server_config import ServerConfig

from server.strategy import WeightedFedAvg


def main() -> None:
    mlflow.set_experiment("federated-fraud-detection")
    strategy = WeightedFedAvg()
    print("Flower server running")
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=10, round_timeout=100000),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
