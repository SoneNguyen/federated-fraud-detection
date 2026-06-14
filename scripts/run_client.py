"""Entry point for federated learning client."""

import os
import sys
import torch
from pathlib import Path
from typing import cast
from collections.abc import Sized

from flwr.client import start_client

from src.model.fraud_mlp import FraudMLP
from src.client.client import FraudClient
from src.data.dataset import make_loaders


def main():
    """Start a federated learning client."""
    cid = int(os.environ["CLIENT_ID"])
    addr = os.environ.get("SERVER_ADDRESS", "localhost:8080")
    path = os.environ["DATA_PATH"]
    epochs = int(os.environ.get("LOCAL_EPOCHS", "2"))
    device_str = os.environ.get("DEVICE", None)  # None = auto-detect

    # Determine device
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Federated Learning Client Configuration")
    print(f"{'='*60}")
    print(f"Client ID: {cid}")
    print(f"Server address: {addr}")
    print(f"Data path: {path}")
    print(f"Local epochs: {epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Load model and data
    model = FraudMLP(device=str(device))
    train_loader, val_loader = make_loaders(
        path,
        val_split=0.15,
        batch_size=512,
        seed=42,
    )

    # Create federated client
    client = FraudClient(
        model=model,
        train_dataset=train_loader.dataset,
        val_loader=val_loader,
        local_epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
    )

    # Connect to server
    start_client(
        server_address=addr,
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()
