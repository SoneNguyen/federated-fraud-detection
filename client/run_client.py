# This script runs the federated learning client. 
# It loads the local dataset, initializes the model and the client, and starts the client to connect to the server for training.
# The client ID, server address, data path, and local epochs can be configured via environment variables.
from collections.abc import Sized
from typing import cast
import os
import torch
from flwr.client import start_numpy_client
from client.model import FraudMLP
from client.fl_client import FraudClient
from client.dataset import make_loaders

def main():
    cid = int(os.environ["CLIENT_ID"])
    addr = os.environ.get("SERVER_ADDRESS", "localhost:8080")
    path = os.environ["DATA_PATH"]
    epochs = int(os.environ.get("LOCAL_EPOCHS", "2"))
    device = os.environ.get("DEVICE", None)  # None = auto-detect (cuda if available, else cpu)
    
    train_l, val_l = make_loaders(path, num_workers=0)
    model = FraudMLP(device=device)
    
    # Print device info
    print(f"[Client {cid}] Using device: {model.device}")
    if torch.cuda.is_available():
        print(f"[Client {cid}] GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"[Client {cid}] Configuration:")
    print(f"  - Data path: {path}")
    print(f"  - Train samples: {len(cast(Sized, train_l.dataset))}")
    print(f"  - Val samples: {len(cast(Sized, val_l.dataset))}")
    print(f"  - Local epochs: {epochs}")
    
    local_epochs = epochs
    lr = float(os.environ.get("LOCAL_LR", "1e-3"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "1e-4"))
    client = FraudClient(model, train_l, val_l, local_epochs=local_epochs, lr=lr, weight_decay=weight_decay)
    print(f"[Client {cid}] Connecting to server at {addr}...")
    start_numpy_client(
        server_address=addr,
        client=client,
        grpc_max_message_length=1024 * 1024 * 512,
    )

if __name__ == "__main__":
    main()

# Run locally (3 terminals):
# python server/fl_server.py
# CLIENT_ID=0 DATA_PATH=data/processed/client_0/... python client/run_client.py
# CLIENT_ID=1 DATA_PATH=data/processed/client_1/... python client/run_client.py
