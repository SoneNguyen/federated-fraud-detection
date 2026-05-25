import os, flwr as fl
from client.model import FraudMLP
from client.fl_client import FraudClient
from client.dataset import make_loaders

def main():
    cid  = int(os.environ["CLIENT_ID"])
    addr = os.environ.get("SERVER_ADDRESS","localhost:8080")
    path = os.environ["DATA_PATH"]
    epochs = int(os.environ.get("LOCAL_EPOCHS","5"))
    train_l, val_l = make_loaders(path)
    model = FraudMLP()
    client = FraudClient(model, train_l, val_l)
    print(f"[Client {cid}] → {addr}")
    fl.client.start_numpy_client(
        server_address=addr, client=client,
        grpc_max_message_length=1024*1024*512)

if __name__ == "__main__":
    main()

# Run locally (3 terminals):
# python server/fl_server.py
# CLIENT_ID=0 DATA_PATH=data/processed/client_0/... python client/run_client.py
# CLIENT_ID=1 DATA_PATH=data/processed/client_1/... python client/run_client.py