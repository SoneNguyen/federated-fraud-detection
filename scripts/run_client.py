"""Entry point for federated learning client."""

import logging
import os

import torch

from flwr.client import start_client
from torch.utils.data import DataLoader

from src.model.fraud_mlp import FraudMLP
from src.client.client import FraudClient
from src.data.dataset import loader_kwargs, split_dataset

logging.getLogger("flwr").setLevel(logging.ERROR)


def main():
    """Start a federated learning client."""
    cid = int(os.environ["CLIENT_ID"])
    addr = os.environ.get("SERVER_ADDRESS", "localhost:8080")
    path = os.environ["DATA_PATH"]
    epochs = int(os.environ.get("LOCAL_EPOCHS", "2"))
    device_str = os.environ.get("DEVICE", None)  # None = auto-detect

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(os.environ.get("MATMUL_PRECISION", "high"))

    batch_size = int(os.environ.get(
        "BATCH_SIZE",
        "2048" if device.type == "cuda" else "512",
    ))
    num_workers = int(os.environ.get(
        "NUM_WORKERS",
        "0" if os.name == "nt" else "2",
    ))
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "4"))
    use_amp = os.environ.get("USE_AMP", "1" if device.type == "cuda" else "0")
    loss_mode = os.environ.get("LOSS_MODE", "hybrid")
    bce_mix = os.environ.get("BCE_MIX", "0.30")

    gpu = ""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        gpu = f" gpu=\"{props.name}\" mem={props.total_memory / 1024**3:.1f}GiB"
    print(
        f"CLIENT start id={cid} addr={addr} data={path} device={device}{gpu} "
        f"bs={batch_size} workers={num_workers} amp={use_amp} "
        f"loss={loss_mode} bce={bce_mix}"
    )

    print("CLIENT phase=model")
    model = FraudMLP(device=str(device))
    print("CLIENT phase=data")
    train_dataset, val_dataset = split_dataset(path, val_split=0.15)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=prefetch_factor,
        ),
    )

    print("CLIENT phase=init")
    client = FraudClient(
        model=model,
        train_dataset=train_dataset,
        val_loader=val_loader,
        local_epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=batch_size,
    )

    print("CLIENT phase=connect")
    try:
        start_client(
            server_address=addr,
            client=client.to_client(),
        )
    except Exception as exc:
        detail = exc.details() if hasattr(exc, "details") else str(exc)
        raise SystemExit(f"CLIENT connect_failed addr={addr} detail={detail}") from None


if __name__ == "__main__":
    main()
