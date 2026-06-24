"""Factory for constructing Flower fraud clients from runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import loader_kwargs, split_dataset
from src.model.fraud_mlp import FraudMLP


def _resolve_device(device_str: str | None) -> torch.device:
    if device_str and device_str.strip().lower() not in {"auto", "none"}:
        return torch.device(device_str)
    if torch.cuda.is_available():
        # Pick the device with the most VRAM (avoids landing on iGPU as cuda:0)
        best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        return torch.device(f"cuda:{best}")
    return torch.device("cpu")


def build_fraud_client(
    *,
    client_id: int,
    data_path: str | Path,
    device_str: str | None = None,
    local_epochs: int | None = None,
    batch_size: int | None = None,
):
    """Build a configured FraudClient instance."""

    os.environ["CLIENT_ID"] = str(client_id)
    device = _resolve_device(device_str)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(os.environ.get("MATMUL_PRECISION", "high"))
    else:
        threads = int(os.environ.get("TORCH_NUM_THREADS", str(max(os.cpu_count() or 1, 1))))
        torch.set_num_threads(max(threads, 1))
        try:
            torch.set_num_interop_threads(max(1, min(threads, 4)))
        except RuntimeError:
            pass

    resolved_batch_size = batch_size or int(
        os.environ.get("BATCH_SIZE", "2048" if device.type == "cuda" else "512")
    )
    num_workers = int(
        os.environ.get("NUM_WORKERS", "0" if os.name == "nt" else "2")
    )
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "4"))

    from src.client.client import FraudClient

    model = FraudMLP(device=str(device))
    train_dataset, val_dataset = split_dataset(str(data_path), val_split=0.15)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs(
            batch_size=resolved_batch_size,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=prefetch_factor,
        ),
    )

    return FraudClient(
        model=model,
        train_dataset=train_dataset,
        val_loader=val_loader,
        local_epochs=local_epochs or int(os.environ.get("LOCAL_EPOCHS", "2")),
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=resolved_batch_size,
    )
