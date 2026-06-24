"""PyTorch datasets and loaders for preprocessed fraud data."""

from __future__ import annotations

import os
from collections.abc import Sized

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from src.data.feature_registry import FEATURE_ORDER, LABEL

PASS_ONLY: list[str] = []

assert LABEL not in FEATURE_ORDER


class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection data stored as Parquet."""

    def __init__(self, path: str):
        df = pd.read_parquet(path)
        missing = [c for c in FEATURE_ORDER + [LABEL] if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"
        self.X = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
        self.y = torch.tensor(df[LABEL].values, dtype=torch.float32)
        self.meta = df[PASS_ONLY].copy() if all(c in df.columns for c in PASS_ONLY) else None
        print(
            f"DATASET rows={len(df):,} features={self.X.shape[1]} "
            f"fraud={df[LABEL].mean()*100:.2f}% path={path}"
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def load_validation_frame(path: str, val_split: float = 0.15) -> pd.DataFrame:
    """Return the deterministic tail validation frame used by make_loaders."""
    df = pd.read_parquet(path)
    n_val = int(len(df) * val_split)
    split = len(df) - n_val
    return df.iloc[split:].reset_index(drop=True)


def split_dataset(path: str, val_split: float = 0.15) -> tuple[Subset, Subset]:
    """Load data once and return deterministic temporal train/validation subsets."""
    ds = FraudDataset(path)
    n_val = int(len(ds) * val_split)
    split = len(ds) - n_val
    return Subset(ds, range(0, split)), Subset(ds, range(split, len(ds)))


def loader_kwargs(
    batch_size: int,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
) -> dict:
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor or 4
    return kwargs


def make_loaders(
    path: str,
    val_split: float = 0.15,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load data and return deterministic temporal (train_loader, val_loader)."""
    train_ds, val_ds = split_dataset(path, val_split)
    ds = train_ds.dataset
    assert isinstance(ds, FraudDataset)

    train_indices = torch.as_tensor(list(train_ds.indices), dtype=torch.long)
    train_labels = ds.y[train_indices].detach().cpu().numpy().astype(int)
    class_counts = np.bincount(train_labels, minlength=2)
    if class_counts[1] == 0:
        sampler = None
    else:
        ratio = min(class_counts[0] / max(class_counts[1], 1), 20.0)
        class_weights = np.array([1.0, ratio])
        sample_weights = class_weights[train_labels].astype(float).tolist()
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    kwargs = loader_kwargs(batch_size, num_workers, pin_memory, prefetch_factor)

    return (
        DataLoader(
            train_ds,
            sampler=sampler,
            shuffle=(sampler is None),
            **kwargs,
        ),
        DataLoader(val_ds, shuffle=False, **kwargs),
    )


if __name__ == "__main__":
    for i in range(3):
        p = f"dataset/processed/client_{i}/transactions_normalized.parquet"
        tl, _ = make_loaders(p)
        X, y = next(iter(tl))
        assert X.shape[1] == len(FEATURE_ORDER), f"Bad feature count: {X.shape[1]}"
        assert set(y.unique().tolist()).issubset({0.0, 1.0})
        print(f"Client {i}: batch X={X.shape}, y={y.shape} OK")
