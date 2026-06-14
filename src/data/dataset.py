# FraudDataset — PyTorch Dataset for loading preprocessed fraud detection data.
# Reads from Parquet files, extracts features and labels based on schema.json.

from __future__ import annotations

import json
import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Load schema from config directory
config_dir = Path(__file__).parent.parent.parent / "config"
with open(config_dir / "schema.json") as f:
    _s = json.load(f)

FEATURE_ORDER = _s["feature_schema"]["feature_order"]
LABEL         = _s["feature_schema"]["label"]["name"]
PASS_ONLY = []

# Hard assertions — fail loudly at import time
assert len(FEATURE_ORDER) == _s["feature_schema"]["total_features"]
assert LABEL not in FEATURE_ORDER
for p in PASS_ONLY:
    assert p not in FEATURE_ORDER, f"{p} must not be in feature_order"


class FraudDataset(Dataset):
    """PyTorch Dataset for fraud detection data stored as Parquet."""

    def __init__(self, path: str):
        df = pd.read_parquet(path)
        missing = [c for c in FEATURE_ORDER + [LABEL] if c not in df.columns]
        assert not missing, f"Missing columns: {missing}"
        self.X = torch.tensor(df[FEATURE_ORDER].values, dtype=torch.float32)
        self.y = torch.tensor(df[LABEL].values, dtype=torch.float32)
        self.meta = df[PASS_ONLY].copy() if all(c in df.columns for c in PASS_ONLY) else None
        print(f"Loaded {len(df):,} rows | fraud={df[LABEL].mean()*100:.2f}% | X.shape={self.X.shape}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]


def make_loaders(
    path: str,
    val_split: float = 0.15,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load data and return (train_loader, val_loader) with class weighting."""
    ds = FraudDataset(path)
    n_val = int(len(ds) * val_split)
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else 2

    # Balance training samples by fraud label to improve learning on rare positives.
    train_labels = ds.y[train_ds.indices].detach().cpu().numpy().astype(int)
    class_counts = np.bincount(train_labels, minlength=2)
    if class_counts[1] == 0:
        sampler = None
    else:
        # Cap oversampling ratio at 20× to avoid memorisation on tiny fraud sets
        ratio = min(class_counts[0] / max(class_counts[1], 1), 20.0)
        class_weights = np.array([1.0, ratio])
        sample_weights = class_weights[train_labels].astype(float).tolist()
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
        ),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


if __name__ == "__main__":
    # Quick check to verify loader shape and labels
    for i in range(3):
        p = f"data/processed/client_{i}/transactions_normalized.parquet"
        tl, vl = make_loaders(p)
        X, y = next(iter(tl))
        assert X.shape[1] == len(FEATURE_ORDER), f"Bad feature count: {X.shape[1]}"
        assert set(y.unique().tolist()).issubset({0.0, 1.0})
        print(f"Client {i}: batch X={X.shape}, y={y.shape} — OK")
