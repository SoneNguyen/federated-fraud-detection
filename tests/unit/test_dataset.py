"""Tests for the FraudDataset DataLoader."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from collections.abc import Sized
from typing import cast

from client.dataset import FEATURE_ORDER, LABEL, FraudDataset, make_loaders


def _write_parquet(path: Path, n: int = 1000, fraud_rate: float = 0.02) -> Path:
    """Write a minimal valid parquet file for testing."""
    rng = np.random.default_rng(42)
    n_fraud = int(n * fraud_rate)
    data: dict[str, object] = {c: rng.standard_normal(n).astype("float32") for c in FEATURE_ORDER}
    data[LABEL] = np.array([1] * n_fraud + [0] * (n - n_fraud), dtype="int8")
    data["orig_currency"] = ["USD"] * n
    data["stale_fx_flag"] = [0] * n
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def test_dataset_loads_correct_shape(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet", n=512)
    ds = FraudDataset(str(p))
    assert len(ds) == 512
    X, y = ds[0]
    assert X.shape == (11,)
    assert y.shape == ()


def test_dataset_feature_count(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet")
    ds = FraudDataset(str(p))
    X, _ = ds[0]
    assert X.shape[0] == 11  # total_features from schema


def test_dataset_labels_binary(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet", n=200)
    ds = FraudDataset(str(p))
    _, y = ds[0]
    assert y.item() in (0.0, 1.0)


def test_dataset_excludes_metadata_from_x(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet")
    ds = FraudDataset(str(p))
    # X must not contain orig_currency or stale_fx_flag columns
    assert ds.X.shape[1] == 11


def test_make_loaders_split_sizes(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet", n=1000)
    train_l, val_l = make_loaders(str(p), val_split=0.15, batch_size=64)
    assert len(cast(Sized, train_l.dataset)) == 850
    assert len(cast(Sized, val_l.dataset)) == 150


def test_make_loaders_batch_shape(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet", n=256)
    train_l, _ = make_loaders(str(p), val_split=0.2, batch_size=32)
    X, y = next(iter(train_l))
    assert X.shape == (32, 11)
    assert y.shape == (32,)
    assert X.dtype == torch.float32


def test_make_loaders_num_workers(tmp_path):
    p = _write_parquet(tmp_path / "test.parquet", n=128)
    train_l, _ = make_loaders(str(p), val_split=0.1, batch_size=16, num_workers=0)
    X, y = next(iter(train_l))
    assert X.shape[1] == 11
    assert y.shape[0] == 16
    assert train_l.num_workers == 0


def test_dataset_missing_column_raises(tmp_path):
    # Write parquet without one required feature
    p = tmp_path / "bad.parquet"
    df = pd.DataFrame({"tx_amount_usd": [1.0], "is_fraud": [0]})
    df.to_parquet(p)
    with pytest.raises(AssertionError, match="Missing columns"):
        FraudDataset(str(p))