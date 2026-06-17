"""Tests for model/evaluate.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import FEATURE_ORDER, LABEL
from src.data.feature_registry import SCHEMA_VERSION
from src.model.fraud_mlp import FraudMLP
from model.evaluate import eval_model, load_test


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = FraudMLP()
    torch.save(model.state_dict(), path)
    return path


def _write_parquet(path: Path, n: int = 2000) -> Path:
    rng = np.random.default_rng(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_fraud = int(n * 0.05)
    data: dict[str, object] = {c: rng.standard_normal(n).astype("float32") for c in FEATURE_ORDER}
    data[LABEL] = np.array([1] * n_fraud + [0] * (n - n_fraud), dtype="int8")
    data["orig_currency"] = ["USD"] * n
    data["stale_fx_flag"] = [0] * n
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def test_load_test_returns_correct_shapes(tmp_path):
    p = _write_parquet(tmp_path / "data.parquet", n=1000)
    X, y = load_test(str(p), test_frac=0.2)
    assert X.shape[1] == len(FEATURE_ORDER)
    assert len(y) == 200
    assert X.dtype == torch.float32


def test_eval_model_returns_expected_keys(tmp_path):
    model = FraudMLP()
    model.eval()
    rng = np.random.default_rng(1)
    n = 500
    n_fraud = 25
    X = torch.tensor(rng.standard_normal((n, len(FEATURE_ORDER))), dtype=torch.float32)
    y = np.array([1] * n_fraud + [0] * (n - n_fraud))
    result = eval_model(model, X, y)
    assert "AUPRC" in result
    assert "AUROC" in result
    assert "F1_best" in result
    assert "threshold" in result
    assert "schema_version" in result
    assert 0.0 <= result["AUPRC"] <= 1.0
    assert 0.0 <= result["AUROC"] <= 1.0


def test_eval_model_schema_version(tmp_path):
    model = FraudMLP()
    X = torch.randn(100, len(FEATURE_ORDER))
    y = np.array([1] * 5 + [0] * 95)
    result = eval_model(model, X, y)
    assert result["schema_version"] == SCHEMA_VERSION


def test_load_test_split_fraction(tmp_path):
    p = _write_parquet(tmp_path / "data.parquet", n=1000)
    X, y = load_test(str(p), test_frac=0.30)
    assert len(y) == 300
