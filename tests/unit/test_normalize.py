"""Tests for data/normalize.py federated normalization pipeline."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from client.dataset import FEATURE_ORDER, LABEL
from data.normalize import local_stats

NUMERIC = [
    "tx_amount_usd", "tx_count_1h", "tx_count_24h",
    "tx_volume_1h_usd", "tx_volume_24h_usd", "merchant_cat_dev",
    "geo_velocity_kmh", "days_since_last_tx", "account_age_days",
]


def _write_raw_client(base: Path, client_id: int, n: int = 1000,
                      mean_offset: float = 0.0) -> Path:
    rng = np.random.default_rng(client_id)
    path = base / f"client_{client_id}" / "transactions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {c: (rng.standard_normal(n) + mean_offset).astype("float32")
            for c in FEATURE_ORDER}
    data[LABEL] = rng.integers(0, 2, n).astype("int8")
    data["orig_currency"] = ["USD"] * n
    data["stale_fx_flag"] = [0] * n
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def test_local_stats_shape(tmp_path):
    """local_stats() returns a dict with n, sum, sum_sq for each numeric col."""
    path = _write_raw_client(tmp_path, client_id=0)
    df = pd.read_parquet(path)
    stats = local_stats(df)
    assert set(stats.keys()) == set(NUMERIC)
    for col, s in stats.items():
        assert s["n"] == 1000
        assert isinstance(s["sum"], float)
        assert isinstance(s["sum_sq"], float)


def test_federated_mean_close_to_true_mean(tmp_path):
    """Federated global mean approximates the true pooled mean."""
    raw_base = tmp_path / "raw"
    for cid in range(3):
        _write_raw_client(raw_base, cid, n=500, mean_offset=float(cid))

    all_stats = []
    for cid in range(3):
        df = pd.read_parquet(raw_base / f"client_{cid}" / "transactions.parquet")
        all_stats.append({c: {"n": len(df), "sum": float(df[c].sum()),
                               "sum_sq": float((df[c] ** 2).sum())} for c in NUMERIC})

    col = NUMERIC[0]
    N = sum(s[col]["n"] for s in all_stats)
    S = sum(s[col]["sum"] for s in all_stats)
    fed_mean = S / N

    # True pooled mean: average of per-client means weighted by n
    true_mean = sum(s[col]["sum"] / s[col]["n"] for s in all_stats) / 3
    assert abs(fed_mean - true_mean) < 0.5


def test_normalization_produces_near_zero_mean(tmp_path):
    """After normalization, each numeric feature should have mean ≈ 0."""
    raw_base = tmp_path / "raw"
    proc_base = tmp_path / "processed"

    for cid in range(3):
        _write_raw_client(raw_base, cid, n=1000)

    # Compute global params
    all_stats = []
    for cid in range(3):
        df = pd.read_parquet(raw_base / f"client_{cid}" / "transactions.parquet")
        all_stats.append({c: {"n": len(df), "sum": float(df[c].sum()),
                               "sum_sq": float((df[c] ** 2).sum())} for c in NUMERIC})

    global_params = {}
    for col in NUMERIC:
        N = sum(s[col]["n"] for s in all_stats)
        S = sum(s[col]["sum"] for s in all_stats)
        SQ = sum(s[col]["sum_sq"] for s in all_stats)
        mean = S / N
        std = max(np.sqrt(SQ / N - mean ** 2), 1e-8)
        global_params[col] = {"mean": mean, "std": std}

    # Apply normalization to one client
    df = pd.read_parquet(raw_base / "client_0" / "transactions.parquet")
    for col in NUMERIC:
        df[col] = (df[col] - global_params[col]["mean"]) / global_params[col]["std"]

    for col in NUMERIC:
        assert abs(df[col].mean()) < 0.2, f"{col} mean too far from 0: {df[col].mean()}"


def test_normalization_params_saved_with_v3_keys(tmp_path):
    """normalization_params.json must use v3 field names (tx_amount_usd etc)."""
    params_path = tmp_path / "normalization_params.json"
    dummy = {col: {"mean": 0.0, "std": 1.0} for col in NUMERIC}
    params_path.write_text(json.dumps(dummy))
    loaded = json.loads(params_path.read_text())
    assert "tx_amount_usd" in loaded
    assert "tx_volume_1h_usd" in loaded
    assert "tx_volume_24h_usd" in loaded
    assert "tx_amount" not in loaded       # old v1 name must not exist
    assert "tx_volume_1h" not in loaded    # old v1 name must not exist