"""Tests for data/loader.py utility functions."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from client.dataset import FEATURE_ORDER, LABEL


def _write_raw(base: Path, client_id: int, n: int = 100) -> Path:
    rng = np.random.default_rng(client_id)
    path = base / f"client_{client_id}" / "transactions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {c: rng.standard_normal(n).astype("float32") for c in FEATURE_ORDER}
    data[LABEL] = rng.integers(0, 2, n).astype("int8")
    data["orig_currency"] = ["USD"] * n
    data["stale_fx_flag"] = [0] * n
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def _write_processed(base: Path, client_id: int, n: int = 200) -> Path:
    rng = np.random.default_rng(client_id + 10)
    path = base / f"client_{client_id}" / "transactions_normalized.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {c: rng.standard_normal(n).astype("float32") for c in FEATURE_ORDER}
    data[LABEL] = rng.integers(0, 2, n).astype("int8")
    data["orig_currency"] = ["USD"] * n
    data["stale_fx_flag"] = [0] * n
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def test_load_raw_returns_dataframe(tmp_path, monkeypatch):
    raw_base = tmp_path / "raw"
    _write_raw(raw_base, client_id=0)
    from data import loader
    monkeypatch.setattr(loader, "load_raw",
        lambda cid, base=str(raw_base): pd.read_parquet(
            Path(base) / f"client_{cid}" / "transactions.parquet"))
    df = loader.load_raw(0, base=str(raw_base))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100


def test_load_processed_returns_dataframe(tmp_path, monkeypatch):
    proc_base = tmp_path / "processed"
    _write_processed(proc_base, client_id=1)
    from data import loader
    df = loader.load_processed(1, base=str(proc_base))
    assert isinstance(df, pd.DataFrame)
    assert LABEL in df.columns


def test_load_reference_window_size(tmp_path):
    proc_base = tmp_path / "processed"
    _write_processed(proc_base, client_id=0, n=500)
    from data import loader
    ref = loader.load_reference_window(0, n=100, base=str(proc_base))
    assert len(ref) == 100


def test_load_raw_missing_raises(tmp_path):
    from data import loader
    with pytest.raises(FileNotFoundError):
        loader.load_raw(99, base=str(tmp_path / "nonexistent"))


def test_load_processed_missing_raises(tmp_path):
    from data import loader
    with pytest.raises(FileNotFoundError):
        loader.load_processed(99, base=str(tmp_path / "nonexistent"))