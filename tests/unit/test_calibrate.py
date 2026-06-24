"""Tests for model/calibrate.py Platt scaling calibration."""
import json
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.data.dataset import FEATURE_ORDER, LABEL
from src.model.fraud_mlp import FraudMLP
from model.calibrate import (
    apply_calibration,
    calibrate,
    fit_platt_scaling,
    get_raw_scores,
)


def _write_checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = FraudMLP()
    torch.save(m.state_dict(), path)
    return path


def _write_parquet(path: Path, n: int = 500) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    n_fraud = int(n * 0.05)
    data: dict[str, object] = {c: rng.standard_normal(n).astype("float32") for c in FEATURE_ORDER}
    data[LABEL] = np.array([1] * n_fraud + [0] * (n - n_fraud), dtype="int8")
    data["orig_currency"] = ["USD"] * n
    data["stale_fx_flag"] = [0] * n
    pd.DataFrame(data).to_parquet(path, index=False)
    return path


def test_fit_platt_scaling_returns_calibrator():
    probs = np.random.default_rng(0).random(200)
    labels = (probs > 0.5).astype(float)
    calibrator = fit_platt_scaling(probs, labels)
    assert hasattr(calibrator, "predict_proba")
    assert hasattr(calibrator, "coef_")


def test_get_raw_scores_shapes(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "checkpoints" / "round_010.pt")
    data = _write_parquet(tmp_path / "data.parquet", n=400)
    model = FraudMLP()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    probs, labels = get_raw_scores(model, str(data), val_split=0.2)
    assert len(probs) == 80
    assert len(labels) == 80
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_calibrate_writes_json(tmp_path):
    ckpt = _write_checkpoint(tmp_path / "checkpoints" / "round_010.pt")
    data = _write_parquet(tmp_path / "data.parquet", n=400)
    out_dir = tmp_path / "checkpoints"
    result = calibrate(str(ckpt), str(data), out_dir=str(out_dir))
    calib_file = out_dir / "calibration_params.json"
    assert calib_file.exists()
    loaded = json.loads(calib_file.read_text())
    assert "platt_coef" in loaded
    assert "platt_intercept" in loaded
    assert loaded["n_val_samples"] == 60  # 15% of 400


def test_apply_calibration(tmp_path):
    # Write a known calibration file
    calib = {
        "platt_coef": 1.0,
        "platt_intercept": 0.0,
        "n_val_samples": 100,
        "val_fraud_rate": 0.05,
    }
    calib_path = tmp_path / "calibration_params.json"
    calib_path.write_text(json.dumps(calib))
    # sigmoid(1.0 * 0.5 + 0.0) is about 0.622.
    result = apply_calibration(0.5, calib_path=str(calib_path))
    assert abs(result - 0.622) < 0.01


def test_apply_calibration_bounds(tmp_path):
    calib = {"platt_coef": 1.0, "platt_intercept": 0.0,
             "n_val_samples": 100, "val_fraud_rate": 0.05}
    calib_path = tmp_path / "calibration_params.json"
    calib_path.write_text(json.dumps(calib))
    for raw in [0.0, 0.5, 1.0]:
        result = apply_calibration(raw, calib_path=str(calib_path))
        assert 0.0 <= result <= 1.0
