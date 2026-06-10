"""Tests for the FastAPI inference gateway."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient


# ── helpers ───────────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "tx_amount_usd": 85.0,
    "tx_count_1h": 1,
    "tx_count_24h": 4,
    "tx_volume_1h_usd": 85.0,
    "tx_volume_24h_usd": 340.0,
    "merchant_cat_dev": -0.1,
    "geo_velocity_kmh": 5.0,
    "dist2_km": 2.0,
    "card6_code": 1,
    "days_since_last_tx": 3.0,
    "account_age_days": 1200,
    "hour_of_day_local": 10,
    "day_of_week": 1,
    "orig_currency": "USD",
    "stale_fx_flag": 0,
}

NORM_PARAMS = {
    "tx_amount_usd":      {"mean": 98.0,   "std": 297.0},
    "tx_count_1h":        {"mean": 2.1,    "std": 1.6},
    "tx_count_24h":       {"mean": 8.4,    "std": 4.1},
    "tx_volume_1h_usd":   {"mean": 297.0,  "std": 540.0},
    "tx_volume_24h_usd":  {"mean": 3281.0, "std": 9589.0},
    "merchant_cat_dev":   {"mean": 0.04,   "std": 1.06},
    "geo_velocity_kmh":   {"mean": 18.9,   "std": 46.6},
    "dist2_km":           {"mean": 3.2,    "std": 4.9},
    "card6_code":         {"mean": 1.3,    "std": 1.1},
    "days_since_last_tx": {"mean": 2.96,   "std": 3.0},
    "account_age_days":   {"mean": 1809.0, "std": 1062.0},
}


def _make_client(tmp_path: Path, fraud_prob: float = 0.05):
    """Create a TestClient with a mocked model and checkpoint."""
    from client.model import FraudMLP

    # Write a real checkpoint so _load_latest_model() finds it
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    model = FraudMLP()
    torch.save(model.state_dict(), ckpt_dir / "round_010.pt")

    # Write normalization params
    norm_path = tmp_path / "contracts" / "normalization_params.json"
    norm_path.parent.mkdir(parents=True)
    norm_path.write_text(json.dumps(NORM_PARAMS))

    with patch("api.main.CHECKPOINT_DIR", ckpt_dir), \
         patch("api.main.NORM_PARAMS_PATH", norm_path):
        from api.main import app
        import api.main as main_mod
        # Force reload with patched paths
        main_mod._model, main_mod._norm_params, main_mod._model_version = \
            main_mod._load_latest_model()
        client = TestClient(app, raise_server_exceptions=True)
        return client


# ── tests ─────────────────────────────────────────────────────────────────────

def test_health_returns_ok(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_valid_payload(tmp_path):
    client = _make_client(tmp_path)
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["prediction"] in (0, 1)
    assert "model_version" in data
    assert "metadata" in data
    assert data["metadata"]["orig_currency"] == "USD"
    assert data["metadata"]["stale_fx_flag"] == 0


def test_predict_invalid_hour(tmp_path):
    client = _make_client(tmp_path)
    bad = {**VALID_PAYLOAD, "hour_of_day_local": 99}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_dow(tmp_path):
    client = _make_client(tmp_path)
    bad = {**VALID_PAYLOAD, "day_of_week": 7}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_negative_amount(tmp_path):
    client = _make_client(tmp_path)
    bad = {**VALID_PAYLOAD, "tx_amount_usd": -1.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_stale_fx_propagated(tmp_path):
    client = _make_client(tmp_path)
    payload = {**VALID_PAYLOAD, "stale_fx_flag": 1, "orig_currency": "EUR"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    meta = resp.json()["metadata"]
    assert meta["stale_fx_flag"] == 1
    assert meta["orig_currency"] == "EUR"


def test_model_version_endpoint(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/model-version")
    assert resp.status_code == 200
    assert "model_version" in resp.json()