"""Tests for the FastAPI inference gateway."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from src.data.dataset import FEATURE_ORDER


VALID_PAYLOAD = {name: 0.0 for name in FEATURE_ORDER} | {
    "tx_amount_usd": 85.0,
    "tx_count_1h": 1.0,
    "tx_count_24h": 4.0,
    "tx_volume_1h_usd": 85.0,
    "tx_volume_24h_usd": 340.0,
    "geo_velocity_kmh": 5.0,
    "dist2_km": 2.0,
    "days_since_last_tx": 3.0,
    "account_age_days": 1200.0,
    "hour_of_day_local": 10.0,
    "day_of_week": 1.0,
    "tx_time_norm": 0.42,
    "week_of_period": 0.12,
    "prod_H": 1.0,
    "card1_norm": 2.0,
    "card2_norm": 1.0,
    "addr1_norm": 50.0,
    "addr2_norm": 3.0,
    "email_domain_match": 1.0,
    "p_email_free": 1.0,
    "r_email_free": 1.0,
    "card3_norm": 25.0,
    "card4_code": 1.0,
    "orig_currency": "USD",
    "stale_fx_flag": 0,
}

NORM_PARAMS = {
    "tx_amount_usd": {"mean": 98.0, "std": 297.0},
    "tx_count_1h": {"mean": 2.1, "std": 1.6},
    "tx_count_24h": {"mean": 8.4, "std": 4.1},
    "tx_volume_1h_usd": {"mean": 297.0, "std": 540.0},
    "tx_volume_24h_usd": {"mean": 3281.0, "std": 9589.0},
    "geo_velocity_kmh": {"mean": 18.9, "std": 46.6},
    "dist2_km": {"mean": 3.2, "std": 4.9},
    "days_since_last_tx": {"mean": 2.96, "std": 3.0},
    "account_age_days": {"mean": 1809.0, "std": 1062.0},
}


def _make_client(tmp_path: Path):
    from api.main import app
    import api.main as main_mod
    from src.model.fraud_mlp import FraudMLP

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    model = FraudMLP()
    torch.save(model.state_dict(), ckpt_dir / "round_010.pt")

    norm_path = tmp_path / "config" / "normalization_params.json"
    norm_path.parent.mkdir(parents=True)
    norm_path.write_text(json.dumps(NORM_PARAMS))

    main_mod.CHECKPOINT_DIR = ckpt_dir
    main_mod.NORM_PARAMS_PATH = norm_path
    main_mod.RESULTS_DIR = tmp_path / "results"
    main_mod.RESULTS_DIR.mkdir()
    (
        main_mod._model,
        main_mod._norm_params,
        main_mod._model_version,
        main_mod._model_record,
        main_mod._decision_threshold,
    ) = main_mod._load_latest_model()
    return TestClient(app, raise_server_exceptions=True)


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


def test_predict_rejects_stale_schema_field(tmp_path):
    client = _make_client(tmp_path)
    bad = {**VALID_PAYLOAD, "M4_flag": 1.0}
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


def test_models_endpoint_lists_checkpoints(tmp_path):
    client = _make_client(tmp_path)
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    assert data["models"][0]["checkpoint"] == "round_010.pt"


def test_select_model_loads_checkpoint(tmp_path):
    client = _make_client(tmp_path)
    resp = client.post("/models/select", json={"checkpoint": "round_010.pt"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "loaded"
    assert data["model_version"] == "round_010"


def test_predict_demo_converts_static_currency(tmp_path):
    client = _make_client(tmp_path)
    resp = client.post(
        "/predict-demo",
        json={
            "amount": 100,
            "currency": "EUR",
            "use_live_fx": False,
            "product": "W",
            "card_type": "debit",
            "card_brand": "visa",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["metadata"]["orig_currency"] == "EUR"
    assert data["metadata"]["amount_usd"] == 108.2
    assert data["metadata"]["fx_source"] == "static"
