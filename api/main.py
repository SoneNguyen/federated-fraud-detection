"""FastAPI inference gateway for federated fraud detection."""

from __future__ import annotations

import json
import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import AccessLogMiddleware, RateLimitMiddleware
from api.model_registry import (
    list_model_records,
    metrics_for_checkpoint,
)
from api.schemas import (
    DemoTransaction,
    FEATURE_ORDER,
    Prediction,
    PredictionMetadata,
    Transaction,
)
from data.fx.converter import FXConversionError, FXConverter
from src.model.fraud_mlp import FraudMLP

logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _norm_params, _model_version, _model_record, _decision_threshold
    try:
        _model, _norm_params, _model_version, _model_record, _decision_threshold = (
            _load_recommended_model()
        )
    except FileNotFoundError as exc:
        logger.warning("Startup: %s", exc)
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "API_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=1000, window_seconds=60)

_model: Optional[FraudMLP] = None
_norm_params: dict = {}
_model_version: str = "not_loaded"
_model_record: dict = {}
_decision_threshold: float = 0.5
_fx = FXConverter()

CHECKPOINT_DIR = Path("outputs/checkpoints")
NORM_PARAMS_PATH = Path("config/normalization_params.json")
RESULTS_DIR = Path("results")


def _load_checkpoint(checkpoint_path: Path) -> tuple[FraudMLP, dict, str, dict, float]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = FraudMLP()
    model.load_state_dict(state_dict)
    model.eval()

    if not NORM_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Normalization params not found at {NORM_PARAMS_PATH}. "
            "Run data/load_ieee_cis.py first."
        )
    with open(NORM_PARAMS_PATH) as f:
        norm = json.load(f)

    record = metrics_for_checkpoint(checkpoint_path, RESULTS_DIR)
    threshold = float(record.get("threshold", 0.5))
    logger.info("MODEL loaded=%s threshold=%.4f", checkpoint_path.name, threshold)
    return model, norm, checkpoint_path.stem, record, threshold


def _load_recommended_model() -> tuple[FraudMLP, dict, str, dict, float]:
    records = list_model_records(CHECKPOINT_DIR, RESULTS_DIR, limit=50)
    for record in records:
        checkpoint_path = CHECKPOINT_DIR / record["checkpoint"]
        try:
            return _load_checkpoint(checkpoint_path)
        except (RuntimeError, ValueError) as exc:
            logger.warning("MODEL skip=%s reason=%s", checkpoint_path.name, exc)
    return _load_latest_model()


def _load_latest_model() -> tuple[FraudMLP, dict, str, dict, float]:
    checkpoints = sorted(
        p for p in CHECKPOINT_DIR.glob("*.pt") if p.name != "rollback_active.pt"
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {CHECKPOINT_DIR.resolve()}. Run FL training first."
        )

    for checkpoint_path in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            return _load_checkpoint(checkpoint_path)
        except (RuntimeError, ValueError) as exc:
            logger.warning("MODEL skip=%s reason=%s", checkpoint_path.name, exc)

    raise FileNotFoundError(
        f"No compatible checkpoints found in {CHECKPOINT_DIR.resolve()}."
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_version": _model_version,
        "threshold": _decision_threshold,
    }


@app.get("/model-version")
async def model_version() -> dict:
    return {
        "model_version": _model_version,
        "threshold": _decision_threshold,
        "model": _model_record,
    }


@app.post("/reload")
async def reload() -> dict:
    global _model, _norm_params, _model_version, _model_record, _decision_threshold
    try:
        _model, _norm_params, _model_version, _model_record, _decision_threshold = (
            _load_recommended_model()
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "status": "reloaded",
        "model_version": _model_version,
        "threshold": _decision_threshold,
    }


@app.get("/models")
async def models() -> dict:
    selected = f"{_model_version}.pt" if _model_version != "not_loaded" else None
    records = list_model_records(CHECKPOINT_DIR, RESULTS_DIR, selected_name=selected)
    return {
        "selected": _model_version,
        "threshold": _decision_threshold,
        "count": len(records),
        "models": records,
    }


@app.post("/models/select")
async def select_model(payload: dict) -> dict:
    global _model, _norm_params, _model_version, _model_record, _decision_threshold
    checkpoint = str(payload.get("checkpoint", "")).strip()
    if not checkpoint:
        raise HTTPException(status_code=400, detail="checkpoint is required")

    candidate = CHECKPOINT_DIR / Path(checkpoint).name
    if candidate.name != checkpoint or not candidate.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint}")

    try:
        _model, _norm_params, _model_version, _model_record, _decision_threshold = (
            _load_checkpoint(candidate)
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Checkpoint is incompatible: {exc}")

    return {
        "status": "loaded",
        "model_version": _model_version,
        "threshold": _decision_threshold,
        "model": _model_record,
    }


@app.post("/predict", response_model=Prediction)
async def predict(tx: Transaction) -> Prediction:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Check that training has completed and a "
                "checkpoint exists in outputs/checkpoints."
            ),
        )

    prob = _score_transaction(tx)
    threshold = _decision_threshold
    return Prediction(
        fraud_probability=round(prob, 6),
        prediction=int(prob >= threshold),
        model_version=_model_version,
        risk_band=_risk_band(prob, threshold),
        decision_label=_decision_label(prob, threshold),
        metadata=PredictionMetadata(
            stale_fx_flag=tx.stale_fx_flag or 0,
            orig_currency=tx.orig_currency or "USD",
            threshold=round(threshold, 6),
            model_score_source=_model_record.get("reason"),
        ),
    )


@app.post("/predict-demo", response_model=Prediction)
async def predict_demo(tx: DemoTransaction) -> Prediction:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        fx = await _fx.to_usd_live(tx.amount, tx.currency, use_live=tx.use_live_fx)
    except FXConversionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    full_tx = Transaction(**_demo_payload(tx, fx))
    prob = _score_transaction(full_tx)
    threshold = _decision_threshold
    return Prediction(
        fraud_probability=round(prob, 6),
        prediction=int(prob >= threshold),
        model_version=_model_version,
        risk_band=_risk_band(prob, threshold),
        decision_label=_decision_label(prob, threshold),
        metadata=PredictionMetadata(
            stale_fx_flag=int(fx["stale"]),
            orig_currency=tx.currency,
            threshold=round(threshold, 6),
            amount_usd=float(fx["amount_usd"]),
            fx_rate=float(fx["rate"]),
            fx_source=str(fx["source"]),
            model_score_source=_model_record.get("reason"),
        ),
    )


def _score_transaction(tx: Transaction) -> float:
    raw_vals = tx.model_dump()
    features: list[float] = []
    for col in FEATURE_ORDER:
        if col not in raw_vals:
            raise HTTPException(status_code=400, detail=f"Missing feature {col}")
        v = float(raw_vals[col])
        if col in _norm_params:
            std = max(float(_norm_params[col]["std"]), 1e-8)
            v = (v - float(_norm_params[col]["mean"])) / std
        features.append(v)

    assert _model is not None
    x = torch.tensor([features], dtype=torch.float32, device=_model.device)
    with torch.no_grad():
        return float(torch.sigmoid(_model(x)).squeeze())


def _demo_payload(tx: DemoTransaction, fx: dict) -> dict:
    adv = tx.advanced
    amount_usd = max(float(fx["amount_usd"]), 0.0)
    log_amount = math.log1p(amount_usd)
    count_1h = max(float(adv.tx_count_1h), 0.0)
    count_24h = max(float(adv.tx_count_24h), count_1h)
    log_count_1h = math.log1p(count_1h)
    log_count_24h = math.log1p(count_24h)
    velocity = max(float(adv.geo_velocity_kmh), 0.0)
    log_velocity = math.log1p(velocity)
    history_count = max(float(adv.history_count), 0.0)
    fraud_rate = max(0.0, min(1.0, float(adv.history_fraud_rate)))
    prior_fraud_count = max(float(adv.prior_fraud_count), 0.0)
    identity_missing = max(0.0, min(1.0, float(adv.identity_missing_rate)))

    payload = _neutral_payload()
    payload.update(
        {
            "tx_amount_usd": log_amount,
            "tx_count_1h": log_count_1h,
            "tx_count_24h": log_count_24h,
            "tx_volume_1h_usd": math.log1p(min(amount_usd * count_1h, 5e8)),
            "tx_volume_24h_usd": math.log1p(min(amount_usd * count_24h, 5e9)),
            "geo_velocity_kmh": log_velocity,
            "dist2_km": math.log1p(max(float(adv.distance_km), 0.0)),
            "amount_x_velocity": min((log_amount * log_velocity) / 10.0, 10.0),
            "amount_per_tx_1h": max(
                -5.0, min(10.0, log_amount - math.log1p(count_1h + 0.1))
            ),
            "amount_per_tx_24h": max(
                -5.0, min(10.0, log_amount - math.log1p(count_24h + 0.1))
            ),
            "spending_velocity_1h": max(
                0.0, min(10.0, log_count_1h * 0.1 + log_amount * 0.9)
            ),
            "days_since_last_tx": math.log1p(min(float(adv.days_since_last_tx), 365.0)),
            "account_age_days": math.log1p(min(float(adv.account_age_days), 10000.0)),
            "hour_of_day_local": float(tx.hour_of_day_local),
            "day_of_week": float(tx.day_of_week),
            "tx_time_norm": float(tx.hour_of_day_local) / 24.0,
            "week_of_period": (
                float(tx.day_of_week) * 24.0 + float(tx.hour_of_day_local)
            )
            / 168.0,
            "risky_hour_flag": float(tx.hour_of_day_local < 6 or tx.hour_of_day_local > 23),
            "early_morning_high_value": float((tx.hour_of_day_local < 6) and log_amount > 5),
            "weekend_high_value": float(tx.day_of_week in {0, 5, 6} and log_amount > 4),
            "email_domain_match": float(tx.email_domain_match),
            "p_email_free": float(tx.payer_free_email),
            "r_email_free": float(tx.receiver_free_email),
            "both_emails_free": float(tx.payer_free_email and tx.receiver_free_email),
            "email_mismatch_high_value": float((not tx.email_domain_match) and log_amount > 4.5),
            "has_device_info": float(adv.device_present),
            "device_type_mobile": float(adv.mobile_device),
            "device_type_desktop": float(not adv.mobile_device),
            "card_device_mismatch": float(adv.card_device_mismatch),
            "new_account_high_value": float(float(adv.account_age_days) < 7 and log_amount > 5),
            "c5_chargeback": math.log1p(min(float(adv.chargeback_count), 100.0)),
            "card1_freq": math.log1p(float(adv.merchant_frequency)),
            "card2_freq": math.log1p(float(adv.merchant_frequency)),
            "addr1_freq": math.log1p(float(adv.merchant_frequency)),
            "p_email_freq": math.log1p(float(adv.merchant_frequency)),
            "r_email_freq": math.log1p(float(adv.merchant_frequency)),
            "device_info_freq": (
                math.log1p(float(adv.merchant_frequency)) if adv.device_present else 0.0
            ),
            "orig_currency": tx.currency,
            "stale_fx_flag": int(fx["stale"]),
        }
    )

    for cat in ["W", "H", "C", "S", "R"]:
        payload[f"prod_{cat}"] = float(tx.product == cat)
    for name in ["debit", "credit", "charge_card", "debit_or_credit"]:
        payload[f"card6_{name}"] = 0.0
    payload[f"card6_{tx.card_type.replace(' ', '_')}"] = 1.0
    payload["card4_code"] = {
        "visa": 0.0,
        "mastercard": 1.0,
        "american express": 2.0,
        "discover": 3.0,
        "other": 4.0,
    }[tx.card_brand]

    for prefix in (
        "card1",
        "card2",
        "addr1",
        "p_email",
        "r_email",
        "device_info",
        "card_pair",
        "email_pair",
    ):
        payload[f"hist_{prefix}_count_log"] = math.log1p(history_count)
        payload[f"hist_{prefix}_fraud_count_log"] = math.log1p(prior_fraud_count)
        payload[f"hist_{prefix}_fraud_rate"] = fraud_rate
        payload[f"hist_{prefix}_amount_mean_log"] = log_amount
        payload[f"hist_{prefix}_amount_ratio"] = 0.0
        payload[f"hist_{prefix}_since_prev_log"] = math.log1p(
            min(float(adv.days_since_last_tx), 365.0) * 86400.0
        )
        payload[f"hist_{prefix}_since_prev_fraud_log"] = math.log1p(365.0 * 86400.0)

    for name in FEATURE_ORDER:
        if name.endswith("_missing") and name.startswith("id_"):
            payload[name] = identity_missing

    signal = float(adv.suspicious_identity_signal)
    for name in ("V258", "V257", "V201"):
        if name in payload:
            payload[name] = signal

    payload.update(tx.feature_overrides)
    return payload


def _neutral_payload() -> dict:
    payload = {
        name: float(_norm_params.get(name, {}).get("mean", 0.0))
        for name in FEATURE_ORDER
    }
    payload["orig_currency"] = "USD"
    payload["stale_fx_flag"] = 0
    return payload


def _risk_band(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "block"
    if prob >= max(0.55, threshold * 0.85):
        return "review"
    if prob >= max(0.25, threshold * 0.55):
        return "watch"
    return "clear"


def _decision_label(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "Block"
    if prob >= max(0.55, threshold * 0.85):
        return "Manual review"
    if prob >= max(0.25, threshold * 0.55):
        return "Monitor"
    return "Approve"
