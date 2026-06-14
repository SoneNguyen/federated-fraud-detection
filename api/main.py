"""FastAPI inference gateway for the Federated Fraud Detection model.

Endpoints:
    GET  /health          — liveness check
    POST /predict         — score a single transaction
    POST /reload          — hot-reload the latest model checkpoint
    GET  /model-version   — return the currently loaded checkpoint name
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException

from api.middleware import AccessLogMiddleware, RateLimitMiddleware
from api.schemas import FEATURE_ORDER, Prediction, PredictionMetadata, Transaction
from src.model.fraud_mlp import FraudMLP

logger = logging.getLogger("api.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _norm_params, _model_version
    try:
        _model, _norm_params, _model_version = _load_latest_model()
    except FileNotFoundError as e:
        logger.warning("Startup: %s", e)
    yield

app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(AccessLogMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=1000, window_seconds=60)

# ── module-level state ────────────────────────────────────────────────────────
_model: Optional[FraudMLP] = None
_norm_params: dict = {}
_model_version: str = "not_loaded"

CHECKPOINT_DIR   = Path("checkpoints")
NORM_PARAMS_PATH = Path("config/normalization_params.json")
NUMERIC_COLS     = [
    "tx_amount_usd", "tx_count_1h", "tx_count_24h",
    "tx_volume_1h_usd", "tx_volume_24h_usd", "merchant_cat_dev",
    "geo_velocity_kmh", "dist2_km", "card6_code",
    "days_since_last_tx", "account_age_days",
]


def _load_latest_model() -> tuple[FraudMLP, dict, str]:
    """Find the latest checkpoint, load weights, and return model + norm params + version."""
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))
    # Exclude rollback file from version selection
    checkpoints = [c for c in checkpoints if c.name != "rollback_active.pt"]

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {CHECKPOINT_DIR.resolve()}. "
            "Run FL training first."
        )

    # Prefer the newest compatible checkpoint. Some checkpoint files may exist
    # in the directory with the wrong format or partial state dicts.
    for checkpoint_path in sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model = FraudMLP()
            model.load_state_dict(state_dict)
            model.eval()
            latest = checkpoint_path
            break
        except (RuntimeError, ValueError) as exc:
            logger.warning(
                "Skipping incompatible checkpoint %s: %s",
                checkpoint_path.name,
                exc,
            )
    else:
        raise FileNotFoundError(
            f"No compatible checkpoints found in {CHECKPOINT_DIR.resolve()}. "
            "Ensure the directory contains valid FraudMLP state_dict files."
        )

    if not NORM_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"Normalization params not found at {NORM_PARAMS_PATH}. "
            "Run data/load_ieee_cis.py first."
        )
    with open(NORM_PARAMS_PATH) as f:
        norm = json.load(f)

    logger.info("Loaded model checkpoint: %s", latest.name)
    return model, norm, latest.stem


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_version": _model_version}


@app.get("/model-version")
async def model_version() -> dict:
    return {"model_version": _model_version}


@app.post("/reload")
async def reload() -> dict:
    """Hot-reload the latest checkpoint from disk (called after rollback)."""
    global _model, _norm_params, _model_version
    try:
        _model, _norm_params, _model_version = _load_latest_model()
        return {"status": "reloaded", "model_version": _model_version}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=Prediction)
async def predict(tx: Transaction) -> Prediction:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check that training has completed and "
                   "a checkpoint exists in the checkpoints/ directory.",
        )

    # Build feature vector in schema-defined order
    raw_vals = tx.model_dump()

    # Normalize numeric features using federated global stats
    features: list[float] = []
    for col in FEATURE_ORDER:
        if col not in raw_vals:
            raise HTTPException(status_code=400, detail=f"Missing feature {col}")
        v = float(raw_vals[col])
        if col in _norm_params:
            v = (v - _norm_params[col]["mean"]) / _norm_params[col]["std"]
        features.append(v)

    x = torch.tensor([features], dtype=torch.float32, device=_model.device)
    with torch.no_grad():
        prob = float(torch.sigmoid(_model(x)).squeeze())

    return Prediction(
        fraud_probability=round(prob, 6),
        prediction=int(prob >= 0.5),
        model_version=_model_version,
        metadata=PredictionMetadata(
            stale_fx_flag=tx.stale_fx_flag or 0,
            orig_currency=tx.orig_currency or "USD",
        ),
    )