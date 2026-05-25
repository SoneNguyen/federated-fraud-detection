# this file defines the pydantic schemas for the API inputs and outputs, as well as the expected feature order for the model.
from pydantic import BaseModel, field_validator
from typing import Optional


class Transaction(BaseModel):
    tx_amount_usd:       float
    tx_count_1h:         int
    tx_count_24h:        int
    tx_volume_1h_usd:    float
    tx_volume_24h_usd:   float
    merchant_cat_dev:    float
    geo_velocity_kmh:    float
    days_since_last_tx:  float
    account_age_days:    int
    hour_of_day_local:   int
    day_of_week:         int

    # Passthrough metadata — not model inputs
    orig_currency:       Optional[str] = "USD"
    stale_fx_flag:       Optional[int] = 0

    @field_validator("hour_of_day_local")
    @classmethod
    def v_hour(cls, v: int) -> int:
        if not (0 <= v <= 23):
            raise ValueError("hour_of_day_local must be 0-23")
        return v

    @field_validator("day_of_week")
    @classmethod
    def v_dow(cls, v: int) -> int:
        if not (0 <= v <= 6):
            raise ValueError("day_of_week must be 0-6")
        return v

    @field_validator("tx_amount_usd")
    @classmethod
    def v_amt(cls, v: float) -> float:
        if v < 0:
            raise ValueError("tx_amount_usd must be non-negative")
        return v

    @field_validator("stale_fx_flag")
    @classmethod
    def v_stale(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("stale_fx_flag must be 0 or 1")
        return v


class PredictionMetadata(BaseModel):
    stale_fx_flag: int
    orig_currency: str


class Prediction(BaseModel):
    fraud_probability: float
    prediction:        int
    model_version:     str
    metadata:          PredictionMetadata


FEATURE_ORDER = [
    "tx_amount_usd", "tx_count_1h", "tx_count_24h",
    "tx_volume_1h_usd", "tx_volume_24h_usd", "merchant_cat_dev",
    "geo_velocity_kmh", "days_since_last_tx", "account_age_days",
    "hour_of_day_local", "day_of_week",
]  # 11 — matches schema