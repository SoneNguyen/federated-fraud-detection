# this file defines the pydantic schemas for the API inputs and outputs, as well as the expected feature order for the model.
from pydantic import BaseModel, field_validator
from typing import Optional

from src.data.dataset import FEATURE_ORDER


class Transaction(BaseModel):
    tx_amount_usd:       float
    tx_count_1h:         int
    tx_count_24h:        int
    tx_volume_1h_usd:    float
    tx_volume_24h_usd:   float
    geo_velocity_kmh:    float
    dist2_km:            float
    card6_code:          int
    days_since_last_tx:  float
    account_age_days:    int
    hour_of_day_local:   int
    day_of_week:         int
    tx_time_norm:        float
    week_of_period:      float
    prod_W:              float
    prod_H:              float
    prod_C:              float
    prod_S:              float
    prod_R:              float
    card1_norm:          float
    card2_norm:          float
    addr1_norm:          float
    addr2_norm:          float
    V258:                float
    V257:                float
    V201:                float
    M4_flag:             float
    M6_flag:             float
    c5_chargeback:       float
    email_domain_match:  float
    p_email_free:        float
    r_email_free:        float
    card3_norm:          float
    card4_code:          int

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

    @field_validator("card6_code")
    @classmethod
    def v_card6_code(cls, v: int) -> int:
        if not (0 <= v <= 4):
            raise ValueError("card6_code must be 0-4")
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