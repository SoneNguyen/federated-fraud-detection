from pydantic import BaseModel, validator
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

    @validator("hour_of_day_local")
    def v_hour(cls, v):
        assert 0 <= v <= 23, "hour_of_day_local must be 0-23"
        return v

    @validator("day_of_week")
    def v_dow(cls, v):
        assert 0 <= v <= 6, "day_of_week must be 0-6"
        return v

    @validator("tx_amount_usd")
    def v_amt(cls, v):
        assert v >= 0, "tx_amount_usd must be non-negative"
        return v

    @validator("stale_fx_flag")
    def v_stale(cls, v):
        assert v in (0, 1), "stale_fx_flag must be 0 or 1"
        return v

class PredictionMetadata(BaseModel):
    stale_fx_flag: int
    orig_currency: str

class Prediction(BaseModel):
    fraud_probability: float
    prediction:        int
    model_version:     str
    metadata:          PredictionMetadata  # v3: separated from model outputs

# api/main.py — feature extraction uses v3 FEATURE_ORDER
FEATURE_ORDER = ["tx_amount_usd","tx_count_1h","tx_count_24h",
                 "tx_volume_1h_usd","tx_volume_24h_usd","merchant_cat_dev",
                 "geo_velocity_kmh","days_since_last_tx","account_age_days",
                 "hour_of_day_local","day_of_week"]  # 11 — matches schema