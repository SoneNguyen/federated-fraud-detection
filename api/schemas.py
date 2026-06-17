"""Pydantic schemas for the inference API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from src.data.feature_registry import FEATURE_ORDER


class _TransactionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    orig_currency: Optional[str] = "USD"
    stale_fx_flag: Optional[int] = 0

    @field_validator("hour_of_day_local", check_fields=False)
    @classmethod
    def v_hour(cls, v: float) -> float:
        if not (0 <= v <= 23):
            raise ValueError("hour_of_day_local must be 0-23")
        return v

    @field_validator("day_of_week", check_fields=False)
    @classmethod
    def v_dow(cls, v: float) -> float:
        if not (0 <= v <= 6):
            raise ValueError("day_of_week must be 0-6")
        return v

    @field_validator("tx_amount_usd", check_fields=False)
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


Transaction = create_model(
    "Transaction",
    __base__=_TransactionBase,
    **{name: (float, ...) for name in FEATURE_ORDER},
)


class PredictionMetadata(BaseModel):
    stale_fx_flag: int
    orig_currency: str
    threshold: float = 0.5
    amount_usd: Optional[float] = None
    fx_rate: Optional[float] = None
    fx_source: Optional[str] = None
    model_score_source: Optional[str] = None


class Prediction(BaseModel):
    fraud_probability: float
    prediction: int
    model_version: str
    risk_band: str = "review"
    decision_label: str = "Review"
    metadata: PredictionMetadata


class DemoAdvancedOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tx_count_1h: float = Field(default=1.0, ge=0, le=500)
    tx_count_24h: float = Field(default=4.0, ge=0, le=5000)
    geo_velocity_kmh: float = Field(default=5.0, ge=0, le=2000)
    distance_km: float = Field(default=2.0, ge=0, le=20000)
    days_since_last_tx: float = Field(default=3.0, ge=0, le=365)
    account_age_days: float = Field(default=365.0, ge=0, le=10000)
    history_count: float = Field(default=2.0, ge=0, le=1_000_000)
    history_fraud_rate: float = Field(default=0.03, ge=0, le=1)
    prior_fraud_count: float = Field(default=0.0, ge=0, le=10000)
    chargeback_count: float = Field(default=0.0, ge=0, le=100)
    merchant_frequency: float = Field(default=25.0, ge=0, le=10_000_000)
    identity_missing_rate: float = Field(default=0.2, ge=0, le=1)
    device_present: bool = True
    mobile_device: bool = True
    card_device_mismatch: bool = False
    suspicious_identity_signal: float = Field(default=0.0, ge=0, le=1)


class DemoTransaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: float = Field(default=125.0, ge=0, le=1_000_000_000)
    currency: str = "USD"
    product: str = "W"
    card_type: str = "debit"
    card_brand: str = "visa"
    email_domain_match: bool = True
    payer_free_email: bool = True
    receiver_free_email: bool = True
    hour_of_day_local: int = Field(default=10, ge=0, le=23)
    day_of_week: int = Field(default=2, ge=0, le=6)
    use_live_fx: bool = True
    advanced: DemoAdvancedOptions = Field(default_factory=DemoAdvancedOptions)
    feature_overrides: dict[str, float] = Field(default_factory=dict)

    @field_validator("currency")
    @classmethod
    def v_currency(cls, v: str) -> str:
        code = (v or "USD").strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("currency must be a 3-letter ISO code")
        return code

    @field_validator("product")
    @classmethod
    def v_product(cls, v: str) -> str:
        product = (v or "W").strip().upper()
        if product not in {"W", "H", "C", "S", "R"}:
            raise ValueError("product must be W, H, C, S, or R")
        return product

    @field_validator("card_type")
    @classmethod
    def v_card_type(cls, v: str) -> str:
        value = (v or "debit").strip().lower().replace("-", " ")
        if value not in {"debit", "credit", "charge card", "debit or credit"}:
            raise ValueError("card_type is not supported")
        return value

    @field_validator("card_brand")
    @classmethod
    def v_card_brand(cls, v: str) -> str:
        value = (v or "visa").strip().lower()
        if value not in {"visa", "mastercard", "american express", "discover", "other"}:
            raise ValueError("card_brand is not supported")
        return value

    @model_validator(mode="after")
    def v_overrides(self) -> "DemoTransaction":
        bad = sorted(set(self.feature_overrides) - set(FEATURE_ORDER))
        if bad:
            raise ValueError(f"Unknown feature override(s): {', '.join(bad[:5])}")
        return self
