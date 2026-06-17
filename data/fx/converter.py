"""Currency conversion helpers for inference demos.

The live provider is Frankfurter's no-key API. Static rates stay available as a
fallback so prediction never depends on external network health.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is on sys.path so absolute package imports work
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx

from data.fx.rates import STATIC_RATES

FX_CACHE_TTL = 3600
FRANKFURTER_RATES_URL = "https://api.frankfurter.dev/v2/rates"


class FXConversionError(ValueError):
    """Raised when no live or fallback rate can convert a currency."""


class FXConverter:
    def __init__(self, rates: dict[str, float] | None = None, cache_ts: float | None = None):
        self._rates = rates or STATIC_RATES
        self._cache_ts = cache_ts or time.time()
        self._live_rates: dict[str, tuple[float, float]] = {}

    def to_usd(self, amount: float, currency: str) -> tuple[float, int]:
        currency = self._normalize_currency(currency)
        rate = self._rates.get(currency)
        if rate is None:
            raise FXConversionError(f"Unsupported currency: {currency}")
        stale = int((time.time() - self._cache_ts) > FX_CACHE_TTL)
        return round(amount * rate, 2), stale

    async def to_usd_live(
        self,
        amount: float,
        currency: str,
        *,
        timeout_seconds: float = 2.5,
        use_live: bool = True,
    ) -> dict[str, Any]:
        """Convert to USD with live-rate cache and static fallback."""
        currency = self._normalize_currency(currency)
        if amount < 0:
            raise FXConversionError("Amount must be non-negative")
        if currency == "USD":
            return {
                "amount_usd": round(amount, 2),
                "rate": 1.0,
                "source": "identity",
                "stale": 0,
                "currency": currency,
            }

        now = time.time()
        if use_live:
            cached = self._live_rates.get(currency)
            if cached and now - cached[1] <= FX_CACHE_TTL:
                rate = cached[0]
                return {
                    "amount_usd": round(amount * rate, 2),
                    "rate": rate,
                    "source": "frankfurter-cache",
                    "stale": 0,
                    "currency": currency,
                }

            try:
                rate = await self._fetch_live_rate(currency, timeout_seconds)
            except Exception:
                rate = None
            if rate is not None:
                self._live_rates[currency] = (rate, now)
                return {
                    "amount_usd": round(amount * rate, 2),
                    "rate": rate,
                    "source": "frankfurter",
                    "stale": 0,
                    "currency": currency,
                }

        rate = self._rates.get(currency)
        if rate is None:
            raise FXConversionError(f"Unsupported currency: {currency}")
        stale = 1 if use_live else int((now - self._cache_ts) > FX_CACHE_TTL)
        return {
            "amount_usd": round(amount * rate, 2),
            "rate": rate,
            "source": "static",
            "stale": stale,
            "currency": currency,
        }

    def is_stale(self) -> bool:
        return (time.time() - self._cache_ts) > FX_CACHE_TTL

    async def _fetch_live_rate(self, currency: str, timeout_seconds: float) -> float | None:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.get(
                FRANKFURTER_RATES_URL,
                params={"base": currency, "quotes": "USD"},
            )
            resp.raise_for_status()
            data = resp.json()
        rate = data.get("rates", {}).get("USD")
        if rate is None:
            return None
        rate = float(rate)
        return rate if rate > 0 else None

    @staticmethod
    def _normalize_currency(currency: str) -> str:
        code = (currency or "USD").strip().upper()
        return code[:3] if len(code) >= 3 else "USD"

# Unit test
if __name__ == "__main__":
    fx = FXConverter()
    usd, stale = fx.to_usd(100.0, "EUR")
    assert abs(usd - 108.2) < 0.01, f"Bad EUR conversion: {usd}"
    assert stale == 0
    usd2, _ = fx.to_usd(100.0, "USD")
    assert usd2 == 100.0
    print("FXConverter OK")
