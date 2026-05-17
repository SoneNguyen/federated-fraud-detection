# data/fx/converter.py
from pathlib import Path
import sys
import time, json

# Ensure project root is on sys.path so absolute package imports work
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fx.rates import STATIC_RATES

# Load config relative to project root to be robust to CWD
with open(ROOT / "contracts" / "schema.json") as f:
    FX_CACHE_TTL = json.load(f)["feature_schema"]["fx_cache_ttl_sec"]

class FXConverter:
    def __init__(self, rates=None, cache_ts=None):
        self._rates = rates or STATIC_RATES
        self._cache_ts = cache_ts or time.time()

    def to_usd(self, amount: float, currency: str) -> tuple[float, int]:
        rate = self._rates.get(currency, 1.0)
        stale = int((time.time() - self._cache_ts) > FX_CACHE_TTL)
        return round(amount * rate, 2), stale

    def is_stale(self) -> bool:
        return (time.time() - self._cache_ts) > FX_CACHE_TTL

# Unit test
if __name__ == "__main__":
    fx = FXConverter()
    usd, stale = fx.to_usd(100.0, "EUR")
    assert abs(usd - 108.2) < 0.01, f"Bad EUR conversion: {usd}"
    assert stale == 0
    usd2, _ = fx.to_usd(100.0, "USD")
    assert usd2 == 100.0
    print("FXConverter OK")