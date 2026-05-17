# data/generate_synthetic.py
from pathlib import Path
import sys
import numpy as np, pandas as pd, json, pytz
from datetime import datetime, timezone

# Ensure project root is on sys.path so package imports work when run directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fx.converter import FXConverter
from data.fx.rates import CLIENT_CURRENCIES

# Load config relative to project root
with open(ROOT / "contracts" / "schema.json") as f:
    _s = json.load(f)
FEATURE_ORDER  = _s["feature_schema"]["feature_order"]   # 11 features
LABEL          = _s["feature_schema"]["label"]["name"]   # is_fraud

CLIENT_CFG = [
    {"id":0,"n":500_000,"fraud_rate":0.008,"seed":0,"tz":"US/Eastern"},
    {"id":1,"n":500_000,"fraud_rate":0.015,"seed":1,"tz":"Europe/Berlin"},
    {"id":2,"n":300_000,"fraud_rate":0.035,"seed":2,"tz":"Asia/Singapore"},
]
fx = FXConverter()

def gen_rows(n, is_fraud, rng, currency, tz_str):
    tz = pytz.timezone(tz_str)
    # Generate UTC timestamps spread across 30 days
    utc_ts = [datetime(2024,4,1,tzinfo=timezone.utc) +
              pd.Timedelta(seconds=int(rng.integers(0, 30*86400)))
              for _ in range(n)]
    local_hours = [ts.astimezone(tz).hour for ts in utc_ts]
    local_dows  = [ts.astimezone(tz).weekday() for ts in utc_ts]
    raw_amounts = (rng.lognormal(3.5,1.5,n) if not is_fraud
                   else rng.lognormal(2.0,2.2,n))
    usd_amounts = [fx.to_usd(a, currency)[0] for a in raw_amounts]
    stale_flags = [fx.to_usd(a, currency)[1] for a in raw_amounts]
    return {
        "tx_amount_usd":       usd_amounts,
        "tx_count_1h":         rng.poisson(2 if not is_fraud else 8, n).tolist(),
        "tx_count_24h":        rng.poisson(8 if not is_fraud else 30, n).tolist(),
        "tx_volume_1h_usd":    [fx.to_usd(v,currency)[0] for v in rng.lognormal(5,1.2,n)],
        "tx_volume_24h_usd":   [fx.to_usd(v,currency)[0] for v in rng.lognormal(7,1.5,n)],
        "merchant_cat_dev":    rng.normal(0 if not is_fraud else 2.5, 1.0 if not is_fraud else 1.5, n).tolist(),
        "geo_velocity_kmh":    rng.exponential(15 if not is_fraud else 250, n).tolist(),
        "days_since_last_tx":  rng.exponential(3 if not is_fraud else 0.5, n).tolist(),
        "account_age_days":    rng.integers(30 if not is_fraud else 1, 3650 if not is_fraud else 60, n).tolist(),
        "hour_of_day_local":   local_hours,
        "day_of_week":         local_dows,
        "orig_currency":       [currency]*n,
        "stale_fx_flag":       stale_flags,
        "is_fraud":            int(is_fraud),
    }

for cfg in CLIENT_CFG:
    rng = np.random.default_rng(cfg["seed"])
    cur = CLIENT_CURRENCIES[cfg["id"]]
    n_fraud = int(cfg["n"] * cfg["fraud_rate"])
    n_legit = cfg["n"] - n_fraud
    fraud_df = pd.DataFrame(gen_rows(n_fraud, True,  rng, cur, cfg["tz"]))
    legit_df = pd.DataFrame(gen_rows(n_legit, False, rng, cur, cfg["tz"]))
    df = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=cfg["seed"])
    # Enforce column order: FEATURE_ORDER + metadata + label
    full_cols = FEATURE_ORDER + ["orig_currency","stale_fx_flag","is_fraud"]
    df = df[full_cols]
    # Assertions
    assert list(df.columns[:11]) == FEATURE_ORDER, "feature_order mismatch"
    assert df.isnull().sum().sum() == 0, "nulls found"
    assert set(df["orig_currency"].unique()) == {cur}
    assert df["hour_of_day_local"].between(0,23).all()
    out = Path(f"data/raw/client_{cfg['id']}")
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out/"transactions.parquet", index=False)
    print(f"Client {cfg['id']} ({cur}/{cfg['tz']}): {len(df):,} rows, "
          f"{df.is_fraud.sum():,} fraud ({df.is_fraud.mean()*100:.2f}%)")
print("All assertions passed")