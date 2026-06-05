# tests/integration/test_drift_simulation.py
import pandas as pd, numpy as np
from drift.detectors import FeatureMonitor, NUMERIC

def make_df(n=10_000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.normal(shift, 1.0, n) for c in NUMERIC})

def make_stale_df(n=10_000, stale_frac=0.8, seed=5):
    df = make_df(n=n, seed=seed)
    # Add stale_fx_flag metadata column — 80% stale
    rng = np.random.default_rng(seed)
    df["stale_fx_flag"] = (rng.random(n) < stale_frac).astype(int)
    return df

ref = make_df(seed=0)
monitor = FeatureMonitor(ref)

# Test 1: no drift → INFO
r = monitor.check(make_df(seed=99))
assert r.severity=="INFO", f"Expected INFO: {r.severity}"
assert r.stale_fx_rate == 0.0
print(f"PASS no-drift: {r.severity}, stale_fx_rate={r.stale_fx_rate}")

# Test 2: severe drift on tx_amount_usd (v3 name) → CRITICAL
severe = make_df(seed=2)
severe["tx_amount_usd"]     = np.random.default_rng(2).normal(-3.0,0.5,10_000)
severe["tx_volume_1h_usd"]  = np.random.default_rng(3).normal(4.0,0.5,10_000)
r = monitor.check(severe)
assert r.severity=="CRITICAL"
assert "tx_amount_usd" in r.triggered_features
print(f"PASS severe drift: {r.severity}, triggered={r.triggered_features}")

# Test 3: stale FX rate escalation (v3 new)
stale_df = make_stale_df(stale_frac=0.8)
r = monitor.check(stale_df)
assert r.stale_fx_rate > 0.7, f"stale_fx_rate too low: {r.stale_fx_rate}"
print(f"PASS stale FX: stale_fx_rate={r.stale_fx_rate:.2f}")
print("All v3 drift simulation tests PASSED")