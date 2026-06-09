"""Demo script: inject severe distribution drift and verify the alert fires.

Run after the full system is up:
    python tests/demo/inject_drift.py

What it does:
1. Loads the processed data for client_0
2. Builds the reference FeatureMonitor from the first 10,000 rows
3. Generates a severely shifted dataset (new fraud pattern: micro-transactions + high velocity)
4. Runs the drift check and asserts CRITICAL fires
5. Prints the triggered features and PSI scores
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from drift.detectors import FeatureMonitor, NUMERIC


def main() -> None:
    processed_path = Path("data/processed/client_0/transactions_normalized.parquet")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_path} not found. Run data/load_ieee_cis.py first."
        )

    df = pd.read_parquet(processed_path)
    reference = df.head(10_000)
    monitor = FeatureMonitor(reference)
    print(f"Reference window: {len(reference):,} rows")

    # Simulate a new fraud pattern:
    # - tx_amount_usd collapses (micro-transactions, card testing)
    # - tx_count_1h explodes (automated burst)
    # - geo_velocity_kmh spikes (impossible travel)
    rng = np.random.default_rng(42)
    n = 5_000
    drifted = pd.DataFrame({c: rng.normal(0.0, 1.0, n) for c in NUMERIC})
    drifted["tx_amount_usd"]    = rng.normal(-3.5, 0.4, n)   # far below reference mean
    drifted["tx_count_1h"]      = rng.normal( 4.5, 0.5, n)   # far above reference mean
    drifted["geo_velocity_kmh"] = rng.normal( 5.0, 0.8, n)   # impossible velocity

    report = monitor.check(drifted)

    print("\n=== Drift Injection Results ===")
    print(f"Severity:           {report.severity}")
    print(f"Triggered features: {report.triggered_features}")
    print(f"stale_fx_rate:      {report.stale_fx_rate}")
    print("\nPSI scores:")
    for feat, score in sorted(report.feature_psi.items(), key=lambda x: -x[1]):
        marker = " ← TRIGGERED" if feat in report.triggered_features else ""
        print(f"  {feat:30s}: {score:.5f}{marker}")

    assert report.severity == "CRITICAL", (
        f"Expected CRITICAL but got {report.severity}. "
        "Check PSI thresholds in drift/detectors.py"
    )
    assert "tx_amount_usd" in report.triggered_features
    print("\n✓ CRITICAL alert confirmed — drift injection successful")


if __name__ == "__main__":
    main()