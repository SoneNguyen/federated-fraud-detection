"""Run a local feature-drift injection check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drift.detectors import NUMERIC, FeatureMonitor


def main() -> None:
    processed_path = Path("dataset/processed/client_0/transactions_normalized.parquet")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"{processed_path} not found. Run dataset/load_ieee_cis.py first."
        )

    df = pd.read_parquet(processed_path)
    reference = df.head(10_000)
    monitor = FeatureMonitor(reference)
    print(f"reference_rows={len(reference):,}")

    rng = np.random.default_rng(42)
    rows = 5_000
    drifted = pd.DataFrame({feature: rng.normal(0.0, 1.0, rows) for feature in NUMERIC})
    drifted["tx_amount_usd"] = rng.normal(-3.5, 0.4, rows)
    drifted["tx_count_1h"] = rng.normal(4.5, 0.5, rows)
    drifted["geo_velocity_kmh"] = rng.normal(5.0, 0.8, rows)

    report = monitor.check(drifted)

    print(f"severity={report.severity}")
    print(f"triggered_features={report.triggered_features}")
    print(f"stale_fx_rate={report.stale_fx_rate}")
    for feature, score in sorted(report.feature_psi.items(), key=lambda item: -item[1]):
        marker = " triggered" if feature in report.triggered_features else ""
        print(f"{feature:30s} psi={score:.5f}{marker}")

    if report.severity != "CRITICAL":
        raise RuntimeError(f"Expected CRITICAL drift, got {report.severity}")
    if "tx_amount_usd" not in report.triggered_features:
        raise RuntimeError("Expected tx_amount_usd to be triggered")

    print("CRITICAL drift injection confirmed")


if __name__ == "__main__":
    main()
