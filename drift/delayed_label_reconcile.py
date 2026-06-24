"""Retrospective metric reconciliation after delayed fraud labels arrive."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score


def reconcile_delayed_labels(
    predictions_log_path: Path | str,
    ground_truth_path: Path | str,
) -> float:
    predictions = pd.read_parquet(predictions_log_path)
    ground_truth = pd.read_parquet(ground_truth_path)
    merged = predictions.merge(ground_truth, on="transaction_id", how="inner")
    auprc = average_precision_score(merged["is_fraud"], merged["fraud_probability"])
    return float(auprc)
