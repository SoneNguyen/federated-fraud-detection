"""Feature-distribution drift detection for transaction features."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.data.feature_registry import FEATURE_ORDER


NUMERIC = FEATURE_ORDER
DEFAULT_REFERENCE_PATH = Path("dataset/drift_ref/reference_window.parquet")


@dataclass(frozen=True)
class DriftReport:
    timestamp: str
    feature_psi: dict[str, float]
    feature_ks_pval: dict[str, float]
    severity: str
    triggered_features: list[str]
    score_shift: float | None = None
    stale_fx_rate: float = 0.0


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate PSI for one numeric feature."""

    eps = 1e-6
    reference_hist, edges = np.histogram(reference, bins=bins)
    current_hist, _ = np.histogram(current, bins=edges)
    reference_pct = (reference_hist + eps) / (len(reference) + eps * bins)
    current_pct = (current_hist + eps) / (len(current) + eps * bins)
    return float(np.sum((reference_pct - current_pct) * np.log(reference_pct / current_pct)))


class FeatureMonitor:
    PSI_WARN = 0.10
    PSI_CRIT = 0.20

    def __init__(
        self,
        reference_df: pd.DataFrame,
        persist_path: Path | str | None = DEFAULT_REFERENCE_PATH,
    ) -> None:
        self.ref = reference_df.reindex(columns=NUMERIC, fill_value=0.0).copy()
        if persist_path is not None:
            output_path = Path(persist_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.ref.to_parquet(output_path, index=False)

    @classmethod
    def from_disk(
        cls,
        path: Path | str = DEFAULT_REFERENCE_PATH,
    ) -> "FeatureMonitor":
        reference_path = Path(path)
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference file not found: {reference_path}")

        reference_df = pd.read_parquet(reference_path)
        obj = object.__new__(cls)
        obj.ref = reference_df.reindex(columns=NUMERIC, fill_value=0.0).copy()
        return obj

    def check(self, current_df: pd.DataFrame) -> DriftReport:
        current = current_df.reindex(columns=NUMERIC, fill_value=0.0)
        psi_scores: dict[str, float] = {}
        ks_pvalues: dict[str, float] = {}
        triggered: list[str] = []

        for feature in NUMERIC:
            reference_values = self.ref[feature].dropna().to_numpy()
            current_values = current[feature].dropna().to_numpy()
            if len(reference_values) == 0 or len(current_values) == 0:
                psi_value = 0.0
                ks_pvalue = 1.0
            else:
                psi_value = population_stability_index(reference_values, current_values)
                ks_result: Any = stats.ks_2samp(reference_values, current_values)
                ks_pvalue = float(ks_result.pvalue)

            psi_scores[feature] = round(psi_value, 5)
            ks_pvalues[feature] = round(ks_pvalue, 5)

            if psi_value > self.PSI_WARN:
                triggered.append(feature)

        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        severity = (
            "CRITICAL"
            if max_psi >= self.PSI_CRIT
            else "WARNING"
            if max_psi >= self.PSI_WARN
            else "INFO"
        )

        stale_fx_rate = 0.0
        if "stale_fx_flag" in current_df.columns:
            stale_fx_rate = float(current_df["stale_fx_flag"].mean())

        return DriftReport(
            timestamp=datetime.now(UTC).isoformat(),
            feature_psi=psi_scores,
            feature_ks_pval=ks_pvalues,
            severity=severity,
            triggered_features=triggered,
            stale_fx_rate=round(stale_fx_rate, 4),
        )
