"""Prediction-score drift monitoring."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

from river.drift import ADWIN


@dataclass(frozen=True)
class PredictionDriftReport:
    timestamp: str
    drift_detected: bool
    score_shift: float
    mean_score_recent: float
    mean_score_reference: float
    severity: str


class PredictionMonitor:
    WARMUP = 1000
    WINDOW = 200
    WARN = 0.05
    CRIT = 0.15

    def __init__(self) -> None:
        self.adwin = ADWIN(delta=0.002)
        self._warmup: list[float] = []
        self._ref_mean: float | None = None
        self._recent: deque[float] = deque(maxlen=self.WINDOW)

    def update(self, probability: float) -> PredictionDriftReport | None:
        self._recent.append(probability)
        adwin_drift = bool(self.adwin.update(probability))

        if self._ref_mean is None:
            self._warmup.append(probability)
            if len(self._warmup) >= self.WARMUP:
                self._ref_mean = sum(self._warmup) / len(self._warmup)
            return None

        if len(self._recent) < self.WINDOW:
            return None

        recent_mean = sum(self._recent) / len(self._recent)
        shift = abs(recent_mean - self._ref_mean)
        severity = (
            "CRITICAL"
            if shift >= self.CRIT or adwin_drift
            else "WARNING"
            if shift >= self.WARN
            else "INFO"
        )

        return PredictionDriftReport(
            timestamp=datetime.now(UTC).isoformat(),
            drift_detected=adwin_drift,
            score_shift=round(shift, 5),
            mean_score_recent=round(recent_mean, 5),
            mean_score_reference=round(self._ref_mean, 5),
            severity=severity,
        )
