"""Concept-drift monitor based on high-confidence pseudo-label rates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class ConceptDriftReport:
    timestamp: str
    pseudo_positive_rate: float
    reference_positive_rate: float
    rate_shift: float
    severity: str


class PseudoLabelMonitor:
    CONFIDENCE = 0.90
    WINDOW = 500
    WARN = 0.30
    CRIT = 0.60

    def __init__(self, reference_positive_rate: float) -> None:
        self.reference_positive_rate = reference_positive_rate
        self.window: deque[int] = deque(maxlen=self.WINDOW)

    def update(self, probability: float) -> ConceptDriftReport | None:
        if probability >= self.CONFIDENCE:
            self.window.append(1)
        elif probability <= 1 - self.CONFIDENCE:
            self.window.append(0)

        if len(self.window) < self.WINDOW:
            return None

        current_rate = sum(self.window) / len(self.window)
        shift = (self.reference_positive_rate - current_rate) / max(
            self.reference_positive_rate,
            1e-8,
        )
        severity = (
            "CRITICAL"
            if shift >= self.CRIT
            else "WARNING"
            if shift >= self.WARN
            else "INFO"
        )

        return ConceptDriftReport(
            timestamp=datetime.now(UTC).isoformat(),
            pseudo_positive_rate=round(current_rate, 5),
            reference_positive_rate=round(self.reference_positive_rate, 5),
            rate_shift=round(shift, 5),
            severity=severity,
        )
