# drift/prediction_monitor.py
from river.drift import ADWIN
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class PredictionDriftReport:
    timestamp:            str
    drift_detected:       bool
    score_shift:          float
    mean_score_recent:    float
    mean_score_reference: float
    severity:             str

class PredictionMonitor:
    WARMUP = 1000
    WINDOW = 200
    WARN = 0.05
    CRIT = 0.15

    def __init__(self):
        self.adwin = ADWIN(delta=0.002)
        self._warmup: list[float] = []
        self._ref_mean: Optional[float] = None
        self._recent: deque[float] = deque(maxlen=self.WINDOW)

    def update(self, prob: float) -> Optional[PredictionDriftReport]:
        self._recent.append(prob)
        adwin_drift = bool(self.adwin.update(prob))
        if self._ref_mean is None:
            self._warmup.append(prob)
            if len(self._warmup) >= self.WARMUP:
                self._ref_mean = sum(self._warmup) / len(self._warmup)
            return None
        if len(self._recent)<self.WINDOW: return None
        rm=sum(self._recent)/len(self._recent)
        shift=abs(rm-self._ref_mean)
        sev="CRITICAL" if shift>=self.CRIT or adwin_drift else "WARNING" if shift>=self.WARN else "INFO"
        return PredictionDriftReport(
            timestamp=datetime.utcnow().isoformat(),
            drift_detected=adwin_drift, score_shift=round(shift,5),
            mean_score_recent=round(rm,5),
            mean_score_reference=round(self._ref_mean,5), severity=sev)