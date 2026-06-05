# drift/concept_drift.py (unchanged from original plan)
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConceptDriftReport:
    timestamp:              str
    pseudo_positive_rate:   float
    reference_positive_rate:float
    rate_shift:             float
    severity:               str

class PseudoLabelMonitor:
    CONF = 0.90; WINDOW = 500
    WARN = 0.30; CRIT = 0.60

    def __init__(self, ref_positive_rate: float):
        self.ref = ref_positive_rate
        self.window = deque(maxlen=self.WINDOW)

    def update(self, prob:float)->Optional[ConceptDriftReport]:
        if prob >= self.CONF:   self.window.append(1)
        elif prob <= 1-self.CONF: self.window.append(0)
        if len(self.window)<self.WINDOW: return None
        cur = sum(self.window)/len(self.window)
        shift = (self.ref-cur)/max(self.ref,1e-8)
        sev = "CRITICAL" if shift>=self.CRIT else "WARNING" if shift>=self.WARN else "INFO"
        return ConceptDriftReport(
            timestamp=datetime.utcnow().isoformat(),
            pseudo_positive_rate=round(cur,5),
            reference_positive_rate=round(self.ref,5),
            rate_shift=round(shift,5), severity=sev)