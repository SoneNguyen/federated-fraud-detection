import requests
import json
from typing import Optional 
from datetime import datetime, timezone
from pathlib import Path

from drift.detectors import DriftReport
from drift.prediction_monitor import PredictionDriftReport
from server.checkpoint_manager import CheckpointManager

class AlertManager:
    STALE_FX_WARN = 0.50  # >50% stale predictions → FX pipeline issue
    API_URL       = "http://api-gateway:8000"
    LOG           = Path("results/drift_alerts.jsonl")

    def __init__(self, ckpt: CheckpointManager):
        self.ckpt = ckpt
        self.LOG.parent.mkdir(exist_ok=True)

    # 2. Wrap PredictionDriftReport in Optional[]
    def evaluate(self, feat: DriftReport, pred: Optional[PredictionDriftReport] = None) -> str:
        rank = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
        sevs = [feat.severity]
        if pred: 
            sevs.append(pred.severity)
            
        # v3: stale FX rate check — escalate to WARNING if FX pipeline degraded
        if feat.stale_fx_rate > self.STALE_FX_WARN:
            sevs.append("WARNING")
            print(f"[ALERT] stale_fx_rate={feat.stale_fx_rate:.2f} — FX pipeline issue")
            
        combined = max(sevs, key=lambda s: rank[s])
        self._log(combined, feat, pred)
        self._dispatch(combined)
        return combined

    def _dispatch(self, severity):
        if severity == "CRITICAL":
            path = self.ckpt.rollback()
            try: 
                requests.post(f"{self.API_URL}/reload", timeout=10)
            except Exception as e: 
                print(f"[ALERT] reload failed: {e}")
        elif severity == "WARNING":
            open("checkpoints/.trigger_emergency_round", "w").close()

    def _log(self, sev, feat, pred):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": sev,
            "triggered": feat.triggered_features,
            "max_psi": max(feat.feature_psi.values()),
            "stale_fx_rate": feat.stale_fx_rate,  # v3
            "score_shift": pred.score_shift if pred else None
        }
        with open(self.LOG, "a") as f: 
            f.write(json.dumps(entry) + "\n")