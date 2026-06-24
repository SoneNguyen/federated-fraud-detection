import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from drift.detectors import DriftReport
from drift.prediction_monitor import PredictionDriftReport
from scripts.run_paths import results_dir as default_results_dir
from src.server.checkpoint_manager import CheckpointManager


class AlertManager:
    STALE_FX_WARN = 0.50
    API_URL = "http://api-gateway:8000"

    def __init__(self, ckpt: CheckpointManager):
        self.ckpt = ckpt
        self.log_path = Path(
            os.environ.get(
                "DRIFT_ALERT_LOG",
                str(default_results_dir() / "drift_alerts.jsonl"),
            )
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        feat: DriftReport,
        pred: Optional[PredictionDriftReport] = None,
    ) -> str:
        rank = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
        severities = [feat.severity]
        if pred:
            severities.append(pred.severity)

        if feat.stale_fx_rate > self.STALE_FX_WARN:
            severities.append("WARNING")
            print(f"[ALERT] stale_fx_rate={feat.stale_fx_rate:.2f} fx_pipeline=degraded")

        combined = max(severities, key=lambda severity: rank[severity])
        self._log(combined, feat, pred)
        self._dispatch(combined)
        return combined

    def _dispatch(self, severity: str) -> None:
        if severity == "CRITICAL":
            self.ckpt.rollback()
            try:
                requests.post(f"{self.API_URL}/reload", timeout=10)
            except Exception as exc:
                print(f"[ALERT] reload_failed={exc}")
        elif severity == "WARNING":
            trigger = self.ckpt.checkpoint_dir / ".trigger_emergency_round"
            trigger.parent.mkdir(parents=True, exist_ok=True)
            trigger.touch()

    def _log(
        self,
        severity: str,
        feat: DriftReport,
        pred: Optional[PredictionDriftReport],
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": severity,
            "triggered": feat.triggered_features,
            "max_psi": max(feat.feature_psi.values()),
            "stale_fx_rate": feat.stale_fx_rate,
            "score_shift": pred.score_shift if pred else None,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
