"""Prometheus exporter helpers for drift and retrospective metrics."""

from __future__ import annotations

from prometheus_client import Gauge, start_http_server


_psi = Gauge("fraud_psi", "PSI per feature", ["feature"])
_severity = Gauge("fraud_drift_sev", "Severity 0=INFO 1=WARN 2=CRIT")
_shift = Gauge("fraud_score_shift", "Prediction score shift")
_stale_fx = Gauge("fraud_stale_fx_rate", "Fraction of stale FX predictions")
_auprc = Gauge("fraud_auprc", "Model AUPRC on retrospective labels")

SEV = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


def push_feature(report) -> None:
    for feature, psi_value in report.feature_psi.items():
        _psi.labels(feature=feature).set(psi_value)
    _severity.set(SEV[report.severity])
    _stale_fx.set(report.stale_fx_rate)


def push_prediction(report) -> None:
    _shift.set(report.score_shift)


def push_auprc(value: float) -> None:
    _auprc.set(value)


def start(port: int = 9090) -> None:
    start_http_server(port)
    print(f"Prometheus metrics available at :{port}/metrics")
