# drift/exporter.py
from prometheus_client import Gauge, start_http_server

_psi        = Gauge('fraud_psi','PSI per feature',['feature'])
_sev        = Gauge('fraud_drift_sev','Severity 0=INFO 1=WARN 2=CRIT')
_shift      = Gauge('fraud_score_shift','Prediction score shift')
_stale_fx   = Gauge('fraud_stale_fx_rate','Fraction of stale FX predictions')  # v3 new
_auprc      = Gauge('fraud_auprc','Model AUPRC on retrospective labels')
SEV = {"INFO":0,"WARNING":1,"CRITICAL":2}

def push_feature(report):
    for feat,p in report.feature_psi.items():
        _psi.labels(feature=feat).set(p)
    _sev.set(SEV[report.severity])
    _stale_fx.set(report.stale_fx_rate)  # v3

def push_prediction(report):
    _shift.set(report.score_shift)

def push_auprc(v): _auprc.set(v)

def start(port=9090):
    start_http_server(port)
    print(f"Prometheus metrics on :{port}/metrics")

# Grafana alert rules:
# fraud_drift_sev == 2              → CRITICAL alert
# fraud_stale_fx_rate > 0.5         → WARNING (FX pipeline issue) [v3]
# fraud_psi{feature='tx_amount_usd'} > 0.25 → WARNING
# fraud_auprc < 0.70                → CRITICAL