"""Tests for the Prometheus drift metrics exporter."""

from drift.detectors import DriftReport
from drift.prediction_monitor import PredictionDriftReport


def _make_feat_report(severity="WARNING", stale=0.1):
    psi = {
        "tx_amount_usd": 0.15,
        "tx_count_1h": 0.01,
        "tx_count_24h": 0.01,
        "tx_volume_1h_usd": 0.01,
        "tx_volume_24h_usd": 0.01,
        "merchant_cat_dev": 0.01,
        "geo_velocity_kmh": 0.01,
        "days_since_last_tx": 0.01,
        "account_age_days": 0.01,
    }
    return DriftReport(
        timestamp="2024-01-01T00:00:00",
        feature_psi=psi,
        feature_ks_pval={k: 0.05 for k in psi},
        severity=severity,
        triggered_features=["tx_amount_usd"],
        stale_fx_rate=stale,
    )


def _make_pred_report(shift=0.08):
    return PredictionDriftReport(
        timestamp="2024-01-01T00:00:00",
        drift_detected=False,
        score_shift=shift,
        mean_score_recent=0.13,
        mean_score_reference=0.05,
        severity="WARNING",
    )


def test_push_feature_sets_psi_gauges():
    from drift import exporter

    exporter.push_feature(_make_feat_report())


def test_push_feature_sets_severity_gauge():
    from drift import exporter

    for severity, expected in [("INFO", 0), ("WARNING", 1), ("CRITICAL", 2)]:
        exporter.push_feature(_make_feat_report(severity=severity))
        assert exporter.SEV[severity] == expected


def test_push_feature_sets_stale_fx_rate():
    from drift import exporter

    exporter.push_feature(_make_feat_report(stale=0.75))


def test_push_prediction_sets_shift_gauge():
    from drift import exporter

    exporter.push_prediction(_make_pred_report(shift=0.12))


def test_push_auprc():
    from drift import exporter

    exporter.push_auprc(0.82)


def test_sev_map_complete():
    from drift import exporter

    assert exporter.SEV["INFO"] == 0
    assert exporter.SEV["WARNING"] == 1
    assert exporter.SEV["CRITICAL"] == 2
