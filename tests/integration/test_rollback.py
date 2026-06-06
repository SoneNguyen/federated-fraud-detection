"""Integration test: CRITICAL drift alert triggers checkpoint rollback."""
from pathlib import Path
import pytest
import torch
from server.checkpoint_manager import CheckpointManager
from drift.alert_manager import AlertManager
from drift.detectors import DriftReport


def _make_critical_report() -> DriftReport:
    # PSI of 0.35 on tx_amount_usd — well above the 0.20 CRITICAL threshold
    psi = {
        "tx_amount_usd": 0.35,
        "tx_count_1h": 0.01,
        "tx_count_24h": 0.01,
        "tx_volume_1h_usd": 0.01,
        "tx_volume_24h_usd": 0.01,
        "merchant_cat_dev": 0.01,
        "geo_velocity_kmh": 0.01,
        "days_since_last_tx": 0.01,
        "account_age_days": 0.01,
    }
    ks = {k: 0.5 for k in psi}
    return DriftReport(
        timestamp="2024-05-13T12:00:00",
        feature_psi=psi,
        feature_ks_pval=ks,
        severity="CRITICAL",
        triggered_features=["tx_amount_usd"],
        stale_fx_rate=0.0,
    )


def test_rollback_creates_rollback_active_file(tmp_path: Path):
    # Create a fake checkpoint
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    fake_weights = {"layer": torch.tensor([1.0, 2.0])}
    torch.save(fake_weights, ckpt_dir / "round_005.pt")

    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    alert_mgr = AlertManager(ckpt=ckpt_mgr)
    # Override log path to tmp
    alert_mgr.LOG = tmp_path / "drift_alerts.jsonl"
    alert_mgr.LOG.parent.mkdir(exist_ok=True)

    report = _make_critical_report()
    severity = alert_mgr.evaluate(report)

    assert severity == "CRITICAL"
    rollback_file = ckpt_dir / "rollback_active.pt"
    assert rollback_file.exists(), "rollback_active.pt must be created on CRITICAL"
    loaded = torch.load(rollback_file)
    assert "layer" in loaded


def test_rollback_log_entry_written(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    torch.save({}, ckpt_dir / "round_001.pt")

    log_path = tmp_path / "alerts.jsonl"
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir)
    alert_mgr = AlertManager(ckpt=ckpt_mgr)
    alert_mgr.LOG = log_path
    log_path.parent.mkdir(exist_ok=True)

    alert_mgr.evaluate(_make_critical_report())

    assert log_path.exists()
    import json
    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert len(entries) == 1
    assert entries[0]["severity"] == "CRITICAL"
    assert "tx_amount_usd" in entries[0]["triggered"]