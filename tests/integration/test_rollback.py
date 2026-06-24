"""Integration tests for drift-triggered checkpoint rollback."""

from pathlib import Path

import torch

from drift.alert_manager import AlertManager
from drift.detectors import DriftReport
from src.server.checkpoint_manager import CheckpointManager


def _make_critical_report() -> DriftReport:
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
    ks = {key: 0.5 for key in psi}
    return DriftReport(
        timestamp="2024-05-13T12:00:00",
        feature_psi=psi,
        feature_ks_pval=ks,
        severity="CRITICAL",
        triggered_features=["tx_amount_usd"],
        stale_fx_rate=0.0,
    )


def test_rollback_creates_rollback_active_file(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    fake_weights = {"layer": torch.tensor([1.0, 2.0])}
    torch.save(fake_weights, checkpoint_dir / "round_005.pt")

    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    alert_manager = AlertManager(ckpt=checkpoint_manager)
    alert_manager.log_path = tmp_path / "drift_alerts.jsonl"
    alert_manager.log_path.parent.mkdir(exist_ok=True)

    severity = alert_manager.evaluate(_make_critical_report())

    assert severity == "CRITICAL"
    rollback_file = checkpoint_dir / "rollback_active.pt"
    assert rollback_file.exists()
    loaded = torch.load(rollback_file)
    assert "layer" in loaded


def test_rollback_log_entry_written(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    torch.save({}, checkpoint_dir / "round_001.pt")

    log_path = tmp_path / "alerts.jsonl"
    checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
    alert_manager = AlertManager(ckpt=checkpoint_manager)
    alert_manager.log_path = log_path
    log_path.parent.mkdir(exist_ok=True)

    alert_manager.evaluate(_make_critical_report())

    assert log_path.exists()
    import json

    entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert len(entries) == 1
    assert entries[0]["severity"] == "CRITICAL"
    assert "tx_amount_usd" in entries[0]["triggered"]
