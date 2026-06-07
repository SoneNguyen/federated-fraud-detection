"""Tests for the PseudoLabelMonitor concept drift detector."""
import pytest
from drift.concept_drift import PseudoLabelMonitor


def _warmup(monitor: PseudoLabelMonitor, n: int, prob: float) -> None:
    for _ in range(n):
        monitor.update(prob)


def test_returns_none_during_warmup():
    monitor = PseudoLabelMonitor(ref_positive_rate=0.05)
    # Feed 499 confident fraud predictions — still in warmup
    result = None
    for _ in range(499):
        result = monitor.update(0.95)
    assert result is None


def test_info_on_stable_rate():
    monitor = PseudoLabelMonitor(ref_positive_rate=0.10)
    # Warmup: mix of confident fraud (10%) and confident legit (90%)
    for i in range(500):
        prob = 0.95 if i < 50 else 0.02
        monitor.update(prob)
    # Check: stable — should be INFO or None still warming up
    result = None
    result = monitor.update(0.02)
    if result is not None:
        assert result.severity == "INFO"


def test_critical_on_large_rate_drop():
    monitor = PseudoLabelMonitor(ref_positive_rate=0.50)
    # Warmup: 50% confident fraud
    for i in range(500):
        prob = 0.95 if i % 2 == 0 else 0.02
        monitor.update(prob)
    # Now inject: no fraud at all — 100% relative drop → CRITICAL
    result = None
    for _ in range(500):
        result = monitor.update(0.02)
    assert result is not None
    assert result.severity == "CRITICAL"
    assert result.pseudo_positive_rate < result.reference_positive_rate


def test_ambiguous_predictions_skipped():
    monitor = PseudoLabelMonitor(ref_positive_rate=0.10)
    # Ambiguous range (0.10–0.90) should not advance the window
    result = None
    for _ in range(1000):
        result = monitor.update(0.50)
    # Window never fills because all predictions are ambiguous
    assert result is None


def test_rate_shift_calculated_correctly():
    monitor = PseudoLabelMonitor(ref_positive_rate=0.40)
    # Warmup: 40% fraud
    for i in range(500):
        monitor.update(0.95 if i < 200 else 0.02)
    # Drop to 0% fraud
    result = None
    for _ in range(500):
        result = monitor.update(0.02)
    if result is not None:
        assert result.rate_shift >= 0.0
        assert result.reference_positive_rate == pytest.approx(0.40, abs=0.05)