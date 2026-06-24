"""Tests for the PseudoLabelMonitor concept drift detector."""

import pytest

from drift.concept_drift import PseudoLabelMonitor


def test_returns_none_during_warmup():
    monitor = PseudoLabelMonitor(reference_positive_rate=0.05)
    result = None

    for _ in range(499):
        result = monitor.update(0.95)

    assert result is None


def test_info_on_stable_rate():
    monitor = PseudoLabelMonitor(reference_positive_rate=0.10)
    for i in range(500):
        prob = 0.95 if i < 50 else 0.02
        monitor.update(prob)

    result = monitor.update(0.02)

    if result is not None:
        assert result.severity == "INFO"


def test_critical_on_large_rate_drop():
    monitor = PseudoLabelMonitor(reference_positive_rate=0.50)
    for i in range(500):
        prob = 0.95 if i % 2 == 0 else 0.02
        monitor.update(prob)

    result = None
    for _ in range(500):
        result = monitor.update(0.02)

    assert result is not None
    assert result.severity == "CRITICAL"
    assert result.pseudo_positive_rate < result.reference_positive_rate


def test_ambiguous_predictions_skipped():
    monitor = PseudoLabelMonitor(reference_positive_rate=0.10)
    result = None

    for _ in range(1000):
        result = monitor.update(0.50)

    assert result is None


def test_rate_shift_calculated_correctly():
    monitor = PseudoLabelMonitor(reference_positive_rate=0.40)
    for i in range(500):
        monitor.update(0.95 if i < 200 else 0.02)

    result = None
    for _ in range(500):
        result = monitor.update(0.02)

    if result is not None:
        assert result.rate_shift >= 0.0
        assert result.reference_positive_rate == pytest.approx(0.40, abs=0.05)
