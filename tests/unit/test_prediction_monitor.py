#  This module contains unit tests for the PredictionMonitor class in the drift.prediction_monitor module.
#  The tests verify that the update method returns a PredictionDriftReport after the warmup and window periods, 
#  and that the report contains the expected values for drift detection, severity, and mean scores.

import unittest
from typing import cast

from drift.prediction_monitor import PredictionMonitor, PredictionDriftReport


class TestPredictionMonitor(unittest.TestCase):

    def test_update_returns_report_after_warmup_and_window(self):
        monitor = PredictionMonitor()

        for _ in range(monitor.WARMUP):
            self.assertIsNone(monitor.update(0.1))

        report = monitor.update(0.1)

        self.assertIsNotNone(report)
        report = cast(PredictionDriftReport, report)
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.severity, "INFO")
        self.assertAlmostEqual(report.mean_score_recent, 0.1, places=2)
        self.assertAlmostEqual(report.mean_score_reference, 0.1, places=2)

    def test_update_returns_report_immediately_after_warmup(self):
        monitor = PredictionMonitor()

        for _ in range(monitor.WARMUP):
            monitor.update(0.3)

        report = monitor.update(0.3)

        self.assertIsNotNone(report)
        report = cast(PredictionDriftReport, report)
        self.assertEqual(report.severity, "INFO")


if __name__ == "__main__":
    unittest.main()
