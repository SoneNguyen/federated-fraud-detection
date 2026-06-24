from src.server.failure_detector import PhiAccrualFailureDetector


def test_phi_detector_stays_low_for_regular_heartbeat() -> None:
    detector = PhiAccrualFailureDetector(threshold=8.0, min_samples=5)
    for second in range(10):
        detector.heartbeat(float(second))

    assert detector.phi(10.0) < 1.0
    assert detector.is_suspect(10.0) is False


def test_phi_detector_flags_long_silence() -> None:
    detector = PhiAccrualFailureDetector(threshold=3.0, min_samples=5)
    for second in range(10):
        detector.heartbeat(float(second))

    assert detector.phi(25.0) >= 3.0
    assert detector.is_suspect(25.0) is True
