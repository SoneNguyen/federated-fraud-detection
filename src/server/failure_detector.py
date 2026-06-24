"""Adaptive failure suspicion utilities for federated clients."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PhiAccrualFailureDetector:
    """Estimate client failure suspicion from heartbeat inter-arrival times.

    The detector follows the Phi Accrual idea: instead of returning a boolean
    timeout, it returns a suspicion score that grows as silence becomes unusual
    compared with recent heartbeat intervals.
    """

    threshold: float = 8.0
    min_samples: int = 5
    window_size: int = 100
    min_stddev: float = 1.0
    heartbeat_times: deque[float] = field(default_factory=deque)
    intervals: deque[float] = field(default_factory=deque)

    def heartbeat(self, timestamp: float) -> None:
        """Record a heartbeat timestamp in seconds."""
        ts = float(timestamp)
        if self.heartbeat_times:
            interval = max(ts - self.heartbeat_times[-1], 1e-6)
            self.intervals.append(interval)
            while len(self.intervals) > self.window_size:
                self.intervals.popleft()
        self.heartbeat_times.append(ts)
        while len(self.heartbeat_times) > self.window_size + 1:
            self.heartbeat_times.popleft()

    def phi(self, timestamp: float) -> float:
        """Return the current suspicion score."""
        if not self.heartbeat_times:
            return 0.0
        elapsed = max(float(timestamp) - self.heartbeat_times[-1], 0.0)
        if len(self.intervals) < self.min_samples:
            mean = self.intervals[-1] if self.intervals else max(elapsed, 1.0)
            return 0.0 if elapsed <= mean * 3.0 else self.threshold

        values = list(self.intervals)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stddev = max(math.sqrt(variance), self.min_stddev)
        survival = max(1.0 - _normal_cdf(elapsed, mean, stddev), 1e-12)
        return -math.log10(survival)

    def is_suspect(self, timestamp: float) -> bool:
        """Return True when the suspicion score crosses the configured threshold."""
        return self.phi(timestamp) >= self.threshold


def _normal_cdf(x: float, mean: float, stddev: float) -> float:
    z = (x - mean) / (stddev * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))
