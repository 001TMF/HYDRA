"""CUSUM (Cumulative Sum) change-point detector.

Standard CUSUM algorithm for detecting mean shifts in streaming metrics.
Maintains positive and negative cumulative sums; signals drift when either
exceeds the configured threshold.
"""

from __future__ import annotations


class CUSUMDetector:
    """Cumulative sum change-point detector.

    Parameters
    ----------
    target : float
        Expected mean of the process (default 0.0).
    threshold : float
        Decision boundary for drift detection (default 5.0).
    drift : float
        Allowable slack / drift parameter (default 0.5).
    """

    def __init__(
        self,
        target: float = 0.0,
        threshold: float = 5.0,
        drift: float = 0.5,
    ) -> None:
        self.target = target
        self.threshold = threshold
        self.drift = drift
        self.s_pos: float = 0.0
        self.s_neg: float = 0.0
        self._drift_detected: bool = False

    def update(self, x: float) -> bool:
        """Feed a new observation and check for drift.

        Standard CUSUM update:
          s_pos = max(0, s_pos + (x - target) - drift)
          s_neg = max(0, s_neg - (x - target) - drift)

        Parameters
        ----------
        x : float
            New observation value.

        Returns
        -------
        bool
            True if drift detected (either accumulator exceeds threshold).
        """
        deviation = x - self.target
        self.s_pos = max(0.0, self.s_pos + deviation - self.drift)
        self.s_neg = max(0.0, self.s_neg - deviation - self.drift)
        self._drift_detected = self.s_pos > self.threshold or self.s_neg > self.threshold
        return self._drift_detected

    def reset(self) -> None:
        """Reset accumulators to zero."""
        self.s_pos = 0.0
        self.s_neg = 0.0
        self._drift_detected = False

    @property
    def drift_detected(self) -> bool:
        """Whether drift has been detected in the most recent update."""
        return self._drift_detected

    @property
    def cumulative_sum(self) -> tuple[float, float]:
        """Current cumulative sums (s_pos, s_neg)."""
        return (self.s_pos, self.s_neg)
