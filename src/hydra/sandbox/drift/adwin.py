"""ADWIN (Adaptive Windowing) drift detector.

Wraps River library's ADWIN implementation for detecting concept drift
in streaming data via adaptive window sizing.
"""

from __future__ import annotations

from river.drift import ADWIN


class ADWINDetector:
    """Adaptive windowing drift detector backed by River.

    Parameters
    ----------
    delta : float
        Confidence parameter for ADWIN (default 0.002).
        Lower values make detection more conservative.
    grace_period : int
        Minimum number of observations before detection activates (default 30).
    """

    def __init__(self, delta: float = 0.002, grace_period: int = 30) -> None:
        self.delta = delta
        self.grace_period = grace_period
        self.detector = ADWIN(delta=delta, grace_period=grace_period)
        self._n_detections: int = 0

    def update(self, value: float) -> bool:
        """Feed a new observation and check for drift.

        Parameters
        ----------
        value : float
            New observation.

        Returns
        -------
        bool
            True if drift detected at this step.
        """
        self.detector.update(value)
        detected = self.detector.drift_detected
        if detected:
            self._n_detections += 1
        return detected

    @property
    def estimation(self) -> float:
        """Current mean estimate from the adaptive window."""
        return self.detector.estimation

    @property
    def n_detections(self) -> int:
        """Total number of drift detections so far."""
        return self._n_detections

    def reset(self) -> None:
        """Reset detector to initial state with same parameters."""
        self.detector = ADWIN(delta=self.delta, grace_period=self.grace_period)
        self._n_detections = 0
