"""Unit tests for the 4 drift detectors (PSI, KS, CUSUM, ADWIN)."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.sandbox.drift import ADWINDetector, CUSUMDetector, check_ks_drift, compute_psi


# ---------------------------------------------------------------------------
# PSI tests
# ---------------------------------------------------------------------------


class TestPSI:
    def test_psi_identical_distributions(self) -> None:
        """PSI of identical distributions should be near zero."""
        np.random.seed(42)
        data = np.random.randn(1000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_psi_shifted_distribution(self) -> None:
        """Shifted distribution should produce PSI >= 0.25 (significant)."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)
        psi = compute_psi(baseline, current)
        assert psi > 0.25

    def test_psi_epsilon_no_nan(self) -> None:
        """PSI should be finite even with zero bins (epsilon smoothing)."""
        np.random.seed(42)
        # Baseline concentrated in narrow range; current spread across wider range
        # Some bins will have zero counts in one distribution
        baseline = np.concatenate([np.zeros(500), np.ones(500)])
        current = np.random.randn(1000)
        psi = compute_psi(baseline, current)
        assert np.isfinite(psi)
        assert psi >= 0.0


# ---------------------------------------------------------------------------
# KS tests
# ---------------------------------------------------------------------------


class TestKS:
    def test_ks_same_distribution(self) -> None:
        """Same distribution should not be flagged as drifted."""
        np.random.seed(42)
        data = np.random.randn(500)
        drifted, stat, pvalue = check_ks_drift(data, data)
        assert not drifted
        assert pvalue > 0.05

    def test_ks_different_distribution(self) -> None:
        """Clearly different distributions should be flagged."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 500)
        current = np.random.normal(3, 1, 500)
        drifted, stat, pvalue = check_ks_drift(baseline, current)
        assert drifted
        assert pvalue < 0.05


# ---------------------------------------------------------------------------
# CUSUM tests
# ---------------------------------------------------------------------------


class TestCUSUM:
    def test_cusum_stable_signal(self) -> None:
        """Stable signal near target should not trigger drift."""
        np.random.seed(42)
        det = CUSUMDetector(target=0.0, threshold=5.0, drift=0.5)
        triggered = False
        for _ in range(100):
            if det.update(np.random.normal(0, 0.1)):
                triggered = True
        assert not triggered

    def test_cusum_mean_shift(self) -> None:
        """Mean shift should trigger drift detection."""
        np.random.seed(42)
        det = CUSUMDetector(target=0.0, threshold=5.0, drift=0.5)
        # Feed 50 values near 0, then 50 at 10
        triggered = False
        for _ in range(50):
            det.update(np.random.normal(0, 0.1))
        for _ in range(50):
            if det.update(10.0):
                triggered = True
        assert triggered
        assert det.drift_detected

    def test_cusum_reset(self) -> None:
        """After reset, accumulators should be zero."""
        det = CUSUMDetector(target=0.0, threshold=5.0, drift=0.5)
        # Trigger drift
        for _ in range(20):
            det.update(10.0)
        assert det.drift_detected
        det.reset()
        assert det.cumulative_sum == (0.0, 0.0)
        assert not det.drift_detected


# ---------------------------------------------------------------------------
# ADWIN tests
# ---------------------------------------------------------------------------


class TestADWIN:
    def test_adwin_stable(self) -> None:
        """Stable distribution should produce zero drift detections."""
        np.random.seed(42)
        det = ADWINDetector(delta=0.002, grace_period=30)
        for _ in range(200):
            det.update(np.random.normal(0, 1))
        assert det.n_detections == 0

    def test_adwin_concept_drift(self) -> None:
        """Concept drift (mean shift) should be detected."""
        np.random.seed(42)
        det = ADWINDetector(delta=0.002, grace_period=30)
        any_drift = False
        for _ in range(100):
            det.update(np.random.normal(0, 1))
        for _ in range(100):
            if det.update(np.random.normal(5, 1)):
                any_drift = True
        assert any_drift
        assert det.n_detections > 0
