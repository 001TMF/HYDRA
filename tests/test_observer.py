"""Integration tests for DriftObserver."""

from __future__ import annotations

import numpy as np
import pytest

from hydra.sandbox.observer import DriftObserver, DriftReport


class TestDriftObserver:
    def test_performance_drift_healthy(self) -> None:
        """Healthy model returns should not trigger degradation flags."""
        np.random.seed(42)
        obs = DriftObserver()
        # Good consistent positive returns
        returns = np.random.normal(0.002, 0.005, 100)
        predictions = np.ones(100)
        actuals = np.ones(100)
        probabilities = np.full(100, 0.8)

        report = obs.check_performance_drift(
            recent_returns=returns,
            predictions=predictions,
            actuals=actuals,
            probabilities=probabilities,
            baseline_sharpe=1.0,
        )
        assert not report.sharpe_degraded
        assert not report.drawdown_alert
        assert not report.hit_rate_degraded

    def test_performance_drift_degraded(self) -> None:
        """Bad returns producing poor Sharpe should flag sharpe_degraded."""
        np.random.seed(42)
        obs = DriftObserver()
        # Very negative returns
        returns = np.random.normal(-0.01, 0.03, 100)
        predictions = np.ones(100)
        # Mostly wrong predictions
        actuals = np.zeros(100)
        probabilities = np.full(100, 0.8)

        report = obs.check_performance_drift(
            recent_returns=returns,
            predictions=predictions,
            actuals=actuals,
            probabilities=probabilities,
            baseline_sharpe=2.0,
        )
        assert report.sharpe_degraded
        assert report.hit_rate_degraded

    def test_feature_drift_detected(self) -> None:
        """Shifted feature distributions should be flagged."""
        np.random.seed(42)
        obs = DriftObserver()
        n_samples = 500
        baseline = np.column_stack([
            np.random.normal(0, 1, n_samples),
            np.random.normal(0, 1, n_samples),
        ])
        current = np.column_stack([
            np.random.normal(5, 1, n_samples),
            np.random.normal(5, 1, n_samples),
        ])
        report = obs.check_feature_drift(
            baseline_features=baseline,
            current_features=current,
            feature_names=["feat_a", "feat_b"],
        )
        assert len(report.drifted_features) > 0
        assert "feat_a" in report.drifted_features
        assert "feat_b" in report.drifted_features

    def test_full_report_needs_diagnosis(self) -> None:
        """Combined degraded performance + feature drift should flag needs_diagnosis."""
        np.random.seed(42)
        obs = DriftObserver()
        n = 100
        returns = np.random.normal(-0.01, 0.03, n)
        predictions = np.ones(n)
        actuals = np.zeros(n)
        probabilities = np.full(n, 0.8)

        baseline_feat = np.random.normal(0, 1, (n, 2))
        current_feat = np.random.normal(5, 1, (n, 2))

        report = obs.get_full_report(
            recent_returns=returns,
            predictions=predictions,
            actuals=actuals,
            probabilities=probabilities,
            baseline_sharpe=2.0,
            baseline_features=baseline_feat,
            current_features=current_feat,
            feature_names=["x1", "x2"],
        )
        assert isinstance(report, DriftReport)
        assert report.needs_diagnosis

    def test_streaming_drift_detection(self) -> None:
        """Streaming detector should flag drift after mean shift."""
        np.random.seed(42)
        obs = DriftObserver()
        # Feed stable values
        for _ in range(100):
            obs.update_streaming("test_metric", np.random.normal(0, 0.1))
        # Feed shifted values
        any_drift = False
        for _ in range(100):
            if obs.update_streaming("test_metric", np.random.normal(10, 0.1)):
                any_drift = True
        assert any_drift
