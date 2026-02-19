"""Tests for CompositeEvaluator -- normalization, weighting, edge cases, metrics.

Covers:
  1. Weight-sum validation
  2. Perfect score (top of every range)
  3. Worst score (bottom of every range)
  4. Mid-range score (~0.5)
  5. Calibration inversion (lower Brier -> higher normalized)
  6. Clipping above-range values
  7. Robustness computation
  8. Simplicity computation
  9. Calibration (Brier score) computation
  10. score_from_backtest integration
  11. All components in [0, 1] with arbitrary inputs
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hydra.sandbox.evaluator import (
    DEFAULT_RANGES,
    DEFAULT_WEIGHTS,
    CompositeEvaluator,
    FitnessScore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(**overrides: float) -> dict[str, float]:
    """Return a complete metrics dict with sensible defaults, applying overrides."""
    base = {
        "sharpe": 1.0,
        "drawdown": -0.10,
        "calibration": 0.15,
        "robustness": 0.6,
        "slippage_adjusted_return": 0.20,
        "simplicity": 0.5,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompositeEvaluator:
    """Core evaluator tests."""

    def test_weights_sum_validation(self) -> None:
        """Weights not summing to 1.0 should raise ValueError."""
        bad_weights = {k: v * 0.5 for k, v in DEFAULT_WEIGHTS.items()}
        with pytest.raises(ValueError, match="sum to 1.0"):
            CompositeEvaluator(weights=bad_weights)

    def test_perfect_score(self) -> None:
        """Metrics at top of each range should yield composite close to 1.0."""
        evaluator = CompositeEvaluator()
        metrics = {
            "sharpe": 3.0,
            "drawdown": 0.0,
            "calibration": 0.0,  # Brier 0 = perfect (inverted -> 1.0)
            "robustness": 1.0,
            "slippage_adjusted_return": 1.0,
            "simplicity": 1.0,
        }
        result = evaluator.score(metrics)
        assert isinstance(result, FitnessScore)
        assert result.composite == pytest.approx(1.0, abs=1e-9)

    def test_worst_score(self) -> None:
        """Metrics at bottom of each range should yield composite close to 0.0."""
        evaluator = CompositeEvaluator()
        metrics = {
            "sharpe": -1.0,
            "drawdown": -0.50,
            "calibration": 0.5,  # Brier 0.5 = random (inverted -> 0.0)
            "robustness": 0.0,
            "slippage_adjusted_return": -0.50,
            "simplicity": 0.0,
        }
        result = evaluator.score(metrics)
        assert result.composite == pytest.approx(0.0, abs=1e-9)

    def test_mid_range_score(self) -> None:
        """Metrics at midpoints should yield composite close to 0.5."""
        evaluator = CompositeEvaluator()
        metrics = {}
        for name, (lo, hi) in DEFAULT_RANGES.items():
            metrics[name] = (lo + hi) / 2.0
        result = evaluator.score(metrics)
        assert result.composite == pytest.approx(0.5, abs=1e-9)

    def test_calibration_inverted(self) -> None:
        """Lower Brier score (better calibration) should produce higher normalized value."""
        evaluator = CompositeEvaluator()

        good_cal = evaluator.score(_make_metrics(calibration=0.1))
        bad_cal = evaluator.score(_make_metrics(calibration=0.4))

        assert good_cal.components["calibration"] > bad_cal.components["calibration"]

    def test_clipping(self) -> None:
        """Values above range max are clipped to normalized 1.0."""
        evaluator = CompositeEvaluator()
        result = evaluator.score(_make_metrics(sharpe=10.0))
        assert result.components["sharpe"] == pytest.approx(1.0, abs=1e-9)

    def test_compute_robustness(self) -> None:
        """Robustness = fraction of folds with Sharpe > 0."""
        fold_sharpes = [0.5, -0.1, 0.3, 0.8, -0.2]
        robustness = CompositeEvaluator.compute_robustness(fold_sharpes)
        assert robustness == pytest.approx(3.0 / 5.0)

    def test_compute_robustness_empty(self) -> None:
        """Empty fold_sharpes returns 0.0."""
        assert CompositeEvaluator.compute_robustness([]) == 0.0

    def test_compute_simplicity(self) -> None:
        """Simplicity for n_features=17, n_estimators=100 is a float in (0, 1)."""
        simplicity = CompositeEvaluator.compute_simplicity(
            n_features=17, n_estimators=100
        )
        assert isinstance(simplicity, float)
        assert 0.0 < simplicity < 1.0

    def test_compute_calibration(self) -> None:
        """Perfect calibration (probs == actuals) should give Brier score close to 0."""
        probs = np.array([0.1, 0.9, 0.5, 0.3])
        actuals = np.array([0.1, 0.9, 0.5, 0.3])
        brier = CompositeEvaluator.compute_calibration(probs, actuals)
        assert brier == pytest.approx(0.0, abs=1e-12)

    def test_score_from_backtest(self) -> None:
        """score_from_backtest extracts metrics from a BacktestResult-like object."""
        evaluator = CompositeEvaluator()

        # Mock backtest result via SimpleNamespace (duck-typed)
        bt = SimpleNamespace(
            sharpe_ratio=1.5,
            max_drawdown=-0.12,
            total_return=0.35,
            fold_sharpes=[0.8, 1.2, -0.1, 0.5, 0.9],
        )
        probs = np.array([0.6, 0.7, 0.4, 0.8, 0.3])
        actuals = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

        result = evaluator.score_from_backtest(
            backtest_result=bt,
            probabilities=probs,
            actuals=actuals,
            n_features=17,
            n_estimators=100,
        )

        assert isinstance(result, FitnessScore)
        assert len(result.components) == 6
        assert all(name in result.components for name in DEFAULT_WEIGHTS)
        assert 0.0 <= result.composite <= 1.0

    def test_components_in_range(self) -> None:
        """All components and composite should be in [0, 1] for arbitrary valid metrics."""
        evaluator = CompositeEvaluator()
        metrics = _make_metrics(
            sharpe=0.8,
            drawdown=-0.25,
            calibration=0.22,
            robustness=0.7,
            slippage_adjusted_return=0.15,
            simplicity=0.4,
        )
        result = evaluator.score(metrics)

        for name, value in result.components.items():
            assert 0.0 <= value <= 1.0, f"{name} component {value} out of [0, 1]"
        assert 0.0 <= result.composite <= 1.0

    def test_missing_metric_raises_key_error(self) -> None:
        """Missing metric in input dict should raise KeyError."""
        evaluator = CompositeEvaluator()
        incomplete = {k: 0.5 for k in list(DEFAULT_WEIGHTS.keys())[:-1]}
        with pytest.raises(KeyError, match="Missing required metric"):
            evaluator.score(incomplete)
