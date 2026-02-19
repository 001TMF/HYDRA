"""Tests for backtest evaluation metrics.

Tests verify:
- Sharpe ratio sign correctness
- Max drawdown calculation
- Hit rate computation
- Edge cases (empty returns, zero std)
"""

import numpy as np
import pytest

from hydra.model.evaluation import BacktestResult, compute_backtest_metrics


class TestComputeBacktestMetrics:
    """Tests for compute_backtest_metrics."""

    def test_sharpe_positive_returns(self):
        """All positive returns produce Sharpe > 0."""
        returns = np.array([0.01, 0.02, 0.005, 0.015, 0.01] * 10)
        preds = np.ones(50, dtype=int)
        actuals = np.ones(50, dtype=int)

        result = compute_backtest_metrics(returns, preds, actuals)
        assert result.sharpe_ratio > 0

    def test_sharpe_negative_returns(self):
        """All negative returns produce Sharpe < 0."""
        returns = np.array([-0.01, -0.02, -0.005, -0.015, -0.01] * 10)
        preds = np.zeros(50, dtype=int)
        actuals = np.ones(50, dtype=int)

        result = compute_backtest_metrics(returns, preds, actuals)
        assert result.sharpe_ratio < 0

    def test_max_drawdown_calculation(self):
        """Known equity curve produces known max drawdown."""
        # Equity: goes up to 1.10, then drops to 0.99, then recovers
        # Drawdown from peak 1.10 to trough 0.99 = (0.99-1.10)/1.10 = -0.10
        returns = np.array([0.05, 0.05, -0.05, -0.06, 0.03, 0.02])
        preds = np.array([1, 1, 0, 0, 1, 1])
        actuals = np.array([1, 1, 0, 0, 1, 1])

        result = compute_backtest_metrics(returns, preds, actuals)

        # Max drawdown should be negative
        assert result.max_drawdown < 0
        # Equity curve: 1.05, 1.1025, 1.047375, 0.984533, 1.014069, 1.034350
        # Peak at 1.1025, trough at 0.984533 => dd = (0.984533-1.1025)/1.1025 = -0.107
        assert result.max_drawdown == pytest.approx(-0.107, abs=0.01)

    def test_hit_rate_perfect(self):
        """All correct predictions produce hit_rate = 1.0."""
        returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
        preds = np.array([1, 0, 1, 0, 1])
        actuals = np.array([1, 0, 1, 0, 1])

        result = compute_backtest_metrics(returns, preds, actuals)
        assert result.hit_rate == 1.0

    def test_hit_rate_random(self):
        """50/50 predictions produce hit_rate close to 0.5."""
        np.random.seed(42)
        n = 1000
        returns = np.random.randn(n) * 0.01
        preds = np.random.randint(0, 2, n)
        actuals = np.random.randint(0, 2, n)

        result = compute_backtest_metrics(returns, preds, actuals)
        assert 0.4 <= result.hit_rate <= 0.6

    def test_empty_returns(self):
        """Empty returns produce zero metrics."""
        result = compute_backtest_metrics(
            np.array([]), np.array([]), np.array([])
        )
        assert result.sharpe_ratio == 0.0
        assert result.total_return == 0.0
        assert result.max_drawdown == 0.0
        assert result.n_trades == 0

    def test_total_return_computation(self):
        """Total return equals cumulative product minus 1."""
        returns = np.array([0.10, -0.05, 0.03])
        preds = np.array([1, 0, 1])
        actuals = np.array([1, 0, 1])

        result = compute_backtest_metrics(returns, preds, actuals)
        expected = (1.10 * 0.95 * 1.03) - 1.0
        assert result.total_return == pytest.approx(expected, abs=1e-10)

    def test_n_trades_counts_nonzero(self):
        """n_trades counts only non-zero return entries."""
        returns = np.array([0.01, 0.0, -0.02, 0.0, 0.005])
        preds = np.array([1, 0, 1, 0, 1])
        actuals = np.array([1, 0, 0, 0, 1])

        result = compute_backtest_metrics(returns, preds, actuals)
        assert result.n_trades == 3

    def test_fold_sharpes_computed(self):
        """Per-fold Sharpe ratios computed when fold_returns provided."""
        fold1 = np.array([0.01, 0.02, -0.005, 0.01, 0.005] * 5)
        fold2 = np.array([-0.01, -0.02, -0.005, -0.01, -0.005] * 5)

        all_returns = np.concatenate([fold1, fold2])
        preds = np.ones(len(all_returns), dtype=int)
        actuals = np.ones(len(all_returns), dtype=int)

        result = compute_backtest_metrics(
            all_returns, preds, actuals, fold_returns=[fold1, fold2]
        )
        assert len(result.fold_sharpes) == 2
        assert result.fold_sharpes[0] > 0  # positive fold
        assert result.fold_sharpes[1] < 0  # negative fold

    def test_equity_curve_length(self):
        """Equity curve has same length as returns."""
        returns = np.array([0.01, -0.01, 0.02])
        preds = np.array([1, 0, 1])
        actuals = np.array([1, 0, 1])

        result = compute_backtest_metrics(returns, preds, actuals)
        assert len(result.equity_curve) == len(returns)
