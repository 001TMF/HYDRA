"""Backtest evaluation metrics computation.

Computes slippage-adjusted backtest metrics from walk-forward engine output:
Sharpe ratio, total return, max drawdown, hit rate, per-fold statistics,
equity curve, and trade log.

All return-based metrics are slippage-adjusted (slippage already applied
to the returns before they reach this module).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BacktestResult:
    """Results from walk-forward backtest.

    All return metrics are slippage-adjusted.
    """

    sharpe_ratio: float  # annualized, slippage-adjusted
    total_return: float  # cumulative return after slippage
    max_drawdown: float  # maximum drawdown (negative value)
    hit_rate: float  # fraction of correct directional predictions
    avg_return_per_trade: float  # average return per trade after slippage
    n_trades: int  # total number of trades executed
    fold_sharpes: list[float] = field(default_factory=list)  # per-fold Sharpe ratios
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_log: list[dict] = field(default_factory=list)  # individual trade records


def compute_backtest_metrics(
    returns: np.ndarray,
    predictions: np.ndarray,
    actuals: np.ndarray,
    fold_returns: list[np.ndarray] | None = None,
    risk_free_rate: float = 0.0,
    trade_log: list[dict] | None = None,
) -> BacktestResult:
    """Compute all backtest metrics from daily returns.

    Parameters
    ----------
    returns : np.ndarray
        Daily returns, already slippage-adjusted.
    predictions : np.ndarray
        Model predictions (0 or 1).
    actuals : np.ndarray
        Actual outcomes (0 or 1).
    fold_returns : list[np.ndarray] | None
        Per-fold return arrays for computing per-fold Sharpe ratios.
    risk_free_rate : float
        Annualized risk-free rate (default 0.0).
    trade_log : list[dict] | None
        Individual trade records from the engine (passed through to result).

    Returns
    -------
    BacktestResult
        Complete backtest metrics.
    """
    if len(returns) == 0:
        return BacktestResult(
            sharpe_ratio=0.0,
            total_return=0.0,
            max_drawdown=0.0,
            hit_rate=0.0,
            avg_return_per_trade=0.0,
            n_trades=0,
            fold_sharpes=[],
            equity_curve=np.array([1.0]),
            trade_log=[],
        )

    # Sharpe ratio: annualized
    sharpe = _compute_sharpe(returns, risk_free_rate)

    # Total return: cumulative product
    total_return = float(np.prod(1 + returns) - 1)

    # Equity curve: cumulative product starting at 1.0
    equity_curve = np.cumprod(1 + returns)

    # Max drawdown from running max of equity curve
    max_drawdown = _compute_max_drawdown(equity_curve)

    # Hit rate: fraction of correct directional predictions
    hit_rate = float(np.mean(predictions == actuals)) if len(actuals) > 0 else 0.0

    # Count actual trades (non-zero returns)
    n_trades = int(np.sum(returns != 0))

    # Average return per trade (non-zero trades only)
    nonzero_returns = returns[returns != 0]
    avg_return_per_trade = (
        float(np.mean(nonzero_returns)) if len(nonzero_returns) > 0 else 0.0
    )

    # Per-fold Sharpe ratios
    fold_sharpes: list[float] = []
    if fold_returns is not None:
        for fr in fold_returns:
            fold_sharpes.append(_compute_sharpe(fr, risk_free_rate))

    return BacktestResult(
        sharpe_ratio=sharpe,
        total_return=total_return,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        avg_return_per_trade=avg_return_per_trade,
        n_trades=n_trades,
        fold_sharpes=fold_sharpes,
        equity_curve=equity_curve,
        trade_log=trade_log if trade_log is not None else [],
    )


def _compute_sharpe(
    returns: np.ndarray, risk_free_rate: float = 0.0
) -> float:
    """Compute annualized Sharpe ratio.

    Sharpe = (mean_return - rf_daily) / std_return * sqrt(252)

    Returns 0.0 if std is zero (avoids division by zero).
    """
    if len(returns) == 0:
        return 0.0
    daily_rf = risk_free_rate / 252.0
    excess = returns - daily_rf
    std = float(np.std(excess, ddof=1)) if len(returns) > 1 else 0.0
    if std == 0:
        return 0.0
    mean_excess = float(np.mean(excess))
    return mean_excess / std * np.sqrt(252)


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum drawdown from equity curve.

    Returns a negative value (e.g., -0.15 for 15% drawdown).
    Returns 0.0 if no drawdown occurs.
    """
    if len(equity_curve) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    return float(np.min(drawdowns))
