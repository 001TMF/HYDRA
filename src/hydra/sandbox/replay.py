"""Market replay engine for bar-by-bar backtesting of pre-trained models.

Unlike WalkForwardEngine (which trains per fold), MarketReplayEngine takes
a pre-trained model and simulates execution over a date range with per-bar
slippage that adapts to each bar's volume and spread conditions.

This is the evaluation mechanism for candidate models in the experiment loop.
The key difference from WalkForwardEngine: volumes[i] and spreads[i] change
per bar, producing volume-adaptive slippage instead of static estimates.

Observer callbacks receive TradeEvent per trade for real-time drift monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from hydra.risk.position_sizing import fractional_kelly, volume_capped_position
from hydra.risk.slippage import estimate_slippage


@dataclass
class TradeEvent:
    """Per-bar trade details emitted to observer callbacks.

    Captures everything needed for drift monitoring and trade analysis.
    """

    bar_idx: int
    direction: int  # +1 long, -1 short
    n_contracts: int
    price: float
    volume: float
    spread: float
    slippage_per_contract: float
    raw_return: float
    net_return: float
    capital_after: float


@dataclass
class ReplayResult:
    """Final output of a replay run.

    Provides the same metrics as BacktestResult for comparison,
    but self-contained to avoid circular import risk.
    """

    sharpe_ratio: float
    total_return: float
    max_drawdown: float  # negative value, e.g. -0.15 for 15% drawdown
    hit_rate: float
    n_trades: int
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    trade_log: list[TradeEvent] = field(default_factory=list)
    daily_returns: np.ndarray = field(default_factory=lambda: np.array([]))


class MarketReplayEngine:
    """Bar-by-bar replay engine with volume-adaptive slippage.

    Takes a pre-trained model and replays historical data one bar at a time.
    For each bar, the model predicts direction, position sizing is computed
    via fractional Kelly + volume cap, and slippage is estimated using
    that bar's actual volume and spread (not static averages).

    Parameters
    ----------
    config : dict | None
        Engine configuration. Keys:
        - initial_capital (float): Starting capital (default 100_000)
        - kelly_fraction (float): Fraction of Kelly to use (default 0.5)
        - max_position_pct (float): Max position as fraction of capital (default 0.10)
        - max_volume_pct (float): Max fraction of daily volume (default 0.02)
        - contract_value (float): Value per contract (default 1000)
        - impact_coefficient (float): Slippage impact coefficient (default 0.1)
        - prediction_threshold (float): Probability threshold for direction (default 0.5)
    """

    DEFAULT_CONFIG = {
        "initial_capital": 100_000.0,
        "kelly_fraction": 0.5,
        "max_position_pct": 0.10,
        "max_volume_pct": 0.02,
        "contract_value": 1000.0,
        "impact_coefficient": 0.1,
        "prediction_threshold": 0.5,
    }

    def __init__(self, config: dict | None = None) -> None:
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._callbacks: list[Callable[[TradeEvent], None]] = []

    def add_callback(self, fn: Callable[[TradeEvent], None]) -> None:
        """Register an observer callback that receives TradeEvent per trade.

        Callbacks are invoked in registration order for each trade event.
        Used by drift monitoring hooks and experiment logging.

        Parameters
        ----------
        fn : Callable[[TradeEvent], None]
            Function to call with each TradeEvent.
        """
        self._callbacks.append(fn)

    def replay(
        self,
        model,
        features: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        spreads: np.ndarray,
    ) -> ReplayResult:
        """Replay historical data bar-by-bar through a pre-trained model.

        For each bar:
        1. model.predict_proba(features[i:i+1]) to get P(up)
        2. Direction from prediction vs threshold
        3. Position sizing via fractional_kelly + volume_capped_position
        4. Per-bar slippage via estimate_slippage with that bar's volume/spread
        5. Compute raw return from price movement
        6. Net return = direction * raw_return - slippage_frac
        7. Create TradeEvent, emit to all callbacks
        8. Track equity, wins/losses for Kelly updates

        Parameters
        ----------
        model
            Pre-trained model with predict_proba(X) -> np.ndarray method.
        features : np.ndarray
            Feature matrix of shape (N, F).
        prices : np.ndarray
            Price series of shape (N,).
        volumes : np.ndarray
            Per-bar volumes of shape (N,).
        spreads : np.ndarray
            Per-bar bid-ask spreads of shape (N,).

        Returns
        -------
        ReplayResult
            Complete replay results with metrics, equity curve, and trade log.
        """
        n_bars = len(features)

        # Handle empty data
        if n_bars == 0:
            return ReplayResult(
                sharpe_ratio=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                hit_rate=0.0,
                n_trades=0,
                equity_curve=np.array([]),
                trade_log=[],
                daily_returns=np.array([]),
            )

        capital = self.config["initial_capital"]
        threshold = self.config["prediction_threshold"]

        equity_curve = np.zeros(n_bars)
        daily_returns = np.zeros(n_bars)
        trade_log: list[TradeEvent] = []

        # Running statistics for Kelly sizing
        wins: list[float] = []
        losses: list[float] = []

        for i in range(n_bars):
            # Get model prediction
            proba = float(model.predict_proba(features[i : i + 1])[0])

            # Direction from prediction vs threshold
            direction = 1 if proba >= threshold else -1

            # Compute raw return from price movement
            raw_return = self._compute_raw_return(prices, i)

            # Position sizing via fractional Kelly
            win_prob = max(0.01, min(0.99, proba if direction == 1 else 1 - proba))
            avg_win = np.mean(wins) if wins else 0.01
            avg_loss = abs(np.mean(losses)) if losses else 0.01

            kelly_pct = fractional_kelly(
                win_prob=win_prob,
                avg_win=avg_win,
                avg_loss=avg_loss,
                fraction=self.config["kelly_fraction"],
                max_position_pct=self.config["max_position_pct"],
            )

            n_contracts = volume_capped_position(
                kelly_pct=kelly_pct,
                capital=capital,
                contract_value=self.config["contract_value"],
                avg_daily_volume=float(volumes[i]),
                max_volume_pct=self.config["max_volume_pct"],
            )

            if n_contracts == 0:
                # No trade -- record zero return
                equity_curve[i] = capital
                daily_returns[i] = 0.0
                continue

            # Per-bar slippage using that bar's volume and spread
            rolling_vol = self._rolling_vol(prices, i)
            slippage_per_contract = estimate_slippage(
                order_size=n_contracts,
                daily_volume=float(volumes[i]),
                spread=float(spreads[i]),
                daily_volatility=rolling_vol,
                impact_coefficient=self.config["impact_coefficient"],
            )

            # Total slippage cost as fraction of position value
            position_value = n_contracts * self.config["contract_value"]
            total_slippage = slippage_per_contract * n_contracts
            slippage_frac = (
                total_slippage / position_value if position_value > 0 else 0.0
            )

            # Net return = directional return - slippage cost
            net_return = direction * raw_return - slippage_frac

            # Scale by position size (fraction of capital deployed)
            capital_return = net_return * kelly_pct

            # Update capital
            capital *= 1 + capital_return
            equity_curve[i] = capital
            daily_returns[i] = capital_return

            # Track wins/losses for Kelly updates
            if capital_return > 0:
                wins.append(capital_return)
            elif capital_return < 0:
                losses.append(capital_return)

            # Create trade event
            event = TradeEvent(
                bar_idx=i,
                direction=direction,
                n_contracts=n_contracts,
                price=float(prices[i]),
                volume=float(volumes[i]),
                spread=float(spreads[i]),
                slippage_per_contract=slippage_per_contract,
                raw_return=raw_return,
                net_return=net_return,
                capital_after=capital,
            )
            trade_log.append(event)

            # Emit to all callbacks
            for cb in self._callbacks:
                cb(event)

        # Fill equity curve for bars where no trade happened (carry forward)
        # First pass already set traded bars; fill gaps with carry-forward
        last_equity = self.config["initial_capital"]
        for i in range(n_bars):
            if equity_curve[i] == 0.0:
                equity_curve[i] = last_equity
            else:
                last_equity = equity_curve[i]

        # Compute final metrics
        n_trades = len(trade_log)
        sharpe = self._compute_sharpe(daily_returns)
        total_return = (equity_curve[-1] / self.config["initial_capital"]) - 1.0
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Hit rate: fraction of trades where direction matched price movement
        if n_trades > 0:
            correct = sum(
                1
                for t in trade_log
                if (t.direction > 0 and t.raw_return > 0)
                or (t.direction < 0 and t.raw_return < 0)
            )
            hit_rate = correct / n_trades
        else:
            hit_rate = 0.0

        return ReplayResult(
            sharpe_ratio=sharpe,
            total_return=total_return,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            n_trades=n_trades,
            equity_curve=equity_curve,
            trade_log=trade_log,
            daily_returns=daily_returns,
        )

    @staticmethod
    def _rolling_vol(
        prices: np.ndarray, idx: int, window: int = 20
    ) -> float:
        """Compute rolling daily volatility from log returns.

        Uses a lookback window of log returns ending at idx.
        Returns 0.02 default if insufficient data (< 2 prices).

        Parameters
        ----------
        prices : np.ndarray
            Price series.
        idx : int
            Current bar index.
        window : int
            Lookback window size (default 20).

        Returns
        -------
        float
            Rolling daily volatility estimate.
        """
        start = max(0, idx - window)
        if idx - start < 2:
            return 0.02  # default volatility

        price_slice = prices[start : idx + 1]
        log_returns = np.diff(np.log(price_slice))
        return float(np.std(log_returns, ddof=1))

    @staticmethod
    def _compute_raw_return(prices: np.ndarray, idx: int) -> float:
        """Compute simple return at index from price series.

        Uses single-period return: (price[idx] - price[idx-1]) / price[idx-1].
        For the first index, returns 0.

        Parameters
        ----------
        prices : np.ndarray
            Price series.
        idx : int
            Current bar index.

        Returns
        -------
        float
            Simple return at the given index.
        """
        if idx <= 0 or idx >= len(prices):
            return 0.0
        return float((prices[idx] - prices[idx - 1]) / prices[idx - 1])

    @staticmethod
    def _compute_sharpe(
        returns: np.ndarray, risk_free_rate: float = 0.0
    ) -> float:
        """Compute annualized Sharpe ratio.

        Sharpe = (mean_return - rf_daily) / std_return * sqrt(252)
        Returns 0.0 if std is zero.

        Parameters
        ----------
        returns : np.ndarray
            Daily returns array.
        risk_free_rate : float
            Annualized risk-free rate (default 0.0).

        Returns
        -------
        float
            Annualized Sharpe ratio.
        """
        if len(returns) == 0:
            return 0.0
        daily_rf = risk_free_rate / 252.0
        excess = returns - daily_rf
        std = float(np.std(excess, ddof=1)) if len(returns) > 1 else 0.0
        if std == 0:
            return 0.0
        return float(np.mean(excess)) / std * float(np.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
        """Compute maximum drawdown from equity curve.

        Returns a negative value (e.g., -0.15 for 15% drawdown).
        Returns 0.0 if no drawdown occurs.

        Parameters
        ----------
        equity_curve : np.ndarray
            Equity curve (capital values over time).

        Returns
        -------
        float
            Maximum drawdown as a negative fraction.
        """
        if len(equity_curve) == 0:
            return 0.0
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return float(np.min(drawdowns))
