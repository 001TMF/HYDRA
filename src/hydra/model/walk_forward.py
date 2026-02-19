"""Walk-forward backtesting engine with purged/embargoed cross-validation.

PurgedWalkForwardSplit: Expanding-window train/test splitter with configurable
embargo gap to prevent temporal leakage.

WalkForwardEngine: Full backtest loop integrating BaselineModel, slippage,
Kelly position sizing, and circuit breakers. Trains per fold, predicts OOS,
applies the full risk stack, and produces BacktestResult.

Timeline per fold: [====TRAIN====][--EMBARGO--][==TEST==]
"""

from __future__ import annotations

import numpy as np

from hydra.model.baseline import BaselineModel
from hydra.model.evaluation import BacktestResult, compute_backtest_metrics
from hydra.risk.circuit_breakers import CircuitBreakerManager
from hydra.risk.position_sizing import fractional_kelly, volume_capped_position
from hydra.risk.slippage import estimate_slippage


class PurgedWalkForwardSplit:
    """Walk-forward cross-validator with embargo gap.

    Timeline per fold: [====TRAIN====][--EMBARGO--][==TEST==]

    Expanding window: train set grows each fold.
    Embargo gap: N trading days removed between train end and test start
    to prevent information leakage from overlapping labels.

    Parameters
    ----------
    n_splits : int
        Maximum number of folds to produce.
    embargo_days : int
        Number of trading days removed between train end and test start.
    min_train_size : int
        Minimum number of training samples for the first fold (~1 year).
    test_size : int
        Number of test samples per fold (~3 months).
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        min_train_size: int = 252,
        test_size: int = 63,
    ) -> None:
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return list of (train_indices, test_indices) tuples.

        Fold k train: [0, train_end_k)
        Fold k test: [train_end_k + embargo_days, train_end_k + embargo_days + test_size)

        Expanding: train_end grows by test_size each fold.

        If n_samples is too small for all n_splits, returns fewer folds.

        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) pairs.
        """
        folds: list[tuple[np.ndarray, np.ndarray]] = []

        for k in range(self.n_splits):
            train_end = self.min_train_size + k * self.test_size
            test_start = train_end + self.embargo_days
            test_end = test_start + self.test_size

            # Not enough data for this fold
            if test_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))

        return folds

    def get_n_splits(self) -> int:
        """Return the configured number of splits."""
        return self.n_splits


class WalkForwardEngine:
    """Runs walk-forward backtest with full risk stack.

    Per fold:
    1. Train BaselineModel on train set
    2. Predict probabilities on test set
    3. For each test day:
       a. Check circuit breakers (skip if triggered)
       b. Compute position size via fractional Kelly + volume cap
       c. Compute slippage via estimate_slippage
       d. Record trade with slippage-adjusted return
    4. Compute per-fold metrics
    5. Aggregate across all folds

    Parameters
    ----------
    config : dict | None
        Engine configuration. Keys:
        - n_splits (int): Number of walk-forward folds (default 5)
        - embargo_days (int): Embargo gap in trading days (default 5)
        - min_train_size (int): Minimum training samples (default 252)
        - test_size (int): Test samples per fold (default 63)
        - initial_capital (float): Starting capital (default 100_000)
        - spread (float): Bid-ask spread for slippage (default 0.50)
        - daily_volume (float): Average daily volume in contracts (default 5000)
        - daily_volatility (float): Daily price volatility (default 0.02)
        - contract_value (float): Value per contract (default 1000)
        - impact_coefficient (float): Slippage impact coefficient (default 0.1)
        - kelly_fraction (float): Fraction of Kelly to use (default 0.5)
        - max_position_pct (float): Max position as fraction of capital (default 0.10)
        - max_volume_pct (float): Max fraction of daily volume (default 0.02)
        - circuit_breakers (dict): CircuitBreakerManager config
        - prediction_horizon (int): Forward-looking days for returns (default 5)
    """

    DEFAULT_CONFIG = {
        "n_splits": 5,
        "embargo_days": 5,
        "min_train_size": 252,
        "test_size": 63,
        "initial_capital": 100_000.0,
        "spread": 0.50,
        "daily_volume": 5000.0,
        "daily_volatility": 0.02,
        "contract_value": 1000.0,
        "impact_coefficient": 0.1,
        "kelly_fraction": 0.5,
        "max_position_pct": 0.10,
        "max_volume_pct": 0.02,
        "circuit_breakers": {},
        "prediction_horizon": 5,
    }

    def __init__(self, config: dict | None = None) -> None:
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prices: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> BacktestResult:
        """Execute full walk-forward backtest.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, F).
        y : np.ndarray
            Binary targets of shape (N,).
        prices : np.ndarray
            Price series for PnL computation of shape (N,).
        feature_names : list[str] | None
            Optional feature names for model training.

        Returns
        -------
        BacktestResult
            Full backtest results with all metrics.
        """
        splitter = PurgedWalkForwardSplit(
            n_splits=self.config["n_splits"],
            embargo_days=self.config["embargo_days"],
            min_train_size=self.config["min_train_size"],
            test_size=self.config["test_size"],
        )
        folds = splitter.split(len(X))

        if len(folds) == 0:
            return BacktestResult(
                sharpe_ratio=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                hit_rate=0.0,
                avg_return_per_trade=0.0,
                n_trades=0,
                fold_sharpes=[],
                equity_curve=np.array([self.config["initial_capital"]]),
                trade_log=[],
            )

        all_returns: list[float] = []
        all_predictions: list[int] = []
        all_actuals: list[int] = []
        fold_returns_list: list[np.ndarray] = []
        trade_log: list[dict] = []

        capital = self.config["initial_capital"]
        peak_equity = capital
        daily_pnl = 0.0

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Train model on train set
            model = BaselineModel()
            model.train(X[train_idx], y[train_idx], feature_names)

            # Predict on test set
            probas = model.predict_proba(X[test_idx])
            preds = (probas >= 0.5).astype(int)

            # Circuit breaker manager (fresh per fold)
            cb_manager = CircuitBreakerManager(
                self.config.get("circuit_breakers", {})
            )

            fold_returns: list[float] = []
            # Track running win statistics for Kelly sizing
            wins: list[float] = []
            losses: list[float] = []

            for i, (pred, proba, actual) in enumerate(
                zip(preds, probas, y[test_idx])
            ):
                test_day_idx = test_idx[i]

                # Compute raw return from price movement
                raw_return = self._compute_raw_return(prices, test_day_idx)

                # Check circuit breakers
                drawdown_pct = (
                    (capital - peak_equity) / peak_equity
                    if peak_equity > 0
                    else 0.0
                )
                position_pct = self.config["max_position_pct"]

                allowed, triggered = cb_manager.check_trade(
                    daily_pnl=daily_pnl / max(capital, 1),
                    peak_equity=peak_equity,
                    current_equity=capital,
                    position_value=position_pct,
                    trade_loss=(
                        losses[-1] if losses else 0.0
                    ),
                )

                if not allowed:
                    # Skip trade, record zero return
                    fold_returns.append(0.0)
                    cb_manager.advance_period()
                    continue

                # Position sizing via fractional Kelly
                win_prob = max(0.01, min(0.99, proba))
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
                    avg_daily_volume=self.config["daily_volume"],
                    max_volume_pct=self.config["max_volume_pct"],
                )

                if n_contracts == 0:
                    fold_returns.append(0.0)
                    cb_manager.advance_period()
                    continue

                # Compute slippage
                slippage_per_contract = estimate_slippage(
                    order_size=n_contracts,
                    daily_volume=self.config["daily_volume"],
                    spread=self.config["spread"],
                    daily_volatility=self.config["daily_volatility"],
                    impact_coefficient=self.config["impact_coefficient"],
                )

                # Total slippage cost as fraction of position value
                position_value = n_contracts * self.config["contract_value"]
                total_slippage = slippage_per_contract * n_contracts
                slippage_frac = (
                    total_slippage / position_value if position_value > 0 else 0.0
                )

                # Direction: long if pred==1, short if pred==0
                direction = 1 if pred == 1 else -1
                # Net return = directional return - slippage cost
                net_return = direction * raw_return - slippage_frac

                # Scale by position size (fraction of capital deployed)
                capital_return = net_return * kelly_pct

                fold_returns.append(capital_return)

                # Update equity tracking
                capital *= 1 + capital_return
                peak_equity = max(peak_equity, capital)
                daily_pnl += capital_return

                # Track wins/losses for Kelly updates
                if capital_return > 0:
                    wins.append(capital_return)
                elif capital_return < 0:
                    losses.append(capital_return)

                # Record trade in log
                trade_log.append({
                    "fold": fold_idx,
                    "test_day": int(test_day_idx),
                    "prediction": int(pred),
                    "probability": float(proba),
                    "actual": int(actual),
                    "direction": direction,
                    "n_contracts": n_contracts,
                    "raw_return": float(raw_return),
                    "slippage_frac": float(slippage_frac),
                    "net_return": float(net_return),
                    "capital_return": float(capital_return),
                    "capital_after": float(capital),
                })

                cb_manager.advance_period()

            fold_arr = np.array(fold_returns)
            fold_returns_list.append(fold_arr)
            all_returns.extend(fold_returns)
            all_predictions.extend(preds.tolist())
            all_actuals.extend(y[test_idx].tolist())

            # Reset daily PnL per fold
            daily_pnl = 0.0

        return compute_backtest_metrics(
            returns=np.array(all_returns),
            predictions=np.array(all_predictions),
            actuals=np.array(all_actuals),
            fold_returns=fold_returns_list,
            trade_log=trade_log,
        )

    @staticmethod
    def _compute_raw_return(prices: np.ndarray, idx: int) -> float:
        """Compute simple return at index from price series.

        Uses single-period return: (price[idx] - price[idx-1]) / price[idx-1].
        For the first index, returns 0.
        """
        if idx <= 0 or idx >= len(prices):
            return 0.0
        return (prices[idx] - prices[idx - 1]) / prices[idx - 1]
