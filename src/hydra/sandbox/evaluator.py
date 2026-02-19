"""Composite fitness evaluator for sandbox candidate models.

Scores candidate models on 6 weighted metrics with min-max normalization.
The composite fitness score is the single number that determines promotion
decisions in the sandbox tournament.

Metrics:
    - Sharpe ratio (0.25): Risk-adjusted return quality
    - Max drawdown (0.20): Downside risk control
    - Calibration (0.15): Brier score -- probability accuracy (inverted: lower is better)
    - Robustness (0.15): Fraction of walk-forward folds with positive Sharpe
    - Slippage-adjusted return (0.15): Net return after transaction costs
    - Simplicity (0.10): Model complexity penalty via 1/log2(params + 1)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "sharpe": 0.25,
    "drawdown": 0.20,
    "calibration": 0.15,
    "robustness": 0.15,
    "slippage_adjusted_return": 0.15,
    "simplicity": 0.10,
}

DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "sharpe": (-1.0, 3.0),
    "drawdown": (-0.50, 0.0),
    "calibration": (0.0, 0.5),  # Brier score: 0.5 is random, 0 is perfect
    "robustness": (0.0, 1.0),
    "slippage_adjusted_return": (-0.50, 1.0),
    "simplicity": (0.0, 1.0),
}

# Metrics where lower raw value means better performance (inverted normalization).
_INVERTED_METRICS: frozenset[str] = frozenset({"calibration"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FitnessScore:
    """Result of composite fitness evaluation.

    Attributes
    ----------
    composite : float
        Weighted sum of normalized metrics, in [0, 1].
    components : dict[str, float]
        Per-metric normalized values, each in [0, 1].
    raw_metrics : dict[str, float]
        Per-metric raw values before normalization.
    weights : dict[str, float]
        The weights used for this evaluation.
    """

    composite: float
    components: dict[str, float] = field(default_factory=dict)
    raw_metrics: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class CompositeEvaluator:
    """Evaluates candidate models using a weighted composite fitness score.

    Each raw metric is min-max normalized to [0, 1] using configurable ranges,
    then combined via a weighted sum.  Calibration (Brier score) is inverted
    so that lower raw values produce higher normalized scores.

    Parameters
    ----------
    weights : dict[str, float] | None
        Metric weights.  Must sum to 1.0 (tolerance 1e-6).
        Defaults to ``DEFAULT_WEIGHTS``.
    ranges : dict[str, tuple[float, float]] | None
        ``{metric: (lo, hi)}`` used for min-max normalization.
        Defaults to ``DEFAULT_RANGES``.

    Raises
    ------
    ValueError
        If weights do not sum to 1.0 within tolerance.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)
        self.ranges = ranges or dict(DEFAULT_RANGES)

        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.6f}"
            )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(self, metrics: dict[str, float]) -> FitnessScore:
        """Compute composite fitness from raw metric values.

        Parameters
        ----------
        metrics : dict[str, float]
            Raw metric values keyed by metric name.  Must contain every
            key present in ``self.weights``.

        Returns
        -------
        FitnessScore
            Composite and per-component scores.

        Raises
        ------
        KeyError
            If a required metric is missing from *metrics*.
        """
        components: dict[str, float] = {}
        raw_metrics: dict[str, float] = {}

        for name in self.weights:
            if name not in metrics:
                raise KeyError(f"Missing required metric: '{name}'")

            raw = float(metrics[name])
            raw_metrics[name] = raw

            lo, hi = self.ranges[name]
            normalized = float(np.clip((raw - lo) / (hi - lo), 0.0, 1.0))

            # Invert metrics where lower raw value is better.
            if name in _INVERTED_METRICS:
                normalized = 1.0 - normalized

            components[name] = normalized

        composite = sum(
            self.weights[name] * components[name] for name in self.weights
        )

        return FitnessScore(
            composite=composite,
            components=components,
            raw_metrics=raw_metrics,
            weights=dict(self.weights),
        )

    # ------------------------------------------------------------------
    # Helper computations
    # ------------------------------------------------------------------

    @staticmethod
    def compute_robustness(fold_sharpes: list[float]) -> float:
        """Fraction of walk-forward folds with Sharpe > 0.

        Returns 0.0 if *fold_sharpes* is empty.
        """
        if not fold_sharpes:
            return 0.0
        positive = sum(1 for s in fold_sharpes if s > 0)
        return positive / len(fold_sharpes)

    @staticmethod
    def compute_simplicity(n_features: int, n_estimators: int) -> float:
        """Model complexity penalty: ``1 / log2(n_features * n_estimators + 1)``.

        Result is clipped to [0, 1].
        """
        complexity = n_features * n_estimators + 1
        return float(np.clip(1.0 / np.log2(complexity), 0.0, 1.0))

    @staticmethod
    def compute_calibration(
        probabilities: np.ndarray, actuals: np.ndarray
    ) -> float:
        """Brier score: mean squared error between probabilities and outcomes.

        Lower is better (0 = perfect calibration).
        """
        return float(np.mean((probabilities - actuals) ** 2))

    # ------------------------------------------------------------------
    # Convenience: score from BacktestResult (duck-typed)
    # ------------------------------------------------------------------

    def score_from_backtest(
        self,
        backtest_result: object,
        probabilities: np.ndarray,
        actuals: np.ndarray,
        n_features: int,
        n_estimators: int,
    ) -> FitnessScore:
        """Score a model using a BacktestResult-like object.

        Uses duck-typing -- accesses ``.sharpe_ratio``, ``.max_drawdown``,
        ``.total_return``, and ``.fold_sharpes`` on *backtest_result* without
        importing the concrete class.

        Parameters
        ----------
        backtest_result : object
            Any object with the four attributes listed above.
        probabilities : np.ndarray
            Predicted probabilities for calibration.
        actuals : np.ndarray
            Actual binary outcomes for calibration.
        n_features : int
            Number of features in the model.
        n_estimators : int
            Number of estimators (trees) in the model.

        Returns
        -------
        FitnessScore
        """
        metrics = {
            "sharpe": backtest_result.sharpe_ratio,  # type: ignore[attr-defined]
            "drawdown": backtest_result.max_drawdown,  # type: ignore[attr-defined]
            "calibration": self.compute_calibration(probabilities, actuals),
            "robustness": self.compute_robustness(backtest_result.fold_sharpes),  # type: ignore[attr-defined]
            "slippage_adjusted_return": backtest_result.total_return,  # type: ignore[attr-defined]
            "simplicity": self.compute_simplicity(n_features, n_estimators),
        }
        return self.score(metrics)
