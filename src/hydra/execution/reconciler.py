"""Slippage reconciler: compare predicted vs actual slippage distributions.

Uses fill data from the ``FillJournal`` to compute bias, RMSE, correlation,
and a pessimism multiplier between the predicted slippage (from
``estimate_slippage()``) and the actual fill slippage observed during paper
trading.  This quantifies how optimistic the simulated slippage model is
relative to real fills.

A minimum of 10 fills is required for meaningful statistics.  The
``is_model_calibrated`` method provides a boolean check against configurable
bias and correlation thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from hydra.execution.fill_journal import FillJournal

logger = structlog.get_logger()


@dataclass
class ReconciliationReport:
    """Statistical comparison of predicted vs actual slippage.

    Parameters
    ----------
    n_fills : int
        Number of fills analyzed.
    mean_predicted : float
        Mean predicted slippage across fills.
    mean_actual : float
        Mean actual slippage across fills.
    bias : float
        ``mean_actual - mean_predicted``.  Positive means the model
        underestimates slippage (reality is worse).
    rmse : float
        Root mean squared error between predicted and actual.
    median_predicted : float
        Median predicted slippage.
    median_actual : float
        Median actual slippage.
    correlation : float
        Pearson correlation between predicted and actual arrays.
    pessimism_multiplier : float
        ``mean_actual / mean_predicted``.  Values > 1 mean reality is
        worse than the model predicts.
    percentile_95_actual : float
        95th percentile of actual slippage.
    percentile_95_predicted : float
        95th percentile of predicted slippage.
    """

    n_fills: int
    mean_predicted: float
    mean_actual: float
    bias: float
    rmse: float
    median_predicted: float
    median_actual: float
    correlation: float
    pessimism_multiplier: float
    percentile_95_actual: float
    percentile_95_predicted: float


class SlippageReconciler:
    """Compare predicted vs actual slippage from fill data.

    Wraps a ``FillJournal`` and computes statistical metrics that
    quantify the gap between the simulated slippage model and real
    IB paper-trading fills.

    Parameters
    ----------
    fill_journal : FillJournal
        The journal containing fill records with slippage data.
    """

    #: Minimum number of fills required for meaningful statistics.
    MIN_FILLS = 10

    def __init__(self, fill_journal: FillJournal) -> None:
        self._journal = fill_journal

    def reconcile(
        self,
        symbol: str | None = None,
        since: str | None = None,
    ) -> ReconciliationReport | None:
        """Compute reconciliation metrics for predicted vs actual slippage.

        Parameters
        ----------
        symbol : str | None
            Restrict analysis to a specific instrument.
        since : str | None
            Inclusive lower bound on timestamp (ISO 8601).

        Returns
        -------
        ReconciliationReport | None
            The report, or ``None`` if fewer than ``MIN_FILLS`` fills
            are available (insufficient data for meaningful statistics).
        """
        pairs = self._journal.get_slippage_pairs(symbol=symbol, since=since)

        if len(pairs) < self.MIN_FILLS:
            logger.warning(
                "reconciliation_insufficient_data",
                n_fills=len(pairs),
                min_required=self.MIN_FILLS,
            )
            return None

        predicted = np.array([p[0] for p in pairs])
        actual = np.array([p[1] for p in pairs])

        mean_predicted = float(np.mean(predicted))
        mean_actual = float(np.mean(actual))
        bias = mean_actual - mean_predicted
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        # Pearson correlation -- handle constant arrays gracefully
        if np.std(predicted) == 0 or np.std(actual) == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(predicted, actual)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0

        pessimism_multiplier = mean_actual / max(mean_predicted, 1e-10)

        report = ReconciliationReport(
            n_fills=len(pairs),
            mean_predicted=mean_predicted,
            mean_actual=mean_actual,
            bias=bias,
            rmse=rmse,
            median_predicted=float(np.median(predicted)),
            median_actual=float(np.median(actual)),
            correlation=correlation,
            pessimism_multiplier=pessimism_multiplier,
            percentile_95_actual=float(np.percentile(actual, 95)),
            percentile_95_predicted=float(np.percentile(predicted, 95)),
        )

        logger.info(
            "reconciliation_complete",
            n_fills=report.n_fills,
            bias=round(report.bias, 4),
            rmse=round(report.rmse, 4),
            correlation=round(report.correlation, 4),
            pessimism_multiplier=round(report.pessimism_multiplier, 2),
        )

        return report

    def is_model_calibrated(
        self,
        max_bias: float = 0.5,
        min_correlation: float = 0.3,
    ) -> tuple[bool, str]:
        """Check whether the slippage model is adequately calibrated.

        Parameters
        ----------
        max_bias : float
            Maximum acceptable absolute bias (default 0.5).
        min_correlation : float
            Minimum acceptable Pearson correlation (default 0.3).

        Returns
        -------
        tuple[bool, str]
            ``(True, "Model calibrated")`` if within thresholds, or
            ``(False, reason)`` explaining the failure.
        """
        report = self.reconcile()

        if report is None:
            return (False, "Insufficient data")

        if abs(report.bias) > max_bias:
            return (
                False,
                f"Bias {report.bias:.4f} exceeds threshold {max_bias}",
            )

        if report.correlation < min_correlation:
            return (
                False,
                f"Correlation {report.correlation:.4f} below threshold {min_correlation}",
            )

        return (True, "Model calibrated")
