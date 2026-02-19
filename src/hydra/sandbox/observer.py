"""DriftObserver -- unified drift monitoring for HYDRA sandbox.

Combines performance drift detection (rolling Sharpe, drawdown, hit rate,
calibration) with feature distribution drift (PSI, KS) and streaming
change detection (ADWIN, CUSUM) into a single DriftReport.

The ``needs_diagnosis`` flag on DriftReport is the trigger for the
autonomous agent loop to initiate a diagnostic cycle in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from hydra.sandbox.drift.adwin import ADWINDetector
from hydra.sandbox.drift.cusum import CUSUMDetector
from hydra.sandbox.drift.ks import check_ks_drift
from hydra.sandbox.drift.psi import compute_psi


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PerformanceDriftReport:
    """Performance metrics and degradation flags."""

    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    calibration: float  # Brier score (lower is better)
    sharpe_degraded: bool
    drawdown_alert: bool
    hit_rate_degraded: bool


@dataclass
class FeatureDriftReport:
    """Feature distribution drift results."""

    psi_scores: dict[str, float]
    ks_results: dict[str, tuple[bool, float, float]]
    drifted_features: list[str]


@dataclass
class DriftReport:
    """Combined drift report from all detection methods."""

    performance: PerformanceDriftReport
    feature: FeatureDriftReport | None
    streaming_alerts: dict[str, bool]
    needs_diagnosis: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "psi_threshold": 0.25,
    "ks_alpha": 0.05,
    "adwin_delta": 0.002,
    "cusum_threshold": 5.0,
    "cusum_drift": 0.5,
    "sharpe_degradation_factor": 0.5,
    "drawdown_threshold": -0.15,
    "hit_rate_floor": 0.45,
    "grace_period": 30,
}


# ---------------------------------------------------------------------------
# DriftObserver
# ---------------------------------------------------------------------------


class DriftObserver:
    """Unified drift monitoring combining batch and streaming detectors.

    Parameters
    ----------
    config : dict | None
        Override default thresholds. Keys are documented in ``_DEFAULT_CONFIG``.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = {**_DEFAULT_CONFIG, **(config or {})}

        # Streaming detectors keyed by metric name
        self._adwin: dict[str, ADWINDetector] = {}
        self._cusum: dict[str, CUSUMDetector] = {}

        # Pre-create detectors for standard monitored metrics
        for metric in ("sharpe", "drawdown", "hit_rate"):
            self._ensure_streaming_detectors(metric)

    # -- internal helpers ---------------------------------------------------

    def _ensure_streaming_detectors(self, metric_name: str) -> None:
        """Lazily create ADWIN + CUSUM pair for *metric_name*."""
        if metric_name not in self._adwin:
            self._adwin[metric_name] = ADWINDetector(
                delta=self.config["adwin_delta"],
                grace_period=self.config["grace_period"],
            )
        if metric_name not in self._cusum:
            self._cusum[metric_name] = CUSUMDetector(
                threshold=self.config["cusum_threshold"],
                drift=self.config["cusum_drift"],
            )

    @staticmethod
    def _compute_sharpe(returns: np.ndarray) -> float:
        """Annualized Sharpe ratio (same formula as evaluation.py)."""
        if len(returns) < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std == 0:
            return 0.0
        return float(np.mean(returns)) / std * np.sqrt(252)

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """Max drawdown from returns array (negative value)."""
        if len(returns) == 0:
            return 0.0
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        return float(np.min(drawdowns))

    # -- public API ---------------------------------------------------------

    def check_performance_drift(
        self,
        recent_returns: np.ndarray,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray,
        baseline_sharpe: float,
    ) -> PerformanceDriftReport:
        """Evaluate performance metrics against degradation thresholds.

        Parameters
        ----------
        recent_returns : np.ndarray
            Recent period daily returns.
        predictions : np.ndarray
            Model directional predictions (0/1).
        actuals : np.ndarray
            Actual outcomes (0/1).
        probabilities : np.ndarray
            Model-output probabilities for calibration scoring.
        baseline_sharpe : float
            Historical baseline Sharpe ratio for comparison.

        Returns
        -------
        PerformanceDriftReport
        """
        recent_returns = np.asarray(recent_returns, dtype=float).ravel()
        predictions = np.asarray(predictions, dtype=float).ravel()
        actuals = np.asarray(actuals, dtype=float).ravel()
        probabilities = np.asarray(probabilities, dtype=float).ravel()

        sharpe = self._compute_sharpe(recent_returns)
        max_dd = self._compute_max_drawdown(recent_returns)
        hit_rate = float(np.mean(predictions == actuals)) if len(actuals) > 0 else 0.0

        # Brier score: mean squared error between probabilities and actuals
        if len(probabilities) > 0 and len(actuals) > 0:
            calibration = float(np.mean((probabilities - actuals) ** 2))
        else:
            calibration = 0.0

        return PerformanceDriftReport(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            hit_rate=hit_rate,
            calibration=calibration,
            sharpe_degraded=sharpe < baseline_sharpe * self.config["sharpe_degradation_factor"],
            drawdown_alert=max_dd < self.config["drawdown_threshold"],
            hit_rate_degraded=hit_rate < self.config["hit_rate_floor"],
        )

    def check_feature_drift(
        self,
        baseline_features: np.ndarray,
        current_features: np.ndarray,
        feature_names: list[str],
    ) -> FeatureDriftReport:
        """Evaluate feature distribution drift via PSI and KS test.

        Parameters
        ----------
        baseline_features : np.ndarray
            2-D array (n_samples, n_features) from baseline period.
        current_features : np.ndarray
            2-D array (n_samples, n_features) from current period.
        feature_names : list[str]
            Column names corresponding to feature columns.

        Returns
        -------
        FeatureDriftReport
        """
        baseline_features = np.asarray(baseline_features, dtype=float)
        current_features = np.asarray(current_features, dtype=float)

        psi_scores: dict[str, float] = {}
        ks_results: dict[str, tuple[bool, float, float]] = {}
        drifted: list[str] = []

        for i, name in enumerate(feature_names):
            bl = baseline_features[:, i]
            cur = current_features[:, i]

            psi_val = compute_psi(bl, cur)
            psi_scores[name] = psi_val

            ks_drifted, ks_stat, ks_pval = check_ks_drift(
                bl, cur, alpha=self.config["ks_alpha"]
            )
            ks_results[name] = (ks_drifted, ks_stat, ks_pval)

            if psi_val >= self.config["psi_threshold"] or ks_drifted:
                drifted.append(name)

        return FeatureDriftReport(
            psi_scores=psi_scores,
            ks_results=ks_results,
            drifted_features=drifted,
        )

    def update_streaming(self, metric_name: str, value: float) -> bool:
        """Feed a new value to streaming detectors for *metric_name*.

        If the metric has not been seen before, new ADWIN + CUSUM
        detectors are created automatically.

        Returns
        -------
        bool
            True if either ADWIN or CUSUM detects drift.
        """
        self._ensure_streaming_detectors(metric_name)
        adwin_drift = self._adwin[metric_name].update(value)
        cusum_drift = self._cusum[metric_name].update(value)
        return adwin_drift or cusum_drift

    def get_full_report(
        self,
        recent_returns: np.ndarray,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray,
        baseline_sharpe: float,
        baseline_features: np.ndarray | None = None,
        current_features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> DriftReport:
        """Generate a comprehensive drift report.

        Parameters
        ----------
        recent_returns, predictions, actuals, probabilities, baseline_sharpe
            Forwarded to :meth:`check_performance_drift`.
        baseline_features, current_features, feature_names
            Optional; forwarded to :meth:`check_feature_drift` if all provided.

        Returns
        -------
        DriftReport
        """
        perf = self.check_performance_drift(
            recent_returns, predictions, actuals, probabilities, baseline_sharpe
        )

        feat: FeatureDriftReport | None = None
        if (
            baseline_features is not None
            and current_features is not None
            and feature_names is not None
        ):
            feat = self.check_feature_drift(baseline_features, current_features, feature_names)

        # Collect streaming alerts
        streaming_alerts: dict[str, bool] = {}
        for name, adwin_det in self._adwin.items():
            cusum_det = self._cusum[name]
            streaming_alerts[name] = adwin_det.detector.drift_detected or cusum_det.drift_detected

        # Determine if diagnosis is needed
        needs_diagnosis = (
            perf.sharpe_degraded
            or perf.drawdown_alert
            or perf.hit_rate_degraded
            or (feat is not None and len(feat.drifted_features) > 0)
            or any(streaming_alerts.values())
        )

        return DriftReport(
            performance=perf,
            feature=feat,
            streaming_alerts=streaming_alerts,
            needs_diagnosis=needs_diagnosis,
        )

    def reset_streaming(self) -> None:
        """Reset all ADWIN and CUSUM streaming detectors."""
        for det in self._adwin.values():
            det.reset()
        for det in self._cusum.values():
            det.reset()
