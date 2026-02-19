"""Tests for the rule-based diagnostician.

Verifies that the Diagnostician maps DriftReport signals to DiagnosisResult
with correct primary causes, confidence levels, evidence, and recommended
mutation types -- all without any LLM dependency.
"""

from __future__ import annotations

import pytest

from hydra.agent.diagnostician import Diagnostician
from hydra.agent.types import DriftCategory, DiagnosisResult
from hydra.sandbox.observer import (
    DriftReport,
    FeatureDriftReport,
    PerformanceDriftReport,
)


# ---------------------------------------------------------------------------
# Test fixtures -- DriftReport builders
# ---------------------------------------------------------------------------


def _perf(
    *,
    sharpe_degraded: bool = False,
    drawdown_alert: bool = False,
    hit_rate_degraded: bool = False,
    calibration: float = 0.1,
) -> PerformanceDriftReport:
    """Build a PerformanceDriftReport with sensible defaults."""
    return PerformanceDriftReport(
        sharpe_ratio=0.5,
        max_drawdown=-0.08,
        hit_rate=0.52,
        calibration=calibration,
        sharpe_degraded=sharpe_degraded,
        drawdown_alert=drawdown_alert,
        hit_rate_degraded=hit_rate_degraded,
    )


def _feat(drifted: list[str] | None = None) -> FeatureDriftReport:
    """Build a FeatureDriftReport with given drifted features."""
    drifted = drifted or []
    return FeatureDriftReport(
        psi_scores={f: 0.3 for f in drifted},
        ks_results={f: (True, 0.4, 0.001) for f in drifted},
        drifted_features=drifted,
    )


def _report(
    *,
    sharpe_degraded: bool = False,
    drawdown_alert: bool = False,
    hit_rate_degraded: bool = False,
    calibration: float = 0.1,
    drifted_features: list[str] | None = None,
    streaming_alerts: dict[str, bool] | None = None,
) -> DriftReport:
    """Build a full DriftReport with controllable flags."""
    feat = _feat(drifted_features) if drifted_features else None
    return DriftReport(
        performance=_perf(
            sharpe_degraded=sharpe_degraded,
            drawdown_alert=drawdown_alert,
            hit_rate_degraded=hit_rate_degraded,
            calibration=calibration,
        ),
        feature=feat,
        streaming_alerts=streaming_alerts or {},
        needs_diagnosis=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiagnosticianRuleBased:
    """Rule-based diagnosis without LLM."""

    def test_sharpe_degraded_produces_performance_cause(self) -> None:
        report = _report(sharpe_degraded=True)
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert isinstance(result, DiagnosisResult)
        assert result.primary_cause == DriftCategory.PERFORMANCE

    def test_feature_drift_with_3_plus_features(self) -> None:
        report = _report(
            drifted_features=["feat_a", "feat_b", "feat_c"],
        )
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert result.primary_cause == DriftCategory.FEATURE_DRIFT
        assert result.confidence == 0.8

    def test_drawdown_alert_with_streaming_produces_regime_change(self) -> None:
        report = _report(
            drawdown_alert=True,
            streaming_alerts={"drawdown": True},
        )
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert result.primary_cause == DriftCategory.REGIME_CHANGE
        assert result.confidence == 0.6

    def test_hit_rate_degraded_with_high_calibration_produces_overfitting(self) -> None:
        report = _report(
            hit_rate_degraded=True,
            calibration=0.35,
        )
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert result.primary_cause == DriftCategory.OVERFITTING
        assert result.confidence == 0.7

    def test_priority_order_feature_drift_over_performance(self) -> None:
        """Feature drift (3+ features) takes priority over sharpe degradation."""
        report = _report(
            sharpe_degraded=True,
            drifted_features=["feat_a", "feat_b", "feat_c"],
        )
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert result.primary_cause == DriftCategory.FEATURE_DRIFT

    def test_default_cause_when_no_strong_signals(self) -> None:
        report = _report()  # no flags set
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert result.primary_cause == DriftCategory.PERFORMANCE
        assert result.confidence == 0.5

    def test_no_llm_client_works(self) -> None:
        diag = Diagnostician(llm_client=None)
        report = _report(sharpe_degraded=True)
        result = diag.diagnose(report)

        assert result.primary_cause == DriftCategory.PERFORMANCE
        assert result.confidence > 0

    def test_evidence_populated_from_drift_flags(self) -> None:
        report = _report(
            sharpe_degraded=True,
            drawdown_alert=True,
            drifted_features=["vol_skew", "oi_concentration", "term_slope"],
        )
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert len(result.evidence) > 0
        # Should mention specific flags
        evidence_text = " ".join(result.evidence).lower()
        assert "sharpe" in evidence_text or "drifted" in evidence_text

    def test_recommended_mutation_types_performance(self) -> None:
        report = _report(sharpe_degraded=True)
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert "hyperparameter" in result.recommended_mutation_types
        assert "feature_remove" in result.recommended_mutation_types

    def test_recommended_mutation_types_feature_drift(self) -> None:
        report = _report(drifted_features=["a", "b", "c"])
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert "feature_engineering" in result.recommended_mutation_types
        assert "hyperparameter" in result.recommended_mutation_types

    def test_recommended_mutation_types_regime_change(self) -> None:
        report = _report(
            drawdown_alert=True,
            streaming_alerts={"drawdown": True},
        )
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert "ensemble_method" in result.recommended_mutation_types
        assert "hyperparameter" in result.recommended_mutation_types

    def test_recommended_mutation_types_overfitting(self) -> None:
        report = _report(hit_rate_degraded=True, calibration=0.35)
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert "hyperparameter" in result.recommended_mutation_types
        assert "feature_remove" in result.recommended_mutation_types

    def test_reasoning_is_populated(self) -> None:
        report = _report(sharpe_degraded=True)
        diag = Diagnostician()
        result = diag.diagnose(report)

        assert len(result.reasoning) > 0
