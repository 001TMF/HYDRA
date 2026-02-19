"""Rule-based diagnostician for the HYDRA agent loop.

Performs structured triage from a DriftReport (produced by the sandbox
observer) to a DiagnosisResult with primary cause, confidence, evidence,
and recommended mutation types.

The diagnostician is entirely deterministic by default -- no LLM required.
When an LLM client is provided and diagnosis confidence is low, it can
optionally enhance the result, but the rule-based path is always the
fallback.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hydra.agent.types import DriftCategory, DiagnosisResult

if TYPE_CHECKING:
    from hydra.sandbox.observer import DriftReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cause -> recommended mutation types mapping
# ---------------------------------------------------------------------------

_CAUSE_TO_MUTATIONS: dict[DriftCategory, list[str]] = {
    DriftCategory.PERFORMANCE: ["hyperparameter", "feature_remove"],
    DriftCategory.FEATURE_DRIFT: ["feature_engineering", "hyperparameter"],
    DriftCategory.REGIME_CHANGE: ["ensemble_method", "hyperparameter"],
    DriftCategory.OVERFITTING: ["hyperparameter", "feature_remove"],
    DriftCategory.DATA_QUALITY: ["feature_remove"],
}


class Diagnostician:
    """Structured triage engine: DriftReport -> DiagnosisResult.

    Parameters
    ----------
    llm_client : object | None
        Optional LLM client for low-confidence enhancement.
        Must implement ``call(response_model=..., messages=...)``
        and raise ``LLMUnavailableError`` on failure.
    """

    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diagnose(self, report: DriftReport) -> DiagnosisResult:
        """Produce a DiagnosisResult from a DriftReport.

        Steps:
        1. Collect evidence from the report flags.
        2. Rule-based classification with priority ordering.
        3. Map cause to recommended mutation types.
        4. Optional LLM enhancement for low-confidence diagnoses.
        5. Build reasoning string.

        Parameters
        ----------
        report : DriftReport
            Combined drift report from the sandbox observer.

        Returns
        -------
        DiagnosisResult
        """
        evidence = self._collect_evidence(report)
        primary_cause, confidence = self._classify(report, evidence)
        recommended = list(_CAUSE_TO_MUTATIONS.get(primary_cause, []))
        reasoning = self._build_reasoning(primary_cause, confidence, evidence)

        result = DiagnosisResult(
            primary_cause=primary_cause,
            confidence=confidence,
            evidence=evidence,
            recommended_mutation_types=recommended,
            reasoning=reasoning,
        )

        # Optional LLM enhancement for low-confidence diagnoses
        if self.llm_client is not None and confidence < 0.6:
            result = self._try_llm_enhance(result, report)

        return result

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_evidence(report: DriftReport) -> list[str]:
        """Extract human-readable evidence strings from report flags."""
        evidence: list[str] = []

        perf = report.performance
        if perf.sharpe_degraded:
            evidence.append(
                f"Sharpe ratio degraded (current: {perf.sharpe_ratio:.3f})"
            )
        if perf.drawdown_alert:
            evidence.append(
                f"Drawdown alert triggered (max drawdown: {perf.max_drawdown:.3f})"
            )
        if perf.hit_rate_degraded:
            evidence.append(
                f"Hit rate degraded (current: {perf.hit_rate:.3f})"
            )

        if report.feature is not None:
            n_drifted = len(report.feature.drifted_features)
            if n_drifted > 0:
                names = ", ".join(report.feature.drifted_features[:5])
                evidence.append(
                    f"{n_drifted} features drifted: {names}"
                )

        for metric_name, alerted in report.streaming_alerts.items():
            if alerted:
                evidence.append(
                    f"Streaming alert for {metric_name}"
                )

        return evidence

    # ------------------------------------------------------------------
    # Rule-based classification (deterministic priority order)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        report: DriftReport,
        evidence: list[str],
    ) -> tuple[DriftCategory, float]:
        """Classify the drift root cause using priority-ordered rules.

        Priority order:
        1. Feature distribution drift (3+ drifted features)
        2. Performance degradation (Sharpe degraded, no feature drift)
        3. Regime change (drawdown alert + streaming alerts)
        4. Overfitting (hit rate degraded + high calibration error)
        5. Default: performance degradation at low confidence
        """
        perf = report.performance

        # 1. Feature distribution drift -- highest priority
        if (
            report.feature is not None
            and len(report.feature.drifted_features) >= 3
        ):
            return DriftCategory.FEATURE_DRIFT, 0.8

        # 2. Performance degradation (Sharpe)
        if perf.sharpe_degraded:
            # Check for overfitting sub-pattern
            if perf.hit_rate_degraded and perf.calibration > 0.3:
                return DriftCategory.OVERFITTING, 0.7
            return DriftCategory.PERFORMANCE, 0.7

        # 3. Regime change (drawdown + streaming confirmation)
        if perf.drawdown_alert and any(
            alerted
            for name, alerted in report.streaming_alerts.items()
            if "drawdown" in name.lower()
        ):
            return DriftCategory.REGIME_CHANGE, 0.6

        # 4. Overfitting (hit rate + calibration)
        if perf.hit_rate_degraded and perf.calibration > 0.3:
            return DriftCategory.OVERFITTING, 0.7

        # 5. Default: performance at low confidence
        return DriftCategory.PERFORMANCE, 0.5

    # ------------------------------------------------------------------
    # Reasoning builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reasoning(
        cause: DriftCategory,
        confidence: float,
        evidence: list[str],
    ) -> str:
        """Build a human-readable reasoning string."""
        if not evidence:
            return (
                f"Diagnosed {cause.value} with confidence {confidence:.2f}. "
                f"No strong signals detected; using default classification."
            )

        evidence_summary = "; ".join(evidence)
        return (
            f"Diagnosed {cause.value} with confidence {confidence:.2f}. "
            f"Evidence: {evidence_summary}."
        )

    # ------------------------------------------------------------------
    # Optional LLM enhancement
    # ------------------------------------------------------------------

    def _try_llm_enhance(
        self,
        baseline: DiagnosisResult,
        report: DriftReport,
    ) -> DiagnosisResult:
        """Attempt LLM-enhanced diagnosis for low-confidence results.

        The LLM is purely additive -- on any failure, the rule-based
        result is returned unchanged.
        """
        try:
            # Build a context summary for the LLM
            context = (
                f"Drift report summary: "
                f"sharpe_degraded={report.performance.sharpe_degraded}, "
                f"drawdown_alert={report.performance.drawdown_alert}, "
                f"hit_rate_degraded={report.performance.hit_rate_degraded}, "
                f"calibration={report.performance.calibration:.3f}"
            )
            if report.feature is not None:
                context += (
                    f", drifted_features={report.feature.drifted_features}"
                )
            context += (
                f", streaming_alerts={report.streaming_alerts}"
            )

            enhanced = self.llm_client.call(
                response_model=DiagnosisResult,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a model diagnostician. Given drift report data, "
                            "identify the most likely root cause of model degradation."
                        ),
                    },
                    {
                        "role": "user",
                        "content": context,
                    },
                ],
            )

            logger.info(
                "LLM enhanced diagnosis: %s (confidence %.2f -> %.2f)",
                enhanced.primary_cause,
                baseline.confidence,
                enhanced.confidence,
            )
            return enhanced

        except Exception:
            logger.debug(
                "LLM enhancement failed, falling back to rule-based result",
                exc_info=True,
            )
            return baseline
