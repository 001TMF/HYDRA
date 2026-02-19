"""Single-head autonomous agent loop (AGNT-01).

Wires all Phase 4 modules into a state machine that runs the complete
observe -> diagnose -> hypothesize -> experiment -> evaluate cycle.

Each step checks autonomy permissions. Dedup, budget, and cooldown are
checked before experiments. Rollback fires on sustained degradation.
Journal logging captures the full cycle. CLI state gating prevents
execution when paused.

The agent loop works entirely rule-based with zero LLM calls when no
API keys are configured (AGNT-10 integration).

Exports:
    - ``AgentLoop``: The autonomous agent loop class.
    - ``AgentPhase``: Enum of loop phases.
    - ``AgentCycleResult``: Dataclass capturing the outcome of one cycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

import structlog

from hydra.agent.autonomy import (
    AutonomyLevel,
    PermissionDeniedError,
    check_permission,
)
from hydra.agent.types import DiagnosisResult, Hypothesis
from hydra.cli.state import AgentState, get_state

if TYPE_CHECKING:
    import numpy as np

    from hydra.agent.budget import MutationBudget
    from hydra.agent.dedup import HypothesisDeduplicator
    from hydra.agent.experiment_runner import ExperimentResult
    from hydra.agent.llm.client import LLMClient
    from hydra.agent.promotion import PromotionEvaluator
    from hydra.agent.rollback import HysteresisRollbackTrigger
    from hydra.sandbox.evaluator import CompositeEvaluator
    from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord
    from hydra.sandbox.observer import DriftObserver
    from hydra.sandbox.registry import ModelRegistry

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------


class AgentPhase(str, Enum):
    """Phases of the autonomous agent loop."""

    IDLE = "idle"
    OBSERVE = "observe"
    DIAGNOSE = "diagnose"
    HYPOTHESIZE = "hypothesize"
    EXPERIMENT = "experiment"
    EVALUATE = "evaluate"
    PROMOTE = "promote"
    COOLDOWN = "cooldown"


# ---------------------------------------------------------------------------
# Cycle result
# ---------------------------------------------------------------------------


@dataclass
class AgentCycleResult:
    """Outcome of a single agent loop cycle.

    Attributes
    ----------
    phase_reached : AgentPhase
        The last phase successfully entered during this cycle.
    diagnosis : DiagnosisResult | None
        Diagnosis output, if the DIAGNOSE phase ran.
    hypothesis : Hypothesis | None
        The mutation hypothesis, if the HYPOTHESIZE phase ran.
    experiment_result : ExperimentResult | None
        Experiment outcome, if the EXPERIMENT phase ran.
    promoted : bool
        Whether the candidate was promoted to champion.
    rolled_back : bool
        Whether a rollback was triggered.
    skipped_reason : str | None
        If the cycle ended early, the human-readable reason why.
    duration_seconds : float
        Wall-clock seconds for the entire cycle.
    """

    phase_reached: AgentPhase
    diagnosis: DiagnosisResult | None = None
    hypothesis: Hypothesis | None = None
    experiment_result: object | None = None  # ExperimentResult
    promoted: bool = False
    rolled_back: bool = False
    skipped_reason: str | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Single-head autonomous agent loop state machine (AGNT-01).

    Runs the complete observe -> diagnose -> hypothesize -> experiment ->
    evaluate -> promote/reject cycle. All dependencies are injected via
    the constructor for testability.

    Parameters
    ----------
    observer : DriftObserver
        Produces drift reports from recent model performance.
    diagnostician : Diagnostician
        Triages drift reports into structured diagnoses.
    hypothesis_engine : HypothesisEngine
        Proposes mutations from the playbook or LLM.
    experiment_runner : ExperimentRunner
        Runs candidate training in subprocess isolation.
    evaluator : CompositeEvaluator
        Scores candidate models on composite fitness.
    journal : ExperimentJournal
        Persistent log of all experiments.
    registry : ModelRegistry
        MLflow model registry for champion/rollback lifecycle.
    rollback_trigger : HysteresisRollbackTrigger
        Detects sustained degradation and triggers rollback.
    promotion_evaluator : PromotionEvaluator
        3-of-5 window evaluation for candidate promotion.
    deduplicator : HypothesisDeduplicator
        Semantic dedup to prevent re-testing similar hypotheses.
    budget : MutationBudget
        Per-cycle experiment limits and cooldown timers.
    autonomy_level : AutonomyLevel
        Current permission level (default SUPERVISED).
    llm_client : LLMClient | None
        Optional LLM client for enhanced diagnosis/hypothesis.
    """

    def __init__(
        self,
        observer: DriftObserver,
        diagnostician: object,  # Diagnostician
        hypothesis_engine: object,  # HypothesisEngine
        experiment_runner: object,  # ExperimentRunner
        evaluator: CompositeEvaluator,
        journal: ExperimentJournal,
        registry: ModelRegistry,
        rollback_trigger: HysteresisRollbackTrigger,
        promotion_evaluator: PromotionEvaluator,
        deduplicator: HypothesisDeduplicator,
        budget: MutationBudget,
        autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
        llm_client: object | None = None,
    ) -> None:
        self.observer = observer
        self.diagnostician = diagnostician
        self.hypothesis_engine = hypothesis_engine
        self.experiment_runner = experiment_runner
        self.evaluator = evaluator
        self.journal = journal
        self.registry = registry
        self.rollback_trigger = rollback_trigger
        self.promotion_evaluator = promotion_evaluator
        self.deduplicator = deduplicator
        self.budget = budget
        self.autonomy_level = autonomy_level
        self.llm_client = llm_client

        self._phase: AgentPhase = AgentPhase.IDLE
        self._current_cycle: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_autonomy_level(self, level: AutonomyLevel) -> None:
        """Update the autonomy level at runtime."""
        self.autonomy_level = level
        logger.info("autonomy_level_changed", level=level.name)

    def run_cycle(
        self,
        recent_returns: np.ndarray,
        predictions: np.ndarray,
        actuals: np.ndarray,
        probabilities: np.ndarray,
        baseline_sharpe: float,
        baseline_features: np.ndarray | None = None,
        current_features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        current_config: dict | None = None,
        champion_fitness: float | None = None,
        candidate_window_scores: list[float] | None = None,
        champion_window_scores: list[float] | None = None,
    ) -> AgentCycleResult:
        """Run one complete cycle through all agent loop phases.

        Parameters
        ----------
        recent_returns : array-like
            Recent period daily returns for performance drift detection.
        predictions : array-like
            Model directional predictions (0/1).
        actuals : array-like
            Actual outcomes (0/1).
        probabilities : array-like
            Model-output probabilities for calibration scoring.
        baseline_sharpe : float
            Historical baseline Sharpe ratio for comparison.
        baseline_features : array-like | None
            2D baseline feature array for feature drift detection.
        current_features : array-like | None
            2D current feature array for feature drift detection.
        feature_names : list[str] | None
            Column names for feature drift detection.
        current_config : dict | None
            Current model configuration for hypothesis resolution.
        champion_fitness : float | None
            Champion model fitness score for comparison.
        candidate_window_scores : list[float] | None
            Pre-computed per-window fitness for 3-of-5 promotion path.
        champion_window_scores : list[float] | None
            Pre-computed per-window champion fitness for 3-of-5 path.

        Returns
        -------
        AgentCycleResult
        """
        start = time.monotonic()
        self._phase = AgentPhase.IDLE

        # ------------------------------------------------------------------
        # 1. Check CLI state
        # ------------------------------------------------------------------
        state = get_state()
        if state == AgentState.PAUSED:
            return AgentCycleResult(
                phase_reached=AgentPhase.IDLE,
                skipped_reason="Agent paused",
                duration_seconds=time.monotonic() - start,
            )

        # ------------------------------------------------------------------
        # 2. OBSERVE
        # ------------------------------------------------------------------
        self._phase = AgentPhase.OBSERVE

        if not check_permission("observe", self.autonomy_level):
            return AgentCycleResult(
                phase_reached=AgentPhase.OBSERVE,
                skipped_reason=(
                    f"Observe blocked at autonomy level {self.autonomy_level.name}"
                ),
                duration_seconds=time.monotonic() - start,
            )

        report = self.observer.get_full_report(
            recent_returns=recent_returns,
            predictions=predictions,
            actuals=actuals,
            probabilities=probabilities,
            baseline_sharpe=baseline_sharpe,
            baseline_features=baseline_features,
            current_features=current_features,
            feature_names=feature_names,
        )

        if not report.needs_diagnosis:
            return AgentCycleResult(
                phase_reached=AgentPhase.OBSERVE,
                skipped_reason="No drift detected",
                duration_seconds=time.monotonic() - start,
            )

        # ------------------------------------------------------------------
        # 3. DIAGNOSE
        # ------------------------------------------------------------------
        self._phase = AgentPhase.DIAGNOSE

        if not check_permission("diagnose", self.autonomy_level):
            return AgentCycleResult(
                phase_reached=AgentPhase.DIAGNOSE,
                skipped_reason=(
                    f"Diagnose blocked at autonomy level {self.autonomy_level.name}"
                ),
                duration_seconds=time.monotonic() - start,
            )

        diagnosis = self.diagnostician.diagnose(report)
        logger.info(
            "diagnosis_complete",
            primary_cause=diagnosis.primary_cause.value,
            confidence=diagnosis.confidence,
        )

        if diagnosis.confidence < 0.3 and not diagnosis.evidence:
            return AgentCycleResult(
                phase_reached=AgentPhase.DIAGNOSE,
                diagnosis=diagnosis,
                skipped_reason="Diagnosis inconclusive",
                duration_seconds=time.monotonic() - start,
            )

        # ------------------------------------------------------------------
        # 4. HYPOTHESIZE
        # ------------------------------------------------------------------
        self._phase = AgentPhase.HYPOTHESIZE

        if not check_permission("hypothesize", self.autonomy_level):
            return AgentCycleResult(
                phase_reached=AgentPhase.HYPOTHESIZE,
                diagnosis=diagnosis,
                skipped_reason=(
                    f"Hypothesize blocked at autonomy level "
                    f"{self.autonomy_level.name}"
                ),
                duration_seconds=time.monotonic() - start,
            )

        hypothesis = self.hypothesis_engine.propose(diagnosis, current_config)

        # Check dedup
        is_dup, sim_score = self.deduplicator.is_duplicate(hypothesis.description)
        if is_dup:
            # Try propose_multiple and pick first non-duplicate
            alternatives = self.hypothesis_engine.propose_multiple(
                diagnosis, n=10, current_config=current_config
            )
            hypothesis = None
            for alt in alternatives:
                alt_dup, _ = self.deduplicator.is_duplicate(alt.description)
                if not alt_dup:
                    hypothesis = alt
                    break

            if hypothesis is None:
                return AgentCycleResult(
                    phase_reached=AgentPhase.HYPOTHESIZE,
                    diagnosis=diagnosis,
                    skipped_reason=(
                        "All hypotheses are duplicates of recent experiments"
                    ),
                    duration_seconds=time.monotonic() - start,
                )

        # Check budget
        can_run, budget_reason = self.budget.can_run(
            hypothesis.mutation_type.value
            if hasattr(hypothesis.mutation_type, "value")
            else str(hypothesis.mutation_type)
        )
        if not can_run:
            return AgentCycleResult(
                phase_reached=AgentPhase.HYPOTHESIZE,
                diagnosis=diagnosis,
                hypothesis=hypothesis,
                skipped_reason=budget_reason,
                duration_seconds=time.monotonic() - start,
            )

        # ------------------------------------------------------------------
        # 5. EXPERIMENT
        # ------------------------------------------------------------------
        self._phase = AgentPhase.EXPERIMENT

        if not check_permission("experiment", self.autonomy_level):
            return AgentCycleResult(
                phase_reached=AgentPhase.EXPERIMENT,
                diagnosis=diagnosis,
                hypothesis=hypothesis,
                skipped_reason=(
                    f"Experiment blocked at autonomy level "
                    f"{self.autonomy_level.name}"
                ),
                duration_seconds=time.monotonic() - start,
            )

        experiment_result = self.experiment_runner.run(
            hypothesis, current_config or {}
        )

        if not experiment_result.success:
            logger.warning(
                "experiment_failed",
                hypothesis=hypothesis.description,
                error=experiment_result.error_message,
            )
            # Record to journal as rejected
            self._log_to_journal(
                hypothesis=hypothesis,
                experiment_result=experiment_result,
                promotion_decision="rejected",
                promotion_reason=(
                    f"Experiment failed: {experiment_result.error_message}"
                ),
            )
            return AgentCycleResult(
                phase_reached=AgentPhase.EXPERIMENT,
                diagnosis=diagnosis,
                hypothesis=hypothesis,
                experiment_result=experiment_result,
                skipped_reason=(
                    f"Experiment failed: {experiment_result.error_message}"
                ),
                duration_seconds=time.monotonic() - start,
            )

        # ------------------------------------------------------------------
        # 6. EVALUATE
        # ------------------------------------------------------------------
        self._phase = AgentPhase.EVALUATE

        # Feed current fitness to rollback trigger
        current_fitness = experiment_result.fitness_score or 0.0
        rollback_fired = self.rollback_trigger.update(
            current_fitness, champion_fitness or 0.0
        )

        if rollback_fired:
            # Attempt registry rollback if method exists
            if hasattr(self.registry, "rollback"):
                try:
                    self.registry.rollback()
                    logger.info("rollback_executed")
                except Exception as exc:
                    logger.warning("rollback_failed", error=str(exc))

            self._log_to_journal(
                hypothesis=hypothesis,
                experiment_result=experiment_result,
                promotion_decision="rejected",
                promotion_reason="Rollback triggered by sustained degradation",
            )
            return AgentCycleResult(
                phase_reached=AgentPhase.EVALUATE,
                diagnosis=diagnosis,
                hypothesis=hypothesis,
                experiment_result=experiment_result,
                rolled_back=True,
                duration_seconds=time.monotonic() - start,
            )

        # Determine promotion
        if (
            candidate_window_scores is not None
            and champion_window_scores is not None
        ):
            # Full 3-of-5 path (AGNT-08) -- caller provides window scores
            promotion_result = self.promotion_evaluator.evaluate(
                candidate_scores=candidate_window_scores,
                champion_scores=champion_window_scores,
            )
            promoted = promotion_result.promoted
            promotion_reason = promotion_result.reason
        else:
            # Honda default: single-score comparison
            experiment_fitness = experiment_result.fitness_score or 0.0
            promoted = experiment_fitness > (champion_fitness or 0.0)
            promotion_reason = (
                "single-score comparison (no window scores provided)"
            )

        # ------------------------------------------------------------------
        # 7. PROMOTE / REJECT
        # ------------------------------------------------------------------
        self._phase = AgentPhase.PROMOTE

        if promoted:
            if not check_permission("promote", self.autonomy_level):
                # Permission denied for promotion -- record as pending
                self._log_to_journal(
                    hypothesis=hypothesis,
                    experiment_result=experiment_result,
                    promotion_decision="pending",
                    promotion_reason=(
                        f"Promote blocked at autonomy level "
                        f"{self.autonomy_level.name}; {promotion_reason}"
                    ),
                )
                return AgentCycleResult(
                    phase_reached=AgentPhase.PROMOTE,
                    diagnosis=diagnosis,
                    hypothesis=hypothesis,
                    experiment_result=experiment_result,
                    promoted=False,
                    skipped_reason=(
                        f"Promote blocked at autonomy level "
                        f"{self.autonomy_level.name}"
                    ),
                    duration_seconds=time.monotonic() - start,
                )

        promotion_decision = "promoted" if promoted else "rejected"

        # Log to journal
        self._log_to_journal(
            hypothesis=hypothesis,
            experiment_result=experiment_result,
            promotion_decision=promotion_decision,
            promotion_reason=promotion_reason,
        )

        # Register in deduplicator
        self.deduplicator.register(hypothesis.description)

        # Record in budget
        mutation_type_value = (
            hypothesis.mutation_type.value
            if hasattr(hypothesis.mutation_type, "value")
            else str(hypothesis.mutation_type)
        )
        self.budget.record_experiment(mutation_type_value, promoted)

        logger.info(
            "cycle_complete",
            promoted=promoted,
            promotion_reason=promotion_reason,
            hypothesis=hypothesis.description,
            fitness=experiment_result.fitness_score,
        )

        return AgentCycleResult(
            phase_reached=AgentPhase.PROMOTE,
            diagnosis=diagnosis,
            hypothesis=hypothesis,
            experiment_result=experiment_result,
            promoted=promoted,
            duration_seconds=time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_to_journal(
        self,
        hypothesis: Hypothesis,
        experiment_result: object,
        promotion_decision: str,
        promotion_reason: str,
    ) -> None:
        """Log an ExperimentRecord to the journal."""
        from hydra.sandbox.journal import ExperimentRecord

        mutation_type_value = (
            hypothesis.mutation_type.value
            if hasattr(hypothesis.mutation_type, "value")
            else str(hypothesis.mutation_type)
        )

        record = ExperimentRecord(
            created_at=datetime.now(timezone.utc).isoformat(),
            hypothesis=hypothesis.description,
            mutation_type=mutation_type_value,
            config_diff=hypothesis.config_diff,
            results=getattr(experiment_result, "metrics", {}),
            champion_metrics=None,
            promotion_decision=promotion_decision,
            promotion_reason=promotion_reason,
        )

        try:
            self.journal.log_experiment(record)
        except Exception as exc:
            logger.warning("journal_log_failed", error=str(exc))
