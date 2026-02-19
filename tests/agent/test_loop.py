"""Unit tests for AgentLoop step transitions.

Each test mocks all dependencies and verifies one specific behavior
of the agent loop state machine -- from CLI state gating through
drift detection, diagnosis, hypothesis generation, experiment
execution, rollback, and promotion.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hydra.agent.autonomy import AutonomyLevel, PermissionDeniedError
from hydra.agent.loop import AgentLoop, AgentCycleResult, AgentPhase
from hydra.agent.types import DriftCategory, DiagnosisResult, Hypothesis, MutationType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_loop(autonomy: AutonomyLevel = AutonomyLevel.AUTONOMOUS) -> AgentLoop:
    """Create an AgentLoop with fully mocked dependencies."""
    loop = AgentLoop(
        observer=MagicMock(),
        diagnostician=MagicMock(),
        hypothesis_engine=MagicMock(),
        experiment_runner=MagicMock(),
        evaluator=MagicMock(),
        journal=MagicMock(),
        registry=MagicMock(),
        rollback_trigger=MagicMock(),
        promotion_evaluator=MagicMock(),
        deduplicator=MagicMock(),
        budget=MagicMock(),
        autonomy_level=autonomy,
        llm_client=None,
    )
    return loop


def _base_kwargs() -> dict:
    """Return minimal kwargs for run_cycle()."""
    return {
        "recent_returns": [0.01, -0.005, 0.02],
        "predictions": [1, 0, 1],
        "actuals": [1, 0, 1],
        "probabilities": [0.8, 0.3, 0.7],
        "baseline_sharpe": 1.5,
    }


def _make_diagnosis(
    confidence: float = 0.7,
    cause: DriftCategory = DriftCategory.PERFORMANCE,
    evidence: list[str] | None = None,
) -> DiagnosisResult:
    return DiagnosisResult(
        primary_cause=cause,
        confidence=confidence,
        evidence=evidence if evidence is not None else ["Sharpe degraded"],
        recommended_mutation_types=["hyperparameter"],
        reasoning="Test diagnosis",
    )


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        mutation_type=MutationType.HYPERPARAMETER,
        description="reduce_learning_rate",
        config_diff={"learning_rate": 0.05},
        expected_impact="Reduce overshoot",
        testable_prediction="Improved Sharpe on next eval",
        source="playbook",
    )


def _make_experiment_result(success: bool = True, fitness: float = 0.8):
    result = MagicMock()
    result.success = success
    result.fitness_score = fitness
    result.metrics = {"sharpe": 1.2}
    result.error_message = None if success else "Training failed"
    return result


# ---------------------------------------------------------------------------
# Test: PAUSED state
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_paused_state_skips_cycle(mock_get_state):
    """When CLI state is PAUSED, the cycle should be skipped immediately."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.PAUSED

    loop = _make_loop()
    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.IDLE
    assert "paused" in result.skipped_reason.lower()
    loop.observer.get_full_report.assert_not_called()


# ---------------------------------------------------------------------------
# Test: No drift detected
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_no_drift_skips_after_observe(mock_get_state):
    """When no drift is detected, cycle should stop after OBSERVE."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = False
    loop.observer.get_full_report.return_value = report

    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.OBSERVE
    assert "No drift" in result.skipped_reason


# ---------------------------------------------------------------------------
# Test: LOCKDOWN blocks observe
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_lockdown_blocks_observe(mock_get_state):
    """At LOCKDOWN level, observe should be blocked."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop(autonomy=AutonomyLevel.LOCKDOWN)
    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.OBSERVE
    assert "blocked" in result.skipped_reason.lower()
    assert "LOCKDOWN" in result.skipped_reason


# ---------------------------------------------------------------------------
# Test: SUPERVISED blocks experiment
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_supervised_blocks_experiment(mock_get_state):
    """At SUPERVISED level, experiment should be blocked."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop(autonomy=AutonomyLevel.SUPERVISED)

    # Set up observe -> diagnose -> hypothesize to succeed
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")

    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.EXPERIMENT
    assert "blocked" in result.skipped_reason.lower()
    assert result.diagnosis is not None
    assert result.hypothesis is not None


# ---------------------------------------------------------------------------
# Test: Duplicate hypothesis
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_duplicate_hypothesis_skips(mock_get_state):
    """When all hypotheses are duplicates, the cycle should skip."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()

    # First hypothesis is duplicate
    loop.deduplicator.is_duplicate.return_value = (True, 0.92)

    # propose_multiple also returns all duplicates
    loop.hypothesis_engine.propose_multiple.return_value = [_make_hypothesis()]

    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.HYPOTHESIZE
    assert "duplicates" in result.skipped_reason.lower()


# ---------------------------------------------------------------------------
# Test: Budget exceeded
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_budget_exceeded_skips(mock_get_state):
    """When budget is exceeded, the cycle should skip before experiment."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (False, "budget exceeded")

    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.HYPOTHESIZE
    assert "budget exceeded" in result.skipped_reason


# ---------------------------------------------------------------------------
# Test: Experiment failure
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_experiment_failure_records_rejection(mock_get_state):
    """When experiment fails, journal should record a rejection."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")

    failed_result = _make_experiment_result(success=False)
    loop.experiment_runner.run.return_value = failed_result

    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.EXPERIMENT
    assert result.experiment_result is failed_result
    # Journal should have been called to record the rejection
    loop.journal.log_experiment.assert_called_once()
    logged_record = loop.journal.log_experiment.call_args[0][0]
    assert logged_record.promotion_decision == "rejected"


# ---------------------------------------------------------------------------
# Test: Rollback triggered
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_rollback_triggered(mock_get_state):
    """When rollback trigger fires, the result should show rolled_back=True."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")
    loop.experiment_runner.run.return_value = _make_experiment_result()

    # Rollback fires
    loop.rollback_trigger.update.return_value = True

    result = loop.run_cycle(**_base_kwargs(), champion_fitness=0.5)

    assert result.phase_reached == AgentPhase.EVALUATE
    assert result.rolled_back is True
    loop.registry.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Successful cycle with simple comparison (no window scores)
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_successful_cycle_simple_comparison(mock_get_state):
    """Full cycle: candidate beats champion via single-score comparison."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")
    loop.experiment_runner.run.return_value = _make_experiment_result(
        success=True, fitness=0.8
    )
    loop.rollback_trigger.update.return_value = False

    result = loop.run_cycle(**_base_kwargs(), champion_fitness=0.5)

    assert result.phase_reached == AgentPhase.PROMOTE
    assert result.promoted is True
    assert result.diagnosis is not None
    assert result.hypothesis is not None
    loop.journal.log_experiment.assert_called_once()
    logged_record = loop.journal.log_experiment.call_args[0][0]
    assert logged_record.promotion_decision == "promoted"


# ---------------------------------------------------------------------------
# Test: Successful cycle with window scores (3-of-5 path)
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_successful_cycle_window_scores_promoted(mock_get_state):
    """Full cycle: 3-of-5 promotion evaluator approves candidate."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")
    loop.experiment_runner.run.return_value = _make_experiment_result()
    loop.rollback_trigger.update.return_value = False

    # Mock 3-of-5 promotion
    promotion_result_mock = MagicMock()
    promotion_result_mock.promoted = True
    promotion_result_mock.windows_won = 3
    promotion_result_mock.reason = "Candidate won 3 of 5 windows"
    loop.promotion_evaluator.evaluate.return_value = promotion_result_mock

    result = loop.run_cycle(
        **_base_kwargs(),
        champion_fitness=0.5,
        candidate_window_scores=[0.8, 0.7, 0.9, 0.4, 0.85],
        champion_window_scores=[0.6, 0.65, 0.7, 0.5, 0.6],
    )

    assert result.promoted is True
    loop.promotion_evaluator.evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Window scores provided but 3-of-5 fails
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_window_scores_promotion_fails(mock_get_state):
    """Window scores provided but candidate wins only 2 of 5."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()
    loop.hypothesis_engine.propose.return_value = _make_hypothesis()
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")
    loop.experiment_runner.run.return_value = _make_experiment_result()
    loop.rollback_trigger.update.return_value = False

    # Mock 3-of-5 rejection
    promotion_result_mock = MagicMock()
    promotion_result_mock.promoted = False
    promotion_result_mock.windows_won = 2
    promotion_result_mock.reason = "Candidate won 2 of 5 windows"
    loop.promotion_evaluator.evaluate.return_value = promotion_result_mock

    result = loop.run_cycle(
        **_base_kwargs(),
        champion_fitness=0.5,
        candidate_window_scores=[0.8, 0.5, 0.4, 0.4, 0.85],
        champion_window_scores=[0.6, 0.65, 0.7, 0.5, 0.6],
    )

    assert result.promoted is False
    loop.promotion_evaluator.evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Journal logging correctness
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_journal_logging_correctness(mock_get_state):
    """Verify ExperimentRecord contains correct hypothesis and mutation_type."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report
    loop.diagnostician.diagnose.return_value = _make_diagnosis()

    hypothesis = _make_hypothesis()
    loop.hypothesis_engine.propose.return_value = hypothesis
    loop.deduplicator.is_duplicate.return_value = (False, 0.0)
    loop.budget.can_run.return_value = (True, "ok")
    loop.experiment_runner.run.return_value = _make_experiment_result(
        success=True, fitness=0.9
    )
    loop.rollback_trigger.update.return_value = False

    result = loop.run_cycle(**_base_kwargs(), champion_fitness=0.5)

    assert result.promoted is True
    loop.journal.log_experiment.assert_called_once()
    logged_record = loop.journal.log_experiment.call_args[0][0]
    assert logged_record.hypothesis == "reduce_learning_rate"
    assert logged_record.mutation_type == "hyperparameter"
    assert logged_record.config_diff == {"learning_rate": 0.05}
    assert logged_record.promotion_decision == "promoted"


# ---------------------------------------------------------------------------
# Test: set_autonomy_level
# ---------------------------------------------------------------------------


def test_set_autonomy_level():
    """Verify set_autonomy_level updates the level."""
    loop = _make_loop(autonomy=AutonomyLevel.SUPERVISED)
    assert loop.autonomy_level == AutonomyLevel.SUPERVISED
    loop.set_autonomy_level(AutonomyLevel.AUTONOMOUS)
    assert loop.autonomy_level == AutonomyLevel.AUTONOMOUS


# ---------------------------------------------------------------------------
# Test: Diagnosis inconclusive (low confidence, no evidence)
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_diagnosis_inconclusive(mock_get_state):
    """When diagnosis confidence < 0.3 and no evidence, cycle should skip."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    loop = _make_loop()
    report = MagicMock()
    report.needs_diagnosis = True
    loop.observer.get_full_report.return_value = report

    # Low confidence, no evidence
    loop.diagnostician.diagnose.return_value = _make_diagnosis(
        confidence=0.2, evidence=[]
    )

    result = loop.run_cycle(**_base_kwargs())

    assert result.phase_reached == AgentPhase.DIAGNOSE
    assert "inconclusive" in result.skipped_reason.lower()
    assert result.diagnosis is not None
