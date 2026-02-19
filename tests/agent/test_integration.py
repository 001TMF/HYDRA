"""Integration test for the full observe-diagnose-hypothesize-experiment-evaluate cycle.

Creates real (non-mock) module instances and runs the complete agent loop
to prove the entire pipeline works end-to-end with zero LLM calls.

Uses:
- Real DriftObserver, Diagnostician, HypothesisEngine, ExperimentRunner
- Real CompositeEvaluator, ExperimentJournal (temp SQLite)
- Real HysteresisRollbackTrigger, PromotionEvaluator
- Mocked SentenceTransformer (to avoid downloading 22M model in CI)
- Mocked get_state (to return RUNNING)
- Mocked ModelRegistry (to avoid MLflow dependency in unit tests)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hydra.agent.autonomy import AutonomyLevel
from hydra.agent.budget import BudgetConfig, MutationBudget
from hydra.agent.diagnostician import Diagnostician
from hydra.agent.experiment_runner import ExperimentRunner
from hydra.agent.hypothesis import HypothesisEngine
from hydra.agent.loop import AgentLoop, AgentPhase
from hydra.agent.promotion import PromotionEvaluator
from hydra.agent.rollback import HysteresisRollbackTrigger, RollbackConfig
from hydra.sandbox.evaluator import CompositeEvaluator
from hydra.sandbox.journal import ExperimentJournal
from hydra.sandbox.observer import DriftObserver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def journal(tmp_path):
    """Create a temporary ExperimentJournal."""
    db_path = tmp_path / "test_journal.db"
    j = ExperimentJournal(db_path)
    yield j
    j.close()


@pytest.fixture
def mock_deduplicator():
    """Create a mock deduplicator that never flags duplicates.

    Avoids downloading the sentence-transformers model in CI.
    """
    dedup = MagicMock()
    dedup.is_duplicate.return_value = (False, 0.0)
    dedup.register.return_value = None
    return dedup


# ---------------------------------------------------------------------------
# Integration test: Full cycle with synthetic drift data
# ---------------------------------------------------------------------------


@patch("hydra.agent.loop.get_state")
def test_full_cycle_end_to_end(mock_get_state, journal, mock_deduplicator):
    """Complete observe-diagnose-hypothesize-experiment-evaluate cycle.

    This test proves the full pipeline works end-to-end with:
    - Zero LLM calls (the 'Honda' path)
    - Real observer detecting actual Sharpe degradation
    - Real diagnostician producing a structured diagnosis
    - Real hypothesis engine selecting from the playbook
    - ExperimentRunner (mocked subprocess, as no real training)
    - Real journal logging the experiment
    """
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    # Create real dependencies
    observer = DriftObserver()
    diagnostician = Diagnostician()
    hypothesis_engine = HypothesisEngine()
    evaluator = CompositeEvaluator()
    rollback_trigger = HysteresisRollbackTrigger(
        RollbackConfig(sustained_periods=5)  # High threshold to avoid triggering
    )
    promotion_evaluator = PromotionEvaluator()
    budget = MutationBudget(BudgetConfig(max_experiments_per_cycle=10))

    # Mock the ExperimentRunner to return a successful result
    # (We cannot run a real subprocess -- no trained model exists yet)
    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.fitness_score = 0.75
    mock_result.metrics = {"sharpe": 1.1, "max_drawdown": -0.08}
    mock_result.error_message = None
    mock_runner.run.return_value = mock_result

    # Mock the ModelRegistry (avoid MLflow dependency in test)
    mock_registry = MagicMock()

    # Create agent loop with AUTONOMOUS level
    loop = AgentLoop(
        observer=observer,
        diagnostician=diagnostician,
        hypothesis_engine=hypothesis_engine,
        experiment_runner=mock_runner,
        evaluator=evaluator,
        journal=journal,
        registry=mock_registry,
        rollback_trigger=rollback_trigger,
        promotion_evaluator=promotion_evaluator,
        deduplicator=mock_deduplicator,
        budget=budget,
        autonomy_level=AutonomyLevel.AUTONOMOUS,
    )

    # Create synthetic drift data: returns that produce sharpe_degraded=True
    # Baseline Sharpe is 2.0; use recent returns that give a very low Sharpe
    rng = np.random.default_rng(42)
    n = 60

    # Mostly negative returns with high variance -> low Sharpe
    recent_returns = rng.normal(loc=-0.005, scale=0.03, size=n)

    # Predictions are mostly wrong
    actuals = (rng.random(n) > 0.5).astype(float)
    predictions = (1 - actuals)  # Invert: all wrong -> low hit rate
    probabilities = rng.random(n) * 0.5  # Low probabilities

    result = loop.run_cycle(
        recent_returns=recent_returns,
        predictions=predictions,
        actuals=actuals,
        probabilities=probabilities,
        baseline_sharpe=2.0,  # High baseline ensures degradation is detected
        current_config={"learning_rate": 0.1, "num_leaves": 31},
        champion_fitness=0.5,
    )

    # Verify: cycle reaches PROMOTE phase (not skipped early)
    assert result.phase_reached == AgentPhase.PROMOTE

    # Verify: diagnosis was produced
    assert result.diagnosis is not None
    assert result.diagnosis.primary_cause is not None
    assert result.diagnosis.confidence > 0

    # Verify: hypothesis was from playbook
    assert result.hypothesis is not None
    assert result.hypothesis.source == "playbook"

    # Verify: experiment ran
    assert result.experiment_result is not None

    # Verify: promoted (candidate 0.75 > champion 0.5)
    assert result.promoted is True

    # Verify: journal has exactly 1 entry
    assert journal.count() == 1
    entries = journal.query()
    assert len(entries) == 1
    assert entries[0].promotion_decision == "promoted"
    assert entries[0].hypothesis == result.hypothesis.description

    # Verify: duration was tracked
    assert result.duration_seconds > 0


@patch("hydra.agent.loop.get_state")
def test_full_cycle_no_drift(mock_get_state, journal, mock_deduplicator):
    """When no drift is detected, cycle should stop at OBSERVE."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    observer = DriftObserver()
    loop = AgentLoop(
        observer=observer,
        diagnostician=Diagnostician(),
        hypothesis_engine=HypothesisEngine(),
        experiment_runner=MagicMock(),
        evaluator=CompositeEvaluator(),
        journal=journal,
        registry=MagicMock(),
        rollback_trigger=HysteresisRollbackTrigger(),
        promotion_evaluator=PromotionEvaluator(),
        deduplicator=mock_deduplicator,
        budget=MutationBudget(),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
    )

    # Generate returns with good performance -> no drift
    rng = np.random.default_rng(123)
    n = 60
    # Strong positive returns, low volatility -> high Sharpe
    recent_returns = rng.normal(loc=0.01, scale=0.005, size=n)
    actuals = (rng.random(n) > 0.3).astype(float)  # 70% hit rate
    predictions = actuals.copy()  # Perfect predictions -> high hit rate
    probabilities = actuals * 0.8 + (1 - actuals) * 0.2

    result = loop.run_cycle(
        recent_returns=recent_returns,
        predictions=predictions,
        actuals=actuals,
        probabilities=probabilities,
        baseline_sharpe=0.5,  # Low baseline -> current performance is much better
    )

    assert result.phase_reached == AgentPhase.OBSERVE
    assert "No drift" in result.skipped_reason
    assert journal.count() == 0


@patch("hydra.agent.loop.get_state")
def test_full_cycle_experiment_rejected(mock_get_state, journal, mock_deduplicator):
    """When candidate does not beat champion, experiment is rejected."""
    from hydra.cli.state import AgentState

    mock_get_state.return_value = AgentState.RUNNING

    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.fitness_score = 0.3  # Lower than champion
    mock_result.metrics = {"sharpe": 0.5}
    mock_result.error_message = None
    mock_runner.run.return_value = mock_result

    loop = AgentLoop(
        observer=DriftObserver(),
        diagnostician=Diagnostician(),
        hypothesis_engine=HypothesisEngine(),
        experiment_runner=mock_runner,
        evaluator=CompositeEvaluator(),
        journal=journal,
        registry=MagicMock(),
        rollback_trigger=HysteresisRollbackTrigger(
            RollbackConfig(sustained_periods=10)
        ),
        promotion_evaluator=PromotionEvaluator(),
        deduplicator=mock_deduplicator,
        budget=MutationBudget(BudgetConfig(max_experiments_per_cycle=10)),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
    )

    rng = np.random.default_rng(42)
    n = 60
    recent_returns = rng.normal(loc=-0.005, scale=0.03, size=n)
    actuals = (rng.random(n) > 0.5).astype(float)
    predictions = (1 - actuals)
    probabilities = rng.random(n) * 0.5

    result = loop.run_cycle(
        recent_returns=recent_returns,
        predictions=predictions,
        actuals=actuals,
        probabilities=probabilities,
        baseline_sharpe=2.0,
        current_config={"learning_rate": 0.1, "num_leaves": 31},
        champion_fitness=0.8,  # Higher than candidate 0.3
    )

    assert result.phase_reached == AgentPhase.PROMOTE
    assert result.promoted is False
    assert journal.count() == 1
    entries = journal.query()
    assert entries[0].promotion_decision == "rejected"
