# HYDRA Agent Core
# Phase 4: Autonomous agent loop with LLM integration

from hydra.agent.loop import AgentLoop, AgentPhase, AgentCycleResult
from hydra.agent.autonomy import AutonomyLevel, check_permission, PermissionDeniedError
from hydra.agent.diagnostician import Diagnostician
from hydra.agent.hypothesis import HypothesisEngine, MUTATION_PLAYBOOK
from hydra.agent.experiment_runner import ExperimentRunner, ExperimentResult
from hydra.agent.rollback import HysteresisRollbackTrigger, RollbackConfig
from hydra.agent.promotion import PromotionEvaluator, PromotionConfig
from hydra.agent.dedup import HypothesisDeduplicator
from hydra.agent.budget import MutationBudget, BudgetConfig

__all__ = [
    "AgentLoop",
    "AgentPhase",
    "AgentCycleResult",
    "AutonomyLevel",
    "check_permission",
    "PermissionDeniedError",
    "Diagnostician",
    "HypothesisEngine",
    "MUTATION_PLAYBOOK",
    "ExperimentRunner",
    "ExperimentResult",
    "HysteresisRollbackTrigger",
    "RollbackConfig",
    "PromotionEvaluator",
    "PromotionConfig",
    "HypothesisDeduplicator",
    "MutationBudget",
    "BudgetConfig",
]
