"""Sandbox and experiment infrastructure for HYDRA.

Provides market replay, experiment management, and model evaluation
tools for the autonomous agent loop.
"""

from hydra.sandbox.evaluator import CompositeEvaluator, FitnessScore
from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord
from hydra.sandbox.registry import ModelRegistry
from hydra.sandbox.observer import DriftObserver, DriftReport, FeatureDriftReport, PerformanceDriftReport
from hydra.sandbox.replay import MarketReplayEngine, ReplayResult, TradeEvent

__all__ = [
    "CompositeEvaluator",
    "DriftObserver",
    "DriftReport",
    "ExperimentJournal",
    "ExperimentRecord",
    "FeatureDriftReport",
    "FitnessScore",
    "MarketReplayEngine",
    "ModelRegistry",
    "PerformanceDriftReport",
    "ReplayResult",
    "TradeEvent",
]
