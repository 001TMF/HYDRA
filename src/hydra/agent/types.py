"""Canonical shared types for the HYDRA agent subsystem.

This module defines the core domain enums and dataclasses used across
all agent components. It is the single source of truth -- other modules
(e.g., llm/schemas.py, diagnostician.py, hypothesis.py) import from here
rather than redefining types.

Zero LLM dependency -- pure dataclasses and enums.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DriftCategory(str, Enum):
    """Root cause categories for model degradation."""

    PERFORMANCE = "performance_degradation"
    FEATURE_DRIFT = "feature_distribution_drift"
    REGIME_CHANGE = "regime_change"
    OVERFITTING = "overfitting"
    DATA_QUALITY = "data_quality_issue"


class MutationType(str, Enum):
    """Categories of model mutations the agent can propose."""

    HYPERPARAMETER = "hyperparameter"
    FEATURE_ADD = "feature_add"
    FEATURE_REMOVE = "feature_remove"
    FEATURE_ENGINEERING = "feature_engineering"
    ENSEMBLE_METHOD = "ensemble_method"
    PREDICTION_TARGET = "prediction_target"
    NEW_DATA_SIGNAL = "new_data_signal"


@dataclass
class DiagnosisResult:
    """Output of the diagnostician: identified root cause with evidence.

    Attributes
    ----------
    primary_cause : DriftCategory
        The most likely root cause of observed drift.
    confidence : float
        Confidence in the diagnosis, 0.0 to 1.0.
    evidence : list[str]
        Human-readable evidence strings supporting the diagnosis.
    recommended_mutation_types : list[str]
        Mutation type values recommended for this root cause.
    reasoning : str
        Explanation of how the diagnosis was reached.
    """

    primary_cause: DriftCategory
    confidence: float
    evidence: list[str] = field(default_factory=list)
    recommended_mutation_types: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class Hypothesis:
    """A concrete mutation proposal generated from the playbook or LLM.

    Attributes
    ----------
    mutation_type : MutationType
        Category of mutation being proposed.
    description : str
        Human-readable description of the proposed change.
    config_diff : dict
        Configuration changes to apply (key -> new value).
    expected_impact : str
        What improvement is expected from this mutation.
    testable_prediction : str
        A falsifiable prediction for sandbox evaluation.
    source : str
        Origin of this hypothesis: "playbook" or "llm".
    """

    mutation_type: MutationType
    description: str
    config_diff: dict
    expected_impact: str
    testable_prediction: str
    source: str  # "playbook" or "llm"
