"""Pydantic wrappers for LLM structured I/O.

These schemas wrap the canonical types from hydra.agent.types and add
Field(description=...) annotations so that the instructor library can
use them as schema hints for structured output extraction from LLMs.

The LLM schemas are intentionally separate from the core domain types:
- Core types (types.py): minimal enums/dataclasses used throughout the agent
- LLM schemas (this file): Pydantic BaseModels with rich descriptions for LLM use

This separation prevents coupling the entire agent to Pydantic while
enabling instructor's schema-guided generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hydra.agent.types import DriftCategory, MutationType


class DiagnosisResultLLM(BaseModel):
    """Structured output from LLM diagnosis of a drift report."""

    primary_cause: DriftCategory = Field(
        description="Root cause category of the model degradation"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this diagnosis (0.0 to 1.0)",
    )
    evidence: list[str] = Field(
        description="Specific signals from the drift report supporting this diagnosis"
    )
    recommended_mutation_types: list[str] = Field(
        description="Which mutation types to try, in priority order"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning explaining this diagnosis"
    )


class HypothesisLLM(BaseModel):
    """Structured hypothesis proposed by a head via LLM."""

    mutation_type: MutationType = Field(
        description="Category of model change to attempt"
    )
    description: str = Field(
        description="Human-readable description of the proposed change"
    )
    config_diff: dict = Field(
        description="Key-value config changes to apply"
    )
    expected_impact: str = Field(
        description="Predicted effect on model performance"
    )
    testable_prediction: str = Field(
        description="Specific measurable outcome to validate"
    )
    source: str = Field(
        default="llm",
        description="Origin: 'playbook' for rule-based, 'llm' for LLM-generated",
    )


class ExperimentConfig(BaseModel):
    """Configuration for running a sandbox experiment from a hypothesis."""

    hypothesis: HypothesisLLM = Field(
        description="The hypothesis to test"
    )
    training_timeout_seconds: int = Field(
        default=300,
        description="Maximum seconds allowed for candidate training",
    )
    base_config: dict = Field(
        default_factory=dict,
        description="Base model configuration to apply the hypothesis diff on top of",
    )
