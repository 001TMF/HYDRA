"""Tests for the LLM client, schemas, and router.

All tests use mocks -- no live API calls. Tests cover:
- Disabled mode (no API keys) raises LLMUnavailableError
- Budget enforcement
- Fallback chain behavior (first fails, second succeeds)
- All providers failing raises LLMUnavailableError
- Cost tracking accumulates correctly
- Router integration with task types
- Schema validation (valid and invalid inputs)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError

from hydra.agent.llm.client import LLMClient, LLMConfig, LLMUnavailableError
from hydra.agent.llm.router import ModelSpec, TaskRouter, TaskType
from hydra.agent.llm.schemas import (
    DiagnosisResultLLM,
    ExperimentConfig,
    HypothesisLLM,
)
from hydra.agent.types import DriftCategory, MutationType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleResponse(BaseModel):
    """Minimal response model for testing."""

    answer: str = Field(description="A short answer")


@pytest.fixture
def disabled_client() -> LLMClient:
    """Client with no API keys -- disabled mode."""
    return LLMClient(LLMConfig())


@pytest.fixture
def mock_config() -> LLMConfig:
    """Config with a fake Together AI key."""
    return LLMConfig(
        together_api_key="fake-together-key-12345",
        daily_budget=20.0,
        max_retries_per_provider=1,
    )


@pytest.fixture
def sample_messages() -> list[dict]:
    return [{"role": "user", "content": "Diagnose this drift report."}]


# ---------------------------------------------------------------------------
# Disabled mode tests (AGNT-10)
# ---------------------------------------------------------------------------


class TestDisabledMode:
    """When no API keys are configured, system must be rule-based."""

    def test_is_available_false_when_no_keys(self, disabled_client: LLMClient) -> None:
        assert disabled_client.is_available is False

    def test_call_raises_unavailable_when_no_keys(
        self, disabled_client: LLMClient, sample_messages: list[dict]
    ) -> None:
        with pytest.raises(LLMUnavailableError, match="No API keys configured"):
            disabled_client.call(SimpleResponse, sample_messages)

    def test_daily_cost_starts_at_zero(self, disabled_client: LLMClient) -> None:
        assert disabled_client.daily_cost == 0.0


# ---------------------------------------------------------------------------
# Budget enforcement tests
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_is_available_false_when_budget_exceeded(
        self, mock_config: LLMConfig
    ) -> None:
        client = LLMClient(mock_config)
        # Manually set cost above budget
        client._daily_cost = 25.0
        assert client.is_available is False

    def test_check_budget_returns_false_after_exceeding(
        self, mock_config: LLMConfig
    ) -> None:
        client = LLMClient(mock_config)
        client._daily_cost = 20.0  # exactly at budget
        assert client._check_budget() is False

    def test_check_budget_returns_true_under_budget(
        self, mock_config: LLMConfig
    ) -> None:
        client = LLMClient(mock_config)
        client._daily_cost = 10.0
        assert client._check_budget() is True

    def test_call_raises_when_budget_exceeded(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        client = LLMClient(mock_config)
        client._daily_cost = 25.0
        with pytest.raises(LLMUnavailableError, match="Daily budget exceeded"):
            client.call(SimpleResponse, sample_messages)

    def test_reset_daily_cost(self, mock_config: LLMConfig) -> None:
        client = LLMClient(mock_config)
        client._daily_cost = 15.0
        client.reset_daily_cost()
        assert client.daily_cost == 0.0


# ---------------------------------------------------------------------------
# Cost tracking tests
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_track_cost_accumulates(self, mock_config: LLMConfig) -> None:
        client = LLMClient(mock_config)
        model_spec = ModelSpec(
            provider="together",
            model_id="test-model",
            cost_per_1m_input=3.0,
            cost_per_1m_output=7.0,
        )
        # 1000 input tokens, 500 output tokens
        client._track_cost(model_spec, input_tokens=1000, output_tokens=500)
        # Expected: (1000/1e6)*3.0 + (500/1e6)*7.0 = 0.003 + 0.0035 = 0.0065
        assert abs(client.daily_cost - 0.0065) < 1e-10

    def test_track_cost_multiple_calls_accumulate(
        self, mock_config: LLMConfig
    ) -> None:
        client = LLMClient(mock_config)
        model_spec = ModelSpec(
            provider="together",
            model_id="test-model",
            cost_per_1m_input=1.0,
            cost_per_1m_output=1.0,
        )
        client._track_cost(model_spec, input_tokens=1_000_000, output_tokens=0)
        assert abs(client.daily_cost - 1.0) < 1e-10
        client._track_cost(model_spec, input_tokens=0, output_tokens=1_000_000)
        assert abs(client.daily_cost - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Fallback chain tests (mock instructor)
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_first_provider_fails_second_succeeds(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        """When first provider in chain fails, client tries the next."""
        client = LLMClient(mock_config)
        expected = SimpleResponse(answer="test answer")

        # Mock the instructor client's chat.completions.create
        mock_instructor = MagicMock()
        # First call raises, second call returns result
        mock_instructor.chat.completions.create.side_effect = [
            RuntimeError("Provider 1 timeout"),
            expected,
        ]

        # Set up a 2-model chain both on "together" provider
        router = TaskRouter(
            chains={
                TaskType.REASONING: [
                    ModelSpec(
                        provider="together",
                        model_id="model-a",
                        cost_per_1m_input=3.0,
                        cost_per_1m_output=7.0,
                    ),
                    ModelSpec(
                        provider="together",
                        model_id="model-b",
                        cost_per_1m_input=0.2,
                        cost_per_1m_output=0.6,
                    ),
                ]
            }
        )
        client._router = router
        client._clients["together"] = mock_instructor

        result = client.call(SimpleResponse, sample_messages, TaskType.REASONING)
        assert result.answer == "test answer"
        assert mock_instructor.chat.completions.create.call_count == 2

    def test_all_providers_fail_raises_unavailable(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        """When all providers in chain fail, LLMUnavailableError is raised."""
        client = LLMClient(mock_config)

        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.side_effect = RuntimeError(
            "All broken"
        )

        router = TaskRouter(
            chains={
                TaskType.REASONING: [
                    ModelSpec(
                        provider="together",
                        model_id="model-a",
                        cost_per_1m_input=3.0,
                        cost_per_1m_output=7.0,
                    ),
                    ModelSpec(
                        provider="together",
                        model_id="model-b",
                        cost_per_1m_input=0.2,
                        cost_per_1m_output=0.6,
                    ),
                ]
            }
        )
        client._router = router
        client._clients["together"] = mock_instructor

        with pytest.raises(LLMUnavailableError, match="All providers"):
            client.call(SimpleResponse, sample_messages, TaskType.REASONING)

    def test_skips_unconfigured_provider(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        """Models from unconfigured providers are skipped gracefully."""
        client = LLMClient(mock_config)
        expected = SimpleResponse(answer="from together")

        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = expected

        # Chain has deepseek (not configured) then together (configured)
        router = TaskRouter(
            chains={
                TaskType.REASONING: [
                    ModelSpec(
                        provider="deepseek",
                        model_id="deepseek-chat",
                        cost_per_1m_input=0.6,
                        cost_per_1m_output=1.7,
                    ),
                    ModelSpec(
                        provider="together",
                        model_id="model-a",
                        cost_per_1m_input=3.0,
                        cost_per_1m_output=7.0,
                    ),
                ]
            }
        )
        client._router = router
        client._clients["together"] = mock_instructor

        result = client.call(SimpleResponse, sample_messages, TaskType.REASONING)
        assert result.answer == "from together"

    def test_successful_call_tracks_cost(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        """Cost is tracked after a successful call."""
        client = LLMClient(mock_config)
        expected = SimpleResponse(answer="ok")

        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = expected

        router = TaskRouter(
            chains={
                TaskType.CLASSIFICATION: [
                    ModelSpec(
                        provider="together",
                        model_id="cheap-model",
                        cost_per_1m_input=0.3,
                        cost_per_1m_output=0.3,
                    ),
                ]
            }
        )
        client._router = router
        client._clients["together"] = mock_instructor

        result = client.call(
            SimpleResponse, sample_messages, TaskType.CLASSIFICATION
        )
        assert result.answer == "ok"
        assert client.daily_cost > 0.0


# ---------------------------------------------------------------------------
# Router integration tests
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    def test_client_uses_task_type_for_routing(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        """Client passes task_type through to router for chain selection."""
        client = LLMClient(mock_config)
        expected = SimpleResponse(answer="classified")

        mock_instructor = MagicMock()
        mock_instructor.chat.completions.create.return_value = expected
        client._clients["together"] = mock_instructor

        # Call with CLASSIFICATION -- should use classification chain
        result = client.call(
            SimpleResponse, sample_messages, TaskType.CLASSIFICATION
        )
        assert result.answer == "classified"

        # Verify the model used was from the classification chain
        call_kwargs = mock_instructor.chat.completions.create.call_args
        model_used = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        # Default classification chain starts with Qwen2.5-7B
        assert "Qwen" in model_used or "qwen" in model_used.lower()

    def test_empty_chain_raises_unavailable(
        self, mock_config: LLMConfig, sample_messages: list[dict]
    ) -> None:
        """Task type with no chain configured raises LLMUnavailableError."""
        client = LLMClient(mock_config)
        client._router = TaskRouter(chains={})

        with pytest.raises(LLMUnavailableError, match="No fallback chain"):
            client.call(SimpleResponse, sample_messages, TaskType.REASONING)


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_diagnosis_result_valid(self) -> None:
        d = DiagnosisResultLLM(
            primary_cause=DriftCategory.PERFORMANCE,
            confidence=0.85,
            evidence=["High PSI on feature X", "Sharpe ratio declined 30%"],
            recommended_mutation_types=["hyperparameter", "feature_remove"],
            reasoning="PSI indicates feature distribution shift",
        )
        assert d.primary_cause == DriftCategory.PERFORMANCE
        assert d.confidence == 0.85
        assert len(d.evidence) == 2

    def test_diagnosis_result_invalid_confidence(self) -> None:
        with pytest.raises(ValidationError):
            DiagnosisResultLLM(
                primary_cause=DriftCategory.PERFORMANCE,
                confidence=1.5,  # out of range
                evidence=[],
                recommended_mutation_types=[],
                reasoning="",
            )

    def test_diagnosis_result_invalid_category(self) -> None:
        with pytest.raises(ValidationError):
            DiagnosisResultLLM(
                primary_cause="not_a_real_category",
                confidence=0.5,
                evidence=[],
                recommended_mutation_types=[],
                reasoning="",
            )

    def test_hypothesis_valid(self) -> None:
        h = HypothesisLLM(
            mutation_type=MutationType.HYPERPARAMETER,
            description="Reduce learning rate by 50%",
            config_diff={"learning_rate": 0.05},
            expected_impact="Improved OOS generalization",
            testable_prediction="OOS Sharpe > 0.3",
        )
        assert h.mutation_type == MutationType.HYPERPARAMETER
        assert h.source == "llm"  # default

    def test_hypothesis_playbook_source(self) -> None:
        h = HypothesisLLM(
            mutation_type=MutationType.FEATURE_ADD,
            description="Add rolling z-scores",
            config_diff={"add_features": ["z_score_30d"]},
            expected_impact="Distribution-invariant features",
            testable_prediction="Feature importance > 0.05",
            source="playbook",
        )
        assert h.source == "playbook"

    def test_experiment_config_defaults(self) -> None:
        h = HypothesisLLM(
            mutation_type=MutationType.HYPERPARAMETER,
            description="test",
            config_diff={},
            expected_impact="none",
            testable_prediction="none",
        )
        ec = ExperimentConfig(hypothesis=h)
        assert ec.training_timeout_seconds == 300
        assert ec.base_config == {}


# ---------------------------------------------------------------------------
# Router unit tests
# ---------------------------------------------------------------------------


class TestRouterUnit:
    def test_default_reasoning_chain_has_two_models(self) -> None:
        r = TaskRouter()
        chain = r.get_fallback_chain(TaskType.REASONING)
        assert len(chain) >= 2

    def test_default_classification_chain(self) -> None:
        r = TaskRouter()
        chain = r.get_fallback_chain(TaskType.CLASSIFICATION)
        assert len(chain) >= 1
        assert "Qwen" in chain[0].model_id

    def test_default_formatting_chain(self) -> None:
        r = TaskRouter()
        chain = r.get_fallback_chain(TaskType.FORMATTING)
        assert len(chain) == 1

    def test_custom_chains(self) -> None:
        custom = ModelSpec(
            provider="test",
            model_id="test-model",
            cost_per_1m_input=0.0,
            cost_per_1m_output=0.0,
        )
        r = TaskRouter(chains={TaskType.REASONING: [custom]})
        chain = r.get_fallback_chain(TaskType.REASONING)
        assert len(chain) == 1
        assert chain[0].model_id == "test-model"

    def test_set_chain_at_runtime(self) -> None:
        r = TaskRouter()
        custom = ModelSpec(
            provider="test",
            model_id="runtime-model",
            cost_per_1m_input=0.0,
            cost_per_1m_output=0.0,
        )
        r.set_chain(TaskType.FORMATTING, [custom])
        chain = r.get_fallback_chain(TaskType.FORMATTING)
        assert chain[0].model_id == "runtime-model"

    def test_supported_task_types(self) -> None:
        r = TaskRouter()
        supported = r.supported_task_types
        assert TaskType.REASONING in supported
        assert TaskType.CLASSIFICATION in supported
        assert TaskType.FORMATTING in supported

    def test_unknown_task_type_returns_empty(self) -> None:
        r = TaskRouter(chains={})
        chain = r.get_fallback_chain(TaskType.REASONING)
        assert chain == []


# ---------------------------------------------------------------------------
# Token estimation tests
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_tokens_from_messages(self) -> None:
        messages = [{"role": "user", "content": "Hello world test message"}]
        tokens = LLMClient._estimate_tokens(messages)
        # ~24 chars / 4 = ~6 tokens
        assert tokens > 0

    def test_estimate_tokens_empty_message(self) -> None:
        messages = [{"role": "user", "content": ""}]
        tokens = LLMClient._estimate_tokens(messages)
        assert tokens >= 1  # minimum 1

    def test_estimate_tokens_from_model(self) -> None:
        model = SimpleResponse(answer="Short answer")
        tokens = LLMClient._estimate_tokens_from_model(model)
        assert tokens > 0
