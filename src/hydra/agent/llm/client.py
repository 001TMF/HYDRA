"""Multi-provider LLM client with fallback chain, cost tracking, and budget cap.

The LLMClient is the gateway for all LLM calls in the agent subsystem.
It wraps the instructor library for structured output extraction with
Pydantic validation and automatic retry.

Key contract:
    try:
        result = client.call(DiagnosisResultLLM, messages, TaskType.REASONING)
    except LLMUnavailableError:
        result = rule_based_diagnosis(drift_report)

When no API keys are configured, the client enters "disabled" mode and
immediately raises LLMUnavailableError on any call(). This satisfies
AGNT-10: the system operates entirely rule-based by default.
"""

from __future__ import annotations

import logging
from typing import TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from hydra.agent.llm.router import ModelSpec, TaskRouter, TaskType

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(Exception):
    """Raised when all LLM providers in the fallback chain have failed.

    Callers catch this and fall back to rule-based logic.
    """


class LLMConfig(BaseModel):
    """Configuration for the LLM client."""

    together_api_key: str | None = Field(
        default=None,
        description="Together AI API key for DeepSeek-R1, Qwen, etc.",
    )
    deepseek_api_key: str | None = Field(
        default=None,
        description="DeepSeek direct API key (optional alternative)",
    )
    daily_budget: float = Field(
        default=20.0,
        description="Maximum daily spend in USD across all providers",
    )
    max_retries_per_provider: int = Field(
        default=2,
        description="Number of instructor retries per provider before moving to next",
    )


class LLMClient:
    """Multi-provider LLM client with fallback chain and cost tracking.

    Uses instructor for structured Pydantic output. Iterates through
    the fallback chain from the TaskRouter. Tracks cost and enforces
    a daily budget cap.

    Parameters
    ----------
    config : LLMConfig
        API keys and budget settings.
    router : TaskRouter | None
        Custom task router. Uses default router if None.
    """

    # Provider configuration: name -> (base_url,)
    _PROVIDER_URLS: dict[str, str] = {
        "together": "https://api.together.ai/v1",
        "deepseek": "https://api.deepseek.com",
    }

    def __init__(
        self,
        config: LLMConfig,
        router: TaskRouter | None = None,
    ) -> None:
        self._config = config
        self._router = router or TaskRouter()
        self._daily_cost: float = 0.0
        self._clients: dict[str, instructor.Instructor] = {}

        # Build instructor-wrapped clients for each configured provider
        if config.together_api_key:
            self._clients["together"] = instructor.from_openai(
                OpenAI(
                    api_key=config.together_api_key,
                    base_url=self._PROVIDER_URLS["together"],
                ),
            )

        if config.deepseek_api_key:
            self._clients["deepseek"] = instructor.from_openai(
                OpenAI(
                    api_key=config.deepseek_api_key,
                    base_url=self._PROVIDER_URLS["deepseek"],
                ),
            )

    @property
    def is_available(self) -> bool:
        """True if at least one API key is configured and budget not exceeded."""
        return bool(self._clients) and self._check_budget()

    @property
    def daily_cost(self) -> float:
        """Current accumulated daily cost in USD."""
        return self._daily_cost

    def call(
        self,
        response_model: type[T],
        messages: list[dict],
        task_type: TaskType = TaskType.REASONING,
    ) -> T:
        """Call LLM with structured output, iterating through fallback chain.

        Parameters
        ----------
        response_model : type[T]
            Pydantic model class for structured output extraction.
        messages : list[dict]
            Chat messages in OpenAI format.
        task_type : TaskType
            Determines which fallback chain to use.

        Returns
        -------
        T
            Validated Pydantic model instance.

        Raises
        ------
        LLMUnavailableError
            When all providers fail or no API keys are configured.
        """
        if not self._clients:
            raise LLMUnavailableError(
                "No API keys configured. System operating in rule-based mode."
            )

        if not self._check_budget():
            raise LLMUnavailableError(
                f"Daily budget exceeded: ${self._daily_cost:.2f} >= "
                f"${self._config.daily_budget:.2f}"
            )

        chain = self._router.get_fallback_chain(task_type)
        if not chain:
            raise LLMUnavailableError(
                f"No fallback chain configured for task type: {task_type.value}"
            )

        last_error: Exception | None = None

        for model_spec in chain:
            client = self._clients.get(model_spec.provider)
            if client is None:
                # Provider not configured, skip to next in chain
                continue

            try:
                result = client.chat.completions.create(
                    model=model_spec.model_id,
                    response_model=response_model,
                    messages=messages,
                    max_retries=self._config.max_retries_per_provider,
                )

                # Estimate cost from message/response lengths
                input_tokens = self._estimate_tokens(messages)
                output_tokens = self._estimate_tokens_from_model(result)
                self._track_cost(model_spec, input_tokens, output_tokens)

                logger.info(
                    "LLM call succeeded",
                    extra={
                        "provider": model_spec.provider,
                        "model": model_spec.model_id,
                        "task_type": task_type.value,
                        "daily_cost": f"${self._daily_cost:.4f}",
                    },
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM provider failed, trying next in chain",
                    extra={
                        "provider": model_spec.provider,
                        "model": model_spec.model_id,
                        "error": str(e),
                    },
                )
                continue

        raise LLMUnavailableError(
            f"All providers in fallback chain failed for {task_type.value}. "
            f"Last error: {last_error}"
        )

    def _track_cost(
        self,
        model_spec: ModelSpec,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Add estimated cost to the daily accumulator."""
        input_cost = (input_tokens / 1_000_000) * model_spec.cost_per_1m_input
        output_cost = (output_tokens / 1_000_000) * model_spec.cost_per_1m_output
        self._daily_cost += input_cost + output_cost

    def _check_budget(self) -> bool:
        """Return True if daily budget has NOT been exceeded."""
        return self._daily_cost < self._config.daily_budget

    def reset_daily_cost(self) -> None:
        """Reset the daily cost tracker. Called by scheduler at midnight."""
        self._daily_cost = 0.0

    @staticmethod
    def _estimate_tokens(messages: list[dict]) -> int:
        """Rough token estimate from message content lengths.

        Approximation: ~4 characters per token (English text average).
        """
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    @staticmethod
    def _estimate_tokens_from_model(result: BaseModel) -> int:
        """Rough token estimate from Pydantic model JSON output."""
        json_str = result.model_dump_json()
        return max(1, len(json_str) // 4)
