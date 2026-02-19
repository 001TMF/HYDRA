"""Task-type to model routing for cost optimization.

Routes each LLM task to the cheapest capable model. The key insight:
reasoning tasks (diagnosis, hypothesis generation) need DeepSeek-R1,
but classification and formatting tasks work fine with much cheaper
models like Qwen2.5-7B-Turbo.

Default chains are based on Together AI pricing as of Feb 2026:
- DeepSeek-R1-0528: $3.00/$7.00 per 1M tokens (reasoning)
- Qwen3-235B: $0.20/$0.60 per 1M tokens (mid-tier)
- Qwen2.5-7B-Turbo: $0.30/$0.30 per 1M tokens (cheap)
- DeepSeek-V3.1: $0.60/$1.70 per 1M tokens (general)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Categories of LLM tasks for model routing."""

    REASONING = "reasoning"
    CLASSIFICATION = "classification"
    FORMATTING = "formatting"


class ModelSpec(BaseModel):
    """Specification for an LLM model endpoint."""

    provider: str = Field(description="Provider name: 'together' or 'deepseek'")
    model_id: str = Field(description="Model identifier for the API call")
    cost_per_1m_input: float = Field(
        description="Cost in USD per 1M input tokens"
    )
    cost_per_1m_output: float = Field(
        description="Cost in USD per 1M output tokens"
    )


# Default model specs
_DEEPSEEK_R1 = ModelSpec(
    provider="together",
    model_id="deepseek-ai/DeepSeek-R1-0528",
    cost_per_1m_input=3.00,
    cost_per_1m_output=7.00,
)

_QWEN3_235B = ModelSpec(
    provider="together",
    model_id="Qwen/Qwen3-235B-A22B-Instruct",
    cost_per_1m_input=0.20,
    cost_per_1m_output=0.60,
)

_QWEN25_7B_TURBO = ModelSpec(
    provider="together",
    model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
    cost_per_1m_input=0.30,
    cost_per_1m_output=0.30,
)

_DEEPSEEK_V3_1 = ModelSpec(
    provider="together",
    model_id="deepseek-ai/DeepSeek-V3.1",
    cost_per_1m_input=0.60,
    cost_per_1m_output=1.70,
)

# Default fallback chains by task type
_DEFAULT_CHAINS: dict[TaskType, list[ModelSpec]] = {
    TaskType.REASONING: [_DEEPSEEK_R1, _QWEN3_235B],
    TaskType.CLASSIFICATION: [_QWEN25_7B_TURBO, _DEEPSEEK_V3_1],
    TaskType.FORMATTING: [_QWEN25_7B_TURBO],
}


class TaskRouter:
    """Routes LLM tasks to the cheapest capable model via fallback chains.

    The router is configurable: pass a custom chains dict to override
    the defaults. This allows testing with mock models and runtime
    reconfiguration based on provider availability.
    """

    def __init__(
        self,
        chains: dict[TaskType, list[ModelSpec]] | None = None,
    ) -> None:
        self._chains = chains if chains is not None else dict(_DEFAULT_CHAINS)

    def get_fallback_chain(self, task_type: TaskType) -> list[ModelSpec]:
        """Return the ordered fallback chain for a task type.

        Returns an empty list if the task type has no configured chain,
        which means the caller should fall back to rule-based immediately.
        """
        return list(self._chains.get(task_type, []))

    def set_chain(
        self, task_type: TaskType, chain: list[ModelSpec]
    ) -> None:
        """Override the fallback chain for a task type at runtime."""
        self._chains[task_type] = list(chain)

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Return task types that have at least one model configured."""
        return [tt for tt, chain in self._chains.items() if chain]
