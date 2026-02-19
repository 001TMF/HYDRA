"""Mutation playbook and hypothesis generation engine.

Maps diagnosed root causes (DiagnosisResult) to candidate mutations from
a curated playbook. Uses round-robin selection to ensure diversity across
experiments. Config diff expressions are resolved against current model
configuration when available.

The hypothesis engine is entirely deterministic by default. An optional
LLM client can be used to generate creative mutations beyond the playbook,
but this is not implemented in Phase 4 wave 1.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from hydra.agent.types import (
    DriftCategory,
    DiagnosisResult,
    Hypothesis,
    MutationType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated mutation playbook
# ---------------------------------------------------------------------------

MUTATION_PLAYBOOK: dict[str, list[dict]] = {
    "performance_degradation": [
        {
            "type": "hyperparameter",
            "name": "reduce_learning_rate",
            "config_diff": {"learning_rate": "current * 0.5"},
            "rationale": "Slower learning rate reduces overshoot on noisy gradients",
        },
        {
            "type": "hyperparameter",
            "name": "increase_regularization",
            "config_diff": {
                "reg_alpha": "current * 2",
                "reg_lambda": "current * 2",
            },
            "rationale": "Stronger L1/L2 regularization reduces model variance",
        },
        {
            "type": "feature_remove",
            "name": "drop_low_importance_features",
            "config_diff": {"drop_features": "bottom 3 by SHAP"},
            "rationale": "Remove noisy features with lowest SHAP importance",
        },
    ],
    "feature_distribution_drift": [
        {
            "type": "feature_engineering",
            "name": "shorten_training_window",
            "config_diff": {"training_window": "current * 0.7"},
            "rationale": "Shorter window weights recent regime more heavily",
        },
        {
            "type": "feature_engineering",
            "name": "add_rolling_z_scores",
            "config_diff": {
                "add_features": ["z_score_30d", "z_score_90d"],
            },
            "rationale": "Z-scored features are distribution-invariant",
        },
    ],
    "regime_change": [
        {
            "type": "ensemble_method",
            "name": "add_regime_conditioning",
            "config_diff": {"ensemble_type": "regime_conditional"},
            "rationale": "Regime-aware ensemble adapts to market state transitions",
        },
        {
            "type": "hyperparameter",
            "name": "increase_num_leaves",
            "config_diff": {"num_leaves": "current * 1.5"},
            "rationale": "More leaves capture regime-specific decision boundaries",
        },
    ],
    "overfitting": [
        {
            "type": "hyperparameter",
            "name": "reduce_num_leaves",
            "config_diff": {"num_leaves": "max(8, current // 2)"},
            "rationale": "Fewer leaves reduce model complexity and overfitting risk",
        },
        {
            "type": "hyperparameter",
            "name": "increase_min_child_samples",
            "config_diff": {"min_child_samples": "current * 2"},
            "rationale": "Higher minimum samples per leaf prevents memorization",
        },
        {
            "type": "feature_remove",
            "name": "add_dropout",
            "config_diff": {"feature_fraction": "current * 0.7"},
            "rationale": "Feature subsampling acts as dropout regularization",
        },
    ],
    "data_quality_issue": [
        {
            "type": "feature_remove",
            "name": "remove_degraded_features",
            "config_diff": {"drop_features": "features_with_high_psi"},
            "rationale": "Remove features with distribution shift indicating data quality issues",
        },
        {
            "type": "hyperparameter",
            "name": "reduce_training_window",
            "config_diff": {"training_window": "current * 0.5"},
            "rationale": "Shorter window avoids stale/corrupted historical data",
        },
    ],
}

# ---------------------------------------------------------------------------
# Config diff resolver
# ---------------------------------------------------------------------------

# Pattern: "current * N" or "current * N.N"
_MULTIPLY_PATTERN = re.compile(r"^current\s*\*\s*([\d.]+)$")

# Pattern: "current // N"
_FLOOR_DIV_PATTERN = re.compile(r"^current\s*//\s*(\d+)$")

# Pattern: "max(N, current // M)"
_MAX_FLOOR_PATTERN = re.compile(r"^max\(\s*(\d+)\s*,\s*current\s*//\s*(\d+)\s*\)$")


def _resolve_expression(expr: str, current_value: float) -> float | str:
    """Resolve a config_diff expression against a current value.

    Supports:
    - "current * N" -> current_value * N
    - "current // N" -> current_value // N
    - "max(N, current // M)" -> max(N, current_value // M)

    Falls back to the raw string if the expression is unresolvable.
    """
    m = _MULTIPLY_PATTERN.match(expr.strip())
    if m:
        return current_value * float(m.group(1))

    m = _FLOOR_DIV_PATTERN.match(expr.strip())
    if m:
        return current_value // int(m.group(1))

    m = _MAX_FLOOR_PATTERN.match(expr.strip())
    if m:
        floor_val = int(m.group(1))
        divisor = int(m.group(2))
        return max(floor_val, current_value // divisor)

    return expr  # Unresolvable -- return raw string


# ---------------------------------------------------------------------------
# HypothesisEngine
# ---------------------------------------------------------------------------


class HypothesisEngine:
    """Generate mutation hypotheses from a curated playbook.

    Parameters
    ----------
    playbook : dict | None
        Custom playbook override. Keys are drift category values (str),
        values are lists of mutation entry dicts.
    llm_client : object | None
        Optional LLM client for creative hypothesis generation
        (not used in wave 1).
    """

    def __init__(
        self,
        playbook: dict[str, list[dict]] | None = None,
        llm_client: Any = None,
    ) -> None:
        self.playbook = playbook if playbook is not None else dict(MUTATION_PLAYBOOK)
        self.llm_client = llm_client
        self._round_robin_idx: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propose(
        self,
        diagnosis: DiagnosisResult,
        current_config: dict | None = None,
    ) -> Hypothesis:
        """Generate a single hypothesis from the playbook.

        Uses round-robin selection within the diagnosed category to
        ensure diversity across repeated calls.

        Parameters
        ----------
        diagnosis : DiagnosisResult
            The diagnosis to generate a mutation for.
        current_config : dict | None
            Current model configuration for resolving expressions
            like "current * 0.5".

        Returns
        -------
        Hypothesis

        Raises
        ------
        ValueError
            If no playbook entries exist for the diagnosed category.
        """
        category = diagnosis.primary_cause.value
        entries = self.playbook.get(category, [])

        if not entries:
            raise ValueError(
                f"No playbook entries for category: {category}"
            )

        # Round-robin selection
        idx = self._round_robin_idx.get(category, 0)
        entry = entries[idx % len(entries)]
        self._round_robin_idx[category] = (idx + 1) % len(entries)

        return self._build_hypothesis(entry, current_config)

    def propose_multiple(
        self,
        diagnosis: DiagnosisResult,
        n: int = 3,
        current_config: dict | None = None,
    ) -> list[Hypothesis]:
        """Generate up to *n* hypotheses from the playbook.

        Returns all available entries if n exceeds the playbook size
        for the diagnosed category.

        Parameters
        ----------
        diagnosis : DiagnosisResult
            The diagnosis to generate mutations for.
        n : int
            Maximum number of hypotheses to generate.
        current_config : dict | None
            Current model configuration for expression resolution.

        Returns
        -------
        list[Hypothesis]
        """
        category = diagnosis.primary_cause.value
        entries = self.playbook.get(category, [])

        if not entries:
            return []

        count = min(n, len(entries))
        hypotheses: list[Hypothesis] = []
        for _ in range(count):
            hyp = self.propose(diagnosis, current_config)
            hypotheses.append(hyp)

        return hypotheses

    def get_playbook_size(self, category: str) -> int:
        """Return the number of mutations available for a category."""
        return len(self.playbook.get(category, []))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_hypothesis(
        self,
        entry: dict,
        current_config: dict | None,
    ) -> Hypothesis:
        """Build a Hypothesis from a playbook entry, resolving config diffs."""
        mutation_type = MutationType(entry["type"])
        config_diff = dict(entry["config_diff"])

        # Resolve expressions against current config
        if current_config is not None:
            config_diff = self._resolve_config_diff(config_diff, current_config)

        return Hypothesis(
            mutation_type=mutation_type,
            description=entry["name"],
            config_diff=config_diff,
            expected_impact=entry["rationale"],
            testable_prediction=(
                f"Applying {entry['name']} should improve "
                f"model performance on the next sandbox evaluation"
            ),
            source="playbook",
        )

    @staticmethod
    def _resolve_config_diff(
        config_diff: dict,
        current_config: dict,
    ) -> dict:
        """Resolve string expressions in config_diff against current_config.

        For each key in config_diff whose value is a string expression
        containing 'current', look up the key in current_config and
        resolve the expression. Non-string values and unresolvable
        expressions are preserved as-is.
        """
        resolved: dict = {}
        for key, value in config_diff.items():
            if isinstance(value, str) and "current" in value:
                if key in current_config:
                    current_val = float(current_config[key])
                    resolved[key] = _resolve_expression(value, current_val)
                else:
                    resolved[key] = value  # No current value to resolve against
            else:
                resolved[key] = value
        return resolved
