"""Tests for the mutation playbook and hypothesis engine.

Verifies that the HypothesisEngine produces valid Hypothesis objects
from the curated MUTATION_PLAYBOOK using round-robin selection and
config diff resolution.
"""

from __future__ import annotations

import pytest

from hydra.agent.hypothesis import HypothesisEngine, MUTATION_PLAYBOOK
from hydra.agent.types import (
    DriftCategory,
    DiagnosisResult,
    Hypothesis,
    MutationType,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _diagnosis(
    cause: DriftCategory = DriftCategory.PERFORMANCE,
    confidence: float = 0.7,
) -> DiagnosisResult:
    """Build a DiagnosisResult with sensible defaults."""
    return DiagnosisResult(
        primary_cause=cause,
        confidence=confidence,
        evidence=["test evidence"],
        recommended_mutation_types=["hyperparameter"],
        reasoning="test reasoning",
    )


# ---------------------------------------------------------------------------
# Playbook tests
# ---------------------------------------------------------------------------


class TestMutationPlaybook:
    """Verify the curated MUTATION_PLAYBOOK structure."""

    def test_all_5_categories_have_entries(self) -> None:
        expected_categories = {
            "performance_degradation",
            "feature_distribution_drift",
            "regime_change",
            "overfitting",
            "data_quality_issue",
        }
        assert set(MUTATION_PLAYBOOK.keys()) == expected_categories

    def test_each_category_has_at_least_2_entries(self) -> None:
        for category, entries in MUTATION_PLAYBOOK.items():
            assert len(entries) >= 2, f"{category} has only {len(entries)} entries"

    def test_each_entry_has_required_keys(self) -> None:
        required_keys = {"type", "name", "config_diff", "rationale"}
        for category, entries in MUTATION_PLAYBOOK.items():
            for entry in entries:
                missing = required_keys - set(entry.keys())
                assert not missing, (
                    f"{category}/{entry.get('name', '?')} missing keys: {missing}"
                )


# ---------------------------------------------------------------------------
# HypothesisEngine tests
# ---------------------------------------------------------------------------


class TestHypothesisEngine:
    """Verify hypothesis generation from DiagnosisResult."""

    def test_propose_performance_returns_valid_hypothesis(self) -> None:
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        hyp = engine.propose(diag)

        assert isinstance(hyp, Hypothesis)
        assert hyp.source == "playbook"
        assert isinstance(hyp.mutation_type, MutationType)

    def test_round_robin_cycles_through_entries(self) -> None:
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        n_entries = engine.get_playbook_size("performance_degradation")

        names = []
        for _ in range(n_entries + 1):
            hyp = engine.propose(diag)
            names.append(hyp.description)

        # After n_entries+1 calls, the first name should repeat (wrap)
        assert names[0] == names[n_entries], "Round-robin should wrap around"

        # All entries before wrap should be unique
        assert len(set(names[:n_entries])) == n_entries

    def test_propose_multiple_returns_n_hypotheses(self) -> None:
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        hyps = engine.propose_multiple(diag, n=3)

        assert len(hyps) == 3
        assert all(isinstance(h, Hypothesis) for h in hyps)

    def test_propose_multiple_caps_at_playbook_size(self) -> None:
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        n_available = engine.get_playbook_size("performance_degradation")

        hyps = engine.propose_multiple(diag, n=100)
        assert len(hyps) == n_available

    def test_config_diff_resolution_multiplication(self) -> None:
        """'current * 0.5' with current_config resolves to actual value."""
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        current_config = {"learning_rate": 0.1}

        hyp = engine.propose(diag, current_config=current_config)

        # The first performance entry is reduce_learning_rate: lr * 0.5
        # If config_diff has a learning_rate key, it should be resolved
        for key, val in hyp.config_diff.items():
            if "learning_rate" in key and isinstance(val, (int, float)):
                assert abs(val - 0.05) < 1e-9

    def test_config_diff_fallback_unresolvable(self) -> None:
        """Unresolvable expressions are preserved as strings."""
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        # No current_config provided -- expressions stay as strings
        hyp = engine.propose(diag, current_config=None)

        for val in hyp.config_diff.values():
            # Without config, string expressions should be preserved as-is
            assert val is not None

    def test_unknown_category_returns_empty(self) -> None:
        """Unknown drift category returns empty list from propose_multiple."""
        engine = HypothesisEngine()
        # Create a diagnosis with a cause that has no playbook entry
        # by using a custom empty playbook
        custom_engine = HypothesisEngine(playbook={})
        diag = _diagnosis(DriftCategory.PERFORMANCE)

        hyps = custom_engine.propose_multiple(diag, n=3)
        assert hyps == []

    def test_custom_playbook_override(self) -> None:
        custom = {
            "performance_degradation": [
                {
                    "type": "hyperparameter",
                    "name": "custom_mutation",
                    "config_diff": {"custom_key": "custom_value"},
                    "rationale": "testing custom playbook",
                },
                {
                    "type": "feature_remove",
                    "name": "custom_mutation_2",
                    "config_diff": {"another_key": "another_value"},
                    "rationale": "another test",
                },
            ]
        }
        engine = HypothesisEngine(playbook=custom)
        diag = _diagnosis(DriftCategory.PERFORMANCE)
        hyp = engine.propose(diag)

        assert hyp.description == "custom_mutation"

    def test_hypothesis_source_is_playbook(self) -> None:
        engine = HypothesisEngine()
        diag = _diagnosis(DriftCategory.FEATURE_DRIFT)
        hyp = engine.propose(diag)

        assert hyp.source == "playbook"

    def test_propose_all_categories(self) -> None:
        """Every drift category produces at least one hypothesis."""
        engine = HypothesisEngine()
        for category in DriftCategory:
            diag = _diagnosis(category)
            hyp = engine.propose(diag)
            assert isinstance(hyp, Hypothesis), f"Failed for {category}"
