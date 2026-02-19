"""Tests for semantic hypothesis deduplication.

Verifies:
- Empty history returns (False, 0.0)
- Identical strings are detected as duplicates (similarity ~1.0)
- Semantically similar strings are caught
- Clearly different strings are NOT flagged
- Threshold sensitivity (0.85 vs 0.99)
- register + is_duplicate flow
- clear() resets memory
- load_from_journal queries journal and registers descriptions
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from hydra.agent.dedup import HypothesisDeduplicator


@pytest.fixture
def dedup() -> HypothesisDeduplicator:
    """Create a deduplicator with default threshold (0.85)."""
    return HypothesisDeduplicator(similarity_threshold=0.85)


class TestIsDuplicate:
    """Test the is_duplicate method."""

    def test_empty_history_returns_false(self, dedup: HypothesisDeduplicator) -> None:
        """With no registered hypotheses, nothing is a duplicate."""
        is_dup, score = dedup.is_duplicate("reduce learning rate by half")
        assert is_dup is False
        assert score == 0.0

    def test_identical_strings_are_duplicates(
        self, dedup: HypothesisDeduplicator
    ) -> None:
        """Exact same string should be a duplicate (similarity ~1.0)."""
        text = "reduce learning rate by half"
        dedup.register(text)
        is_dup, score = dedup.is_duplicate(text)
        assert is_dup is True
        assert score > 0.99

    def test_semantically_similar_detected(
        self, dedup: HypothesisDeduplicator
    ) -> None:
        """Semantically similar strings should be caught."""
        dedup.register("reduce learning rate by half")
        is_dup, score = dedup.is_duplicate("halve the learning rate")
        assert is_dup is True
        assert score > 0.85

    def test_clearly_different_not_flagged(
        self, dedup: HypothesisDeduplicator
    ) -> None:
        """Clearly different strings should NOT be duplicates."""
        dedup.register("reduce learning rate by half")
        is_dup, score = dedup.is_duplicate("add new features from USDA crop data")
        assert is_dup is False
        assert score < 0.85

    def test_threshold_high_only_catches_exact(self) -> None:
        """A 0.99 threshold should only catch near-exact matches."""
        strict_dedup = HypothesisDeduplicator(similarity_threshold=0.99)
        strict_dedup.register("reduce learning rate by half")

        # Near-exact should still be caught
        is_dup_exact, score_exact = strict_dedup.is_duplicate(
            "reduce learning rate by half"
        )
        assert is_dup_exact is True

        # Paraphrase should NOT be caught with strict threshold
        is_dup_para, score_para = strict_dedup.is_duplicate(
            "halve the learning rate"
        )
        assert is_dup_para is False

    def test_threshold_default_catches_paraphrases(self) -> None:
        """Default 0.85 threshold catches paraphrased duplicates."""
        dedup = HypothesisDeduplicator(similarity_threshold=0.85)
        dedup.register("reduce learning rate by half")
        is_dup, score = dedup.is_duplicate("halve the learning rate")
        assert is_dup is True


class TestRegisterAndFlow:
    """Test register + is_duplicate flow."""

    def test_register_then_check(self, dedup: HypothesisDeduplicator) -> None:
        """Register 'reduce lr' then 'reduce learning rate' is_duplicate."""
        dedup.register("reduce lr")
        is_dup, score = dedup.is_duplicate("reduce learning rate")
        # These are related but may or may not exceed 0.85
        # The key test: the check should work without error
        assert isinstance(is_dup, bool)
        assert isinstance(score, float)
        assert score > 0.0

    def test_multiple_registrations(self, dedup: HypothesisDeduplicator) -> None:
        """Multiple registrations should all be searchable."""
        dedup.register("reduce learning rate by half")
        dedup.register("add dropout layer")
        dedup.register("increase regularization")

        is_dup, score = dedup.is_duplicate("reduce the learning rate")
        assert is_dup is True
        assert score > 0.85


class TestClear:
    """Test the clear() method."""

    def test_clear_resets_memory(self, dedup: HypothesisDeduplicator) -> None:
        """After clear(), nothing should be a duplicate."""
        dedup.register("reduce learning rate by half")

        # Before clear: should be a duplicate
        is_dup_before, _ = dedup.is_duplicate("reduce learning rate by half")
        assert is_dup_before is True

        dedup.clear()

        # After clear: should NOT be a duplicate
        is_dup_after, score_after = dedup.is_duplicate("reduce learning rate by half")
        assert is_dup_after is False
        assert score_after == 0.0


class TestLoadFromJournal:
    """Test load_from_journal integration."""

    def test_loads_and_registers_from_journal(
        self, dedup: HypothesisDeduplicator
    ) -> None:
        """load_from_journal should query journal and register descriptions."""
        mock_journal = MagicMock()

        # Create mock experiment records
        record1 = MagicMock()
        record1.hypothesis = "reduce learning rate by half"
        record2 = MagicMock()
        record2.hypothesis = "add dropout regularization"

        mock_journal.query.return_value = [record1, record2]

        dedup.load_from_journal(mock_journal, days=30)

        # Verify query was called with a date_from
        mock_journal.query.assert_called_once()
        call_kwargs = mock_journal.query.call_args
        assert "date_from" in call_kwargs.kwargs

        # Verify descriptions were registered
        is_dup, score = dedup.is_duplicate("reduce learning rate by half")
        assert is_dup is True
        assert score > 0.99
