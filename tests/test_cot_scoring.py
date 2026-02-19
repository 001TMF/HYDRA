"""TDD tests for COT sentiment scoring.

Tests cover the compute_cot_sentiment function and SentimentScore dataclass,
validating percentile-rank-based sentiment scoring with confidence weighting
derived from open interest magnitude and positioning concentration.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra.signals.sentiment.cot_scoring import SentimentScore, compute_cot_sentiment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(n: int = 52, low: float = -5000.0, high: float = 5000.0) -> np.ndarray:
    """Generate a linearly spaced history array (simulates 52-week positioning)."""
    return np.linspace(low, high, n)


def _make_oi_history(n: int = 52, low: float = 50000.0, high: float = 150000.0) -> np.ndarray:
    """Generate a linearly spaced OI history array."""
    return np.linspace(low, high, n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCOTSentimentScoring:
    """Tests for compute_cot_sentiment."""

    def test_neutral_on_insufficient_history(self):
        """< 4 weeks of history should return neutral score with zero confidence."""
        short_history = np.array([100.0, 200.0, 300.0])  # only 3 weeks
        short_oi = np.array([50000.0, 60000.0, 70000.0])

        result = compute_cot_sentiment(
            managed_money_net=200.0,
            producer_net=-100.0,
            total_oi=60000.0,
            history_managed=short_history,
            history_oi=short_oi,
        )

        assert isinstance(result, SentimentScore)
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.components == {}

    def test_extreme_bullish(self):
        """managed_money_net at max of 52-week history -> score near +1.0."""
        history = _make_history(52, low=-5000.0, high=5000.0)
        oi_history = _make_oi_history(52)

        result = compute_cot_sentiment(
            managed_money_net=5000.0,  # at the maximum
            producer_net=-3000.0,
            total_oi=150000.0,
            history_managed=history,
            history_oi=oi_history,
        )

        assert result.score >= 0.9, f"Expected score >= 0.9 for extreme bullish, got {result.score}"
        assert result.score <= 1.0

    def test_extreme_bearish(self):
        """managed_money_net at min of 52-week history -> score near -1.0."""
        history = _make_history(52, low=-5000.0, high=5000.0)
        oi_history = _make_oi_history(52)

        result = compute_cot_sentiment(
            managed_money_net=-5000.0,  # at the minimum
            producer_net=3000.0,
            total_oi=50000.0,
            history_managed=history,
            history_oi=oi_history,
        )

        assert result.score <= -0.9, f"Expected score <= -0.9 for extreme bearish, got {result.score}"
        assert result.score >= -1.0

    def test_median_positioning(self):
        """managed_money_net at median of history -> score near 0.0."""
        history = _make_history(52, low=-5000.0, high=5000.0)
        oi_history = _make_oi_history(52)

        result = compute_cot_sentiment(
            managed_money_net=0.0,  # at the median
            producer_net=0.0,
            total_oi=100000.0,
            history_managed=history,
            history_oi=oi_history,
        )

        assert abs(result.score) < 0.15, f"Expected score near 0.0 for median, got {result.score}"

    def test_confidence_scales_with_oi(self):
        """Higher total_oi relative to history produces higher confidence."""
        history = _make_history(52)
        oi_history = _make_oi_history(52, low=50000.0, high=150000.0)

        # Low OI scenario
        result_low = compute_cot_sentiment(
            managed_money_net=0.0,
            producer_net=0.0,
            total_oi=50000.0,  # at the minimum of OI history
            history_managed=history,
            history_oi=oi_history,
        )

        # High OI scenario
        result_high = compute_cot_sentiment(
            managed_money_net=0.0,
            producer_net=0.0,
            total_oi=150000.0,  # at the maximum of OI history
            history_managed=history,
            history_oi=oi_history,
        )

        assert result_high.confidence > result_low.confidence, (
            f"High OI confidence ({result_high.confidence}) should exceed "
            f"low OI confidence ({result_low.confidence})"
        )

    def test_confidence_scales_with_concentration(self):
        """Higher abs(managed_money_net)/total_oi increases confidence."""
        history = _make_history(52)
        oi_history = _make_oi_history(52)

        # Low concentration: small net position relative to OI
        result_low = compute_cot_sentiment(
            managed_money_net=100.0,
            producer_net=-50.0,
            total_oi=150000.0,  # large OI, small net => low concentration
            history_managed=history,
            history_oi=oi_history,
        )

        # High concentration: large net position relative to OI
        result_high = compute_cot_sentiment(
            managed_money_net=4000.0,
            producer_net=-2000.0,
            total_oi=10000.0,  # small OI, large net => high concentration
            history_managed=history,
            history_oi=oi_history,
        )

        assert result_high.confidence > result_low.confidence, (
            f"High concentration confidence ({result_high.confidence}) should exceed "
            f"low concentration confidence ({result_low.confidence})"
        )

    def test_score_clamped_to_bounds(self):
        """Score always in [-1, +1], confidence always in [0, 1]."""
        history = _make_history(52)
        oi_history = _make_oi_history(52)

        # Test with values well beyond history range
        for managed_net in [-100000.0, -5000.0, 0.0, 5000.0, 100000.0]:
            for total_oi in [1.0, 100000.0, 10000000.0]:
                result = compute_cot_sentiment(
                    managed_money_net=managed_net,
                    producer_net=0.0,
                    total_oi=total_oi,
                    history_managed=history,
                    history_oi=oi_history,
                )

                assert -1.0 <= result.score <= 1.0, (
                    f"Score {result.score} out of bounds for net={managed_net}, oi={total_oi}"
                )
                assert 0.0 <= result.confidence <= 1.0, (
                    f"Confidence {result.confidence} out of bounds for net={managed_net}, oi={total_oi}"
                )

    def test_components_populated(self):
        """Components dict contains managed_money_pct_rank, oi_rank, concentration."""
        history = _make_history(52)
        oi_history = _make_oi_history(52)

        result = compute_cot_sentiment(
            managed_money_net=2000.0,
            producer_net=-1000.0,
            total_oi=100000.0,
            history_managed=history,
            history_oi=oi_history,
        )

        assert "managed_money_pct_rank" in result.components, "Missing managed_money_pct_rank"
        assert "oi_rank" in result.components, "Missing oi_rank"
        assert "concentration" in result.components, "Missing concentration"

        # Validate component types
        assert isinstance(result.components["managed_money_pct_rank"], float)
        assert isinstance(result.components["oi_rank"], float)
        assert isinstance(result.components["concentration"], float)

        # Validate component ranges
        assert 0.0 <= result.components["managed_money_pct_rank"] <= 1.0
        assert 0.0 <= result.components["oi_rank"] <= 1.0
        assert result.components["concentration"] >= 0.0
