"""Tests for 3-of-5 window promotion evaluation (AGNT-08).

Verifies that the promotion evaluator correctly requires a candidate to
beat the champion on a majority of independent evaluation windows.
"""

import pytest

from hydra.agent.promotion import PromotionConfig, PromotionEvaluator, PromotionResult


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestPromotionDefaults:
    def test_default_config(self):
        cfg = PromotionConfig()
        assert cfg.n_windows == 5
        assert cfg.required_wins == 3
        assert cfg.min_improvement == 0.0

    def test_result_structure(self):
        result = PromotionResult(
            promoted=True,
            windows_won=3,
            window_scores=[(0.8, 0.7), (0.6, 0.65), (0.9, 0.85), (0.7, 0.68), (0.5, 0.55)],
            reason="Candidate won 3 of 5 windows",
        )
        assert result.promoted is True
        assert result.windows_won == 3
        assert len(result.window_scores) == 5


# ---------------------------------------------------------------------------
# Promotion decisions
# ---------------------------------------------------------------------------

class TestPromotionDecisions:
    def test_candidate_wins_3_of_5_promoted(self):
        """Candidate winning 3 of 5 windows -> promoted."""
        evaluator = PromotionEvaluator()
        candidate = [0.80, 0.50, 0.90, 0.85, 0.40]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        # Candidate wins windows 0, 2, 3 (3 wins)
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is True
        assert result.windows_won == 3

    def test_candidate_wins_2_of_5_not_promoted(self):
        """Candidate winning only 2 of 5 windows -> not promoted."""
        evaluator = PromotionEvaluator()
        candidate = [0.80, 0.50, 0.90, 0.40, 0.40]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        # Candidate wins windows 0, 2 (2 wins)
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is False
        assert result.windows_won == 2

    def test_candidate_wins_all_5_promoted(self):
        """Candidate winning 5 of 5 windows -> promoted."""
        evaluator = PromotionEvaluator()
        candidate = [0.90, 0.80, 0.95, 0.85, 0.70]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is True
        assert result.windows_won == 5

    def test_candidate_wins_0_of_5_not_promoted(self):
        """Candidate winning 0 of 5 windows -> not promoted."""
        evaluator = PromotionEvaluator()
        candidate = [0.60, 0.50, 0.80, 0.70, 0.40]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is False
        assert result.windows_won == 0


# ---------------------------------------------------------------------------
# Tie handling
# ---------------------------------------------------------------------------

class TestTieHandling:
    def test_tied_scores_candidate_does_not_win(self):
        """Equal fitness -> candidate does NOT win that window (must strictly beat)."""
        evaluator = PromotionEvaluator()
        candidate = [0.80, 0.60, 0.90, 0.80, 0.50]
        champion = [0.80, 0.60, 0.85, 0.80, 0.50]
        # Only window 2 is a strict win (0.90 > 0.85)
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is False
        assert result.windows_won == 1

    def test_all_tied_not_promoted(self):
        """All tied -> 0 wins -> not promoted."""
        evaluator = PromotionEvaluator()
        scores = [0.70, 0.65, 0.80, 0.75, 0.60]
        result = evaluator.evaluate(scores, scores)
        assert result.promoted is False
        assert result.windows_won == 0


# ---------------------------------------------------------------------------
# Min improvement margin
# ---------------------------------------------------------------------------

class TestMinImprovement:
    def test_min_improvement_requires_margin(self):
        """With min_improvement > 0, candidate must exceed champion by margin."""
        cfg = PromotionConfig(min_improvement=0.05)
        evaluator = PromotionEvaluator(cfg)
        # Candidate beats champion by 0.01 on all windows (not enough with 0.05 margin)
        candidate = [0.71, 0.61, 0.86, 0.81, 0.51]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is False
        assert result.windows_won == 0

    def test_min_improvement_satisfied(self):
        """Candidate exceeding champion by more than margin wins the window."""
        cfg = PromotionConfig(min_improvement=0.05)
        evaluator = PromotionEvaluator(cfg)
        # Candidate beats champion by > 0.05 on windows 0, 2, 3
        candidate = [0.80, 0.50, 0.95, 0.90, 0.40]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        result = evaluator.evaluate(candidate, champion)
        assert result.promoted is True
        assert result.windows_won == 3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestPromotionValidation:
    def test_mismatched_lengths_raises_value_error(self):
        evaluator = PromotionEvaluator()
        with pytest.raises(ValueError, match="length"):
            evaluator.evaluate([0.8, 0.7, 0.9], [0.7, 0.6, 0.8, 0.7, 0.5])

    def test_wrong_number_of_windows_raises_value_error(self):
        evaluator = PromotionEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate([0.8, 0.7, 0.9], [0.7, 0.6, 0.8])

    def test_empty_lists_raises_value_error(self):
        evaluator = PromotionEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate([], [])


# ---------------------------------------------------------------------------
# Window scores audit trail
# ---------------------------------------------------------------------------

class TestWindowScoresAudit:
    def test_window_scores_captured_in_result(self):
        evaluator = PromotionEvaluator()
        candidate = [0.80, 0.50, 0.90, 0.85, 0.40]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        result = evaluator.evaluate(candidate, champion)
        assert len(result.window_scores) == 5
        assert result.window_scores[0] == (0.80, 0.70)
        assert result.window_scores[1] == (0.50, 0.60)

    def test_reason_is_descriptive(self):
        evaluator = PromotionEvaluator()
        candidate = [0.80, 0.50, 0.90, 0.85, 0.40]
        champion = [0.70, 0.60, 0.85, 0.80, 0.50]
        result = evaluator.evaluate(candidate, champion)
        assert "3" in result.reason
        assert "5" in result.reason
