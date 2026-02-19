"""3-of-5 window promotion evaluation for the HYDRA agent loop (AGNT-08).

Evaluates whether a candidate model should be promoted to champion by
comparing their fitness scores across multiple independent evaluation
windows. The candidate must win a majority (default 3 of 5) of windows,
where winning means strictly beating the champion's score (plus an
optional minimum improvement margin).

This prevents lucky single-window flukes from causing promotion and
ensures genuine outperformance across diverse market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromotionConfig:
    """Configuration for the promotion evaluator."""

    n_windows: int = 5
    """Number of independent evaluation windows."""

    required_wins: int = 3
    """Number of windows the candidate must win for promotion."""

    min_improvement: float = 0.0
    """Minimum improvement margin: candidate must beat champion by this amount.
    Default 0.0 means strictly greater (any positive difference counts)."""


@dataclass
class PromotionResult:
    """Result of a promotion evaluation with full audit trail."""

    promoted: bool
    """Whether the candidate should be promoted."""

    windows_won: int
    """Number of evaluation windows the candidate won."""

    window_scores: list[tuple[float, float]]
    """Per-window (candidate_fitness, champion_fitness) pairs."""

    reason: str
    """Human-readable explanation of the decision."""


class PromotionEvaluator:
    """Evaluates candidate vs champion across independent windows.

    Usage::

        evaluator = PromotionEvaluator()
        result = evaluator.evaluate(candidate_scores, champion_scores)
        if result.promoted:
            # promote candidate to champion
            ...
    """

    def __init__(self, config: PromotionConfig | None = None) -> None:
        self._config = config or PromotionConfig()

    def evaluate(
        self,
        candidate_scores: list[float],
        champion_scores: list[float],
    ) -> PromotionResult:
        """Compare candidate and champion fitness across evaluation windows.

        Args:
            candidate_scores: Fitness scores for the candidate on each window.
            champion_scores: Fitness scores for the champion on each window.

        Returns:
            PromotionResult with promotion decision, win count, and audit trail.

        Raises:
            ValueError: If score lists have different lengths or don't match
                n_windows configuration.
        """
        n = self._config.n_windows

        if len(candidate_scores) != len(champion_scores):
            msg = (
                f"candidate_scores length ({len(candidate_scores)}) != "
                f"champion_scores length ({len(champion_scores)})"
            )
            raise ValueError(msg)

        if len(candidate_scores) != n:
            msg = (
                f"Expected {n} window scores, "
                f"got {len(candidate_scores)}"
            )
            raise ValueError(msg)

        window_scores: list[tuple[float, float]] = []
        wins = 0

        for c_score, ch_score in zip(candidate_scores, champion_scores):
            window_scores.append((c_score, ch_score))
            if c_score > ch_score + self._config.min_improvement:
                wins += 1

        promoted = wins >= self._config.required_wins
        reason = (
            f"Candidate won {wins} of {n} windows "
            f"(required: {self._config.required_wins})"
        )

        return PromotionResult(
            promoted=promoted,
            windows_won=wins,
            window_scores=window_scores,
            reason=reason,
        )
