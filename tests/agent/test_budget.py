"""Tests for mutation budgets and cooldown timers.

Verifies:
- Fresh budget allows experiments
- Total budget limit enforcement
- Per-category limit enforcement
- Cooldown after rejection
- No cooldown on promotion
- reset_cycle resets counts but NOT cooldowns
- Custom BudgetConfig values
- load_cooldowns_from_journal reconstruction
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from hydra.agent.budget import BudgetConfig, MutationBudget


@pytest.fixture
def budget() -> MutationBudget:
    """Create a MutationBudget with default config."""
    return MutationBudget()


@pytest.fixture
def custom_budget() -> MutationBudget:
    """Create a MutationBudget with custom config."""
    config = BudgetConfig(
        max_experiments_per_cycle=3,
        max_per_category={"hyperparameter": 1, "feature_add": 1},
        cooldown_days={"hyperparameter": 2, "feature_add": 5},
    )
    return MutationBudget(config=config)


class TestCanRun:
    """Test the can_run method."""

    def test_fresh_budget_allows(self, budget: MutationBudget) -> None:
        """A fresh budget should allow experiments."""
        allowed, reason = budget.can_run("hyperparameter")
        assert allowed is True

    def test_total_budget_exceeded(self, budget: MutationBudget) -> None:
        """After max_experiments_per_cycle, all types are blocked."""
        for _ in range(budget.config.max_experiments_per_cycle):
            budget.record_experiment("hyperparameter", promoted=True)

        allowed, reason = budget.can_run("hyperparameter")
        assert allowed is False
        assert "budget" in reason.lower() or "limit" in reason.lower() or "cycle" in reason.lower()

    def test_per_category_limit(self, budget: MutationBudget) -> None:
        """Per-category limit blocks after N experiments of that type."""
        hp_limit = budget.config.max_per_category.get("hyperparameter", 3)

        for _ in range(hp_limit):
            budget.record_experiment("hyperparameter", promoted=True)

        allowed, reason = budget.can_run("hyperparameter")
        assert allowed is False

        # But a different category should still be allowed
        allowed2, _ = budget.can_run("feature_add")
        assert allowed2 is True


class TestCooldown:
    """Test cooldown timer behavior."""

    def test_cooldown_after_rejection(self, budget: MutationBudget) -> None:
        """After a rejection, that category should be on cooldown."""
        budget.record_experiment("hyperparameter", promoted=False)

        allowed, reason = budget.can_run("hyperparameter")
        assert allowed is False
        assert "cooldown" in reason.lower()

    def test_no_cooldown_on_promotion(self, budget: MutationBudget) -> None:
        """Promoted experiments should NOT trigger cooldown."""
        budget.record_experiment("hyperparameter", promoted=True)

        # Should still be allowed (assuming per-category limit not hit)
        allowed, _ = budget.can_run("hyperparameter")
        assert allowed is True

    def test_cooldown_different_category_unaffected(
        self, budget: MutationBudget
    ) -> None:
        """Cooldown on one category should not affect another."""
        budget.record_experiment("hyperparameter", promoted=False)

        # hyperparameter should be blocked
        allowed_hp, _ = budget.can_run("hyperparameter")
        assert allowed_hp is False

        # feature_add should be fine
        allowed_fa, _ = budget.can_run("feature_add")
        assert allowed_fa is True


class TestResetCycle:
    """Test reset_cycle behavior."""

    def test_reset_resets_counts(self, budget: MutationBudget) -> None:
        """reset_cycle should reset per-cycle counts."""
        budget.record_experiment("hyperparameter", promoted=True)
        budget.record_experiment("hyperparameter", promoted=True)

        budget.reset_cycle()

        # Should be allowed again
        allowed, _ = budget.can_run("hyperparameter")
        assert allowed is True

    def test_reset_preserves_cooldowns(self, budget: MutationBudget) -> None:
        """reset_cycle should NOT reset cooldown timers."""
        budget.record_experiment("hyperparameter", promoted=False)

        budget.reset_cycle()

        # Cooldown should still be active
        allowed, reason = budget.can_run("hyperparameter")
        assert allowed is False
        assert "cooldown" in reason.lower()


class TestCustomConfig:
    """Test custom BudgetConfig values."""

    def test_custom_max_experiments(self, custom_budget: MutationBudget) -> None:
        """Custom max_experiments_per_cycle should be respected."""
        assert custom_budget.config.max_experiments_per_cycle == 3

        # Use up the budget
        for _ in range(3):
            custom_budget.record_experiment("hyperparameter", promoted=True)

        allowed, _ = custom_budget.can_run("feature_add")
        assert allowed is False

    def test_custom_per_category(self, custom_budget: MutationBudget) -> None:
        """Custom per-category limits should be respected."""
        custom_budget.record_experiment("hyperparameter", promoted=True)

        # hp limit is 1 -- should be blocked now
        allowed, _ = custom_budget.can_run("hyperparameter")
        assert allowed is False

    def test_default_config_values(self) -> None:
        """Default BudgetConfig should have expected values."""
        config = BudgetConfig()
        assert config.max_experiments_per_cycle == 5
        assert config.max_per_category["hyperparameter"] == 3
        assert config.cooldown_days["hyperparameter"] == 3
        assert config.cooldown_days["feature_add"] == 7


class TestLoadCooldownsFromJournal:
    """Test load_cooldowns_from_journal reconstruction."""

    def test_reconstructs_cooldowns(self) -> None:
        """Should reconstruct cooldown state from rejected experiments."""
        budget = MutationBudget()
        mock_journal = MagicMock()

        now = datetime.now(timezone.utc)

        # Create mock rejected experiment within cooldown window
        record_rejected = MagicMock()
        record_rejected.mutation_type = "hyperparameter"
        record_rejected.promotion_decision = "rejected"
        record_rejected.created_at = (now - timedelta(days=1)).isoformat()

        # Create mock promoted experiment (should not trigger cooldown)
        record_promoted = MagicMock()
        record_promoted.mutation_type = "feature_add"
        record_promoted.promotion_decision = "promoted"
        record_promoted.created_at = (now - timedelta(days=1)).isoformat()

        mock_journal.query.return_value = [record_rejected, record_promoted]

        budget.load_cooldowns_from_journal(mock_journal)

        # hyperparameter should be on cooldown (rejected within window)
        allowed_hp, reason = budget.can_run("hyperparameter")
        assert allowed_hp is False
        assert "cooldown" in reason.lower()

        # feature_add should be allowed (was promoted, not rejected)
        allowed_fa, _ = budget.can_run("feature_add")
        assert allowed_fa is True

    def test_expired_cooldowns_not_reconstructed(self) -> None:
        """Rejected experiments outside cooldown window should NOT block."""
        budget = MutationBudget()
        mock_journal = MagicMock()

        now = datetime.now(timezone.utc)

        # Create rejected experiment OUTSIDE cooldown window (hp cooldown = 3 days)
        old_record = MagicMock()
        old_record.mutation_type = "hyperparameter"
        old_record.promotion_decision = "rejected"
        old_record.created_at = (now - timedelta(days=10)).isoformat()

        mock_journal.query.return_value = [old_record]

        budget.load_cooldowns_from_journal(mock_journal)

        # Should be allowed (cooldown expired)
        allowed, _ = budget.can_run("hyperparameter")
        assert allowed is True
