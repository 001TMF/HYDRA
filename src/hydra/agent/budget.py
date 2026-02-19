"""Mutation budgets and cooldown timers (AGNT-09 layers 2 and 3).

Provides two mechanisms to prevent degenerate experiment loops:

1. **Per-cycle budgets**: Limit total experiments per agent cycle and
   per mutation category. Prevents resource exhaustion.

2. **Cooldown timers**: After a rejected experiment, the mutation category
   is blocked for a configurable number of days. Prevents retrying
   the same approach repeatedly after failure.

Cooldowns persist across cycles (``reset_cycle()`` only resets counts).
On startup, ``load_cooldowns_from_journal()`` reconstructs cooldown state
from the persistent experiment journal.

Exports:
    - ``BudgetConfig``: Configuration dataclass for budget limits.
    - ``MutationBudget``: Main budget enforcement class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import structlog

logger = structlog.get_logger()

# Default per-category limits per cycle
_DEFAULT_MAX_PER_CATEGORY: dict[str, int] = {
    "hyperparameter": 3,
    "feature_add": 2,
    "feature_remove": 2,
    "feature_engineering": 2,
    "ensemble_method": 1,
    "prediction_target": 1,
}

# Default cooldown days after rejection
_DEFAULT_COOLDOWN_DAYS: dict[str, int] = {
    "hyperparameter": 3,
    "feature_add": 7,
    "feature_remove": 7,
    "ensemble_method": 14,
}


@dataclass
class BudgetConfig:
    """Configuration for mutation budgets and cooldowns.

    Attributes
    ----------
    max_experiments_per_cycle : int
        Total experiments allowed per agent cycle. Default 5.
    max_per_category : dict[str, int]
        Per-category limits per cycle.
    cooldown_days : dict[str, int]
        Minimum days between retrying a mutation type after rejection.
    """

    max_experiments_per_cycle: int = 5
    max_per_category: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_MAX_PER_CATEGORY)
    )
    cooldown_days: dict[str, int] = field(
        default_factory=lambda: dict(_DEFAULT_COOLDOWN_DAYS)
    )


class MutationBudget:
    """Enforce per-cycle mutation budgets and cooldown timers.

    Parameters
    ----------
    config : BudgetConfig | None
        Budget configuration. Uses defaults if None.
    """

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()
        self._cycle_counts: dict[str, int] = {}
        self._total_this_cycle: int = 0
        self._cooldown_until: dict[str, datetime] = {}

    def can_run(self, mutation_type: str) -> tuple[bool, str]:
        """Check if budget and cooldown allow this mutation type.

        Checks are evaluated in order:
        1. Total experiments this cycle < max_experiments_per_cycle
        2. Category count < max_per_category
        3. No active cooldown for this category

        Parameters
        ----------
        mutation_type : str
            The mutation category to check.

        Returns
        -------
        tuple[bool, str]
            ``(allowed, reason)``. If allowed, reason is "ok".
        """
        # Check 1: Total cycle budget
        if self._total_this_cycle >= self.config.max_experiments_per_cycle:
            return (
                False,
                f"Cycle budget exceeded: {self._total_this_cycle}/{self.config.max_experiments_per_cycle}",
            )

        # Check 2: Per-category limit
        category_limit = self.config.max_per_category.get(mutation_type)
        if category_limit is not None:
            current_count = self._cycle_counts.get(mutation_type, 0)
            if current_count >= category_limit:
                return (
                    False,
                    f"Category limit reached for {mutation_type}: {current_count}/{category_limit}",
                )

        # Check 3: Cooldown timer
        cooldown_end = self._cooldown_until.get(mutation_type)
        if cooldown_end is not None:
            now = datetime.now(timezone.utc)
            if now < cooldown_end:
                days_left = (cooldown_end - now).days + 1
                return (
                    False,
                    f"Cooldown active for {mutation_type}: {days_left} day(s) remaining",
                )

        return (True, "ok")

    def record_experiment(self, mutation_type: str, promoted: bool) -> None:
        """Record an experiment and update counters/cooldowns.

        Parameters
        ----------
        mutation_type : str
            The mutation category.
        promoted : bool
            Whether the experiment was promoted. If False, a cooldown
            timer is started for this category.
        """
        self._total_this_cycle += 1
        self._cycle_counts[mutation_type] = (
            self._cycle_counts.get(mutation_type, 0) + 1
        )

        if not promoted:
            cooldown_days = self.config.cooldown_days.get(mutation_type, 0)
            if cooldown_days > 0:
                self._cooldown_until[mutation_type] = datetime.now(
                    timezone.utc
                ) + timedelta(days=cooldown_days)
                logger.info(
                    "cooldown_started",
                    mutation_type=mutation_type,
                    cooldown_days=cooldown_days,
                )

    def reset_cycle(self) -> None:
        """Reset per-cycle counters.

        Called at the start of each agent cycle. Cooldown timers are
        NOT reset -- they persist across cycles.
        """
        self._cycle_counts.clear()
        self._total_this_cycle = 0
        logger.debug("budget_cycle_reset")

    def load_cooldowns_from_journal(self, journal: object) -> None:
        """Reconstruct cooldown state from the experiment journal.

        Scans recent rejected experiments and marks categories as on
        cooldown if the rejection occurred within the cooldown window.

        Parameters
        ----------
        journal : ExperimentJournal
            The experiment journal. Typed as ``object`` to avoid
            circular import; expects ``.query(outcome=...)`` returning
            records with ``.mutation_type``, ``.promotion_decision``,
            and ``.created_at`` attributes.
        """
        now = datetime.now(timezone.utc)

        # Query all recent rejected experiments
        rejected = journal.query(outcome="rejected")

        for record in rejected:
            # Defensive: skip non-rejected records in case query returns mixed results
            if getattr(record, "promotion_decision", None) != "rejected":
                continue

            mutation_type = record.mutation_type
            cooldown_days = self.config.cooldown_days.get(mutation_type, 0)

            if cooldown_days <= 0:
                continue

            # Parse created_at timestamp
            try:
                created = datetime.fromisoformat(record.created_at)
                # Ensure timezone-aware
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            # Check if still within cooldown window
            cooldown_end = created + timedelta(days=cooldown_days)
            if now < cooldown_end:
                # Only update if this extends the current cooldown
                existing = self._cooldown_until.get(mutation_type)
                if existing is None or cooldown_end > existing:
                    self._cooldown_until[mutation_type] = cooldown_end

        logger.info(
            "cooldowns_loaded_from_journal",
            active_cooldowns=list(self._cooldown_until.keys()),
        )
