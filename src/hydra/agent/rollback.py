"""Hysteresis-based rollback trigger for the HYDRA agent loop (AGNT-07).

Detects sustained model degradation and triggers rollback to the previous
champion model. Uses hysteresis (cooldown + re-arming) to prevent flapping
between rollback and re-promotion.

The trigger fires only after *sustained_periods* consecutive degraded checks,
where degradation is defined as fitness dropping more than *degradation_threshold*
below the champion's fitness. After triggering, a cooldown period prevents
immediate re-triggering, and the trigger must observe *recovery_periods*
consecutive healthy checks before re-arming.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RollbackConfig:
    """Configuration for the hysteresis rollback trigger."""

    degradation_threshold: float = 0.15
    """Minimum relative fitness drop to count as degraded (15% default)."""

    sustained_periods: int = 3
    """Consecutive degraded periods required before triggering rollback."""

    recovery_periods: int = 5
    """Consecutive healthy periods required to re-arm after cooldown expires."""

    cooldown_after_rollback: int = 10
    """Periods after rollback during which no re-trigger is possible."""


class HysteresisRollbackTrigger:
    """Stateful rollback trigger with hysteresis to prevent flapping.

    Usage::

        trigger = HysteresisRollbackTrigger()
        if trigger.update(current_fitness, champion_fitness):
            # execute rollback
            ...
    """

    def __init__(self, config: RollbackConfig | None = None) -> None:
        self._config = config or RollbackConfig()
        self._armed: bool = True
        self._degraded_count: int = 0
        self._healthy_count: int = 0
        self._cooldown: int = 0

    # -- Public API ----------------------------------------------------------

    def update(self, current_fitness: float, champion_fitness: float) -> bool:
        """Process one evaluation period and return True if rollback should fire.

        Args:
            current_fitness: Fitness of the currently deployed model.
            champion_fitness: Fitness of the champion model (baseline).

        Returns:
            True if rollback should be executed this period.
        """
        # During cooldown: tick down and return False
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        # Determine if this period is degraded
        # Degradation means the drop strictly exceeds the threshold
        if champion_fitness == 0:
            degraded = False
        else:
            drop = (champion_fitness - current_fitness) / champion_fitness
            degraded = drop > self._config.degradation_threshold

        if degraded:
            self._degraded_count += 1
            self._healthy_count = 0
        else:
            self._healthy_count += 1
            self._degraded_count = 0

        # Check for rollback trigger (only when armed)
        if self._armed and self._degraded_count >= self._config.sustained_periods:
            self._armed = False
            self._cooldown = self._config.cooldown_after_rollback
            self._degraded_count = 0
            self._healthy_count = 0
            return True

        # Check for re-arming (only when disarmed and cooldown expired)
        if not self._armed and self._healthy_count >= self._config.recovery_periods:
            self._armed = True
            self._healthy_count = 0

        return False

    def reset(self) -> None:
        """Reset all state to initial values."""
        self._armed = True
        self._degraded_count = 0
        self._healthy_count = 0
        self._cooldown = 0

    # -- Properties ----------------------------------------------------------

    @property
    def is_armed(self) -> bool:
        """Whether the trigger is armed and will fire on sustained degradation."""
        return self._armed

    @property
    def cooldown_remaining(self) -> int:
        """Number of periods remaining in cooldown."""
        return self._cooldown

    @property
    def consecutive_degraded(self) -> int:
        """Number of consecutive degraded periods observed."""
        return self._degraded_count
