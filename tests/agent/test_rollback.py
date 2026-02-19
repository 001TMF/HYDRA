"""Tests for hysteresis-based rollback trigger (AGNT-07).

Verifies that the rollback trigger fires only on sustained degradation,
respects cooldown periods, and re-arms after recovery.
"""

import pytest

from hydra.agent.rollback import HysteresisRollbackTrigger, RollbackConfig


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestRollbackDefaults:
    def test_default_config_values(self):
        cfg = RollbackConfig()
        assert cfg.degradation_threshold == 0.15
        assert cfg.sustained_periods == 3
        assert cfg.recovery_periods == 5
        assert cfg.cooldown_after_rollback == 10

    def test_trigger_starts_armed(self):
        trigger = HysteresisRollbackTrigger()
        assert trigger.is_armed is True
        assert trigger.cooldown_remaining == 0
        assert trigger.consecutive_degraded == 0


# ---------------------------------------------------------------------------
# Core rollback logic
# ---------------------------------------------------------------------------

class TestRollbackTrigger:
    def test_single_bad_period_does_not_trigger(self):
        """One degraded period is not enough (needs sustained_periods=3)."""
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        bad = champion * (1 - 0.20)  # 20% drop, exceeds 15% threshold
        result = trigger.update(bad, champion)
        assert result is False

    def test_two_bad_periods_does_not_trigger(self):
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        bad = champion * (1 - 0.20)
        trigger.update(bad, champion)
        result = trigger.update(bad, champion)
        assert result is False

    def test_three_consecutive_degraded_triggers_rollback(self):
        """Three consecutive degraded periods should trigger rollback."""
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        bad = champion * (1 - 0.20)  # 20% drop
        trigger.update(bad, champion)
        trigger.update(bad, champion)
        result = trigger.update(bad, champion)
        assert result is True

    def test_interleaved_good_bad_resets_count(self):
        """A healthy period between bad periods resets the degraded counter."""
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        bad = champion * (1 - 0.20)
        good = champion * 0.95  # 5% drop, within threshold

        trigger.update(bad, champion)
        trigger.update(bad, champion)
        trigger.update(good, champion)  # resets degraded count
        trigger.update(bad, champion)
        result = trigger.update(bad, champion)
        assert result is False  # only 2 consecutive bad after reset

    def test_exactly_at_threshold_does_not_trigger(self):
        """Exactly at the threshold (15% drop) does NOT trigger -- must exceed."""
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        at_threshold = champion * (1 - 0.15)  # exactly 15%
        for _ in range(10):
            result = trigger.update(at_threshold, champion)
        assert result is False


# ---------------------------------------------------------------------------
# Cooldown and re-arming
# ---------------------------------------------------------------------------

class TestCooldownAndRearm:
    def test_cooldown_prevents_retrigger(self):
        """After rollback, cooldown prevents re-trigger for N periods."""
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        bad = champion * (1 - 0.20)

        # Trigger rollback
        trigger.update(bad, champion)
        trigger.update(bad, champion)
        assert trigger.update(bad, champion) is True  # triggers

        # During cooldown, even bad periods don't trigger
        assert trigger.cooldown_remaining == 10
        for _ in range(10):
            result = trigger.update(bad, champion)
            assert result is False

        # After cooldown expires, still need to re-arm first
        assert trigger.cooldown_remaining == 0

    def test_rearm_after_recovery(self):
        """After cooldown, trigger re-arms after recovery_periods healthy checks."""
        cfg = RollbackConfig(
            sustained_periods=2,
            recovery_periods=3,
            cooldown_after_rollback=2,
        )
        trigger = HysteresisRollbackTrigger(cfg)
        champion = 1.0
        bad = champion * (1 - 0.20)
        good = champion * 0.95

        # Trigger rollback
        trigger.update(bad, champion)
        assert trigger.update(bad, champion) is True

        # Cooldown
        trigger.update(good, champion)
        trigger.update(good, champion)
        assert trigger.cooldown_remaining == 0
        assert trigger.is_armed is False

        # Recovery: need 3 healthy periods to re-arm
        trigger.update(good, champion)
        trigger.update(good, champion)
        assert trigger.is_armed is False
        trigger.update(good, champion)
        assert trigger.is_armed is True

    def test_second_rollback_after_rearm(self):
        """After re-arming, a new sustained degradation triggers again."""
        cfg = RollbackConfig(
            sustained_periods=2,
            recovery_periods=2,
            cooldown_after_rollback=2,
        )
        trigger = HysteresisRollbackTrigger(cfg)
        champion = 1.0
        bad = champion * (1 - 0.20)
        good = champion * 0.95

        # First rollback
        trigger.update(bad, champion)
        assert trigger.update(bad, champion) is True

        # Cooldown (2 periods)
        trigger.update(good, champion)
        trigger.update(good, champion)

        # Recovery (2 healthy periods to re-arm)
        trigger.update(good, champion)
        trigger.update(good, champion)
        assert trigger.is_armed is True

        # Second rollback
        trigger.update(bad, champion)
        result = trigger.update(bad, champion)
        assert result is True


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_sustained_periods_one_triggers_on_first_bad(self):
        """sustained_periods=1 means first bad check triggers immediately."""
        cfg = RollbackConfig(sustained_periods=1)
        trigger = HysteresisRollbackTrigger(cfg)
        champion = 1.0
        bad = champion * (1 - 0.20)
        result = trigger.update(bad, champion)
        assert result is True

    def test_custom_threshold(self):
        """Custom threshold changes what counts as degraded."""
        cfg = RollbackConfig(degradation_threshold=0.05)  # 5% threshold
        trigger = HysteresisRollbackTrigger(cfg)
        champion = 1.0
        slightly_bad = champion * (1 - 0.06)  # 6% drop exceeds 5% threshold
        for _ in range(3):
            trigger.update(slightly_bad, champion)
        # Should have triggered on 3rd (default sustained_periods)
        # Check via is_armed
        assert trigger.is_armed is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestRollbackReset:
    def test_reset_clears_all_state(self):
        """Reset returns trigger to initial state."""
        trigger = HysteresisRollbackTrigger()
        champion = 1.0
        bad = champion * (1 - 0.20)

        # Build up some state
        trigger.update(bad, champion)
        trigger.update(bad, champion)
        assert trigger.update(bad, champion) is True  # trigger
        assert trigger.is_armed is False
        assert trigger.cooldown_remaining > 0

        # Reset
        trigger.reset()
        assert trigger.is_armed is True
        assert trigger.cooldown_remaining == 0
        assert trigger.consecutive_degraded == 0
