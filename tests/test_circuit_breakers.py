"""Tests for circuit breaker state machine and manager."""

import pytest

from hydra.risk.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerState,
)


class TestCircuitBreakerState:
    """Tests for individual CircuitBreaker state machine."""

    def test_initial_state_is_active(self):
        """New breaker starts in ACTIVE state."""
        breaker = CircuitBreaker(name="test", threshold=-0.02)
        assert breaker.state == CircuitBreakerState.ACTIVE

    def test_daily_loss_triggers_breaker(self):
        """Loss exceeding daily_loss threshold triggers breaker."""
        breaker = CircuitBreaker(name="daily_loss", threshold=-0.02)
        # Value of -0.03 is below threshold of -0.02 -> triggers
        result = breaker.check(-0.03)
        assert result is False  # Not OK to trade
        assert breaker.state == CircuitBreakerState.TRIGGERED

    def test_value_within_threshold_allows_trade(self):
        """Value within threshold allows trading."""
        breaker = CircuitBreaker(name="daily_loss", threshold=-0.02)
        result = breaker.check(-0.01)  # -0.01 > -0.02 -> OK
        assert result is True
        assert breaker.state == CircuitBreakerState.ACTIVE

    def test_drawdown_triggers_breaker(self):
        """Drawdown exceeding threshold triggers breaker."""
        breaker = CircuitBreaker(name="drawdown", threshold=-0.05)
        result = breaker.check(-0.06)
        assert result is False
        assert breaker.state == CircuitBreakerState.TRIGGERED

    def test_position_size_triggers_breaker(self):
        """Position exceeding max triggers breaker.

        For position size, the threshold is a positive max (e.g., 0.10).
        Values ABOVE the threshold trigger the breaker.
        """
        breaker = CircuitBreaker(
            name="position_size", threshold=0.10, upper_bound=True
        )
        result = breaker.check(0.15)
        assert result is False
        assert breaker.state == CircuitBreakerState.TRIGGERED

    def test_single_trade_loss_triggers_breaker(self):
        """Single trade loss exceeding threshold triggers."""
        breaker = CircuitBreaker(name="single_trade_loss", threshold=-0.01)
        result = breaker.check(-0.015)
        assert result is False
        assert breaker.state == CircuitBreakerState.TRIGGERED

    def test_cooldown_resets_to_active(self):
        """After cooldown period, breaker transitions back to ACTIVE."""
        breaker = CircuitBreaker(
            name="test", threshold=-0.02, cooldown_periods=2
        )
        # Trigger it
        breaker.check(-0.03)
        assert breaker.state == CircuitBreakerState.TRIGGERED

        # First advance: TRIGGERED -> COOLDOWN
        breaker.update()
        assert breaker.state == CircuitBreakerState.COOLDOWN

        # Second advance: still COOLDOWN (1 of 2 periods)
        breaker.update()
        assert breaker.state == CircuitBreakerState.COOLDOWN

        # Third advance: COOLDOWN -> ACTIVE (2 periods elapsed)
        breaker.update()
        assert breaker.state == CircuitBreakerState.ACTIVE

    def test_single_period_cooldown(self):
        """Default cooldown_periods=1: TRIGGERED -> COOLDOWN -> ACTIVE."""
        breaker = CircuitBreaker(name="test", threshold=-0.02, cooldown_periods=1)
        breaker.check(-0.03)
        assert breaker.state == CircuitBreakerState.TRIGGERED

        # TRIGGERED -> COOLDOWN
        breaker.update()
        assert breaker.state == CircuitBreakerState.COOLDOWN

        # COOLDOWN -> ACTIVE (1 period elapsed)
        breaker.update()
        assert breaker.state == CircuitBreakerState.ACTIVE


class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager which manages 4 independent breakers."""

    @pytest.fixture
    def default_config(self):
        return {
            "max_daily_loss": -0.02,
            "max_drawdown": -0.05,
            "max_position_size": 0.10,
            "max_single_trade_loss": -0.01,
        }

    def test_manager_allows_trade_when_all_clear(self, default_config):
        """All breakers ACTIVE -> trade allowed."""
        manager = CircuitBreakerManager(default_config)
        allowed, triggered = manager.check_trade(
            daily_pnl=-0.005,
            peak_equity=100_000,
            current_equity=98_000,
            position_value=0.05,
            trade_loss=-0.003,
        )
        assert allowed is True
        assert triggered == []

    def test_manager_checks_all_breakers(self, default_config):
        """CircuitBreakerManager denies trade if ANY breaker is triggered."""
        manager = CircuitBreakerManager(default_config)
        allowed, triggered = manager.check_trade(
            daily_pnl=-0.03,  # exceeds max_daily_loss=-0.02
            peak_equity=100_000,
            current_equity=98_000,
            position_value=0.05,
            trade_loss=-0.003,
        )
        assert allowed is False
        assert "max_daily_loss" in triggered

    def test_manager_multiple_breakers_triggered(self, default_config):
        """Multiple breakers can trigger simultaneously."""
        manager = CircuitBreakerManager(default_config)
        allowed, triggered = manager.check_trade(
            daily_pnl=-0.03,     # exceeds max_daily_loss=-0.02
            peak_equity=100_000,
            current_equity=93_000,  # drawdown = -0.07, exceeds -0.05
            position_value=0.15,  # exceeds max_position_size=0.10
            trade_loss=-0.02,     # exceeds max_single_trade_loss=-0.01
        )
        assert allowed is False
        assert len(triggered) >= 2

    def test_manager_advance_period(self, default_config):
        """advance_period moves all breakers forward in state machine."""
        manager = CircuitBreakerManager(default_config)
        # Trigger daily loss
        manager.check_trade(
            daily_pnl=-0.03,
            peak_equity=100_000,
            current_equity=98_000,
            position_value=0.05,
            trade_loss=-0.003,
        )

        # Advance through cooldown
        manager.advance_period()
        manager.advance_period()

        # After cooldown, should allow trades again
        allowed, triggered = manager.check_trade(
            daily_pnl=-0.005,
            peak_equity=100_000,
            current_equity=98_000,
            position_value=0.05,
            trade_loss=-0.003,
        )
        assert allowed is True
        assert triggered == []
