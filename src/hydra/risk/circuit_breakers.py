"""Circuit breaker state machine for trading risk management.

Implements four independent circuit breakers:
    - max_daily_loss: Halts on excessive daily P&L loss
    - max_drawdown: Halts on excessive drawdown from peak equity
    - max_position_size: Halts on oversized positions
    - max_single_trade_loss: Halts on excessive single-trade loss

State machine: ACTIVE -> TRIGGERED -> COOLDOWN -> ACTIVE

Each breaker operates independently. The CircuitBreakerManager checks all
breakers pre-trade and returns a combined allow/deny decision.
"""

from enum import Enum


class CircuitBreakerState(Enum):
    """State of a circuit breaker in its lifecycle."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    COOLDOWN = "cooldown"


class CircuitBreaker:
    """Single circuit breaker with threshold and cooldown.

    For lower-bound breakers (losses), the breaker triggers when the
    current value falls BELOW the threshold (e.g., -0.03 < -0.02).

    For upper-bound breakers (position size), set upper_bound=True
    and the breaker triggers when the value EXCEEDS the threshold.
    """

    def __init__(
        self,
        name: str,
        threshold: float,
        cooldown_periods: int = 1,
        upper_bound: bool = False,
    ):
        self.name = name
        self.threshold = threshold
        self.cooldown_periods = cooldown_periods
        self.upper_bound = upper_bound
        self.state = CircuitBreakerState.ACTIVE
        self._cooldown_remaining = 0

    def check(self, current_value: float) -> bool:
        """Check if trading is allowed given the current value.

        Args:
            current_value: The metric to check against the threshold.

        Returns:
            True if OK to trade, False if breaker triggered/active.
        """
        if self.state != CircuitBreakerState.ACTIVE:
            return False

        if self.upper_bound:
            breached = current_value > self.threshold
        else:
            breached = current_value < self.threshold

        if breached:
            self.state = CircuitBreakerState.TRIGGERED
            self._cooldown_remaining = self.cooldown_periods
            return False

        return True

    def update(self) -> None:
        """Advance the state machine by one period.

        TRIGGERED -> COOLDOWN (begins cooldown countdown)
        COOLDOWN -> COOLDOWN (if cooldown_remaining > 0)
        COOLDOWN -> ACTIVE (when cooldown complete)
        """
        if self.state == CircuitBreakerState.TRIGGERED:
            self.state = CircuitBreakerState.COOLDOWN

        elif self.state == CircuitBreakerState.COOLDOWN:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self.state = CircuitBreakerState.ACTIVE


class CircuitBreakerManager:
    """Manages 4 independent breakers: daily_loss, drawdown, position_size, single_trade_loss.

    Default thresholds:
        - max_daily_loss: -0.02 (2% of capital)
        - max_drawdown: -0.05 (5% from peak)
        - max_position_size: 0.10 (10% of capital in one position)
        - max_single_trade_loss: -0.01 (1% of capital per trade)
    """

    DEFAULT_CONFIG = {
        "max_daily_loss": -0.02,
        "max_drawdown": -0.05,
        "max_position_size": 0.10,
        "max_single_trade_loss": -0.01,
    }

    def __init__(self, config: dict | None = None):
        """Initialize circuit breaker manager.

        Args:
            config: Dictionary with threshold values for each breaker.
                Missing keys use defaults.
        """
        cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        self.breakers = {
            "max_daily_loss": CircuitBreaker(
                name="max_daily_loss",
                threshold=cfg["max_daily_loss"],
            ),
            "max_drawdown": CircuitBreaker(
                name="max_drawdown",
                threshold=cfg["max_drawdown"],
            ),
            "max_position_size": CircuitBreaker(
                name="max_position_size",
                threshold=cfg["max_position_size"],
                upper_bound=True,
            ),
            "max_single_trade_loss": CircuitBreaker(
                name="max_single_trade_loss",
                threshold=cfg["max_single_trade_loss"],
            ),
        }

    def check_trade(
        self,
        daily_pnl: float,
        peak_equity: float,
        current_equity: float,
        position_value: float,
        trade_loss: float,
    ) -> tuple[bool, list[str]]:
        """Check all circuit breakers pre-trade.

        Args:
            daily_pnl: Today's P&L as fraction of capital (negative = loss).
            peak_equity: Peak equity value for drawdown calculation.
            current_equity: Current equity value.
            position_value: Proposed position as fraction of capital.
            trade_loss: Loss from the most recent trade as fraction of capital.

        Returns:
            Tuple of (allowed, list_of_triggered_breaker_names).
            allowed is True only if ALL breakers allow the trade.
        """
        drawdown = (current_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        checks = {
            "max_daily_loss": daily_pnl,
            "max_drawdown": drawdown,
            "max_position_size": position_value,
            "max_single_trade_loss": trade_loss,
        }

        triggered: list[str] = []
        for name, value in checks.items():
            if not self.breakers[name].check(value):
                triggered.append(name)

        allowed = len(triggered) == 0
        return allowed, triggered

    def advance_period(self) -> None:
        """Advance cooldown for all breakers by one period."""
        for breaker in self.breakers.values():
            breaker.update()
