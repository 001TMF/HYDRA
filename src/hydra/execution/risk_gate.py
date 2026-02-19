"""RiskGate: mandatory pre-trade circuit breaker middleware (EXEC-04).

Every order submission MUST pass through RiskGate.submit(). There is NO
code path from the order manager to BrokerGateway that bypasses this
class. RiskGate owns the only reference to the broker's submit method.

Architecture:
    OrderManager -> RiskGate.submit() -> CircuitBreakerManager.check_trade()
                                      -> BrokerGateway.submit_order() (if allowed)
                                      -> None (if blocked)

The RiskGate does NOT expose a raw submit_order passthrough. The only
way to submit an order is through RiskGate.submit() which includes
the mandatory risk check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ib_async import Contract, Trade

    from hydra.execution.broker import BrokerGateway
    from hydra.risk.circuit_breakers import CircuitBreakerManager

logger = structlog.get_logger(__name__)


class RiskGate:
    """Mandatory pre-trade risk check middleware.

    Every order passes through here. No bypass path exists.
    The broker property provides read-only access for non-order
    operations (positions, account summary) but does NOT expose
    submit_order -- callers must go through RiskGate.submit().
    """

    def __init__(
        self,
        broker: BrokerGateway,
        breakers: CircuitBreakerManager,
    ) -> None:
        self._broker = broker
        self._breakers = breakers

    async def submit(
        self,
        contract: Contract,
        order: object,
        daily_pnl: float,
        peak_equity: float,
        current_equity: float,
        position_value: float,
        trade_loss: float,
    ) -> Trade | None:
        """Check all circuit breakers then submit order if allowed.

        Args:
            contract: The IB contract to trade.
            order: The IB order object (LimitOrder, etc.).
            daily_pnl: Today's P&L as fraction of capital (negative = loss).
            peak_equity: Peak equity value for drawdown calculation.
            current_equity: Current equity value.
            position_value: Proposed position as fraction of capital.
            trade_loss: Loss from the most recent trade as fraction of capital.

        Returns:
            Trade object if order was submitted, None if blocked by risk checks.
        """
        logger.info(
            "risk_gate_checking",
            contract=str(contract),
            daily_pnl=daily_pnl,
            position_value=position_value,
            trade_loss=trade_loss,
        )

        allowed, triggered = self._breakers.check_trade(
            daily_pnl=daily_pnl,
            peak_equity=peak_equity,
            current_equity=current_equity,
            position_value=position_value,
            trade_loss=trade_loss,
        )

        if not allowed:
            logger.warning(
                "order_blocked_by_risk",
                contract=str(contract),
                triggered=triggered,
                daily_pnl=daily_pnl,
                peak_equity=peak_equity,
                current_equity=current_equity,
                position_value=position_value,
                trade_loss=trade_loss,
            )
            return None

        trade = await self._broker.submit_order(contract, order)
        logger.info(
            "order_allowed_by_risk",
            contract=str(contract),
        )
        return trade

    async def cancel(self, trade: Trade) -> None:
        """Cancel an open order via the broker."""
        await self._broker.cancel_order(trade)

    @property
    def broker(self) -> BrokerGateway:
        """Read-only access to the broker for non-order operations.

        Use this for positions, account summary, connection status, etc.
        This does NOT expose submit_order -- callers must use
        RiskGate.submit() which includes mandatory risk checks.
        """
        return self._broker
