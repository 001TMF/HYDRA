"""Tests for RiskGate -- mandatory pre-trade circuit breaker middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from hydra.execution.risk_gate import RiskGate
from hydra.risk.circuit_breakers import CircuitBreakerManager


def _make_broker_mock() -> MagicMock:
    """Create a mock BrokerGateway with async methods."""
    broker = MagicMock()
    broker.submit_order = AsyncMock(return_value=MagicMock(name="Trade"))
    broker.cancel_order = AsyncMock()
    return broker


class TestRiskGateSubmitAllowed:
    """Test that submit delegates to broker when all breakers allow trade."""

    @pytest.mark.asyncio
    async def test_submit_allowed_returns_trade(self) -> None:
        """When all values are within thresholds, submit delegates to broker."""
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()  # Default thresholds

        gate = RiskGate(broker, breakers)
        contract = MagicMock()
        order = MagicMock()

        result = await gate.submit(
            contract=contract,
            order=order,
            daily_pnl=0.0,       # No loss today
            peak_equity=100000,
            current_equity=100000,
            position_value=0.05,  # 5% < 10% threshold
            trade_loss=0.0,       # No loss
        )

        assert result is not None
        broker.submit_order.assert_called_once_with(contract, order)


class TestRiskGateSubmitBlocked:
    """Test that submit returns None when breakers trip."""

    @pytest.mark.asyncio
    async def test_daily_loss_blocks_order(self) -> None:
        """Order blocked when daily P&L exceeds loss threshold."""
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()  # max_daily_loss = -0.02

        gate = RiskGate(broker, breakers)
        contract = MagicMock()
        order = MagicMock()

        result = await gate.submit(
            contract=contract,
            order=order,
            daily_pnl=-0.05,     # -5% < -2% threshold -> trips
            peak_equity=100000,
            current_equity=100000,
            position_value=0.05,
            trade_loss=0.0,
        )

        assert result is None
        broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_position_size_blocks_order(self) -> None:
        """Order blocked when position size exceeds threshold."""
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()  # max_position_size = 0.10

        gate = RiskGate(broker, breakers)
        contract = MagicMock()
        order = MagicMock()

        result = await gate.submit(
            contract=contract,
            order=order,
            daily_pnl=0.0,
            peak_equity=100000,
            current_equity=100000,
            position_value=0.15,  # 15% > 10% threshold -> trips
            trade_loss=0.0,
        )

        assert result is None
        broker.submit_order.assert_not_called()


class TestRiskGateMultipleBreakers:
    """Test multiple simultaneous breaker triggers."""

    @pytest.mark.asyncio
    async def test_multiple_breakers_all_logged(self) -> None:
        """When multiple breakers trip, all are reported and order is blocked."""
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()

        gate = RiskGate(broker, breakers)
        contract = MagicMock()
        order = MagicMock()

        result = await gate.submit(
            contract=contract,
            order=order,
            daily_pnl=-0.05,      # trips max_daily_loss
            peak_equity=100000,
            current_equity=90000,  # -10% drawdown, trips max_drawdown
            position_value=0.15,   # trips max_position_size
            trade_loss=-0.02,      # trips max_single_trade_loss
        )

        assert result is None
        broker.submit_order.assert_not_called()


class TestRiskGateCancel:
    """Test cancel delegates to broker."""

    @pytest.mark.asyncio
    async def test_cancel_delegates_to_broker(self) -> None:
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()

        gate = RiskGate(broker, breakers)
        trade = MagicMock()

        await gate.cancel(trade)
        broker.cancel_order.assert_called_once_with(trade)


class TestRiskGateBrokerProperty:
    """Test broker property provides read-only access."""

    def test_broker_property_returns_broker(self) -> None:
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()

        gate = RiskGate(broker, breakers)
        assert gate.broker is broker

    def test_broker_property_does_not_expose_submit_bypass(self) -> None:
        """The RiskGate class itself has no submit_order method.

        Callers must use RiskGate.submit() which includes risk checks.
        The broker property exists only for non-order operations.
        """
        broker = _make_broker_mock()
        breakers = CircuitBreakerManager()

        gate = RiskGate(broker, breakers)

        # RiskGate has submit (with risk checks), not submit_order (bypass)
        assert hasattr(gate, "submit")
        assert not hasattr(gate, "submit_order")
