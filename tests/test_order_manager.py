"""Tests for OrderManager -- smart order routing for thin commodity futures.

Tests cover:
- Routing decision (small vs large orders)
- LimitOrder enforcement (no MarketOrder ever)
- TWAP slicing math (remainder distribution)
- RiskGate delegation (never direct to broker)
- Risk-blocked order handling (None entries in result)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hydra.execution.order_manager import OrderManager


def _make_risk_gate_mock(trade_return: object | None = None) -> MagicMock:
    """Create a mock RiskGate.

    Args:
        trade_return: What submit() returns. If None, creates a mock Trade
            with 'Filled' status. Pass explicit None value via
            _make_risk_gate_mock_blocked() instead.
    """
    gate = MagicMock()
    if trade_return is None:
        trade_mock = MagicMock()
        trade_mock.orderStatus.status = "Filled"
        trade_return = trade_mock
    gate.submit = AsyncMock(return_value=trade_return)
    return gate


def _make_risk_gate_mock_blocked() -> MagicMock:
    """Create a mock RiskGate that blocks (returns None)."""
    gate = MagicMock()
    gate.submit = AsyncMock(return_value=None)
    return gate


def _make_contract() -> MagicMock:
    """Create a mock IB contract."""
    return MagicMock(name="MockContract")


def _default_risk_kwargs() -> dict:
    """Risk params that pass all circuit breakers."""
    return {
        "daily_pnl": 0.0,
        "peak_equity": 100000,
        "current_equity": 100000,
        "position_value": 0.05,
        "trade_loss": 0.0,
    }


class TestRoutingDecision:
    """Test that OrderManager routes to correct strategy based on ADV."""

    @pytest.mark.asyncio
    async def test_small_order_uses_limit_patience(self) -> None:
        """Orders below TWAP threshold route to limit-with-patience."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, twap_volume_threshold=0.01)

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=5,
                adv=1000.0,  # 5/1000 = 0.005 < 0.01 -> limit
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

        # limit_with_patience called once -> single Trade in list
        assert len(result) == 1
        assert result[0] is not None
        # Only one submit call (filled immediately)
        gate.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_large_order_uses_twap(self) -> None:
        """Orders at or above TWAP threshold route to TWAP slicing."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(
            gate,
            twap_volume_threshold=0.01,
            twap_slices=5,
            patience_seconds=300,
        )

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=10,
                adv=100.0,  # 10/100 = 0.10 >= 0.01 -> TWAP
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

        # TWAP with 5 slices -> 5 trades in list
        assert len(result) == 5
        # Each slice calls submit once (if filled immediately)
        assert gate.submit.call_count == 5

    @pytest.mark.asyncio
    async def test_exact_threshold_uses_twap(self) -> None:
        """Orders at exactly the TWAP threshold use TWAP."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, twap_volume_threshold=0.01, twap_slices=3)

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await mgr.route_order(
                contract=_make_contract(),
                direction="SELL",
                n_contracts=1,
                adv=100.0,  # 1/100 = 0.01 == threshold -> TWAP
                mid_price=50.0,
                **_default_risk_kwargs(),
            )

        # TWAP with 3 slices
        assert len(result) == 3


class TestLimitOrderEnforcement:
    """Test that only LimitOrder is ever used -- no MarketOrder."""

    @pytest.mark.asyncio
    async def test_limit_order_created(self) -> None:
        """All orders submitted are LimitOrder type."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate)

        with (
            patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock),
            patch("hydra.execution.order_manager.LimitOrder") as mock_limit,
        ):
            mock_limit.return_value = MagicMock()
            await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=2,
                adv=1000.0,
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

            # LimitOrder was constructed
            mock_limit.assert_called()
            call_args = mock_limit.call_args
            assert call_args[0][0] == "BUY"  # direction
            assert call_args[0][1] == 2       # n_contracts
            assert call_args[0][2] == 100.0   # mid_price

    def test_no_market_order_import(self) -> None:
        """order_manager.py does not import or reference MarketOrder."""
        import inspect

        from hydra.execution import order_manager

        source = inspect.getsource(order_manager)
        assert "MarketOrder" not in source


class TestTwapSlicingMath:
    """Test TWAP slice distribution with remainders."""

    @pytest.mark.asyncio
    async def test_13_contracts_5_slices(self) -> None:
        """13 contracts with 5 slices distributes as [3, 3, 3, 2, 2]."""
        slice_sizes: list[int] = []

        gate = MagicMock()

        async def capture_submit(contract, order, **kwargs):
            # Capture the n_contracts from the LimitOrder constructor
            trade = MagicMock()
            trade.orderStatus.status = "Filled"
            return trade

        gate.submit = AsyncMock(side_effect=capture_submit)

        mgr = OrderManager(gate, twap_volume_threshold=0.01, twap_slices=5)

        with (
            patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock),
            patch("hydra.execution.order_manager.LimitOrder") as mock_limit,
        ):
            mock_limit.return_value = MagicMock()
            await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=13,
                adv=100.0,  # 13/100 = 0.13 >= 0.01 -> TWAP
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

            # Extract n_contracts from each LimitOrder call
            for call in mock_limit.call_args_list:
                slice_sizes.append(call[0][1])

        # 13 // 5 = 2, remainder = 3
        # First 3 slices get 3, last 2 slices get 2
        assert slice_sizes == [3, 3, 3, 2, 2]
        assert sum(slice_sizes) == 13

    @pytest.mark.asyncio
    async def test_10_contracts_5_slices_even(self) -> None:
        """10 contracts with 5 slices distributes evenly as [2, 2, 2, 2, 2]."""
        slice_sizes: list[int] = []

        gate = MagicMock()
        gate.submit = AsyncMock(
            return_value=MagicMock(orderStatus=MagicMock(status="Filled")),
        )

        mgr = OrderManager(gate, twap_volume_threshold=0.01, twap_slices=5)

        with (
            patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock),
            patch("hydra.execution.order_manager.LimitOrder") as mock_limit,
        ):
            mock_limit.return_value = MagicMock()
            await mgr.route_order(
                contract=_make_contract(),
                direction="SELL",
                n_contracts=10,
                adv=100.0,
                mid_price=50.0,
                **_default_risk_kwargs(),
            )

            for call in mock_limit.call_args_list:
                slice_sizes.append(call[0][1])

        assert slice_sizes == [2, 2, 2, 2, 2]
        assert sum(slice_sizes) == 10


class TestRiskGateDelegation:
    """Test that all orders go through RiskGate, never directly to broker."""

    @pytest.mark.asyncio
    async def test_submit_goes_through_risk_gate(self) -> None:
        """OrderManager calls risk_gate.submit, not broker.submit_order."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate)

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=1,
                adv=1000.0,
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

        # risk_gate.submit was called
        gate.submit.assert_called()

        # Verify risk params were passed
        call_kwargs = gate.submit.call_args[1]
        assert "daily_pnl" in call_kwargs
        assert "peak_equity" in call_kwargs
        assert "current_equity" in call_kwargs
        assert "position_value" in call_kwargs
        assert "trade_loss" in call_kwargs

    def test_no_direct_broker_reference(self) -> None:
        """OrderManager source code does not import or call submit_order."""
        import inspect

        from hydra.execution import order_manager

        source = inspect.getsource(order_manager)
        assert "submit_order" not in source
        # No import of BrokerGateway -- orders go through RiskGate only
        assert "from hydra.execution.broker" not in source


class TestRiskBlocked:
    """Test behavior when RiskGate blocks orders (returns None)."""

    @pytest.mark.asyncio
    async def test_blocked_limit_returns_none_entry(self) -> None:
        """When risk_gate.submit returns None, result list contains None."""
        gate = _make_risk_gate_mock_blocked()
        mgr = OrderManager(gate)

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=1,
                adv=1000.0,
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

        assert len(result) == 1
        assert result[0] is None

    @pytest.mark.asyncio
    async def test_blocked_twap_returns_none_entries(self) -> None:
        """When risk_gate blocks during TWAP, result contains None entries."""
        gate = _make_risk_gate_mock_blocked()
        mgr = OrderManager(gate, twap_volume_threshold=0.01, twap_slices=3)

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await mgr.route_order(
                contract=_make_contract(),
                direction="SELL",
                n_contracts=10,
                adv=100.0,  # 10/100 = 0.10 >= 0.01 -> TWAP
                mid_price=50.0,
                **_default_risk_kwargs(),
            )

        # Each of 3 slices was blocked
        assert len(result) == 3
        assert all(t is None for t in result)


class TestPriceAdjustment:
    """Test price stepping and spread crossing logic."""

    def test_step_price_buy_increases(self) -> None:
        """Buying steps price UP toward market."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, price_step_pct=0.001)

        stepped = mgr._step_price(100.0, "BUY")
        assert stepped == pytest.approx(100.1)  # 100 + 100*0.001

    def test_step_price_sell_decreases(self) -> None:
        """Selling steps price DOWN toward market."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, price_step_pct=0.001)

        stepped = mgr._step_price(100.0, "SELL")
        assert stepped == pytest.approx(99.9)  # 100 - 100*0.001

    def test_cross_spread_buy_jumps_higher(self) -> None:
        """Spread crossing for BUY goes to approximate ask."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, price_step_pct=0.001)

        crossed = mgr._cross_spread_price(100.0, "BUY")
        assert crossed == pytest.approx(101.0)  # 100 + 100*0.001*10

    def test_cross_spread_sell_jumps_lower(self) -> None:
        """Spread crossing for SELL goes to approximate bid."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, price_step_pct=0.001)

        crossed = mgr._cross_spread_price(100.0, "SELL")
        assert crossed == pytest.approx(99.0)  # 100 - 100*0.001*10


class TestZeroAdv:
    """Test edge case where ADV is zero (avoid division by zero)."""

    @pytest.mark.asyncio
    async def test_zero_adv_uses_twap(self) -> None:
        """ADV of 0 -> max(0, 1) = 1 -> high participation -> TWAP."""
        gate = _make_risk_gate_mock()
        mgr = OrderManager(gate, twap_volume_threshold=0.01, twap_slices=2)

        with patch("hydra.execution.order_manager.asyncio.sleep", new_callable=AsyncMock):
            result = await mgr.route_order(
                contract=_make_contract(),
                direction="BUY",
                n_contracts=1,
                adv=0.0,  # Zero ADV -> participation = 1/1 = 1.0 >= 0.01
                mid_price=100.0,
                **_default_risk_kwargs(),
            )

        # TWAP with 2 slices
        assert len(result) == 2
