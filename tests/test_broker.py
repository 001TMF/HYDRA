"""Tests for BrokerGateway -- ib_async wrapper with connection management."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hydra.execution.broker import BrokerGateway


class TestBrokerGatewayInit:
    """Test BrokerGateway initialization and defaults."""

    def test_default_port_is_paper(self) -> None:
        gw = BrokerGateway()
        assert gw._port == 4002

    def test_default_client_id(self) -> None:
        gw = BrokerGateway()
        assert gw._client_id == 1

    def test_default_is_paper(self) -> None:
        gw = BrokerGateway()
        assert gw.is_paper is True

    def test_default_host(self) -> None:
        gw = BrokerGateway()
        assert gw._host == "127.0.0.1"

    def test_instantiation_without_running_ib(self) -> None:
        """BrokerGateway can be created without an active IB connection."""
        gw = BrokerGateway()
        assert gw.ib is not None
        assert gw.is_connected() is False


class TestIsPaperProperty:
    """Test is_paper detection for all 4 port values."""

    def test_gateway_paper_4002(self) -> None:
        gw = BrokerGateway(port=4002)
        assert gw.is_paper is True

    def test_tws_paper_7497(self) -> None:
        gw = BrokerGateway(port=7497)
        assert gw.is_paper is True

    def test_gateway_live_4001(self) -> None:
        gw = BrokerGateway(port=4001)
        assert gw.is_paper is False

    def test_tws_live_7496(self) -> None:
        gw = BrokerGateway(port=7496)
        assert gw.is_paper is False


class TestSubmitOrder:
    """Test submit_order delegates to ib.placeOrder."""

    @pytest.mark.asyncio
    async def test_submit_order_delegates(self) -> None:
        gw = BrokerGateway()
        mock_trade = MagicMock()
        gw.ib.placeOrder = MagicMock(return_value=mock_trade)

        contract = MagicMock()
        order = MagicMock()
        result = await gw.submit_order(contract, order)

        gw.ib.placeOrder.assert_called_once_with(contract, order)
        assert result is mock_trade


class TestCancelOrder:
    """Test cancel_order delegates to ib.cancelOrder."""

    @pytest.mark.asyncio
    async def test_cancel_order_delegates(self) -> None:
        gw = BrokerGateway()
        gw.ib.cancelOrder = MagicMock()

        mock_trade = MagicMock()
        mock_trade.order = MagicMock()
        mock_trade.order.orderId = 42

        await gw.cancel_order(mock_trade)
        gw.ib.cancelOrder.assert_called_once_with(mock_trade.order)


class TestReconnect:
    """Test reconnection with exponential backoff."""

    @pytest.mark.asyncio
    async def test_reconnect_gives_up_after_max_retries(self) -> None:
        """After MAX_RECONNECT_ATTEMPTS failures, raises ConnectionError."""
        gw = BrokerGateway()
        gw.MAX_RECONNECT_ATTEMPTS = 3
        gw.INITIAL_BACKOFF_SECONDS = 0.01  # Fast for tests
        gw.MAX_BACKOFF_SECONDS = 0.05

        gw.ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("refused"))

        with pytest.raises(ConnectionError, match="Failed to reconnect after 3 attempts"):
            await gw.reconnect()

        assert gw.ib.connectAsync.call_count == 3

    @pytest.mark.asyncio
    async def test_reconnect_succeeds_on_retry(self) -> None:
        """Reconnection succeeds after initial failures."""
        gw = BrokerGateway()
        gw.INITIAL_BACKOFF_SECONDS = 0.01

        # Fail twice, then succeed
        gw.ib.connectAsync = AsyncMock(
            side_effect=[
                ConnectionRefusedError("refused"),
                ConnectionRefusedError("refused"),
                None,  # success
            ]
        )

        await gw.reconnect()  # Should not raise
        assert gw.ib.connectAsync.call_count == 3


class TestQualifyContract:
    """Test qualify_contract wraps qualifyContractsAsync."""

    @pytest.mark.asyncio
    async def test_qualify_returns_first_result(self) -> None:
        gw = BrokerGateway()
        mock_contract = MagicMock()
        qualified_contract = MagicMock()
        gw.ib.qualifyContractsAsync = AsyncMock(return_value=[qualified_contract])

        result = await gw.qualify_contract(mock_contract)
        assert result is qualified_contract

    @pytest.mark.asyncio
    async def test_qualify_returns_original_on_empty(self) -> None:
        gw = BrokerGateway()
        mock_contract = MagicMock()
        gw.ib.qualifyContractsAsync = AsyncMock(return_value=[])

        result = await gw.qualify_contract(mock_contract)
        assert result is mock_contract
