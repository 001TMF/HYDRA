"""BrokerGateway: ib_async wrapper with connection management.

Wraps ib_async.IB into a clean async interface with:
- Automatic reconnection with exponential backoff
- Paper/live mode detection based on port
- Client ID allocation (1=trading, 2=diagnostic, 3=CLI)
- State resynchronization after reconnection

Default port is 4002 (IB Gateway paper trading). Live port 4001
requires explicit opt-in. The is_paper property distinguishes modes.
"""

from __future__ import annotations

import asyncio

import structlog
from ib_async import IB, Contract, Trade

logger = structlog.get_logger(__name__)


class BrokerGateway:
    """Wraps ib_async.IB with reconnection, health checks, and order lifecycle.

    Port mapping:
        4001 = IB Gateway live
        4002 = IB Gateway paper (default)
        7496 = TWS live
        7497 = TWS paper

    Client ID allocation:
        1 = trading (default)
        2 = diagnostic tools
        3 = CLI ad-hoc queries
    """

    LIVE_PORTS: set[int] = {4001, 7496}
    PAPER_PORTS: set[int] = {4002, 7497}

    MAX_RECONNECT_ATTEMPTS: int = 10
    INITIAL_BACKOFF_SECONDS: float = 1.0
    MAX_BACKOFF_SECONDS: float = 30.0

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        readonly: bool = False,
    ) -> None:
        self.ib = IB()
        self._host = host
        self._port = port
        self._client_id = client_id
        self._readonly = readonly

        mode = "PAPER" if self.is_paper else "LIVE"
        logger.info(
            "broker_gateway_init",
            host=host,
            port=port,
            client_id=client_id,
            readonly=readonly,
            mode=mode,
        )

    @property
    def is_paper(self) -> bool:
        """Return True if connected to a paper trading port."""
        return self._port in self.PAPER_PORTS

    async def connect(self) -> None:
        """Connect to IB Gateway/TWS and register disconnect handler."""
        mode = "PAPER" if self.is_paper else "LIVE"
        logger.info(
            "broker_connecting",
            host=self._host,
            port=self._port,
            client_id=self._client_id,
            mode=mode,
        )
        await self.ib.connectAsync(
            self._host,
            self._port,
            clientId=self._client_id,
            readonly=self._readonly,
        )
        self.ib.disconnectedEvent += self._on_disconnect
        logger.info("broker_connected", mode=mode)

    async def disconnect(self) -> None:
        """Disconnect from IB Gateway/TWS."""
        self.ib.disconnect()
        logger.info("broker_disconnected")

    async def reconnect(self) -> None:
        """Reconnect with exponential backoff.

        Attempts up to MAX_RECONNECT_ATTEMPTS with backoff starting at
        INITIAL_BACKOFF_SECONDS and doubling up to MAX_BACKOFF_SECONDS.

        Raises:
            ConnectionError: After exhausting all retry attempts.
        """
        backoff = self.INITIAL_BACKOFF_SECONDS
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            logger.info(
                "broker_reconnect_attempt",
                attempt=attempt,
                max_attempts=self.MAX_RECONNECT_ATTEMPTS,
                backoff_seconds=backoff,
            )
            try:
                await self.ib.connectAsync(
                    self._host,
                    self._port,
                    clientId=self._client_id,
                    readonly=self._readonly,
                )
                logger.info("broker_reconnected", attempt=attempt)
                return
            except Exception:
                logger.warning(
                    "broker_reconnect_failed",
                    attempt=attempt,
                    backoff_seconds=backoff,
                )
                if attempt < self.MAX_RECONNECT_ATTEMPTS:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF_SECONDS)

        msg = (
            f"Failed to reconnect after {self.MAX_RECONNECT_ATTEMPTS} attempts"
        )
        logger.error("broker_reconnect_exhausted")
        raise ConnectionError(msg)

    async def submit_order(self, contract: Contract, order: object) -> Trade:
        """Submit an order to IB. Returns the Trade object for monitoring."""
        trade = self.ib.placeOrder(contract, order)
        logger.info(
            "order_submitted",
            contract=str(contract),
            order_type=type(order).__name__,
        )
        return trade

    async def cancel_order(self, trade: Trade) -> None:
        """Cancel an open order."""
        self.ib.cancelOrder(trade.order)
        logger.info("order_cancelled", order_id=trade.order.orderId)

    async def qualify_contract(self, contract: Contract) -> Contract:
        """Qualify a contract against IB's database.

        Returns the fully qualified contract with conId populated.
        """
        qualified = await self.ib.qualifyContractsAsync(contract)
        if qualified:
            return qualified[0]
        return contract

    async def get_positions(self) -> list:
        """Return current positions from IB."""
        return self.ib.positions()

    async def get_account_summary(self) -> list:
        """Return account summary values from IB."""
        return self.ib.accountSummary()

    def is_connected(self) -> bool:
        """Return True if currently connected to IB."""
        return self.ib.isConnected()

    async def resync_state(self) -> None:
        """Resynchronize local state with broker after reconnection.

        Requests open orders and positions to ensure local state
        matches IB's server state (per Pitfall 6 from research).
        """
        logger.info("broker_resyncing_state")
        self.ib.reqOpenOrders()
        self.ib.reqPositions()
        logger.info("broker_state_resynced")

    async def _on_disconnect(self) -> None:
        """Auto-reconnect handler registered on disconnectedEvent."""
        logger.warning("broker_disconnected_unexpected")
        try:
            await self.reconnect()
            await self.resync_state()
        except ConnectionError:
            logger.error("broker_auto_reconnect_failed")
