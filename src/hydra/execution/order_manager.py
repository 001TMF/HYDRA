"""OrderManager: smart order routing for thin commodity futures markets (EXEC-03).

Routes orders through RiskGate using two strategies:
- Limit-with-patience: for small orders (< 1% ADV), place limit at mid,
  wait, then step toward market if unfilled.
- Custom TWAP: for large orders (>= 1% ADV), slice into N time-spaced
  limit orders with randomized intervals.

All code paths produce LimitOrder only -- no other order type is used.

Architecture:
    OrderManager -> RiskGate.submit() -> CircuitBreakerManager -> broker
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING

import structlog
from ib_async import LimitOrder

if TYPE_CHECKING:
    from ib_async import Contract, Trade

    from hydra.execution.risk_gate import RiskGate

logger = structlog.get_logger(__name__)


class OrderManager:
    """Smart order routing for thin commodity futures markets.

    Routes orders through RiskGate (never directly to broker) using
    volume-aware strategies. Small orders get limit-with-patience,
    large orders get custom TWAP slicing with randomized intervals.

    All orders are LimitOrder -- no other order type is ever created.

    Args:
        risk_gate: Mandatory risk middleware. All orders route through here.
        patience_seconds: Max wait time for limit order fill (default 300s = 5 min).
        twap_volume_threshold: Orders > this fraction of ADV get TWAP (default 1%).
        twap_slices: Number of TWAP time slices (default 5).
        twap_jitter_pct: Randomize slice timing by +/- this fraction (default 20%).
        price_step_pct: Price step toward market when patience exhausted (default 0.1%).
    """

    def __init__(
        self,
        risk_gate: RiskGate,
        patience_seconds: int = 300,
        twap_volume_threshold: float = 0.01,
        twap_slices: int = 5,
        twap_jitter_pct: float = 0.20,
        price_step_pct: float = 0.001,
    ) -> None:
        self._risk_gate = risk_gate
        self._patience_seconds = patience_seconds
        self._twap_volume_threshold = twap_volume_threshold
        self._twap_slices = twap_slices
        self._twap_jitter_pct = twap_jitter_pct
        self._price_step_pct = price_step_pct

    async def route_order(
        self,
        contract: Contract,
        direction: str,
        n_contracts: int,
        adv: float,
        mid_price: float,
        daily_pnl: float,
        peak_equity: float,
        current_equity: float,
        position_value: float,
        trade_loss: float,
    ) -> list[Trade | None]:
        """Route an order using volume-aware strategy selection.

        Args:
            contract: The IB contract to trade.
            direction: "BUY" or "SELL" (ib_async convention).
            n_contracts: Number of contracts to trade.
            adv: Average daily volume for this contract.
            mid_price: Current mid-price for limit order placement.
            daily_pnl: Today's P&L as fraction of capital.
            peak_equity: Peak equity for drawdown calculation.
            current_equity: Current equity value.
            position_value: Proposed position as fraction of capital.
            trade_loss: Loss from most recent trade as fraction of capital.

        Returns:
            List of Trade objects (single for limit, multiple for TWAP).
            None entries indicate risk-blocked orders.
        """
        participation_rate = n_contracts / max(adv, 1)
        risk_params = _compute_risk_params(
            daily_pnl, peak_equity, current_equity, position_value, trade_loss,
        )

        if participation_rate >= self._twap_volume_threshold:
            logger.info(
                "routing_twap",
                contract=str(contract),
                direction=direction,
                n_contracts=n_contracts,
                participation_rate=participation_rate,
                threshold=self._twap_volume_threshold,
            )
            return await self._twap_slice(
                contract, direction, n_contracts, mid_price, risk_params,
            )

        logger.info(
            "routing_limit_patience",
            contract=str(contract),
            direction=direction,
            n_contracts=n_contracts,
            participation_rate=participation_rate,
            threshold=self._twap_volume_threshold,
        )
        return await self._limit_with_patience(
            contract, direction, n_contracts, mid_price, risk_params,
        )

    async def _limit_with_patience(
        self,
        contract: Contract,
        direction: str,
        n_contracts: int,
        mid_price: float,
        risk_params: dict,
    ) -> list[Trade | None]:
        """Place limit at mid, wait, then step toward market if unfilled.

        Three stages:
        1. Limit at mid-price, wait patience_seconds.
        2. Step price toward market by price_step_pct, wait patience_seconds / 2.
        3. Cross the spread (buy at ask, sell at bid), wait patience_seconds / 4.

        Returns list with a single Trade (or None if risk-blocked).
        """
        order = LimitOrder(direction, n_contracts, mid_price)

        logger.info(
            "limit_patience_initial",
            contract=str(contract),
            direction=direction,
            n_contracts=n_contracts,
            limit_price=mid_price,
        )

        trade = await self._risk_gate.submit(contract, order, **risk_params)
        if trade is None:
            return [None]

        # Stage 1: wait at mid-price
        trade = await self._wait_for_fill(trade, self._patience_seconds)
        if self._is_filled(trade):
            return [trade]

        # Stage 2: step price toward market
        stepped_price = self._step_price(mid_price, direction)
        logger.info(
            "limit_patience_stepping",
            contract=str(contract),
            direction=direction,
            original_price=mid_price,
            stepped_price=stepped_price,
        )
        order.lmtPrice = stepped_price
        trade = await self._risk_gate.submit(contract, order, **risk_params)
        if trade is None:
            return [None]

        trade = await self._wait_for_fill(trade, self._patience_seconds // 2)
        if self._is_filled(trade):
            return [trade]

        # Stage 3: cross the spread
        crossed_price = self._cross_spread_price(mid_price, direction)
        logger.info(
            "limit_patience_crossing_spread",
            contract=str(contract),
            direction=direction,
            crossed_price=crossed_price,
        )
        order.lmtPrice = crossed_price
        trade = await self._risk_gate.submit(contract, order, **risk_params)
        if trade is None:
            return [None]

        await self._wait_for_fill(trade, self._patience_seconds // 4)
        return [trade]

    async def _twap_slice(
        self,
        contract: Contract,
        direction: str,
        n_contracts: int,
        mid_price: float,
        risk_params: dict,
    ) -> list[Trade | None]:
        """Slice order into N time-spaced limit orders with jittered intervals.

        Each slice reuses _limit_with_patience for fill logic.

        Returns list of Trade objects from all slices.
        """
        slice_size = n_contracts // self._twap_slices
        remainder = n_contracts % self._twap_slices
        base_interval = self._patience_seconds / self._twap_slices

        logger.info(
            "twap_plan",
            contract=str(contract),
            direction=direction,
            total_contracts=n_contracts,
            n_slices=self._twap_slices,
            slice_size=slice_size,
            remainder=remainder,
            base_interval=base_interval,
        )

        trades: list[Trade | None] = []
        for i in range(self._twap_slices):
            # Jittered delay (skip for first slice)
            if i > 0:
                jitter = random.uniform(
                    -self._twap_jitter_pct, self._twap_jitter_pct,
                )
                delay = base_interval * (1 + jitter)
                await asyncio.sleep(delay)

            # Distribute remainder across first slices
            slice_contracts = slice_size + (1 if i < remainder else 0)

            logger.info(
                "twap_slice_executing",
                slice_index=i,
                slice_contracts=slice_contracts,
                total_slices=self._twap_slices,
            )

            slice_trades = await self._limit_with_patience(
                contract, direction, slice_contracts, mid_price, risk_params,
            )
            trades.extend(slice_trades)

        return trades

    async def _wait_for_fill(self, trade: Trade, timeout: int) -> Trade:
        """Poll trade status until filled or timeout."""
        elapsed = 0
        while elapsed < timeout:
            if self._is_filled(trade):
                return trade
            await asyncio.sleep(1)
            elapsed += 1
        return trade

    @staticmethod
    def _is_filled(trade: Trade) -> bool:
        """Check if trade is completely filled."""
        if trade is None:
            return False
        return trade.orderStatus.status == "Filled"

    def _step_price(self, mid_price: float, direction: str) -> float:
        """Step price toward market by price_step_pct."""
        step = mid_price * self._price_step_pct
        if direction == "BUY":
            return mid_price + step
        return mid_price - step

    def _cross_spread_price(self, mid_price: float, direction: str) -> float:
        """Cross the spread: buy at ask, sell at bid.

        Uses a larger step (10x price_step_pct) as spread-crossing approximation.
        """
        spread_step = mid_price * self._price_step_pct * 10
        if direction == "BUY":
            return mid_price + spread_step
        return mid_price - spread_step


def _compute_risk_params(
    daily_pnl: float,
    peak_equity: float,
    current_equity: float,
    position_value: float,
    trade_loss: float,
) -> dict:
    """Package risk parameters into dict for passing to RiskGate.submit().

    These parameters map directly to RiskGate.submit() keyword arguments
    for circuit breaker evaluation.
    """
    return {
        "daily_pnl": daily_pnl,
        "peak_equity": peak_equity,
        "current_equity": current_equity,
        "position_value": position_value,
        "trade_loss": trade_loss,
    }
