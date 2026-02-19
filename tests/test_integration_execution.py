"""Integration tests for the full execution pipeline.

Tests verify that the complete pipeline works together end-to-end:
    BrokerGateway -> RiskGate -> OrderManager -> FillJournal -> SlippageReconciler

All tests use mocked IB (no real IB Gateway needed for CI). Tests that need
a real IB Gateway are marked with @pytest.mark.skipif(no IB_GATEWAY_HOST).

Tests:
    - full_pipeline_paper_order_flow: order routed, risk allowed, fill logged
    - full_pipeline_risk_blocked: risk gate blocks, broker never called
    - reconciler_integration: 20 synthetic fills, verify bias/RMSE/correlation
    - runner_daily_cycle_mock: full daily cycle with mocked deps
    - port_safety: paper/live port detection and live mode env var guard
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from hydra.execution.broker import BrokerGateway
from hydra.execution.fill_journal import FillJournal, FillRecord
from hydra.execution.order_manager import OrderManager
from hydra.execution.reconciler import ReconciliationReport, SlippageReconciler
from hydra.execution.risk_gate import RiskGate
from hydra.execution.runner import PaperTradingRunner
from hydra.risk.circuit_breakers import CircuitBreakerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ib():
    """Create a mock ib_async.IB instance with standard stubs."""
    mock_ib = MagicMock()
    mock_ib.connectAsync = AsyncMock()
    mock_ib.disconnect = MagicMock()
    mock_ib.isConnected = MagicMock(return_value=True)
    mock_ib.placeOrder = MagicMock()
    mock_ib.cancelOrder = MagicMock()
    mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])
    mock_ib.positions = MagicMock(return_value=[])
    mock_ib.accountSummary = MagicMock(return_value=[])
    mock_ib.reqOpenOrders = MagicMock()
    mock_ib.reqPositions = MagicMock()
    mock_ib.disconnectedEvent = MagicMock()
    mock_ib.disconnectedEvent.__iadd__ = MagicMock(return_value=mock_ib.disconnectedEvent)
    return mock_ib


def _make_mock_trade_with_fill(fill_price: float = 350.5, n_shares: int = 1):
    """Create a mock Trade object with a single fill."""
    mock_fill = MagicMock()
    mock_fill.execution.price = fill_price
    mock_fill.execution.shares = n_shares
    mock_fill.time = 50.0

    mock_order_status = MagicMock()
    mock_order_status.status = "Filled"

    mock_trade = MagicMock()
    mock_trade.fills = [mock_fill]
    mock_trade.order.orderId = 42
    mock_trade.orderStatus = mock_order_status

    return mock_trade


# ---------------------------------------------------------------------------
# Test: full_pipeline_paper_order_flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_paper_order_flow(tmp_path: Path) -> None:
    """End-to-end: order flows through RiskGate, broker submits, fill logged.

    Pipeline: OrderManager -> RiskGate -> CircuitBreakerManager -> BrokerGateway -> FillJournal
    """
    # -- Setup: Real components with mocked IB --
    mock_ib = _make_mock_ib()
    mock_trade = _make_mock_trade_with_fill(fill_price=350.5)
    mock_ib.placeOrder.return_value = mock_trade

    with patch("hydra.execution.broker.IB", return_value=mock_ib):
        broker = BrokerGateway(port=4002, client_id=1)

    breakers = CircuitBreakerManager()
    risk_gate = RiskGate(broker=broker, breakers=breakers)
    order_manager = OrderManager(
        risk_gate=risk_gate,
        patience_seconds=0,  # No waiting in tests
    )
    fill_journal = FillJournal(db_path=tmp_path / "fills.db")

    try:
        # -- Execute: Route an order with safe risk params --
        contract = MagicMock()
        contract.__str__ = lambda self: "ZO-FUT"

        trades = await order_manager.route_order(
            contract=contract,
            direction="BUY",
            n_contracts=1,
            adv=500.0,
            mid_price=350.0,
            daily_pnl=0.0,       # No loss -- within threshold
            peak_equity=100_000,
            current_equity=100_000,
            position_value=0.01,  # 1% -- within 10% threshold
            trade_loss=0.0,
        )

        # -- Verify: risk_gate allowed the order --
        assert len(trades) >= 1
        assert trades[0] is not None, "Risk gate should have allowed the order"

        # -- Verify: broker received the order --
        mock_ib.placeOrder.assert_called()

        # -- Verify: log fill to journal --
        for trade in trades:
            if trade is None:
                continue
            for fill in trade.fills:
                record = FillRecord(
                    timestamp="2026-02-19T14:00:00Z",
                    symbol="ZO",
                    direction=1,
                    n_contracts=fill.execution.shares,
                    order_price=350.0,
                    fill_price=fill.execution.price,
                    predicted_slippage=0.15,
                    actual_slippage=abs(fill.execution.price - 350.0),
                    volume_at_fill=500.0,
                    spread_at_fill=0.25,
                    fill_latency_ms=50.0,
                    order_id=trade.order.orderId,
                )
                fill_journal.log_fill(record)

        # Verify fill was logged
        fills = fill_journal.get_fills(symbol="ZO")
        assert len(fills) >= 1
        assert fills[0].symbol == "ZO"
        assert fills[0].fill_price == 350.5
        assert fills[0].order_id == 42

    finally:
        fill_journal.close()


# ---------------------------------------------------------------------------
# Test: full_pipeline_risk_blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_pipeline_risk_blocked(tmp_path: Path) -> None:
    """End-to-end: risk gate blocks order, broker never called, no fill logged.

    Sets daily_pnl to -0.05 which exceeds the -0.02 threshold.
    """
    mock_ib = _make_mock_ib()

    with patch("hydra.execution.broker.IB", return_value=mock_ib):
        broker = BrokerGateway(port=4002, client_id=1)

    breakers = CircuitBreakerManager()
    risk_gate = RiskGate(broker=broker, breakers=breakers)
    order_manager = OrderManager(
        risk_gate=risk_gate,
        patience_seconds=0,
    )
    fill_journal = FillJournal(db_path=tmp_path / "fills.db")

    try:
        contract = MagicMock()
        contract.__str__ = lambda self: "ZO-FUT"

        trades = await order_manager.route_order(
            contract=contract,
            direction="BUY",
            n_contracts=1,
            adv=500.0,
            mid_price=350.0,
            daily_pnl=-0.05,     # Exceeds -0.02 threshold -> BLOCKED
            peak_equity=100_000,
            current_equity=100_000,
            position_value=0.01,
            trade_loss=0.0,
        )

        # -- Verify: RiskGate blocked the order --
        assert len(trades) >= 1
        assert trades[0] is None, "Risk gate should have blocked the order"

        # -- Verify: broker submit_order was NOT called --
        mock_ib.placeOrder.assert_not_called()

        # -- Verify: no fills logged --
        assert fill_journal.count() == 0

    finally:
        fill_journal.close()


# ---------------------------------------------------------------------------
# Test: reconciler_integration
# ---------------------------------------------------------------------------


def test_reconciler_integration(tmp_path: Path) -> None:
    """Insert 20 synthetic fills, verify reconciler computes correct stats."""
    fill_journal = FillJournal(db_path=tmp_path / "fills.db")

    try:
        # Insert 20 fills with known predicted/actual slippage
        rng = np.random.default_rng(seed=42)
        predicted_values = rng.uniform(0.1, 0.5, size=20)
        # actual = predicted + small noise (model is slightly optimistic)
        noise = rng.normal(0.05, 0.03, size=20)
        actual_values = predicted_values + noise

        for i in range(20):
            record = FillRecord(
                timestamp=f"2026-02-{i + 1:02d}T14:00:00Z",
                symbol="ZO",
                direction=1,
                n_contracts=1,
                order_price=350.0,
                fill_price=350.0 + actual_values[i],
                predicted_slippage=float(predicted_values[i]),
                actual_slippage=float(actual_values[i]),
                volume_at_fill=500.0,
                spread_at_fill=0.25,
                fill_latency_ms=50.0,
            )
            fill_journal.log_fill(record)

        assert fill_journal.count() == 20

        # Run reconciliation
        reconciler = SlippageReconciler(fill_journal)
        report = reconciler.reconcile()

        # -- Verify report is not None (we have 20 fills >= MIN_FILLS=10) --
        assert report is not None
        assert isinstance(report, ReconciliationReport)
        assert report.n_fills == 20

        # -- Verify bias is positive (actuals > predicted by ~0.05) --
        assert report.bias > 0.0, f"Expected positive bias, got {report.bias}"
        assert abs(report.bias - 0.05) < 0.05, (
            f"Bias {report.bias:.4f} should be near 0.05 (within tolerance)"
        )

        # -- Verify RMSE is reasonable (noise std ~0.03, bias ~0.05) --
        assert report.rmse > 0.0
        assert report.rmse < 0.2, f"RMSE {report.rmse:.4f} unexpectedly large"

        # -- Verify correlation is strong (predicted + noise is highly correlated) --
        assert report.correlation > 0.8, (
            f"Expected high correlation, got {report.correlation:.4f}"
        )

        # -- Verify pessimism multiplier > 1 (reality is worse than model) --
        assert report.pessimism_multiplier > 1.0

        # -- Verify calibration check --
        calibrated, reason = reconciler.is_model_calibrated(
            max_bias=0.5, min_correlation=0.3,
        )
        assert calibrated, f"Model should be calibrated, but: {reason}"

    finally:
        fill_journal.close()


# ---------------------------------------------------------------------------
# Test: runner_daily_cycle_mock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_daily_cycle_mock() -> None:
    """PaperTradingRunner.run_daily_cycle calls all components in order."""
    broker = MagicMock()
    broker.is_connected = MagicMock(return_value=True)
    broker._port = 4002
    broker.connect = AsyncMock()
    broker.disconnect = AsyncMock()
    broker.get_account_summary = AsyncMock(return_value=[])
    broker.get_positions = AsyncMock(return_value=[])

    risk_gate = AsyncMock()
    order_manager = AsyncMock()

    mock_trade = _make_mock_trade_with_fill()
    order_manager.route_order = AsyncMock(return_value=[mock_trade])

    fill_journal = MagicMock()
    fill_journal.log_fill = MagicMock(return_value=1)
    fill_journal.close = MagicMock()

    agent_loop = MagicMock()
    agent_result = MagicMock()
    agent_result.phase_reached.value = "observe"
    agent_result.promoted = False
    agent_result.rolled_back = False
    agent_result.skipped_reason = "No drift detected"
    agent_loop.run_cycle = MagicMock(return_value=agent_result)

    model = MagicMock()
    model.is_fitted = True
    model.predict = MagicMock(return_value=np.array([1]))

    reconciler = MagicMock()

    runner = PaperTradingRunner(
        broker=broker,
        risk_gate=risk_gate,
        order_manager=order_manager,
        fill_journal=fill_journal,
        agent_loop=agent_loop,
        model=model,
        reconciler=reconciler,
    )

    result = await runner.run_daily_cycle()

    # Verify broker connection was checked
    broker.is_connected.assert_called()

    # Verify agent_loop.run_cycle was called
    agent_loop.run_cycle.assert_called_once()

    # Verify model was called
    model.predict.assert_called_once()

    # Verify order manager was called
    order_manager.route_order.assert_awaited_once()

    # Verify results
    assert result["cycle_time"] is not None
    assert result["agent_result"] is not None
    assert result["agent_result"]["phase_reached"] == "observe"
    assert result["signal"]["direction"] in ("BUY", "SELL")
    assert result["n_fills"] >= 0


# ---------------------------------------------------------------------------
# Test: port_safety
# ---------------------------------------------------------------------------


class TestPortSafety:
    """Verify port-based paper/live detection and live mode env var guard."""

    def test_paper_port_4002(self) -> None:
        """BrokerGateway(port=4002) reports is_paper == True."""
        gw = BrokerGateway(port=4002)
        assert gw.is_paper is True

    def test_live_port_4001(self) -> None:
        """BrokerGateway(port=4001) reports is_paper == False."""
        gw = BrokerGateway(port=4001)
        assert gw.is_paper is False

    def test_paper_port_7497(self) -> None:
        """TWS paper port 7497 is detected as paper."""
        gw = BrokerGateway(port=7497)
        assert gw.is_paper is True

    def test_live_port_7496(self) -> None:
        """TWS live port 7496 is detected as live."""
        gw = BrokerGateway(port=7496)
        assert gw.is_paper is False

    @pytest.mark.asyncio
    async def test_live_mode_rejects_without_env_var(self) -> None:
        """PaperTradingRunner in live mode raises ValueError without env var."""
        broker = MagicMock()
        broker._port = 4001
        broker.connect = AsyncMock()

        runner = PaperTradingRunner(
            broker=broker,
            risk_gate=MagicMock(),
            order_manager=MagicMock(),
            fill_journal=MagicMock(),
            agent_loop=MagicMock(),
            model=MagicMock(),
            reconciler=MagicMock(),
            config={"trading_mode": "live"},
        )

        # Ensure env var is NOT set
        env_copy = os.environ.copy()
        env_copy.pop("HYDRA_LIVE_CONFIRMED", None)

        with patch.dict(os.environ, env_copy, clear=True):
            with pytest.raises(ValueError, match="HYDRA_LIVE_CONFIRMED"):
                await runner.start()

    @pytest.mark.asyncio
    async def test_live_mode_succeeds_with_env_var(self) -> None:
        """PaperTradingRunner in live mode succeeds with env var set."""
        broker = MagicMock()
        broker._port = 4001
        broker.connect = AsyncMock()

        runner = PaperTradingRunner(
            broker=broker,
            risk_gate=MagicMock(),
            order_manager=MagicMock(),
            fill_journal=MagicMock(),
            agent_loop=MagicMock(),
            model=MagicMock(),
            reconciler=MagicMock(),
            config={"trading_mode": "live"},
        )

        with patch.dict(os.environ, {"HYDRA_LIVE_CONFIRMED": "true"}):
            mock_sched = MagicMock()
            with patch(
                "hydra.execution.runner.AsyncIOScheduler",
                return_value=mock_sched,
            ):
                await runner.start()

        broker.connect.assert_awaited_once()
