"""Tests for PaperTradingRunner daily cycle orchestrator.

Tests cover:
- Start validates trading_mode and rejects live without env var
- run_daily_cycle calls broker, agent_loop, model, order_manager in order
- run_daily_cycle skips execution when model is not fitted
- Fills are logged to fill_journal after order execution
- Stop shuts down scheduler and disconnects broker
- run_reconciliation delegates to reconciler

All dependencies are mocked (no IB connectivity or real scheduling).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from hydra.execution.runner import PaperTradingRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_runner(trading_mode: str = "paper", model_fitted: bool = True) -> tuple:
    """Create a PaperTradingRunner with all dependencies mocked."""
    broker = MagicMock()
    # is_connected is a sync method -- use MagicMock, not AsyncMock
    broker.is_connected = MagicMock(return_value=True)
    broker._port = 4002
    broker.connect = AsyncMock()
    broker.disconnect = AsyncMock()
    broker.reconnect = AsyncMock()
    broker.get_account_summary = AsyncMock(return_value=[])
    broker.get_positions = AsyncMock(return_value=[])

    risk_gate = AsyncMock()

    order_manager = AsyncMock()
    # route_order returns a list of mock trades with fills
    mock_fill = MagicMock()
    mock_fill.execution.price = 350.5
    mock_fill.execution.shares = 1
    mock_fill.time = 50.0

    mock_trade = MagicMock()
    mock_trade.fills = [mock_fill]
    mock_trade.order.orderId = 42
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
    model.is_fitted = model_fitted
    import numpy as np
    model.predict = MagicMock(return_value=np.array([1]))

    reconciler = MagicMock()
    reconciler.reconcile = MagicMock(return_value=None)

    config = {"trading_mode": trading_mode}

    runner = PaperTradingRunner(
        broker=broker,
        risk_gate=risk_gate,
        order_manager=order_manager,
        fill_journal=fill_journal,
        agent_loop=agent_loop,
        model=model,
        reconciler=reconciler,
        config=config,
    )

    return runner, {
        "broker": broker,
        "risk_gate": risk_gate,
        "order_manager": order_manager,
        "fill_journal": fill_journal,
        "agent_loop": agent_loop,
        "model": model,
        "reconciler": reconciler,
    }


# ---------------------------------------------------------------------------
# Tests: start / stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_paper_mode_succeeds():
    """Start in paper mode connects broker and starts scheduler."""
    runner, mocks = _make_runner(trading_mode="paper")
    mock_sched = MagicMock()

    with patch("hydra.execution.runner.AsyncIOScheduler", return_value=mock_sched):
        await runner.start()

    mocks["broker"].connect.assert_awaited_once()
    mock_sched.add_job.assert_called_once()
    mock_sched.start.assert_called_once()


@pytest.mark.asyncio
async def test_start_live_mode_rejects_without_env_var():
    """Start in live mode raises ValueError without HYDRA_LIVE_CONFIRMED."""
    runner, _ = _make_runner(trading_mode="live")

    # Ensure env var is NOT set
    env_copy = os.environ.copy()
    env_copy.pop("HYDRA_LIVE_CONFIRMED", None)

    with patch.dict(os.environ, env_copy, clear=True):
        with pytest.raises(ValueError, match="HYDRA_LIVE_CONFIRMED"):
            await runner.start()


@pytest.mark.asyncio
async def test_start_live_mode_succeeds_with_env_var():
    """Start in live mode succeeds when HYDRA_LIVE_CONFIRMED=true."""
    runner, mocks = _make_runner(trading_mode="live")

    with patch.dict(os.environ, {"HYDRA_LIVE_CONFIRMED": "true"}):
        mock_sched = MagicMock()
        with patch("hydra.execution.runner.AsyncIOScheduler", return_value=mock_sched):
            await runner.start()

        mocks["broker"].connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_shuts_down_scheduler_and_disconnects():
    """Stop shuts down scheduler, disconnects broker, closes journal."""
    runner, mocks = _make_runner()

    # Set up a mock scheduler
    mock_sched = MagicMock()
    runner._scheduler = mock_sched

    await runner.stop()

    mock_sched.shutdown.assert_called_once_with(wait=False)
    mocks["broker"].disconnect.assert_awaited_once()
    mocks["fill_journal"].close.assert_called_once()
    assert runner._scheduler is None


# ---------------------------------------------------------------------------
# Tests: run_daily_cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_daily_cycle_calls_components_in_order():
    """run_daily_cycle calls broker, agent_loop, model, order_manager."""
    runner, mocks = _make_runner(model_fitted=True)

    result = await runner.run_daily_cycle()

    # Verify broker connection was checked
    mocks["broker"].is_connected.assert_called()

    # Verify agent loop was called
    mocks["agent_loop"].run_cycle.assert_called_once()

    # Verify model was called
    mocks["model"].predict.assert_called_once()

    # Verify order manager was called
    mocks["order_manager"].route_order.assert_awaited_once()

    # Verify result structure
    assert "cycle_time" in result
    assert result["agent_result"] is not None
    assert result["signal"] is not None
    assert result["signal"]["direction"] in ("BUY", "SELL")


@pytest.mark.asyncio
async def test_run_daily_cycle_skips_when_model_not_fitted():
    """run_daily_cycle skips order execution when model is not fitted."""
    runner, mocks = _make_runner(model_fitted=False)

    result = await runner.run_daily_cycle()

    # Agent loop should still run
    mocks["agent_loop"].run_cycle.assert_called_once()

    # Model predict should NOT be called (is_fitted is False, so skip)
    mocks["model"].predict.assert_not_called()

    # Order manager should NOT be called
    mocks["order_manager"].route_order.assert_not_awaited()

    # Signal should be None
    assert result["signal"] is None


@pytest.mark.asyncio
async def test_run_daily_cycle_logs_fills():
    """Fills are logged to fill_journal after order execution."""
    runner, mocks = _make_runner(model_fitted=True)

    result = await runner.run_daily_cycle()

    # FillJournal.log_fill should have been called for each fill
    mocks["fill_journal"].log_fill.assert_called_once()
    assert result["n_fills"] == 1


@pytest.mark.asyncio
async def test_run_daily_cycle_handles_blocked_orders():
    """Blocked orders (None) are counted but not logged as fills."""
    runner, mocks = _make_runner(model_fitted=True)

    # Make order_manager return a blocked order
    mocks["order_manager"].route_order = AsyncMock(return_value=[None])

    result = await runner.run_daily_cycle()

    assert result["n_orders_blocked"] == 1
    assert result["n_fills"] == 0
    mocks["fill_journal"].log_fill.assert_not_called()


@pytest.mark.asyncio
async def test_run_daily_cycle_reconnects_when_disconnected():
    """run_daily_cycle attempts reconnect when broker is not connected."""
    runner, mocks = _make_runner(model_fitted=True)
    mocks["broker"].is_connected.return_value = False

    result = await runner.run_daily_cycle()

    mocks["broker"].reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_daily_cycle_skips_on_reconnect_failure():
    """run_daily_cycle skips cycle when reconnect fails."""
    runner, mocks = _make_runner(model_fitted=True)
    mocks["broker"].is_connected.return_value = False
    mocks["broker"].reconnect = AsyncMock(side_effect=ConnectionError("Failed"))

    result = await runner.run_daily_cycle()

    # Should return early with no signal
    assert result["signal"] is None
    assert result["n_fills"] == 0
    mocks["agent_loop"].run_cycle.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: run_reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_reconciliation_delegates_to_reconciler():
    """run_reconciliation passes through to SlippageReconciler."""
    runner, mocks = _make_runner()

    result = await runner.run_reconciliation(symbol="ZO")

    mocks["reconciler"].reconcile.assert_called_once_with(symbol="ZO")


@pytest.mark.asyncio
async def test_run_reconciliation_returns_none_for_insufficient_data():
    """run_reconciliation returns None when reconciler returns None."""
    runner, mocks = _make_runner()
    mocks["reconciler"].reconcile.return_value = None

    result = await runner.run_reconciliation()

    assert result is None
