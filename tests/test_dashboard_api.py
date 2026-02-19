"""Tests for HYDRA dashboard API endpoints with empty and populated data."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from hydra.dashboard.app import create_app
from hydra.execution.fill_journal import FillJournal, FillRecord
from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord


@pytest.fixture
def app_empty(tmp_path):
    """Create app with empty data_dir (no databases)."""
    return create_app(data_dir=str(tmp_path))


@pytest.fixture
def app_with_data(tmp_path):
    """Create app with pre-populated FillJournal and ExperimentJournal."""
    fj = FillJournal(tmp_path / "fill_journal.db")
    fills = [
        FillRecord(
            timestamp="2026-02-19T14:00:00Z",
            symbol="ZO",
            direction=1,
            n_contracts=2,
            order_price=322.0,
            fill_price=322.05,
            predicted_slippage=0.04,
            actual_slippage=0.05,
            volume_at_fill=150.0,
            spread_at_fill=0.25,
            fill_latency_ms=45.0,
        ),
        FillRecord(
            timestamp="2026-02-19T14:05:00Z",
            symbol="HE",
            direction=-1,
            n_contracts=1,
            order_price=85.50,
            fill_price=85.48,
            predicted_slippage=0.03,
            actual_slippage=0.02,
            volume_at_fill=200.0,
            spread_at_fill=0.10,
            fill_latency_ms=32.0,
        ),
        FillRecord(
            timestamp="2026-02-19T14:10:00Z",
            symbol="ZO",
            direction=1,
            n_contracts=3,
            order_price=322.50,
            fill_price=322.56,
            predicted_slippage=0.05,
            actual_slippage=0.06,
            volume_at_fill=120.0,
            spread_at_fill=0.30,
            fill_latency_ms=55.0,
        ),
        FillRecord(
            timestamp="2026-02-19T14:15:00Z",
            symbol="LE",
            direction=-1,
            n_contracts=2,
            order_price=190.00,
            fill_price=189.95,
            predicted_slippage=0.06,
            actual_slippage=0.05,
            volume_at_fill=80.0,
            spread_at_fill=0.15,
            fill_latency_ms=40.0,
        ),
        FillRecord(
            timestamp="2026-02-19T14:20:00Z",
            symbol="ZO",
            direction=1,
            n_contracts=1,
            order_price=323.00,
            fill_price=323.03,
            predicted_slippage=0.02,
            actual_slippage=0.03,
            volume_at_fill=170.0,
            spread_at_fill=0.20,
            fill_latency_ms=28.0,
        ),
    ]
    for fill in fills:
        fj.log_fill(fill)
    fj.close()

    ej = ExperimentJournal(tmp_path / "experiment_journal.db")
    experiments = [
        ExperimentRecord(
            created_at="2026-02-18T10:00:00Z",
            hypothesis="Increase num_leaves",
            mutation_type="hyperparameter",
            config_diff={"num_leaves": {"old": 31, "new": 63}},
            results={"sharpe": 0.45},
            promotion_decision="rejected",
        ),
        ExperimentRecord(
            created_at="2026-02-18T12:00:00Z",
            hypothesis="Add COT feature",
            mutation_type="feature_add",
            config_diff={"features": {"added": "cot_sentiment_z"}},
            results={"sharpe": 0.72},
            promotion_decision="promoted",
        ),
        ExperimentRecord(
            created_at="2026-02-18T14:00:00Z",
            hypothesis="Lower learning rate",
            mutation_type="hyperparameter",
            config_diff={"learning_rate": {"old": 0.1, "new": 0.05}},
            results={"sharpe": 0.55},
            promotion_decision="pending",
        ),
    ]
    for exp in experiments:
        ej.log_experiment(exp)
    ej.close()

    return create_app(data_dir=str(tmp_path))


@pytest_asyncio.fixture
async def async_client_empty(app_empty):
    transport = ASGITransport(app=app_empty)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def async_client_data(app_with_data):
    transport = ASGITransport(app=app_with_data)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------------------------------------------------------------------------
# /api/fills/recent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fills_recent_empty(async_client_empty):
    resp = await async_client_empty.get("/api/fills/recent")
    assert resp.status_code == 200
    assert "No fill data" in resp.text


@pytest.mark.asyncio
async def test_fills_recent_with_data(async_client_data):
    resp = await async_client_data.get("/api/fills/recent")
    assert resp.status_code == 200
    assert "ZO" in resp.text
    assert "<tr>" in resp.text


# ---------------------------------------------------------------------------
# /api/agent/state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_state_endpoint(async_client_empty):
    resp = await async_client_empty.get("/api/agent/state")
    assert resp.status_code == 200
    body = resp.text.upper()
    assert "PAUSED" in body or "RUNNING" in body


# ---------------------------------------------------------------------------
# /api/system/status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_status_endpoint(async_client_empty):
    resp = await async_client_empty.get("/api/system/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "agent_state" in data
    assert "db_status" in data
    assert "fill_journal" in data["db_status"]
    assert "experiment_journal" in data["db_status"]


# ---------------------------------------------------------------------------
# /api/health with populated DBs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_with_populated_dbs(async_client_data):
    resp = await async_client_data.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["api"] == "ok"
    assert data["fill_journal"] == "ok"
    assert data["experiment_journal"] == "ok"


# ---------------------------------------------------------------------------
# /api/fills/summary with data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fills_summary_with_data(async_client_data):
    resp = await async_client_data.get("/api/fills/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fill_count"] > 0
    assert data["fill_count"] == 5
    assert len(data["recent_fills"]) == 5
