"""Tests for all 5 HYDRA dashboard page routes with empty and populated data."""

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
            hypothesis="Increase num_leaves for better accuracy",
            mutation_type="hyperparameter",
            config_diff={"num_leaves": {"old": 31, "new": 63}},
            results={"sharpe": 0.45, "hit_rate": 0.52},
            promotion_decision="rejected",
            promotion_reason="Sharpe below champion",
        ),
        ExperimentRecord(
            created_at="2026-02-18T12:00:00Z",
            hypothesis="Add COT sentiment feature",
            mutation_type="feature_add",
            config_diff={"features": {"added": "cot_sentiment_z"}},
            results={"sharpe": 0.72, "hit_rate": 0.58},
            promotion_decision="promoted",
            promotion_reason="Sharpe improvement +0.15",
        ),
        ExperimentRecord(
            created_at="2026-02-18T14:00:00Z",
            hypothesis="Lower learning rate for stability",
            mutation_type="hyperparameter",
            config_diff={"learning_rate": {"old": 0.1, "new": 0.05}},
            results={"sharpe": 0.55, "hit_rate": 0.54},
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
# Index page tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_empty(async_client_empty):
    resp = await async_client_empty.get("/")
    assert resp.status_code == 200
    assert "HYDRA" in resp.text
    assert "0" in resp.text


@pytest.mark.asyncio
async def test_index_with_data(async_client_data):
    resp = await async_client_data.get("/")
    assert resp.status_code == 200
    assert "HYDRA" in resp.text
    assert "5" in resp.text


# ---------------------------------------------------------------------------
# Fills page tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fills_page_empty(async_client_empty):
    resp = await async_client_empty.get("/fills")
    assert resp.status_code == 200
    assert "No fills" in resp.text or "0 fills" in resp.text


@pytest.mark.asyncio
async def test_fills_page_with_data(async_client_data):
    resp = await async_client_data.get("/fills")
    assert resp.status_code == 200
    assert "<tr>" in resp.text
    assert "ZO" in resp.text
    assert "5 fills" in resp.text


# ---------------------------------------------------------------------------
# Agent page tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_page_empty(async_client_empty):
    resp = await async_client_empty.get("/agent")
    assert resp.status_code == 200
    body = resp.text.lower()
    assert "paused" in body or "running" in body


@pytest.mark.asyncio
async def test_agent_page_with_data(async_client_data):
    resp = await async_client_data.get("/agent")
    assert resp.status_code == 200
    assert "hyperparameter" in resp.text or "feature_add" in resp.text
    assert "3" in resp.text  # experiment count


# ---------------------------------------------------------------------------
# Drift page tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_page(async_client_empty):
    resp = await async_client_empty.get("/drift")
    assert resp.status_code == 200
    assert "Drift" in resp.text
    assert "Sharpe" in resp.text


# ---------------------------------------------------------------------------
# System page tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_page_empty(async_client_empty):
    resp = await async_client_empty.get("/system")
    assert resp.status_code == 200
    body = resp.text.upper()
    assert "UNAVAILABLE" in body


@pytest.mark.asyncio
async def test_system_page_with_data(async_client_data):
    resp = await async_client_data.get("/system")
    assert resp.status_code == 200
    body = resp.text.upper()
    assert "OK" in body


# ---------------------------------------------------------------------------
# All pages return HTML
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_pages_return_html(async_client_empty):
    pages = ["/", "/fills", "/agent", "/drift", "/system"]
    for page in pages:
        resp = await async_client_empty.get(page)
        assert resp.status_code == 200, f"{page} returned {resp.status_code}"
        assert "text/html" in resp.headers["content-type"], f"{page} not HTML"
