"""Tests for enhanced HYDRA dashboard: new pages, API endpoints, and data access."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from hydra.dashboard.app import create_app
from hydra.execution.fill_journal import FillJournal, FillRecord
from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
            results={"sharpe": 0.45, "hit_rate": 0.52},
            promotion_decision="rejected",
            promotion_reason="Sharpe below champion",
        ),
        ExperimentRecord(
            created_at="2026-02-18T12:00:00Z",
            hypothesis="Add COT feature",
            mutation_type="feature_add",
            config_diff={"features": {"added": "cot_sentiment_z"}},
            results={"sharpe": 0.72, "hit_rate": 0.58},
            promotion_decision="promoted",
            promotion_reason="Sharpe improvement +0.15",
        ),
        ExperimentRecord(
            created_at="2026-02-18T14:00:00Z",
            hypothesis="Lower learning rate",
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
async def client_empty(app_empty):
    transport = ASGITransport(app=app_empty)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def client_data(app_with_data):
    transport = ASGITransport(app=app_with_data)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# All 7 pages return HTML
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_seven_pages_return_html(client_empty):
    pages = ["/", "/fills", "/agent", "/model", "/drift", "/data", "/system"]
    for page in pages:
        resp = await client_empty.get(page)
        assert resp.status_code == 200, f"{page} returned {resp.status_code}"
        assert "text/html" in resp.headers["content-type"], f"{page} not HTML"


# ---------------------------------------------------------------------------
# New page routes: /model, /data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_page_empty(client_empty):
    resp = await client_empty.get("/model")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Model" in resp.text or "model" in resp.text.lower()


@pytest.mark.asyncio
async def test_data_page_empty(client_empty):
    resp = await client_empty.get("/data")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Data" in resp.text or "Feature" in resp.text


@pytest.mark.asyncio
async def test_drift_page_loads(client_empty):
    resp = await client_empty.get("/drift")
    assert resp.status_code == 200
    assert "Drift" in resp.text


# ---------------------------------------------------------------------------
# Enhanced pages with data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_with_data_shows_fill_count(client_data):
    resp = await client_data.get("/")
    assert resp.status_code == 200
    assert "2" in resp.text


@pytest.mark.asyncio
async def test_agent_with_data_shows_experiments(client_data):
    resp = await client_data.get("/agent")
    assert resp.status_code == 200
    assert "hyperparameter" in resp.text or "feature_add" in resp.text


@pytest.mark.asyncio
async def test_agent_with_data_shows_outcome(client_data):
    resp = await client_data.get("/agent")
    assert resp.status_code == 200
    assert "promoted" in resp.text or "Promoted" in resp.text


@pytest.mark.asyncio
async def test_fills_with_data_shows_symbol(client_data):
    resp = await client_data.get("/fills")
    assert resp.status_code == 200
    assert "ZO" in resp.text


@pytest.mark.asyncio
async def test_fills_with_data_shows_count(client_data):
    resp = await client_data.get("/fills")
    assert resp.status_code == 200
    assert "2 fills" in resp.text


@pytest.mark.asyncio
async def test_system_with_data_shows_ok(client_data):
    resp = await client_data.get("/system")
    assert resp.status_code == 200
    assert "ok" in resp.text.lower() or "OK" in resp.text.upper()


# ---------------------------------------------------------------------------
# Agent API: /api/agent/summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_summary_empty(client_empty):
    resp = await client_empty.get("/api/agent/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["promoted"] == 0
    assert data["rejected"] == 0
    assert "rate" in data


@pytest.mark.asyncio
async def test_agent_summary_with_data(client_data):
    resp = await client_data.get("/api/agent/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    assert data["promoted"] == 1
    assert data["rejected"] == 1
    assert "rate" in data


# ---------------------------------------------------------------------------
# Agent API: /api/agent/mutation-breakdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_mutation_breakdown_empty(client_empty):
    resp = await client_empty.get("/api/agent/mutation-breakdown")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_agent_mutation_breakdown_with_data(client_data):
    resp = await client_data.get("/api/agent/mutation-breakdown")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert "hyperparameter" in data
    assert data["hyperparameter"] == 2
    assert "feature_add" in data
    assert data["feature_add"] == 1


# ---------------------------------------------------------------------------
# Agent API: /api/agent/experiments/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_experiment_detail_no_db(client_empty):
    resp = await client_empty.get("/api/agent/experiments/1")
    assert resp.status_code == 200
    assert "detail-panel" in resp.text


@pytest.mark.asyncio
async def test_experiment_detail_with_data(client_data):
    resp = await client_data.get("/api/agent/experiments/1")
    assert resp.status_code == 200
    assert "detail-panel" in resp.text
    assert "num_leaves" in resp.text


@pytest.mark.asyncio
async def test_experiment_detail_promoted_shows_reason(client_data):
    resp = await client_data.get("/api/agent/experiments/2")
    assert resp.status_code == 200
    assert "Sharpe improvement" in resp.text


@pytest.mark.asyncio
async def test_experiment_detail_not_found(client_data):
    resp = await client_data.get("/api/agent/experiments/999")
    assert resp.status_code == 200
    assert "not found" in resp.text.lower()


# ---------------------------------------------------------------------------
# Data API endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feature_stats_empty(client_empty):
    resp = await client_empty.get("/api/data/feature-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_latest_features_empty(client_empty):
    resp = await client_empty.get("/api/data/latest-features")
    assert resp.status_code == 200
    data = resp.json()
    assert data == []


# ---------------------------------------------------------------------------
# System API endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_components_empty(client_empty):
    resp = await client_empty.get("/api/system/components")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 3


@pytest.mark.asyncio
async def test_system_circuit_breakers(client_empty):
    resp = await client_empty.get("/api/system/circuit-breakers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1


@pytest.mark.asyncio
async def test_system_disk_usage(client_empty):
    resp = await client_empty.get("/api/system/disk-usage")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_system_config(client_empty):
    resp = await client_empty.get("/api/system/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "data_dir" in data
    assert "agent_state" in data


# ---------------------------------------------------------------------------
# Model API endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_champion_no_mlflow(client_empty):
    resp = await client_empty.get("/api/model/champion")
    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_model_versions_no_mlflow(client_empty):
    resp = await client_empty.get("/api/model/versions")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_model_fitness_with_data(client_data):
    resp = await client_data.get("/api/model/fitness")
    assert resp.status_code == 200
    data = resp.json()
    assert "sharpe" in data


# ---------------------------------------------------------------------------
# Agent page context: exp_summary and mutation_breakdown in template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_page_shows_kpi_counts(client_data):
    """Verify the agent page renders exp_summary values."""
    resp = await client_data.get("/agent")
    assert resp.status_code == 200
    # Total = 3, Promoted = 1, Rejected = 1
    text = resp.text
    assert "3" in text
    assert "1" in text


@pytest.mark.asyncio
async def test_agent_page_empty_no_experiments(client_empty):
    """Agent page gracefully handles no experiment data."""
    resp = await client_empty.get("/agent")
    assert resp.status_code == 200
    assert "No experiments" in resp.text or "0" in resp.text


# ---------------------------------------------------------------------------
# Fills page: slippage_data passes through to template
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fills_page_includes_slippage_chart(client_data):
    """Fills page includes slippage scatter chart when data exists."""
    resp = await client_data.get("/fills")
    assert resp.status_code == 200
    assert "slippage" in resp.text.lower()
