"""Tests for HYDRA dashboard app factory, health endpoint, and routes."""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from hydra.dashboard.app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app(data_dir=str(tmp_path))


@pytest_asyncio.fixture
async def async_client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def test_create_app(tmp_path):
    from pathlib import Path

    app = create_app(data_dir=str(tmp_path))
    assert app.title == "HYDRA Dashboard"
    assert isinstance(app.state.data_dir, Path)


@pytest.mark.asyncio
async def test_health_endpoint_no_databases(async_client):
    resp = await async_client.get("/api/health")
    assert resp.status_code == 503
    data = resp.json()
    assert data["api"] == "ok"
    unavailable = [v for v in data.values() if v == "unavailable"]
    assert len(unavailable) >= 1


@pytest.mark.asyncio
async def test_health_endpoint_with_databases(tmp_path):
    from hydra.execution.fill_journal import FillJournal
    from hydra.sandbox.journal import ExperimentJournal

    fj = FillJournal(tmp_path / "fill_journal.db")
    fj.close()
    ej = ExperimentJournal(tmp_path / "experiment_journal.db")
    ej.close()

    app = create_app(data_dir=str(tmp_path))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["api"] == "ok"
    assert data["fill_journal"] == "ok"
    assert data["experiment_journal"] == "ok"
    assert data["agent_state"] in ("running", "paused")


@pytest.mark.asyncio
async def test_index_page_returns_html(async_client):
    resp = await async_client.get("/")
    assert resp.status_code == 200
    assert "HYDRA" in resp.text
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_fills_summary_empty(async_client):
    resp = await async_client.get("/api/fills/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["fill_count"] == 0


@pytest.mark.asyncio
async def test_static_css_served(async_client):
    resp = await async_client.get("/static/style.css")
    assert resp.status_code == 200
