"""System API endpoints for the HYDRA dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from hydra.dashboard.data_access import DashboardData

router = APIRouter()


@router.get("/components")
async def components(request: Request):
    """Return component health status."""
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(request.app.state.data_dir, runner)
    return JSONResponse(content=data.get_component_health())


@router.get("/circuit-breakers")
async def circuit_breakers(request: Request):
    """Return circuit breaker states."""
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(request.app.state.data_dir, runner)
    return JSONResponse(content=data.get_circuit_breaker_states())


@router.get("/config")
async def config(request: Request):
    """Return runtime configuration."""
    from hydra.cli.state import get_state
    data_dir = request.app.state.data_dir
    return JSONResponse(content={
        "data_dir": str(data_dir),
        "trading_mode": "paper",
        "agent_state": get_state().value,
    })


@router.get("/disk-usage")
async def disk_usage(request: Request):
    """Return disk usage for SQLite databases."""
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(request.app.state.data_dir, runner)
    return JSONResponse(content=data.get_disk_usage())
