"""Data/feature store API endpoints for the HYDRA dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from starlette.responses import JSONResponse

from hydra.dashboard.data_access import DashboardData

router = APIRouter()


@router.get("/feature-stats")
async def feature_stats(request: Request):
    """Return feature store statistics."""
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(request.app.state.data_dir, runner)
    return JSONResponse(content=data.get_feature_store_stats())


@router.get("/latest-features")
async def latest_features(request: Request, market: str | None = Query(None)):
    """Return latest feature values, optionally filtered by market."""
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(request.app.state.data_dir, runner)
    return JSONResponse(content=data.get_latest_features(market=market))
