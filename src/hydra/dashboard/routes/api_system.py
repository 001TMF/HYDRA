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


@router.post("/trigger-cycle")
async def trigger_cycle(request: Request):
    """Trigger an immediate daily cycle via the runner."""
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return JSONResponse(
            content={"error": "Runner not active"}, status_code=503
        )
    import asyncio

    asyncio.create_task(runner.run_daily_cycle())
    return JSONResponse(content={"status": "cycle_triggered"})


@router.post("/backfill")
async def backfill(request: Request):
    """Backfill 1 year of historical futures + COT data for all configured markets.

    Fetches ~250 daily bars per market from IB and a full year of COT reports.
    Features are written to the feature store with correct per-bar dates.
    """
    import asyncio
    from datetime import datetime, timezone

    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return JSONResponse(
            content={"error": "Runner not active"}, status_code=503
        )

    async def _run_backfill():
        from hydra.data.ingestion.ib_futures import IBFuturesIngestPipeline
        from hydra.data.ingestion.cot import COTIngestPipeline

        now = datetime.now(timezone.utc)
        results = {}

        configs = runner._market_configs or []
        pipelines = runner._market_pipelines or {}

        for cfg in configs:
            market = cfg.symbol
            market_results = {"futures": False, "cot": False}

            for pipeline in pipelines.get(market, []):
                try:
                    if isinstance(pipeline, IBFuturesIngestPipeline):
                        market_results["futures"] = await pipeline.run_backfill_async(
                            market, now, duration="1 Y"
                        )
                    elif isinstance(pipeline, COTIngestPipeline):
                        market_results["cot"] = pipeline.run(market, now)
                except Exception as exc:
                    import structlog
                    structlog.get_logger().warning(
                        "backfill_pipeline_failed",
                        market=market,
                        pipeline=type(pipeline).__name__,
                        error=str(exc),
                    )

            results[market] = market_results

        return results

    asyncio.create_task(_run_backfill())
    return JSONResponse(content={"status": "backfill_started"})
