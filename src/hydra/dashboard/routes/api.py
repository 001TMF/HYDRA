"""JSON API endpoints for the HYDRA dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health(request: Request):
    """Health check endpoint.

    Returns status of API, SQLite databases, and agent state.
    Status 200 if all OK, 503 if any component unavailable.
    """
    data_dir = request.app.state.data_dir
    status = {"api": "ok"}
    any_unavailable = False

    # Check FillJournal
    fj = None
    try:
        from hydra.execution.fill_journal import FillJournal

        fj = FillJournal(data_dir / "fill_journal.db")
        status["fill_journal"] = "ok"
    except Exception:
        status["fill_journal"] = "unavailable"
        any_unavailable = True
    finally:
        if fj is not None:
            fj.close()

    # Check ExperimentJournal
    ej = None
    try:
        from hydra.sandbox.journal import ExperimentJournal

        ej = ExperimentJournal(data_dir / "experiment_journal.db")
        status["experiment_journal"] = "ok"
    except Exception:
        status["experiment_journal"] = "unavailable"
        any_unavailable = True
    finally:
        if ej is not None:
            ej.close()

    # Agent state
    from hydra.cli.state import get_state

    status["agent_state"] = get_state().value

    status_code = 503 if any_unavailable else 200
    return JSONResponse(content=status, status_code=status_code)


@router.get("/fills/summary")
async def fills_summary(request: Request):
    """Return fill count and last 5 fills for dashboard polling."""
    data_dir = request.app.state.data_dir
    fj = None
    try:
        from hydra.execution.fill_journal import FillJournal

        fj = FillJournal(data_dir / "fill_journal.db")
        fill_count = fj.count()
        recent = fj.get_fills(limit=5)
        recent_fills = [
            {
                "timestamp": r.timestamp,
                "symbol": r.symbol,
                "direction": r.direction,
                "fill_price": r.fill_price,
                "actual_slippage": r.actual_slippage,
            }
            for r in recent
        ]
        return {"fill_count": fill_count, "recent_fills": recent_fills}
    except Exception:
        return {"fill_count": 0, "recent_fills": []}
    finally:
        if fj is not None:
            fj.close()
