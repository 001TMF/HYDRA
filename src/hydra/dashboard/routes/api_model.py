"""Model API endpoints for the HYDRA dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

router = APIRouter()


@router.get("/champion")
async def champion(request: Request):
    """Return champion model info."""
    try:
        from hydra.sandbox.registry import ModelRegistry

        reg = ModelRegistry()
        info = reg.get_champion_info()
        return JSONResponse(content=info)
    except Exception:
        return JSONResponse(content={"error": "No champion model"}, status_code=404)


@router.get("/versions")
async def versions(request: Request):
    """Return all model versions."""
    try:
        from hydra.sandbox.registry import ModelRegistry

        reg = ModelRegistry()
        return JSONResponse(content=reg.list_versions())
    except Exception:
        return JSONResponse(content=[], status_code=200)


@router.get("/importance")
async def importance(request: Request):
    """Return feature importance from runner's model."""
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return JSONResponse(content={"error": "Runner not active"}, status_code=404)
    try:
        model = runner._model
        if not model.is_fitted:
            return JSONResponse(
                content={"error": "Model not fitted"}, status_code=404
            )
        return JSONResponse(content=model.feature_importance())
    except Exception:
        return JSONResponse(content={"error": "Cannot get importance"}, status_code=500)


@router.get("/fitness")
async def fitness(request: Request):
    """Return fitness breakdown from latest promoted experiment."""
    data_dir = request.app.state.data_dir
    exp_db = data_dir / "experiment_journal.db"
    if not exp_db.exists():
        return JSONResponse(content={"error": "No experiments"}, status_code=404)
    try:
        from hydra.sandbox.journal import ExperimentJournal

        ej = ExperimentJournal(exp_db)
        try:
            promoted = ej.query(outcome="promoted", limit=1)
            if not promoted:
                return JSONResponse(
                    content={"error": "No promoted experiments"}, status_code=404
                )
            results = promoted[0].results
            if not isinstance(results, dict):
                return JSONResponse(
                    content={"error": "Invalid results format"}, status_code=500
                )
            return JSONResponse(content=results)
        finally:
            ej.close()
    except Exception:
        return JSONResponse(content={"error": "Query failed"}, status_code=500)
