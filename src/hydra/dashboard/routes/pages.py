"""HTML page routes for the HYDRA dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter()


def _open_fill_journal(data_dir: Path):
    """Try to open FillJournal, return None on failure."""
    fill_db = data_dir / "fill_journal.db"
    if not fill_db.exists():
        return None
    try:
        from hydra.execution.fill_journal import FillJournal

        return FillJournal(fill_db)
    except Exception:
        return None


def _open_experiment_journal(data_dir: Path):
    """Try to open ExperimentJournal, return None on failure."""
    exp_db = data_dir / "experiment_journal.db"
    if not exp_db.exists():
        return None
    try:
        from hydra.sandbox.journal import ExperimentJournal

        return ExperimentJournal(exp_db)
    except Exception:
        return None


@router.get("/")
async def index(request: Request):
    """Render the overview page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates

    fill_count = 0
    recent_fills = []
    fj = _open_fill_journal(data_dir)
    if fj is not None:
        try:
            fill_count = fj.count()
            recent_fills = fj.get_fills(limit=3)
        finally:
            fj.close()

    from hydra.cli.state import get_state

    agent_state = get_state().value

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_page": "overview",
            "fill_count": fill_count,
            "agent_state": agent_state,
            "recent_fills": recent_fills,
        },
    )


@router.get("/fills")
async def fills(request: Request):
    """Render the fill journal page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates

    fill_count = 0
    fill_list = []
    reconciliation = None
    fj = _open_fill_journal(data_dir)
    if fj is not None:
        try:
            fill_count = fj.count()
            fill_list = fj.get_fills(limit=50)
            if fill_count >= 10:
                try:
                    from hydra.execution.reconciler import SlippageReconciler

                    reconciler = SlippageReconciler(fj)
                    reconciliation = reconciler.reconcile()
                except Exception:
                    pass
        finally:
            fj.close()

    return templates.TemplateResponse(
        request,
        "fills.html",
        {
            "active_page": "fills",
            "fills": fill_list,
            "fill_count": fill_count,
            "reconciliation": reconciliation,
        },
    )


@router.get("/agent")
async def agent(request: Request):
    """Render the agent loop status page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates

    from hydra.cli.state import get_state

    agent_state = get_state().value

    experiment_count = 0
    experiments = []
    ej = _open_experiment_journal(data_dir)
    if ej is not None:
        try:
            experiment_count = ej.count()
            experiments = ej.query(limit=20)
        finally:
            ej.close()

    return templates.TemplateResponse(
        request,
        "agent.html",
        {
            "active_page": "agent",
            "agent_state": agent_state,
            "experiments": experiments,
            "experiment_count": experiment_count,
        },
    )


@router.get("/drift")
async def drift(request: Request):
    """Render the drift monitoring page."""
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "drift.html",
        {"active_page": "drift"},
    )


@router.get("/system")
async def system(request: Request):
    """Render the system health page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates

    from hydra.cli.state import get_state

    agent_state = get_state().value

    db_status = {"fill_journal": "unavailable", "experiment_journal": "unavailable"}

    fj = _open_fill_journal(data_dir)
    reconciliation = None
    if fj is not None:
        try:
            db_status["fill_journal"] = "ok"
            fill_count = fj.count()
            if fill_count >= 10:
                try:
                    from hydra.execution.reconciler import SlippageReconciler

                    reconciler = SlippageReconciler(fj)
                    reconciliation = reconciler.reconcile()
                except Exception:
                    pass
        finally:
            fj.close()

    ej = _open_experiment_journal(data_dir)
    if ej is not None:
        try:
            db_status["experiment_journal"] = "ok"
        finally:
            ej.close()

    return templates.TemplateResponse(
        request,
        "system.html",
        {
            "active_page": "system",
            "agent_state": agent_state,
            "db_status": db_status,
            "reconciliation": reconciliation,
        },
    )
