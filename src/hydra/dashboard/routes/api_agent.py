"""Agent API endpoints for the HYDRA dashboard."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from starlette.responses import HTMLResponse, JSONResponse

router = APIRouter()


@router.get("/state", response_class=HTMLResponse)
async def agent_state(request: Request):
    """Return HTML fragment with agent state badge for htmx swap."""
    from hydra.cli.state import get_state

    state = get_state().value
    if state == "running":
        css_class = "status-ok"
        return HTMLResponse(f'<div class="kpi-value {css_class}">RUNNING</div>')
    else:
        css_class = "status-warn"
        return HTMLResponse(f'<div class="kpi-value {css_class}">PAUSED</div>')


@router.get("/summary")
async def agent_summary(request: Request):
    """Return experiment summary stats (total, promoted, rejected, rate)."""
    data_dir = request.app.state.data_dir
    runner = getattr(request.app.state, "runner", None)

    from hydra.dashboard.data_access import DashboardData

    data = DashboardData(data_dir, runner)
    return JSONResponse(content=data.get_experiment_summary())


@router.get("/mutation-breakdown")
async def mutation_breakdown(request: Request):
    """Return experiment count per mutation type."""
    data_dir = request.app.state.data_dir
    runner = getattr(request.app.state, "runner", None)

    from hydra.dashboard.data_access import DashboardData

    data = DashboardData(data_dir, runner)
    return JSONResponse(content=data.get_mutation_breakdown())


@router.get("/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_detail(request: Request, experiment_id: int):
    """Return HTML fragment with full experiment detail for HTMX expansion."""
    data_dir = request.app.state.data_dir
    exp_db = data_dir / "experiment_journal.db"
    if not exp_db.exists():
        return HTMLResponse('<div class="detail-panel"><p>No experiment data</p></div>')

    try:
        from hydra.sandbox.journal import ExperimentJournal

        ej = ExperimentJournal(exp_db)
        try:
            exp = ej.get_experiment(experiment_id)
            if exp is None:
                return HTMLResponse('<div class="detail-panel"><p>Experiment not found</p></div>')

            parts = ['<div class="detail-panel">']

            if exp.config_diff:
                parts.append("<h4>Config Diff</h4>")
                parts.append(f"<pre>{json.dumps(exp.config_diff, indent=2)}</pre>")

            if exp.results:
                parts.append("<h4>Results</h4>")
                parts.append(f"<pre>{json.dumps(exp.results, indent=2)}</pre>")

            if exp.champion_metrics:
                parts.append("<h4>Champion Metrics</h4>")
                parts.append(f"<pre>{json.dumps(exp.champion_metrics, indent=2)}</pre>")

            if exp.promotion_reason:
                parts.append(f"<h4>Promotion Reason</h4><p>{exp.promotion_reason}</p>")

            if exp.tags:
                parts.append("<h4>Tags</h4>")
                parts.append(f"<pre>{json.dumps(exp.tags, indent=2)}</pre>")

            parts.append("</div>")
            return HTMLResponse("\n".join(parts))
        finally:
            ej.close()
    except Exception:
        return HTMLResponse('<div class="detail-panel"><p>Error loading experiment</p></div>')
