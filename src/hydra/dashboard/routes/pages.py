"""HTML page routes for the HYDRA dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from hydra.dashboard.data_access import DashboardData

router = APIRouter()


@router.get("/")
async def index(request: Request):
    """Render the overview page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates
    runner = getattr(request.app.state, "runner", None)

    from hydra.cli.state import get_state

    agent_state = get_state().value
    data = DashboardData(data_dir, runner)

    fill_count = 0
    fj = data.get_fill_journal()
    if fj is not None:
        try:
            fill_count = fj.count()
        finally:
            fj.close()

    experiment_count = 0
    ej = data.get_experiment_journal()
    if ej is not None:
        try:
            experiment_count = ej.count()
        finally:
            ej.close()

    champion_version = None
    try:
        reg = data.get_model_registry()
        if reg is not None:
            info = reg.get_champion_info()
            champion_version = info.get("version")
    except Exception:
        pass

    components = data.get_component_health()

    # Slippage scatter data: [[predicted, actual], ...]
    slippage_data = None
    fj2 = data.get_fill_journal()
    if fj2 is not None:
        try:
            fills = fj2.get_fills(limit=30)
            if fills:
                slippage_data = [
                    [f.predicted_slippage, f.actual_slippage]
                    for f in fills
                    if f.predicted_slippage is not None and f.actual_slippage is not None
                ]
                if not slippage_data:
                    slippage_data = None
        finally:
            fj2.close()

    # Build recent activity from fills + experiments
    recent_activity = []
    fj3 = data.get_fill_journal()
    if fj3 is not None:
        try:
            for f in fj3.get_fills(limit=5):
                direction = "BUY" if f.direction == 1 else "SELL"
                recent_activity.append({
                    "timestamp": f.timestamp,
                    "type": "Fill",
                    "badge": "ok",
                    "summary": f"{direction} {f.n_contracts}x {f.symbol} @ {f.fill_price:.4f}",
                    "status": f"slippage {f.actual_slippage:.4f}",
                    "status_color": "ok",
                })
        finally:
            fj3.close()

    ej2 = data.get_experiment_journal()
    if ej2 is not None:
        try:
            for e in ej2.query(limit=5):
                decision = getattr(e, 'promotion_decision', None) or "pending"
                recent_activity.append({
                    "timestamp": getattr(e, 'created_at', '') or '',
                    "type": "Experiment",
                    "badge": "warn",
                    "summary": f"{getattr(e, 'mutation_type', 'unknown')} mutation",
                    "status": decision,
                    "status_color": "ok" if decision == "promoted" else ("alert" if decision == "rejected" else "warn"),
                })
        finally:
            ej2.close()

    # Sort combined activity by timestamp descending
    recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_activity = recent_activity[:10]

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_page": "overview",
            "fill_count": fill_count,
            "experiment_count": experiment_count,
            "agent_state": agent_state,
            "champion_version": champion_version,
            "components": components,
            "slippage_data": slippage_data,
            "recent_activity": recent_activity,
        },
    )


@router.get("/fills")
async def fills(request: Request):
    """Render the fill journal page with slippage chart."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(data_dir, runner)

    fill_count = 0
    fill_list = []
    reconciliation = None
    slippage_data = []
    fj = data.get_fill_journal()
    if fj is not None:
        try:
            fill_count = fj.count()
            fill_list = fj.get_fills(limit=100)
            slippage_data = fj.get_slippage_pairs()
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
            "slippage_data": slippage_data,
        },
    )


@router.get("/agent")
async def agent(request: Request):
    """Render the agent loop page with stats, charts, and detail expansion."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(data_dir, runner)

    from hydra.cli.state import get_state

    agent_state = get_state().value

    experiment_count = 0
    experiments = []
    ej = data.get_experiment_journal()
    if ej is not None:
        try:
            experiment_count = ej.count()
            experiments = ej.query(limit=50)
        finally:
            ej.close()

    exp_summary = data.get_experiment_summary()
    mutation_breakdown = data.get_mutation_breakdown()

    return templates.TemplateResponse(
        request,
        "agent.html",
        {
            "active_page": "agent",
            "agent_state": agent_state,
            "experiments": experiments,
            "experiment_count": experiment_count,
            "exp_summary": exp_summary,
            "mutation_breakdown": mutation_breakdown,
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


@router.get("/model")
async def model(request: Request):
    """Render the model page."""
    templates = request.app.state.templates
    runner = getattr(request.app.state, "runner", None)
    data_dir = request.app.state.data_dir
    data = DashboardData(data_dir, runner)

    champion_version = None
    champion_fitness = None
    try:
        reg = data.get_model_registry()
        if reg is not None:
            info = reg.get_champion_info()
            champion_version = info.get("version")
    except Exception:
        pass

    champion_fitness = data.get_champion_fitness()
    experiment_summary = data.get_experiment_summary()
    mutation_breakdown = data.get_mutation_breakdown()

    return templates.TemplateResponse(
        request,
        "model.html",
        {
            "active_page": "model",
            "champion_version": champion_version,
            "champion_fitness": champion_fitness,
            "experiment_summary": experiment_summary,
            "mutation_breakdown": mutation_breakdown,
        },
    )


@router.get("/data")
async def data_page(request: Request):
    """Render the data & feature store page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates
    runner = getattr(request.app.state, "runner", None)
    data = DashboardData(data_dir, runner)

    fs_stats = data.get_feature_store_stats()
    latest_features = data.get_latest_features(limit=50)

    return templates.TemplateResponse(
        request,
        "data.html",
        {
            "active_page": "data",
            "fs_stats": fs_stats,
            "latest_features": latest_features,
            "ib_futures_status": None,
            "ib_options_status": None,
        },
    )


@router.get("/system")
async def system(request: Request):
    """Render the system health page."""
    data_dir = request.app.state.data_dir
    templates = request.app.state.templates
    runner = getattr(request.app.state, "runner", None)

    from hydra.cli.state import get_state

    agent_state = get_state().value
    data = DashboardData(data_dir, runner)

    components = data.get_component_health()
    circuit_breakers = data.get_circuit_breaker_states()
    disk_usage = data.get_disk_usage()

    reconciliation = None
    fj = data.get_fill_journal()
    if fj is not None:
        try:
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

    return templates.TemplateResponse(
        request,
        "system.html",
        {
            "active_page": "system",
            "agent_state": agent_state,
            "components": components,
            "circuit_breakers": circuit_breakers,
            "disk_usage": disk_usage,
            "data_dir": str(data_dir),
            "reconciliation": reconciliation,
        },
    )
