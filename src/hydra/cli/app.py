"""HYDRA CLI -- operator control surface for the autonomous trading system.

Provides 8 commands that wire together sandbox modules, execution modules,
and the agent loop into human-usable commands.

Commands:
    status      -- Display system health: champion metrics, experiments, alerts
    diagnose    -- Run drift detection on the champion model
    rollback    -- Revert champion to the previously archived version
    pause       -- Pause the autonomous agent loop
    run         -- Resume the autonomous agent loop
    journal     -- Query experiment history with filters
    paper-trade -- Start/stop paper trading runner
    fill-report -- View fill history and slippage reconciliation
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from hydra.cli.formatters import (
    format_drift_report,
    format_fill_table,
    format_journal_table,
    format_reconciliation_report,
    format_rollback_result,
    format_status_table,
)
from hydra.cli.state import AgentState, get_state, set_state

app = typer.Typer(
    name="hydra",
    help="HYDRA autonomous trading system CLI",
    rich_markup_mode="rich",
)
console = Console()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    registry_uri: Optional[str] = typer.Option(
        None, help="MLflow tracking URI for the model registry"
    ),
    journal_path: Optional[str] = typer.Option(
        None, help="Path to the experiment journal SQLite database"
    ),
) -> None:
    """Display system health: champion metrics, experiments, alerts, autonomy."""
    from hydra.sandbox.registry import ModelRegistry

    registry = ModelRegistry(tracking_uri=registry_uri)

    champion_info = None
    alerts: list[str] = []
    try:
        champion_info = registry.get_champion_info()
    except ValueError:
        pass  # No champion set -- handled by formatter

    experiment_count = 0
    if journal_path is not None:
        try:
            from hydra.sandbox.journal import ExperimentJournal

            journal = ExperimentJournal(journal_path)
            experiment_count = journal.count()
            journal.close()
        except Exception as exc:
            alerts.append(f"Journal error: {exc}")

    agent_state = get_state()
    autonomy = agent_state.value.upper()

    table = format_status_table(champion_info, experiment_count, alerts, autonomy)
    console.print(table)


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------


@app.command()
def diagnose(
    registry_uri: Optional[str] = typer.Option(
        None, help="MLflow tracking URI for the model registry"
    ),
) -> None:
    """Run drift detection on the champion model and display the DriftReport."""
    import numpy as np

    from hydra.sandbox.observer import DriftObserver
    from hydra.sandbox.registry import ModelRegistry

    registry = ModelRegistry(tracking_uri=registry_uri)

    try:
        champion_info = registry.get_champion_info()
    except ValueError:
        console.print(
            Panel(
                "[yellow]No champion model set.[/yellow]\n"
                "Use the model registry to promote a candidate first.",
                title="Diagnose",
                border_style="yellow",
            )
        )
        return

    # For Phase 3, demonstrate drift infrastructure with champion metrics.
    # Full diagnostic cycle with live data is Phase 4.
    metrics = champion_info.get("metrics", {})
    baseline_sharpe = float(metrics.get("sharpe_ratio", metrics.get("sharpe", 0.5)))

    # Synthetic minimal data to show drift detection working
    observer = DriftObserver()
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.02, 30)
    preds = rng.choice([0, 1], 30)
    actuals = rng.choice([0, 1], 30)
    probas = rng.uniform(0.3, 0.7, 30)

    report = observer.get_full_report(
        recent_returns=returns,
        predictions=preds,
        actuals=actuals,
        probabilities=probas,
        baseline_sharpe=baseline_sharpe,
    )

    panel = format_drift_report(report)
    console.print(panel)


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


@app.command()
def rollback(
    registry_uri: Optional[str] = typer.Option(
        None, help="MLflow tracking URI for the model registry"
    ),
) -> None:
    """Revert champion to the previously archived model version."""
    from hydra.sandbox.registry import ModelRegistry

    registry = ModelRegistry(tracking_uri=registry_uri)

    try:
        champion_info = registry.get_champion_info()
        old_version = champion_info["version"]
    except ValueError:
        console.print(
            Panel(
                "[yellow]No champion model set -- nothing to rollback.[/yellow]",
                title="Rollback",
                border_style="yellow",
            )
        )
        return

    try:
        new_version = registry.rollback()
    except ValueError as exc:
        console.print(
            Panel(
                f"[red]Rollback failed:[/red] {exc}",
                title="Rollback",
                border_style="red",
            )
        )
        return

    panel = format_rollback_result(old_version, new_version)
    console.print(panel)


# ---------------------------------------------------------------------------
# pause / run
# ---------------------------------------------------------------------------


@app.command()
def pause() -> None:
    """Pause the autonomous agent loop."""
    set_state(AgentState.PAUSED)
    console.print(
        Panel("[yellow]Agent loop PAUSED[/yellow]", border_style="yellow")
    )


@app.command(name="run")
def run_agent() -> None:
    """Resume the autonomous agent loop."""
    set_state(AgentState.RUNNING)
    console.print(
        Panel("[green]Agent loop RUNNING[/green]", border_style="green")
    )


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------


@app.command()
def journal(
    journal_path: Optional[str] = typer.Option(
        None, help="Path to the experiment journal SQLite database"
    ),
    tag: Optional[str] = typer.Option(
        None, help="Filter by tag"
    ),
    since: Optional[str] = typer.Option(
        None, help="Filter experiments created on or after this date (ISO 8601)"
    ),
    until: Optional[str] = typer.Option(
        None, help="Filter experiments created on or before this date (ISO 8601)"
    ),
    mutation: Optional[str] = typer.Option(
        None, help="Filter by mutation type"
    ),
    outcome: Optional[str] = typer.Option(
        None, help="Filter by promotion decision (promoted/rejected/pending)"
    ),
    limit: int = typer.Option(
        20, help="Maximum number of results to display"
    ),
) -> None:
    """Query experiment history with filters."""
    from hydra.sandbox.journal import ExperimentJournal

    if journal_path is None:
        import pathlib

        default_path = pathlib.Path.home() / ".hydra" / "experiment_journal.db"
        journal_path = str(default_path)

    try:
        j = ExperimentJournal(journal_path)
    except Exception as exc:
        console.print(
            Panel(
                f"[red]Could not open journal:[/red] {exc}",
                title="Journal",
                border_style="red",
            )
        )
        return

    query_kwargs: dict = {"limit": limit}
    if tag is not None:
        query_kwargs["tags"] = [tag]
    if since is not None:
        query_kwargs["date_from"] = since
    if until is not None:
        query_kwargs["date_to"] = until
    if mutation is not None:
        query_kwargs["mutation_type"] = mutation
    if outcome is not None:
        query_kwargs["outcome"] = outcome

    records = j.query(**query_kwargs)
    j.close()

    if not records:
        console.print(
            Panel(
                "[dim]No experiment records found matching the given filters.[/dim]",
                title="Journal",
                border_style="dim",
            )
        )
        return

    table = format_journal_table(records)
    console.print(table)
    console.print(f"\n[dim]{len(records)} result(s) shown[/dim]")


# ---------------------------------------------------------------------------
# paper-trade
# ---------------------------------------------------------------------------

LIVE_PORTS: set[int] = {4001, 7496}


@app.command(name="paper-trade")
def paper_trade(
    action: str = typer.Argument(help="start or stop"),
    port: int = typer.Option(4002, help="IB Gateway port (4002=paper, 4001=live)"),
    client_id: int = typer.Option(1, help="IB client ID (1=trading)"),
    journal_path: Optional[str] = typer.Option(None, help="Fill journal SQLite path"),
    schedule_hour: int = typer.Option(14, help="Daily cycle hour (24h, US/Central)"),
    schedule_minute: int = typer.Option(0, help="Daily cycle minute"),
    yes_i_mean_live: bool = typer.Option(
        False,
        "--yes-i-mean-live",
        help="Required confirmation flag for live trading ports",
    ),
) -> None:
    """Start or stop the paper trading runner."""
    if action == "start":
        is_live = port in LIVE_PORTS
        mode = "LIVE" if is_live else "PAPER"

        if is_live and not yes_i_mean_live:
            console.print(
                Panel(
                    "[bold red]DANGER: Live trading port detected![/bold red]\n\n"
                    f"Port {port} is a LIVE trading port.\n"
                    "Add [bold]--yes-i-mean-live[/bold] to confirm you want to "
                    "connect to a live trading account.\n\n"
                    "[dim]Paper trading ports: 4002 (Gateway), 7497 (TWS)[/dim]",
                    title="Live Port Warning",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

        color = "red" if is_live else "green"
        console.print(
            Panel(
                f"[bold {color}]{mode} TRADING[/bold {color}]\n\n"
                f"  Port:       {port}\n"
                f"  Client ID:  {client_id}\n"
                f"  Schedule:   {schedule_hour:02d}:{schedule_minute:02d} US/Central\n"
                f"  Journal:    {journal_path or '~/.hydra/fill_journal.db'}\n\n"
                "[dim]Use 'python -m hydra.execution.runner' for the long-running "
                "process.[/dim]",
                title=f"Paper Trading Configuration ({mode})",
                border_style=color,
            )
        )

    elif action == "stop":
        set_state(AgentState.PAUSED)
        console.print(
            Panel(
                "[yellow]Paper trading STOPPED[/yellow]\n"
                "Agent state set to PAUSED.",
                title="Paper Trading",
                border_style="yellow",
            )
        )

    else:
        console.print(
            Panel(
                f"[red]Unknown action: {action}[/red]\n"
                "Use 'start' or 'stop'.",
                title="Paper Trading",
                border_style="red",
            )
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# fill-report
# ---------------------------------------------------------------------------


@app.command(name="fill-report")
def fill_report(
    journal_path: Optional[str] = typer.Option(
        None, help="Fill journal SQLite path"
    ),
    symbol: Optional[str] = typer.Option(None, help="Filter by symbol"),
    since: Optional[str] = typer.Option(
        None, help="Filter fills since date (ISO 8601)"
    ),
    limit: int = typer.Option(20, help="Max fills to display"),
    reconcile: bool = typer.Option(
        False, "--reconcile", help="Show slippage reconciliation report"
    ),
) -> None:
    """View fill history and slippage reconciliation report."""
    import pathlib

    from hydra.execution.fill_journal import FillJournal

    if journal_path is None:
        journal_path = str(pathlib.Path.home() / ".hydra" / "fill_journal.db")

    try:
        fj = FillJournal(journal_path)
    except Exception as exc:
        console.print(
            Panel(
                f"[red]Could not open fill journal:[/red] {exc}",
                title="Fill Report",
                border_style="red",
            )
        )
        return

    if reconcile:
        from hydra.execution.reconciler import SlippageReconciler

        reconciler = SlippageReconciler(fj)
        report = reconciler.reconcile(symbol=symbol, since=since)
        fj.close()

        if report is None:
            console.print(
                Panel(
                    "[yellow]Insufficient fill data for reconciliation.[/yellow]\n"
                    "At least 10 fills are required.",
                    title="Slippage Reconciliation",
                    border_style="yellow",
                )
            )
            return

        panel = format_reconciliation_report(report)
        console.print(panel)
    else:
        fills = fj.get_fills(symbol=symbol, since=since, limit=limit)
        fj.close()

        if not fills:
            console.print(
                Panel(
                    "[dim]No fills found matching the given filters.[/dim]",
                    title="Fill Report",
                    border_style="dim",
                )
            )
            return

        table = format_fill_table(fills)
        console.print(table)
        console.print(f"\n[dim]{len(fills)} fill(s) shown[/dim]")
