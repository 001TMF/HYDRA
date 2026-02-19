"""Rich output formatters for HYDRA CLI.

Each function accepts plain data and returns a Rich renderable (Table,
Panel, etc.).  The caller is responsible for printing via
``console.print()``.  This separation keeps the formatters testable
without capturing stdout.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def format_status_table(
    champion_info: dict | None,
    experiment_count: int,
    alerts: list[str],
    autonomy: str,
) -> Table:
    """Build a Rich Table showing system health at a glance.

    Parameters
    ----------
    champion_info : dict | None
        Output of ``ModelRegistry.get_champion_info()``, or None if no
        champion is set.  Expected keys: ``version``, ``metrics``.
    experiment_count : int
        Total experiments logged in the journal.
    alerts : list[str]
        Active alert strings (e.g. from drift detection).
    autonomy : str
        Current agent state label (e.g. "RUNNING", "PAUSED").
    """
    table = Table(title="HYDRA System Status", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Status", justify="center")

    if champion_info is not None:
        metrics = champion_info.get("metrics", {})
        version = champion_info.get("version", "?")

        table.add_row(
            "Champion Version",
            str(version),
            "[green]SET[/green]",
        )

        sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe"))
        if sharpe is not None:
            sharpe_val = float(sharpe)
            status = (
                "[green]OK[/green]" if sharpe_val > 0.5
                else "[yellow]WATCH[/yellow]" if sharpe_val > 0
                else "[red]ALERT[/red]"
            )
            table.add_row("Sharpe Ratio", f"{sharpe_val:.3f}", status)

        drawdown = metrics.get("max_drawdown", metrics.get("drawdown"))
        if drawdown is not None:
            dd_val = float(drawdown)
            status = (
                "[green]OK[/green]" if dd_val > -0.10
                else "[yellow]WATCH[/yellow]" if dd_val > -0.20
                else "[red]ALERT[/red]"
            )
            table.add_row("Max Drawdown", f"{dd_val:.2%}", status)

        hit_rate = metrics.get("hit_rate", metrics.get("accuracy"))
        if hit_rate is not None:
            hr_val = float(hit_rate)
            status = (
                "[green]OK[/green]" if hr_val > 0.55
                else "[yellow]WATCH[/yellow]" if hr_val > 0.45
                else "[red]ALERT[/red]"
            )
            table.add_row("Hit Rate", f"{hr_val:.1%}", status)
    else:
        table.add_row(
            "Champion",
            "None",
            "[yellow]NOT SET[/yellow]",
        )

    table.add_row(
        "Active Experiments",
        str(experiment_count),
        "[green]OK[/green]" if experiment_count < 10 else "[yellow]BUSY[/yellow]",
    )

    alert_status = (
        "[red]ALERT[/red]" if alerts
        else "[green]OK[/green]"
    )
    table.add_row(
        "Alerts",
        str(len(alerts)) if alerts else "0",
        alert_status,
    )

    autonomy_color = "green" if autonomy == "RUNNING" else "yellow"
    table.add_row(
        "Autonomy",
        autonomy,
        f"[{autonomy_color}]{autonomy}[/{autonomy_color}]",
    )

    return table


def format_drift_report(report) -> Panel:
    """Render a DriftReport as a Rich Panel.

    Parameters
    ----------
    report : DriftReport
        Combined drift report from ``DriftObserver.get_full_report()``.
    """
    lines: list[str] = []

    # Performance section
    perf = report.performance
    lines.append("[bold]Performance Metrics[/bold]")
    lines.append(
        f"  Sharpe Ratio:  {perf.sharpe_ratio:.3f}  "
        f"{'[red]DEGRADED[/red]' if perf.sharpe_degraded else '[green]OK[/green]'}"
    )
    lines.append(
        f"  Max Drawdown:  {perf.max_drawdown:.2%}  "
        f"{'[red]ALERT[/red]' if perf.drawdown_alert else '[green]OK[/green]'}"
    )
    lines.append(
        f"  Hit Rate:      {perf.hit_rate:.1%}  "
        f"{'[red]DEGRADED[/red]' if perf.hit_rate_degraded else '[green]OK[/green]'}"
    )
    lines.append(f"  Calibration:   {perf.calibration:.4f}")

    # Feature drift section
    if report.feature is not None:
        lines.append("")
        lines.append("[bold]Feature Drift[/bold]")
        if report.feature.drifted_features:
            for feat in report.feature.drifted_features:
                psi = report.feature.psi_scores.get(feat, 0.0)
                lines.append(f"  [red]{feat}[/red]: PSI={psi:.3f}")
        else:
            lines.append("  [green]No feature drift detected[/green]")

    # Streaming alerts
    if report.streaming_alerts:
        lines.append("")
        lines.append("[bold]Streaming Alerts[/bold]")
        for name, alerted in report.streaming_alerts.items():
            status = "[red]DRIFT[/red]" if alerted else "[green]OK[/green]"
            lines.append(f"  {name}: {status}")

    # Overall status
    lines.append("")
    if report.needs_diagnosis:
        lines.append("[bold red]DRIFT DETECTED -- Diagnosis recommended[/bold red]")
    else:
        lines.append("[bold green]HEALTHY -- No drift detected[/bold green]")

    return Panel("\n".join(lines), title="Drift Report", border_style="blue")


def format_journal_table(records: list) -> Table:
    """Render experiment journal records as a Rich Table.

    Parameters
    ----------
    records : list[ExperimentRecord]
        Journal records from ``ExperimentJournal.query()``.
    """
    table = Table(title="Experiment Journal", show_lines=True)
    table.add_column("ID", justify="right", style="dim")
    table.add_column("Date", style="cyan")
    table.add_column("Hypothesis", max_width=40)
    table.add_column("Mutation", style="magenta")
    table.add_column("Decision", justify="center")
    table.add_column("Sharpe", justify="right")

    for rec in records:
        # Truncate hypothesis to 40 chars
        hypothesis = rec.hypothesis
        if len(hypothesis) > 40:
            hypothesis = hypothesis[:37] + "..."

        # Color code decision
        decision = rec.promotion_decision
        if decision == "promoted":
            decision_styled = "[green]promoted[/green]"
        elif decision == "rejected":
            decision_styled = "[red]rejected[/red]"
        else:
            decision_styled = "[yellow]pending[/yellow]"

        # Extract Sharpe from results dict
        sharpe = rec.results.get("sharpe_ratio", rec.results.get("sharpe", ""))
        sharpe_str = f"{float(sharpe):.3f}" if sharpe != "" else "-"

        table.add_row(
            str(rec.id or ""),
            rec.created_at[:10] if rec.created_at else "",
            hypothesis,
            rec.mutation_type,
            decision_styled,
            sharpe_str,
        )

    return table


def format_rollback_result(old_version: int, new_version: int) -> Panel:
    """Render a rollback confirmation as a Rich Panel.

    Parameters
    ----------
    old_version : int
        The champion version before rollback.
    new_version : int
        The restored champion version after rollback.
    """
    text = Text.from_markup(
        f"[bold]Rollback Complete[/bold]\n\n"
        f"  Previous champion: v{old_version}\n"
        f"  Restored champion: [green]v{new_version}[/green]\n"
    )
    return Panel(text, title="Model Rollback", border_style="yellow")


def format_fill_table(fills: list) -> Table:
    """Render fill records as a Rich Table.

    Parameters
    ----------
    fills : list[FillRecord]
        Fill records from ``FillJournal.get_fills()``.
    """
    table = Table(title="Fill Report", show_lines=True)
    table.add_column("Time", style="cyan")
    table.add_column("Symbol", style="bold")
    table.add_column("Dir", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Order Price", justify="right")
    table.add_column("Fill Price", justify="right")
    table.add_column("Pred Slip", justify="right")
    table.add_column("Actual Slip", justify="right")
    table.add_column("Latency", justify="right")

    for fill in fills:
        # Color-code direction
        if fill.direction == 1:
            dir_styled = "[green]BUY[/green]"
        else:
            dir_styled = "[red]SELL[/red]"

        # Timestamp: show first 19 chars (date + time)
        ts = fill.timestamp[:19] if len(fill.timestamp) > 19 else fill.timestamp

        table.add_row(
            ts,
            fill.symbol,
            dir_styled,
            str(fill.n_contracts),
            f"{fill.order_price:.2f}",
            f"{fill.fill_price:.2f}",
            f"{fill.predicted_slippage:.4f}",
            f"{fill.actual_slippage:.4f}",
            f"{fill.fill_latency_ms:.0f}ms",
        )

    return table


def format_reconciliation_report(report) -> Panel:
    """Render a ReconciliationReport as a Rich Panel.

    Parameters
    ----------
    report : ReconciliationReport
        Output of ``SlippageReconciler.reconcile()``.
    """
    lines: list[str] = []

    lines.append("[bold]Slippage Reconciliation[/bold]")
    lines.append("")
    lines.append(f"  N Fills:           {report.n_fills}")
    lines.append(f"  Mean Predicted:    {report.mean_predicted:.4f}")
    lines.append(f"  Mean Actual:       {report.mean_actual:.4f}")

    # Color-code bias
    abs_bias = abs(report.bias)
    if abs_bias < 0.1:
        bias_color = "green"
    elif abs_bias < 0.5:
        bias_color = "yellow"
    else:
        bias_color = "red"
    lines.append(
        f"  Bias:              [{bias_color}]{report.bias:.4f}[/{bias_color}]"
    )

    lines.append(f"  RMSE:              {report.rmse:.4f}")
    lines.append(f"  Correlation:       {report.correlation:.4f}")
    lines.append(
        f"  Pessimism Mult:    {report.pessimism_multiplier:.2f}"
    )

    # Recommendation
    lines.append("")
    if report.pessimism_multiplier > 1.5:
        lines.append(
            "[bold red]WARNING: Paper fills are significantly optimistic.[/bold red]"
        )
        lines.append(
            f"  Actual slippage is {report.pessimism_multiplier:.1f}x the "
            "predicted slippage."
        )
        lines.append(
            "  Consider increasing the impact coefficient in the slippage model."
        )
    elif report.pessimism_multiplier > 1.2:
        lines.append(
            "[yellow]WATCH: Paper fills are somewhat optimistic.[/yellow]"
        )
    else:
        lines.append(
            "[green]Slippage model is well-calibrated.[/green]"
        )

    return Panel(
        "\n".join(lines),
        title="Slippage Reconciliation Report",
        border_style="blue",
    )
