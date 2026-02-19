"""Tests for HYDRA CLI commands.

Covers all 6 commands (status, diagnose, rollback, pause, run, journal)
using typer.testing.CliRunner with tmp_path-based paths for isolation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

from hydra.cli.app import app
from hydra.cli.state import AgentState, STATE_FILE
from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord

runner = CliRunner()


def _make_record(**overrides) -> ExperimentRecord:
    """Factory for ExperimentRecord with sensible defaults."""
    defaults = dict(
        hypothesis="Test hypothesis",
        mutation_type="hyperparameter",
        config_diff={"lr": 0.05},
        results={"sharpe_ratio": 0.8, "max_drawdown": -0.05},
        promotion_decision="rejected",
        tags=["test"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


class TestCLI:
    """Tests for all CLI commands."""

    def test_status_no_champion(self, tmp_path):
        """Status with fresh (empty) registry shows 'None' for champion."""
        uri = f"file://{tmp_path / 'mlruns'}"
        result = runner.invoke(app, ["status", "--registry-uri", uri])
        assert result.exit_code == 0
        assert "None" in result.output or "NOT SET" in result.output

    def test_pause_and_run(self, tmp_path, monkeypatch):
        """Pause sets PAUSED state, run sets RUNNING state."""
        state_file = tmp_path / "agent_state.json"
        monkeypatch.setattr("hydra.cli.state.STATE_FILE", state_file)
        monkeypatch.setattr("hydra.cli.app.set_state", lambda s: _patched_set_state(s, state_file))
        monkeypatch.setattr("hydra.cli.app.get_state", lambda: _patched_get_state(state_file))

        result = runner.invoke(app, ["pause"])
        assert result.exit_code == 0
        assert "PAUSED" in result.output

        # Verify state file written
        data = json.loads(state_file.read_text())
        assert data["state"] == "paused"

        result = runner.invoke(app, ["run"])
        assert result.exit_code == 0
        assert "RUNNING" in result.output

        data = json.loads(state_file.read_text())
        assert data["state"] == "running"

    def test_rollback_no_archived(self, tmp_path):
        """Rollback with no archived model shows error message gracefully."""
        uri = f"file://{tmp_path / 'mlruns'}"
        result = runner.invoke(app, ["rollback", "--registry-uri", uri])
        assert result.exit_code == 0
        # Either 'nothing to rollback' or 'No champion'
        assert "rollback" in result.output.lower() or "champion" in result.output.lower()

    def test_journal_empty(self, tmp_path):
        """Journal with empty database shows no-results message."""
        db_path = str(tmp_path / "journal.db")
        result = runner.invoke(app, ["journal", "--journal-path", db_path])
        assert result.exit_code == 0
        assert "No experiment records" in result.output or "0" in result.output

    def test_journal_with_data(self, tmp_path):
        """Journal with pre-populated data shows experiment records."""
        db_path = str(tmp_path / "journal.db")
        journal = ExperimentJournal(db_path)
        for i in range(3):
            journal.log_experiment(
                _make_record(
                    hypothesis=f"Experiment {i}",
                    mutation_type="hyperparameter",
                    results={"sharpe_ratio": 0.5 + i * 0.2},
                    promotion_decision=["promoted", "rejected", "pending"][i],
                )
            )
        journal.close()

        result = runner.invoke(app, ["journal", "--journal-path", db_path])
        assert result.exit_code == 0
        assert "Experiment 0" in result.output
        assert "Experiment 1" in result.output
        assert "Experiment 2" in result.output
        assert "3 result" in result.output

    def test_journal_filtered(self, tmp_path):
        """Journal with --mutation filter shows only matching records."""
        db_path = str(tmp_path / "journal.db")
        journal = ExperimentJournal(db_path)
        journal.log_experiment(
            _make_record(
                hypothesis="HP tune",
                mutation_type="hyperparameter",
            )
        )
        journal.log_experiment(
            _make_record(
                hypothesis="Add feature X",
                mutation_type="feature_add",
            )
        )
        journal.log_experiment(
            _make_record(
                hypothesis="HP tune 2",
                mutation_type="hyperparameter",
            )
        )
        journal.close()

        result = runner.invoke(
            app, ["journal", "--journal-path", db_path, "--mutation", "hyperparameter"]
        )
        assert result.exit_code == 0
        assert "HP tune" in result.output
        # feature_add record should not appear
        assert "Add feature X" not in result.output
        assert "2 result" in result.output

    def test_cli_help(self):
        """--help lists all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "status" in result.output
        assert "diagnose" in result.output
        assert "rollback" in result.output
        assert "pause" in result.output
        assert "run" in result.output
        assert "journal" in result.output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patched_set_state(state: AgentState, state_file) -> None:
    """Write state to a custom path (for test isolation)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state.value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_file.write_text(json.dumps(payload, indent=2))


def _patched_get_state(state_file) -> AgentState:
    """Read state from a custom path (for test isolation)."""
    if not state_file.exists():
        return AgentState.PAUSED
    try:
        data = json.loads(state_file.read_text())
        return AgentState(data["state"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return AgentState.PAUSED
