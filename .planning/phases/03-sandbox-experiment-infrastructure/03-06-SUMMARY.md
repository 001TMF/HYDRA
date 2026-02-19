---
phase: 03-sandbox-experiment-infrastructure
plan: 06
subsystem: cli
tags: [cli, typer, rich, operator-interface]
dependency_graph:
  requires: [03-02, 03-03, 03-04, 03-05]
  provides: [hydra-cli, agent-state-management]
  affects: [phase-4-agent-loop]
tech_stack:
  added: [typer, rich]
  patterns: [typer-cli, rich-formatters, json-state-file]
key_files:
  created:
    - src/hydra/cli/__init__.py
    - src/hydra/cli/app.py
    - src/hydra/cli/formatters.py
    - src/hydra/cli/state.py
    - tests/test_cli.py
  modified:
    - pyproject.toml
    - uv.lock
decisions:
  - Formatters return Rich renderables (Table/Panel) rather than printing directly for testability
  - AgentState defaults to PAUSED when state file missing for safety
  - Diagnose uses synthetic data in Phase 3; live data integration deferred to Phase 4
  - Journal defaults to ~/.hydra/experiment_journal.db when no path provided
metrics:
  duration: 8min
  completed: 2026-02-19T11:37:00Z
  tasks_completed: 2
  tasks_total: 2
  test_count: 7
  test_pass: 7
---

# Phase 03 Plan 06: HYDRA CLI Summary

Typer CLI with 6 Rich-formatted commands (status, diagnose, rollback, pause, run, journal) wiring all Phase 3 sandbox modules into an operator control surface.

## What Was Built

### Task 1: CLI Package with State Management and Rich Formatters
**Commit:** `e48e07c`

- **`src/hydra/cli/state.py`**: `AgentState` enum (RUNNING/PAUSED) with JSON file persistence at `~/.hydra/agent_state.json`. Safe default to PAUSED when file missing.
- **`src/hydra/cli/formatters.py`**: Four Rich formatters that return renderables (not print):
  - `format_status_table()` -- System health table with color-coded metrics
  - `format_drift_report()` -- DriftReport panel with performance/feature/streaming sections
  - `format_journal_table()` -- Experiment records table with colored decisions
  - `format_rollback_result()` -- Rollback confirmation panel
- **`pyproject.toml`**: Added `typer[all]>=0.12` dependency and `[project.scripts] hydra = "hydra.cli.app:app"` entry point

### Task 2: Typer App with 6 Commands and Test Suite
**Commit:** `f53f2e1`

- **`src/hydra/cli/app.py`**: Typer app with all 6 commands:
  - `status` -- Displays champion metrics, experiment count, alerts, autonomy level via ModelRegistry and ExperimentJournal
  - `diagnose` -- Runs drift detection via DriftObserver and displays DriftReport
  - `rollback` -- Calls ModelRegistry.rollback() with version confirmation
  - `pause` -- Sets AgentState.PAUSED via JSON state file
  - `run` -- Sets AgentState.RUNNING via JSON state file
  - `journal` -- Queries ExperimentJournal with --tag, --since, --until, --mutation, --outcome filters
- **`tests/test_cli.py`**: 7 tests using typer.testing.CliRunner:
  1. `test_status_no_champion` -- Fresh registry shows no champion
  2. `test_pause_and_run` -- Toggle agent state with file verification
  3. `test_rollback_no_archived` -- Graceful error handling
  4. `test_journal_empty` -- Empty journal shows no-results message
  5. `test_journal_with_data` -- Pre-populated journal displays records
  6. `test_journal_filtered` -- Mutation type filter works correctly
  7. `test_cli_help` -- Help lists all 6 commands

## Key Integration Points

| CLI Command | Sandbox Module | Method Used |
|-------------|---------------|-------------|
| status | ModelRegistry | `get_champion_info()` |
| status | ExperimentJournal | `count()` |
| diagnose | DriftObserver | `get_full_report()` |
| rollback | ModelRegistry | `rollback()` |
| journal | ExperimentJournal | `query()` |
| pause/run | AgentState | `set_state()` / `get_state()` |

## Decisions Made

1. **Formatters return renderables, not print**: All 4 formatters in `formatters.py` return Rich Table/Panel objects. The calling command does `console.print()`. This makes formatters unit-testable without stdout capture.

2. **PAUSED safe default**: When `~/.hydra/agent_state.json` doesn't exist, `get_state()` returns `AgentState.PAUSED`. The agent loop won't run without explicit operator activation.

3. **Diagnose uses synthetic data in Phase 3**: The `diagnose` command uses random synthetic data to demonstrate drift detection infrastructure. Full diagnostic cycle with live model data is Phase 4 scope.

4. **Journal path defaults to ~/.hydra/**: When `--journal-path` is not provided, journal command uses `~/.hydra/experiment_journal.db` as default location.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

```
tests/test_cli.py::TestCLI::test_status_no_champion PASSED
tests/test_cli.py::TestCLI::test_pause_and_run PASSED
tests/test_cli.py::TestCLI::test_rollback_no_archived PASSED
tests/test_cli.py::TestCLI::test_journal_empty PASSED
tests/test_cli.py::TestCLI::test_journal_with_data PASSED
tests/test_cli.py::TestCLI::test_journal_filtered PASSED
tests/test_cli.py::TestCLI::test_cli_help PASSED
======================== 7 passed in 2.33s =========================
```

All success criteria met:
1. `status` shows champion health, experiment count, alerts, autonomy level in Rich table
2. `diagnose` triggers drift check and displays DriftReport
3. `rollback` swaps champion to archived version with confirmation
4. `pause` and `run` toggle agent state via JSON file
5. `journal` queries with --tag, --since, --until, --mutation, --outcome filters
6. All output uses Rich tables, panels, and colored text (CLI-06)
7. All 7 tests pass

## Self-Check: PASSED

- All 5 created files verified on disk
- Commit e48e07c verified in git log
- Commit f53f2e1 verified in git log
