---
phase: "03-sandbox-experiment-infrastructure"
plan: "02"
subsystem: "model-registry"
tags: ["mlflow", "model-lifecycle", "registry", "champion-candidate"]
dependency_graph:
  requires: ["lightgbm", "mlflow"]
  provides: ["ModelRegistry"]
  affects: ["sandbox", "cli", "agent-loop"]
tech_stack:
  added: ["mlflow>=3.9"]
  patterns: ["alias-based lifecycle", "explicit logging", "local file backend"]
key_files:
  created:
    - src/hydra/sandbox/registry.py
    - tests/test_registry.py
  modified:
    - pyproject.toml
    - src/hydra/sandbox/__init__.py
decisions:
  - "Absolute file:// tracking URI default prevents MLflow relative-path confusion"
  - "Alias-based lifecycle (champion/archived) instead of deprecated MLflow stages"
  - "Explicit logging (not autolog) for full control over what gets tracked"
metrics:
  duration: "3min"
  completed: "2026-02-19"
---

# Phase 3 Plan 2: MLflow Model Registry Summary

MLflow model registry wrapper with alias-based champion/candidate/archived lifecycle, explicit logging of LightGBM models with full config snapshots and metrics, and rollback via archived alias swap.

## What Was Built

### ModelRegistry (`src/hydra/sandbox/registry.py`)

Thin wrapper over MLflow enforcing HYDRA's model lifecycle conventions:

- **`log_candidate(model, metrics, config, tags, run_name)`** -- Starts an MLflow run, logs params/metrics/tags explicitly, logs the underlying LGBMClassifier via `mlflow.lightgbm.log_model`, registers the model version. Returns `(run_id, version_number)`.

- **`promote_to_champion(version)`** -- Sets the "champion" alias on the specified version. If a champion already exists, it first gets the "archived" alias (single rollback target).

- **`rollback()`** -- Swaps champion alias to the previously archived version. Raises `ValueError` if no archived version exists.

- **`load_champion()`** -- Loads the champion model via `mlflow.lightgbm.load_model("models:/{name}@champion")`. Returns a LightGBM Booster ready for prediction.

- **`get_champion_info()`** -- Returns dict with version, run_id, metrics, tags, and creation timestamp from the champion's MLflow run.

- **`list_versions()`** -- Lists all registered model versions with their aliases.

Default tracking URI uses absolute path (`file://{project_root}/mlruns`) to prevent the relative-path pitfall documented in research.

### Test Suite (`tests/test_registry.py`)

6 tests using real MLflow operations against temporary directories:

| Test | What It Validates |
|------|-------------------|
| `test_log_candidate` | Returns valid (run_id, version) tuple |
| `test_promote_and_load_champion` | Promoted model loads and predicts on dummy data |
| `test_promote_archives_previous` | Promoting B archives A with "archived" alias |
| `test_rollback` | Rollback restores previously archived version |
| `test_rollback_no_archived_raises` | ValueError when no archived model exists |
| `test_list_versions` | Returns correct structure for multiple candidates |

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **Absolute tracking URI default**: `file://{project_root}/mlruns` using `pathlib.Path(__file__).resolve().parents[3]` to compute project root. Prevents MLflow's known relative-path confusion.

2. **Alias-based lifecycle**: Uses `set_registered_model_alias` / `get_model_version_by_alias` instead of deprecated `transition_model_version_stage`. Future-proof per MLflow 3.x migration.

3. **Explicit logging over autolog**: `mlflow.log_params`, `mlflow.log_metrics`, `mlflow.set_tag` called individually for full control over what gets tracked. No hidden autolog side effects.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | `97f5aaa` | ModelRegistry class with 6 methods, mlflow dependency, sandbox export |
| 2 | `d05c96d` | 6 tests covering logging, promotion, archival, rollback, error handling |

## Key Files

| File | Purpose |
|------|---------|
| `src/hydra/sandbox/registry.py` | ModelRegistry wrapper over MLflow with lifecycle conventions |
| `tests/test_registry.py` | 6 tests with real MLflow operations against temp directories |
| `pyproject.toml` | mlflow>=3.9 dependency (shared commit with 03-04) |
| `src/hydra/sandbox/__init__.py` | ModelRegistry added to sandbox exports |

## Self-Check: PASSED

- FOUND: src/hydra/sandbox/registry.py
- FOUND: tests/test_registry.py
- FOUND: commit 97f5aaa (Task 1)
- FOUND: commit d05c96d (Task 2)
