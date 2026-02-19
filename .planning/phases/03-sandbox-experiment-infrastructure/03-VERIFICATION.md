---
phase: 03-sandbox-experiment-infrastructure
verified: 2026-02-19T12:00:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
---

# Phase 3: Sandbox & Experiment Infrastructure Verification Report

**Phase Goal:** A safe experimentation environment exists where models can be trained, evaluated, versioned, and compared through replay of historical data with realistic slippage, and an operator can inspect and control the system via CLI

**Verified:** 2026-02-19
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Replay engine feeds historical bars one-by-one through a pre-trained model with volume-adaptive slippage and produces a ReplayResult with equity curve, trade log, and metrics | VERIFIED | `replay.py` loop at lines 174-262: per-bar `estimate_slippage(order_size=n_contracts, daily_volume=volumes[i], spread=spreads[i])` with unique i per bar; ReplayResult includes equity_curve, trade_log, sharpe_ratio, max_drawdown, hit_rate |
| 2  | Slippage varies per bar based on that bar's volume and spread, not a static estimate | VERIFIED | `test_slippage_varies_with_volume` passes: two replays with high/low volumes produce different total slippage sums |
| 3  | Observer callbacks receive trade events during replay | VERIFIED | `add_callback()` in replay.py line 97; callbacks invoked for every trade at line 261; `test_callback_receives_events` passes |
| 4  | Models are logged to MLflow with full config snapshot, all metrics, and custom tags | VERIFIED | `registry.py` `log_candidate()` calls `mlflow.log_params(config)`, `mlflow.log_metrics(metrics)`, `mlflow.lightgbm.log_model(model.model, ...)` |
| 5  | Champion/candidate/archived lifecycle enforced via MLflow aliases | VERIFIED | `promote_to_champion()` calls `set_registered_model_alias("champion", ...)` and `set_registered_model_alias("archived", ...)`; `test_promote_archives_previous` passes |
| 6  | Promoting a candidate to champion automatically archives the current champion | VERIFIED | `promote_to_champion()` first archives current before setting new champion (lines 114-128 in registry.py) |
| 7  | Rollback swaps champion alias to the previously archived version | VERIFIED | `rollback()` calls `get_model_version_by_alias("archived")` then `promote_to_champion()`; `test_rollback` passes |
| 8  | Every experiment is logged with hypothesis, config diff, results, promotion decision, and tags; journal queryable by tag, date range, mutation type, and outcome | VERIFIED | ExperimentRecord dataclass has all required fields; `query()` supports all four filter types with AND logic; 10 journal tests all pass |
| 9  | PSI, KS, CUSUM, ADWIN detect distribution drift with correct algorithms | VERIFIED | PSI uses quantile bins + epsilon smoothing; KS wraps scipy.stats.ks_2samp; CUSUM uses standard update formula; ADWIN wraps river.drift.ADWIN; 10 drift unit tests pass |
| 10 | DriftObserver combines performance drift and feature distribution drift into a unified DriftReport with needs_diagnosis flag | VERIFIED | `get_full_report()` calls check_performance_drift + check_feature_drift + streaming alerts; sets needs_diagnosis; 5 observer tests pass |
| 11 | Composite fitness score combines 6 metrics with specified weights (Sharpe 0.25, drawdown 0.20, calibration 0.15, robustness 0.15, slippage-adjusted return 0.15, simplicity 0.10) with min-max normalization | VERIFIED | DEFAULT_WEIGHTS constant matches spec exactly; `score()` normalizes each metric via np.clip((raw-lo)/(hi-lo), 0, 1) with calibration inverted; 13 evaluator tests pass |
| 12 | Operator CLI commands (status, diagnose, rollback, pause, run, journal) use Rich formatting and are wired to all sandbox modules | VERIFIED | app.py imports and calls ModelRegistry, ExperimentJournal, DriftObserver; formatters return Rich Table/Panel; entry point `hydra = "hydra.cli.app:app"` in pyproject.toml; 7 CLI tests pass |

**Score:** 12/12 truths verified

---

### Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `src/hydra/sandbox/__init__.py` | VERIFIED | Package init exists; sandbox modules importable |
| `src/hydra/sandbox/replay.py` | VERIFIED | MarketReplayEngine, ReplayResult, TradeEvent present; 408 lines of substantive implementation |
| `src/hydra/sandbox/registry.py` | VERIFIED | ModelRegistry with log_candidate, promote_to_champion, rollback, load_champion, get_champion_info, list_versions |
| `src/hydra/sandbox/journal.py` | VERIFIED | ExperimentJournal with SQLite WAL, full query layer; ExperimentRecord dataclass |
| `src/hydra/sandbox/evaluator.py` | VERIFIED | CompositeEvaluator with DEFAULT_WEIGHTS matching spec, FitnessScore dataclass |
| `src/hydra/sandbox/observer.py` | VERIFIED | DriftObserver, DriftReport, PerformanceDriftReport, FeatureDriftReport |
| `src/hydra/sandbox/drift/psi.py` | VERIFIED | compute_psi with quantile bins and epsilon smoothing |
| `src/hydra/sandbox/drift/ks.py` | VERIFIED | check_ks_drift wrapping scipy.stats.ks_2samp |
| `src/hydra/sandbox/drift/cusum.py` | VERIFIED | CUSUMDetector with update/reset/drift_detected |
| `src/hydra/sandbox/drift/adwin.py` | VERIFIED | ADWINDetector wrapping river.drift.ADWIN |
| `src/hydra/cli/app.py` | VERIFIED | Typer app with all 6 commands |
| `src/hydra/cli/formatters.py` | VERIFIED | format_status_table, format_drift_report, format_journal_table, format_rollback_result — all return Rich renderables |
| `src/hydra/cli/state.py` | VERIFIED | AgentState enum, get_state/set_state via JSON file |
| `tests/test_replay.py` | VERIFIED | 5 tests, all pass |
| `tests/test_registry.py` | VERIFIED | 6 tests, all pass |
| `tests/test_journal.py` | VERIFIED | 10 tests, all pass |
| `tests/test_drift.py` | VERIFIED | 10 tests, all pass |
| `tests/test_observer.py` | VERIFIED | 5 tests, all pass |
| `tests/test_evaluator.py` | VERIFIED | 13 tests, all pass |
| `tests/test_cli.py` | VERIFIED | 7 tests, all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `sandbox/replay.py` | `risk/slippage.py` | `estimate_slippage()` per bar | WIRED | Lines 21-22 import; line 213 calls with per-bar volumes[i]/spreads[i] |
| `sandbox/replay.py` | `risk/position_sizing.py` | `fractional_kelly`, `volume_capped_position` | WIRED | Lines 21-22 import; lines 189, 197 call |
| `sandbox/registry.py` | `mlflow.lightgbm` | `log_model` and `load_model` | WIRED | Lines 17, 90, 173 |
| `sandbox/registry.py` | `mlflow.MlflowClient` | Alias-based lifecycle | WIRED | Line 18 import; `set_registered_model_alias` used for champion/archived |
| `sandbox/journal.py` | `sqlite3` | WAL mode, indexed columns | WIRED | Line 109: `PRAGMA journal_mode=WAL`; indexes created at lines 132-143 |
| `sandbox/journal.py` | `registry.py` | `run_id`, `model_version` fields | WIRED | Lines 87-88 in ExperimentRecord; inserted at line 176 |
| `sandbox/drift/adwin.py` | `river.drift.ADWIN` | River library wrapper | WIRED | Line 9: `from river.drift import ADWIN`; line 27: instantiated |
| `sandbox/drift/ks.py` | `scipy.stats.ks_2samp` | Two-sample KS test | WIRED | Line 10 import; line 41 call |
| `sandbox/observer.py` | `sandbox/drift/` | All 4 detectors imported | WIRED | Lines 18-21: imports compute_psi, check_ks_drift, CUSUMDetector, ADWINDetector |
| `cli/app.py` | `sandbox/registry.py` | status, rollback, diagnose commands | WIRED | Lines 54, 98, 152 import and call ModelRegistry |
| `cli/app.py` | `sandbox/journal.py` | journal command, status count | WIRED | Lines 68, 238 import and call ExperimentJournal |
| `cli/app.py` | `sandbox/observer.py` | diagnose command | WIRED | Line 97 import; line 121 instantiates DriftObserver |
| `cli/app.py` | `sandbox/evaluator.py` | (not directly wired in Phase 3) | INFO | CompositeEvaluator referenced in PLAN but diagnose uses synthetic data per Phase 3 design decision; evaluator is independently tested and used by Phase 4 |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SBOX-01 | 03-01 | Market replay with volume-adaptive slippage | SATISFIED | replay.py uses volumes[i]/spreads[i] per bar; test_slippage_varies_with_volume proves volume-adaptation |
| SBOX-02 | 03-02 | MLflow registry with config snapshot, metrics, champion/candidate/archived lifecycle | SATISFIED | registry.py log_candidate + alias-based lifecycle; 6 registry tests pass |
| SBOX-03 | 03-03 | Experiment journal with hypothesis, config diff, results, promotion decision, tags | SATISFIED | ExperimentJournal with ExperimentRecord; all 10 journal tests pass |
| SBOX-04 | 03-04 | Observer detects drift via rolling performance metrics and feature distribution drift | SATISFIED | DriftObserver with PSI, KS, CUSUM, ADWIN + PerformanceDriftReport; 5 observer tests pass |
| SBOX-05 | 03-05 | 6-metric composite fitness with specified weights | SATISFIED | DEFAULT_WEIGHTS exactly matches spec; calibration inverted; all 13 evaluator tests pass |
| SBOX-06 | 03-03 | Journal queryable by tag, date range, mutation type, outcome | SATISFIED | query() with all 4 filter types, AND logic; test_query_by_tag, test_query_by_date_range, test_query_by_mutation_type, test_query_by_outcome, test_query_combined_filters all pass |
| CLI-01 | 03-06 | `status` shows model health, experiments, alerts, autonomy level | SATISFIED | status command calls get_champion_info(), journal.count(), get_state(); format_status_table with color-coded rows |
| CLI-02 | 03-06 | `diagnose` forces diagnostic cycle on champion | SATISFIED | diagnose calls DriftObserver.get_full_report() and displays DriftReport panel. Note: Phase 3 uses synthetic data for the drift check per documented design decision — live data integration is Phase 4 scope |
| CLI-03 | 03-06 | `rollback` reverts to previous champion | SATISFIED | rollback command calls registry.rollback() and displays format_rollback_result() with version confirmation |
| CLI-04 | 03-06 | `pause`/`run` halt and resume agent loop | SATISFIED | pause/run_agent commands call set_state(); JSON state file at ~/.hydra/agent_state.json; test_pause_and_run verifies file state |
| CLI-05 | 03-06 | `journal query` searches by tag, date, mutation type | SATISFIED | journal command passes --tag, --since, --until, --mutation, --outcome to ExperimentJournal.query(); test_journal_filtered passes |
| CLI-06 | 03-06 | CLI uses Rich-formatted output with tables, colored alerts, panels | SATISFIED | formatters.py returns Table/Panel renderables; all formatters use Rich markup with color-coded status |

**All 12 requirements (SBOX-01 through SBOX-06, CLI-01 through CLI-06) SATISFIED.**

No orphaned requirements found — all Phase 3 requirements in REQUIREMENTS.md are covered by the 6 plans.

---

### Anti-Patterns Found

No anti-patterns detected. Scanned for TODO/FIXME/placeholder comments and empty implementations across all sandbox and CLI source files. None found.

**Notable design comment:** `diagnose` command uses synthetic data in Phase 3 (explicitly documented: "For Phase 3, demonstrate drift infrastructure with champion metrics. Full diagnostic cycle with live data is Phase 4."). This is an intentional scope boundary documented in the plan, not a stub — the DriftObserver itself is fully implemented and tested with real behavior.

---

### Human Verification Required

#### 1. CLI Rich Output Rendering

**Test:** Run `uv run hydra status` and `uv run hydra --help` in a terminal.
**Expected:** Rich-formatted table with colored Metric/Value/Status columns; help lists all 6 commands.
**Why human:** Cannot verify terminal color rendering and visual layout programmatically.

#### 2. `diagnose` with Real Champion Model

**Test:** Log a model via ModelRegistry, promote to champion, then run `uv run hydra diagnose`.
**Expected:** DriftReport panel renders with champion's Sharpe retrieved from MLflow and drift detection output.
**Why human:** Full end-to-end flow with a live MLflow registry requires manual setup.

#### 3. Agent State Persistence Across Sessions

**Test:** Run `uv run hydra pause`, exit shell, open new shell, run `uv run hydra status`.
**Expected:** Autonomy shows PAUSED (state persists in ~/.hydra/agent_state.json).
**Why human:** Cross-session file persistence behavior depends on OS file system and shell environment.

---

### Test Run Summary

All 56 tests pass (`uv run python -m pytest tests/test_replay.py tests/test_registry.py tests/test_journal.py tests/test_drift.py tests/test_observer.py tests/test_evaluator.py tests/test_cli.py`):

- test_replay.py: 5/5 passed
- test_registry.py: 6/6 passed
- test_journal.py: 10/10 passed
- test_drift.py: 10/10 passed
- test_observer.py: 5/5 passed
- test_evaluator.py: 13/13 passed
- test_cli.py: 7/7 passed

Warnings are MLflow FutureWarnings about filesystem tracking backend deprecation (cosmetic, no functional impact in Phase 3; relevant for Phase 4+ configuration).

---

_Verified: 2026-02-19_
_Verifier: Claude (gsd-verifier)_
