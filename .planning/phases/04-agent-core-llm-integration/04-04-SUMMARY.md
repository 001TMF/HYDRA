---
phase: 04-agent-core-llm-integration
plan: 04
subsystem: agent
tags: [subprocess, isolation, dedup, embeddings, sentence-transformers, budget, cooldown, experiment-runner]

# Dependency graph
requires:
  - phase: 03-sandbox-experiment-infrastructure
    provides: "ExperimentJournal for dedup/budget journal integration"
provides:
  - "ExperimentRunner: subprocess-isolated candidate training with configurable timeout"
  - "HypothesisDeduplicator: semantic dedup via all-MiniLM-L6-v2 embeddings (cosine > 0.85 rejects)"
  - "MutationBudget + BudgetConfig: per-cycle limits, per-category caps, cooldown timers"
affects: [04-05-agent-loop-assembly, agent-loop, experiment-pipeline]

# Tech tracking
tech-stack:
  added: [sentence-transformers]
  patterns: [subprocess-isolation, expression-config-merging, three-layer-loop-prevention]

key-files:
  created:
    - src/hydra/agent/experiment_runner.py
    - src/hydra/agent/_train_candidate.py
    - src/hydra/agent/dedup.py
    - src/hydra/agent/budget.py
    - tests/agent/test_experiment_runner.py
    - tests/agent/test_dedup.py
    - tests/agent/test_budget.py
  modified: []

key-decisions:
  - "Defensive filtering in load_cooldowns_from_journal checks promotion_decision field on each record, not just relying on query filter"
  - "Expression resolution via regex for config merging supports +, -, *, / operators with 'current' token"
  - "sentence-transformers installed as blocking dependency fix (04-01 not yet executed)"

patterns-established:
  - "Subprocess isolation: temp JSON config file -> subprocess.run -> parse stdout JSON -> cleanup"
  - "Three-layer loop prevention: semantic dedup (0.85 cosine) + per-category budgets + cooldown timers"
  - "Config expression resolution: 'current * 0.5' evaluates against base config value"

requirements-completed: [AGNT-04, AGNT-09]

# Metrics
duration: 8min
completed: 2026-02-19
---

# Phase 04 Plan 04: Experiment Runner, Semantic Dedup, and Mutation Budgets Summary

**Subprocess-isolated experiment runner with configurable timeout, plus three-layer defense against degenerate loops (semantic dedup via all-MiniLM-L6-v2, per-category mutation budgets, cooldown timers)**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-19T13:12:18Z
- **Completed:** 2026-02-19T13:21:00Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- ExperimentRunner executes candidate training in isolated subprocess with configurable timeout, crash handling, and structured ExperimentResult
- Config merging resolves numeric expressions like "current * 0.5" against base config values
- HypothesisDeduplicator uses all-MiniLM-L6-v2 (22M params, CPU) for semantic similarity detection at 0.85 threshold
- MutationBudget enforces per-cycle total limits, per-category caps, and cooldown timers after rejection
- Both dedup and budget integrate with ExperimentJournal for persistent state reconstruction on startup
- 39 tests covering all success/failure paths, edge cases, and integration points

## Task Commits

Each task was committed atomically:

1. **Task 1: Experiment runner with subprocess isolation and timeout** - `5974338` (feat)
2. **Task 2: Semantic dedup and mutation budgets with cooldowns** - `5513e60` (feat)

## Files Created/Modified
- `src/hydra/agent/experiment_runner.py` - ExperimentRunner class with subprocess isolation, timeout, config merging
- `src/hydra/agent/_train_candidate.py` - Subprocess entry point stub (full training deferred to 04-05)
- `src/hydra/agent/dedup.py` - HypothesisDeduplicator with sentence-transformers embeddings
- `src/hydra/agent/budget.py` - MutationBudget + BudgetConfig with cooldown timers
- `tests/agent/test_experiment_runner.py` - 16 tests for runner, config merging, error handling
- `tests/agent/test_dedup.py` - 10 tests for similarity detection, threshold tuning, journal integration
- `tests/agent/test_budget.py` - 13 tests for budget limits, cooldowns, cycle resets, journal reconstruction

## Decisions Made
- Defensive filtering in `load_cooldowns_from_journal` checks `promotion_decision` field on each record rather than relying solely on journal query filter -- prevents bugs from mock/test inconsistencies and future query changes
- Expression resolution supports four arithmetic operators (+, -, *, /) via regex pattern matching on "current" token
- sentence-transformers installed directly as blocking dependency fix since 04-01 (which was supposed to install it) has not yet executed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed sentence-transformers dependency**
- **Found during:** Pre-execution setup
- **Issue:** Plan states "sentence-transformers is already added by 04-01" but 04-01 has not been executed yet
- **Fix:** Ran `uv add sentence-transformers` to unblock Task 2
- **Files modified:** pyproject.toml, uv.lock
- **Verification:** `import sentence_transformers` succeeds
- **Committed in:** Not committed separately (dependency change tracked by uv.lock)

**2. [Rule 1 - Bug] Defensive promotion_decision check in load_cooldowns_from_journal**
- **Found during:** Task 2 test execution
- **Issue:** `load_cooldowns_from_journal` processed all records from journal query result without checking `promotion_decision`, causing promoted records to incorrectly trigger cooldowns
- **Fix:** Added `if getattr(record, "promotion_decision", None) != "rejected": continue` guard
- **Files modified:** src/hydra/agent/budget.py
- **Verification:** test_reconstructs_cooldowns passes -- promoted experiments do not trigger cooldown
- **Committed in:** 5513e60 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking dependency, 1 bug fix)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Experiment runner, dedup, and budget modules are ready for wiring into the agent loop in Plan 04-05
- _train_candidate is a stub -- full training integration (BaselineModel + MarketReplayEngine) deferred to Plan 05
- Three-layer loop prevention (dedup + budgets + cooldowns) operates independently with zero LLM calls

## Self-Check: PASSED

All 7 files verified present. Both task commits (5974338, 5513e60) confirmed in git log.

---
*Phase: 04-agent-core-llm-integration*
*Completed: 2026-02-19*
