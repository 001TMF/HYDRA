---
phase: 04-agent-core-llm-integration
plan: 02
subsystem: agent
tags: [autonomy, rollback, promotion, guardrails, hysteresis, safety]

# Dependency graph
requires:
  - phase: 03-sandbox-experiment-infrastructure
    provides: "Sandbox evaluation and fitness scoring infrastructure"
provides:
  - "AutonomyLevel enum with 4-level action gating (LOCKDOWN/SUPERVISED/SEMI_AUTO/AUTONOMOUS)"
  - "HysteresisRollbackTrigger with sustained degradation detection and cooldown"
  - "PromotionEvaluator with 3-of-5 independent window evaluation"
affects: [04-05-agent-loop-wiring, 04-agent-core-llm-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [IntEnum-based permission gating, hysteresis state machine, dataclass config objects]

key-files:
  created:
    - src/hydra/agent/autonomy.py
    - src/hydra/agent/rollback.py
    - src/hydra/agent/promotion.py
    - tests/agent/__init__.py
    - tests/agent/test_autonomy.py
    - tests/agent/test_rollback.py
    - tests/agent/test_promotion.py
  modified: []

key-decisions:
  - "Minimum-level comparison (level >= required) for permission gating -- simpler than nested dict from research"
  - "Strict inequality for degradation threshold -- exactly at threshold is NOT degraded"
  - "Strict inequality for promotion wins -- tied fitness does NOT count as candidate win"

patterns-established:
  - "IntEnum permission gating: ordered levels with >= comparison for action authorization"
  - "Hysteresis state machine: armed/disarmed with cooldown to prevent flapping"
  - "Dataclass config objects: RollbackConfig/PromotionConfig with sensible defaults"

requirements-completed: [AGNT-06, AGNT-07, AGNT-08]

# Metrics
duration: 4min
completed: 2026-02-19
---

# Phase 4 Plan 02: Guardrails Summary

**4-level autonomy gating, hysteresis rollback trigger, and 3-of-5 window promotion evaluator -- pure logic guardrails with 79 tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T13:12:13Z
- **Completed:** 2026-02-19T13:15:48Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- AutonomyLevel IntEnum with check_permission/require_permission gating all 6 agent actions across 4 levels
- HysteresisRollbackTrigger fires only on sustained degradation (3 consecutive periods), with cooldown and re-arming to prevent flapping
- PromotionEvaluator requires candidate to beat champion on 3 of 5 independent windows with per-window audit trail
- 79 tests covering every level/action combination, threshold edge cases, cooldown lifecycle, and promotion tie handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Autonomy levels and permission gating** - `87ab05f` (feat)
2. **Task 2: Hysteresis rollback and 3-of-5 promotion** - `5d576fb` (feat)

## Files Created/Modified
- `src/hydra/agent/autonomy.py` - AutonomyLevel IntEnum, PERMISSIONS dict, check/require/get_allowed_actions
- `src/hydra/agent/rollback.py` - RollbackConfig dataclass, HysteresisRollbackTrigger with armed/cooldown state machine
- `src/hydra/agent/promotion.py` - PromotionConfig/PromotionResult dataclasses, PromotionEvaluator with window comparison
- `tests/agent/__init__.py` - Test package init (idempotent creation for parallel plan execution)
- `tests/agent/test_autonomy.py` - 51 tests: all level/action permutations, error messages, unknown actions
- `tests/agent/test_rollback.py` - 13 tests: sustained degradation, cooldown, re-arming, threshold edge, reset
- `tests/agent/test_promotion.py` - 15 tests: win counts, ties, min_improvement, validation, audit trail

## Decisions Made
- Minimum-level comparison (`level >= PERMISSIONS[action]`) for permission gating instead of nested dict from research -- simpler, less error-prone
- Strict inequality for degradation threshold: exactly at threshold is NOT degraded (conservative -- avoids false rollbacks)
- Strict inequality for promotion wins: tied fitness does NOT count as candidate win (conservative -- prevents undeserved promotions)
- Unknown actions return False from check_permission rather than raising KeyError -- defensive coding for extensibility

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All three guardrail modules are pure logic with no LLM dependency -- ready for wiring into agent loop (04-05)
- HysteresisRollbackTrigger.update() returns bool consumed by loop.py EVALUATE phase
- PromotionEvaluator.evaluate() returns PromotionResult consumed by loop.py EVALUATE phase
- AutonomyLevel used as guard at start of each loop step via require_permission()

## Self-Check: PASSED

- All 7 created files verified present on disk
- Commit 87ab05f (Task 1) verified in git log
- Commit 5d576fb (Task 2) verified in git log
- 79/79 tests passing

---
*Phase: 04-agent-core-llm-integration*
*Completed: 2026-02-19*
