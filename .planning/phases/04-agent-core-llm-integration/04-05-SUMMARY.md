---
phase: 04-agent-core-llm-integration
plan: 05
subsystem: agent
tags: [agent-loop, state-machine, integration, autonomous, observe-diagnose-hypothesize-experiment-evaluate]

# Dependency graph
requires:
  - phase: 04-agent-core-llm-integration
    plan: 01
    provides: "LLMClient with multi-provider fallback chain"
  - phase: 04-agent-core-llm-integration
    plan: 02
    provides: "AutonomyLevel, HysteresisRollbackTrigger, PromotionEvaluator"
  - phase: 04-agent-core-llm-integration
    plan: 03
    provides: "Diagnostician, HypothesisEngine, MUTATION_PLAYBOOK"
  - phase: 04-agent-core-llm-integration
    plan: 04
    provides: "ExperimentRunner, HypothesisDeduplicator, MutationBudget"
provides:
  - "AgentLoop: complete autonomous agent loop state machine"
  - "AgentPhase enum for loop phase tracking"
  - "AgentCycleResult dataclass for cycle outcome reporting"
  - "Public API in hydra.agent.__init__ exporting all Phase 4 modules"
affects: [agent-loop, phase-05-deployment]

# Tech tracking
tech-stack:
  added: []
  patterns: [constructor-injection, state-machine, guard-rail-chain, dual-promotion-path]

key-files:
  created:
    - src/hydra/agent/loop.py
    - tests/agent/test_loop.py
    - tests/agent/test_integration.py
  modified:
    - src/hydra/agent/__init__.py

key-decisions:
  - "Dual promotion path: 3-of-5 PromotionEvaluator when window scores provided, single-score comparison as Honda default"
  - "check_permission returns graceful skip (not PermissionDeniedError) at each loop step for testability"
  - "Diagnosis inconclusive threshold: confidence < 0.3 AND no evidence causes early exit"

patterns-established:
  - "Constructor injection for all dependencies -- no global state, fully testable"
  - "Guard rail chain: CLI state -> autonomy -> dedup -> budget -> experiment"
  - "Phase-aware skip reasons: AgentCycleResult.skipped_reason documents exactly why a cycle ended early"

requirements-completed: [AGNT-01]

# Metrics
duration: 8min
completed: 2026-02-19
---

# Phase 4 Plan 05: Agent Loop Wiring Summary

**Single-head autonomous agent loop state machine wiring all Phase 4 modules into the complete observe-diagnose-hypothesize-experiment-evaluate cycle with zero LLM calls**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-19T13:23:51Z
- **Completed:** 2026-02-19T13:32:03Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- AgentLoop class wires 11 dependencies (observer, diagnostician, hypothesis engine, experiment runner, evaluator, journal, registry, rollback trigger, promotion evaluator, deduplicator, budget) into a single state machine
- Each step checks autonomy permissions via check_permission() for graceful skip behavior
- Dedup, budget, and cooldown guards are enforced before experiments
- Rollback trigger fires on sustained degradation and calls registry.rollback()
- Dual promotion path: 3-of-5 PromotionEvaluator when caller provides window scores, single-score comparison as Honda default
- Journal logging captures the full cycle outcome with hypothesis, config diff, and promotion decision
- CLI state gating prevents execution when agent is PAUSED
- Updated hydra.agent.__init__.py with complete public API exports for all Phase 4 modules
- 14 unit tests covering every state transition and guard rail
- 3 integration tests proving full pipeline end-to-end with zero LLM calls
- All 193 agent tests passing (17 new + 176 existing)

## Task Commits

Each task was committed atomically:

1. **Task 1: Agent loop state machine with step transitions** - `622e087` (feat)
2. **Task 2: Unit tests and integration test for agent loop** - `fe34acf` (test)

## Files Created/Modified
- `src/hydra/agent/loop.py` - AgentLoop, AgentPhase, AgentCycleResult -- the complete autonomous agent loop state machine
- `src/hydra/agent/__init__.py` - Updated with public API exports for AgentLoop, AutonomyLevel, Diagnostician, HypothesisEngine, ExperimentRunner, rollback, promotion, dedup, budget
- `tests/agent/test_loop.py` - 14 unit tests: PAUSED gating, LOCKDOWN/SUPERVISED blocks, dedup, budget, experiment failure, rollback, simple and 3-of-5 promotion, journal logging, diagnosis inconclusive
- `tests/agent/test_integration.py` - 3 integration tests: full cycle with drift detection, no-drift early exit, rejected experiment

## Decisions Made
- **Dual promotion path**: When `candidate_window_scores` and `champion_window_scores` are both provided, the full 3-of-5 PromotionEvaluator is used (AGNT-08). When absent, a simple single-score comparison serves as the Honda default for unit tests and early integration callers.
- **Graceful permission checks**: Uses `check_permission()` (returns bool) instead of `require_permission()` (raises exception) at each loop step. This produces clean `skipped_reason` strings in AgentCycleResult rather than requiring try/except boilerplate in every step.
- **Diagnosis inconclusive threshold**: Confidence < 0.3 combined with empty evidence list causes early exit. This prevents wasting experiment budget on diagnoses with no signal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed evidence default in test helper**
- **Found during:** Task 2 test execution
- **Issue:** `_make_diagnosis(evidence=[])` was coerced to `["Sharpe degraded"]` by `evidence or default` pattern (empty list is falsy)
- **Fix:** Changed to `evidence if evidence is not None else ["Sharpe degraded"]` for explicit None checking
- **Files modified:** tests/agent/test_loop.py
- **Verification:** test_diagnosis_inconclusive passes correctly
- **Committed in:** fe34acf (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test helper)
**Impact on plan:** Trivial fix. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required. The entire agent loop works with zero API keys (AGNT-10 rule-based mode).

## Next Phase Readiness
- Phase 4 is now complete: all 5 plans executed
- The "Honda engine" is fully assembled: drift detection -> diagnosis -> hypothesis -> experiment -> evaluation -> promotion/rollback
- All guardrails (autonomy, dedup, budget, cooldowns, rollback) are wired in and tested
- Zero LLM dependency: the system operates entirely rule-based when no API keys are configured
- Ready for multi-head architecture and deployment integration in future phases

## Self-Check: PASSED

- All 4 files verified present on disk
- Commit 622e087 (Task 1) verified in git log
- Commit fe34acf (Task 2) verified in git log
- 193/193 agent tests passing
- All plan verification commands succeed

---
*Phase: 04-agent-core-llm-integration*
*Completed: 2026-02-19*
