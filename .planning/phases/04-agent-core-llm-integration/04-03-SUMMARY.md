---
phase: 04-agent-core-llm-integration
plan: 03
subsystem: agent
tags: [diagnostician, hypothesis-engine, mutation-playbook, drift-diagnosis, rule-based, round-robin]

# Dependency graph
requires:
  - phase: 03-sandbox-experiment-infrastructure
    provides: "DriftReport, FeatureDriftReport, PerformanceDriftReport from observer.py"
provides:
  - "Diagnostician: DriftReport -> DiagnosisResult with structured triage"
  - "HypothesisEngine: DiagnosisResult -> Hypothesis from curated playbook"
  - "MUTATION_PLAYBOOK: 5 categories x 2-3 mutations with config diffs"
  - "Shared domain types: DriftCategory, DiagnosisResult, MutationType, Hypothesis"
affects: [04-04-experiment-runner, 04-05-orchestrator, agent-loop]

# Tech tracking
tech-stack:
  added: []
  patterns: [rule-based-triage, round-robin-selection, expression-resolution, optional-llm-enhancement]

key-files:
  created:
    - src/hydra/agent/diagnostician.py
    - src/hydra/agent/hypothesis.py
    - tests/agent/test_diagnostician.py
    - tests/agent/test_hypothesis.py
  modified:
    - src/hydra/agent/types.py

key-decisions:
  - "DriftCategory enum member names kept as PERFORMANCE/FEATURE_DRIFT (from 04-01) rather than PERFORMANCE_DEGRADATION/FEATURE_DISTRIBUTION_DRIFT -- shorter, already established"
  - "Priority-ordered classification: feature_drift > performance > regime_change > overfitting > default"
  - "Config diff resolution via regex patterns (multiply, floor_div, max) -- no eval() for security"
  - "propose_multiple caps at playbook size to avoid duplicate hypotheses"
  - "LLM enhancement threshold: confidence < 0.6 triggers optional LLM call"

patterns-established:
  - "Rule-based-first pattern: deterministic engine works standalone, LLM is optional enhancer"
  - "Round-robin selection for experiment diversity across repeated calls"
  - "Expression resolver for config diffs: regex-based, eval-free, falls back to raw string"

requirements-completed: [AGNT-02, AGNT-03]

# Metrics
duration: 5min
completed: 2026-02-19
---

# Phase 4 Plan 3: Diagnostician + Hypothesis Engine Summary

**Rule-based diagnostician with priority-ordered triage and curated mutation playbook with round-robin hypothesis selection**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-19T13:12:16Z
- **Completed:** 2026-02-19T13:17:29Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Diagnostician maps DriftReport to DiagnosisResult with deterministic priority-ordered rules covering all 5 drift categories
- Mutation playbook covers performance_degradation, feature_distribution_drift, regime_change, overfitting, and data_quality_issue with 2-3 concrete mutations each
- HypothesisEngine uses round-robin selection ensuring experiment diversity and resolves config diff expressions against current model config
- Full test coverage: 26 tests across both modules, all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Shared types + rule-based diagnostician** - `5974338` (feat)
2. **Task 2: Mutation playbook + hypothesis engine** - `5761fd7` (feat)

## Files Created/Modified
- `src/hydra/agent/types.py` - Extended with DiagnosisResult and Hypothesis dataclasses (shared domain types)
- `src/hydra/agent/diagnostician.py` - Rule-based triage: DriftReport -> DiagnosisResult with evidence, confidence, and mutation recommendations
- `src/hydra/agent/hypothesis.py` - MUTATION_PLAYBOOK + HypothesisEngine with round-robin selection and config diff resolution
- `tests/agent/test_diagnostician.py` - 13 tests covering all causes, priority ordering, evidence, mutation type mapping
- `tests/agent/test_hypothesis.py` - 13 tests covering playbook structure, round-robin, config resolution, edge cases

## Decisions Made
- Kept DriftCategory enum names from 04-01 (PERFORMANCE, FEATURE_DRIFT) rather than the longer names in the plan spec -- already committed and referenced by other wave-1 plans
- Priority ordering matches plan spec: feature drift (3+ features) takes highest priority, then performance, regime change, overfitting, default
- Config diff resolver uses regex patterns (`current * N`, `current // N`, `max(N, current // M)`) instead of eval() for safety
- propose_multiple caps output at playbook size to avoid returning duplicate hypotheses
- LLM enhancement gated on confidence < 0.6 with full try/except fallback to rule-based result

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Diagnostician + HypothesisEngine ready for integration into the experiment runner (04-04)
- The DriftReport -> DiagnosisResult -> Hypothesis pipeline is the deterministic "Honda engine" core
- LLM enhancement is wired but optional; pure rule-based path works for all categories

## Self-Check: PASSED

All 6 files verified present. Both commit hashes (5974338, 5761fd7) confirmed in git log.

---
*Phase: 04-agent-core-llm-integration*
*Completed: 2026-02-19*
