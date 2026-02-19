---
phase: 03-sandbox-experiment-infrastructure
plan: 04
subsystem: drift-detection
tags: [psi, ks-test, cusum, adwin, river, drift-detection, streaming]

# Dependency graph
requires:
  - phase: 02-signal-generation-pipeline
    provides: "evaluation.py Sharpe/drawdown formulas reused in DriftObserver"
provides:
  - "PSI batch distribution drift detector with epsilon smoothing"
  - "KS two-sample test wrapper for feature drift"
  - "CUSUM streaming change-point detector"
  - "ADWIN adaptive windowing drift detector via River"
  - "DriftObserver combining performance + feature + streaming drift into DriftReport"
  - "needs_diagnosis flag for triggering Phase 4 diagnostic cycles"
affects: [04-agent-loop, drift-monitoring, model-lifecycle]

# Tech tracking
tech-stack:
  added: [river>=0.21]
  patterns: [streaming-detector-pair, unified-drift-report, configurable-thresholds]

key-files:
  created:
    - src/hydra/sandbox/drift/__init__.py
    - src/hydra/sandbox/drift/psi.py
    - src/hydra/sandbox/drift/ks.py
    - src/hydra/sandbox/drift/cusum.py
    - src/hydra/sandbox/drift/adwin.py
    - src/hydra/sandbox/observer.py
    - tests/test_drift.py
    - tests/test_observer.py
  modified:
    - pyproject.toml
    - src/hydra/sandbox/__init__.py

key-decisions:
  - "River library (v0.23) for ADWIN adaptive windowing -- avoids reimplementing complex algorithm"
  - "ADWIN + CUSUM paired per streaming metric for complementary detection (adaptive window + cumulative sum)"
  - "Epsilon smoothing (1e-4) with re-normalization for PSI zero-bin handling"
  - "Adaptive n_bins reduction when baseline has fewer unique values than requested bins"

patterns-established:
  - "Streaming detector pair: each monitored metric gets both ADWIN and CUSUM detectors"
  - "Unified DriftReport: needs_diagnosis flag aggregates all drift signals into single boolean trigger"
  - "Configurable thresholds via config dict for per-environment tuning"

requirements-completed: [SBOX-04]

# Metrics
duration: 3min
completed: 2026-02-19
---

# Phase 3 Plan 4: Drift Detection + Observer Summary

**PSI/KS/CUSUM/ADWIN drift detection toolkit with DriftObserver combining performance, feature, and streaming drift into unified DriftReport with needs_diagnosis trigger**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T11:19:55Z
- **Completed:** 2026-02-19T11:23:25Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments
- Four drift detectors: PSI (batch distribution), KS (two-sample test), CUSUM (streaming change-point), ADWIN (adaptive windowing via River)
- DriftObserver with three monitoring modes: performance drift (Sharpe, drawdown, hit rate, calibration), feature distribution drift (PSI + KS per column), streaming drift (ADWIN + CUSUM pairs)
- Unified DriftReport with needs_diagnosis flag that triggers Phase 4 diagnostic cycles
- 15 tests covering all detectors and observer integration (all pass)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 4 drift detectors in drift subpackage** - `ab8f72b` (feat)
2. **Task 2: Create DriftObserver and test all components** - `4696d4e` (feat)

## Files Created/Modified
- `src/hydra/sandbox/drift/__init__.py` - Package exports for all 4 detectors
- `src/hydra/sandbox/drift/psi.py` - PSI computation with quantile binning and epsilon smoothing
- `src/hydra/sandbox/drift/ks.py` - KS two-sample test wrapper with configurable alpha
- `src/hydra/sandbox/drift/cusum.py` - CUSUM streaming change-point detector
- `src/hydra/sandbox/drift/adwin.py` - ADWIN wrapper around River library
- `src/hydra/sandbox/observer.py` - DriftObserver with PerformanceDriftReport, FeatureDriftReport, DriftReport
- `tests/test_drift.py` - 10 unit tests for PSI, KS, CUSUM, ADWIN
- `tests/test_observer.py` - 5 integration tests for DriftObserver
- `pyproject.toml` - Added river>=0.21 dependency
- `src/hydra/sandbox/__init__.py` - Added DriftObserver, DriftReport exports

## Decisions Made
- Used River library (v0.23 installed) for ADWIN implementation rather than reimplementing adaptive windowing from scratch
- Paired ADWIN + CUSUM per streaming metric for complementary detection (adaptive window detects distribution changes, cumulative sum detects persistent shifts)
- Epsilon smoothing (1e-4) with re-normalization handles zero-bin edge cases in PSI without NaN/Inf
- Adaptive n_bins reduction when baseline has fewer unique values than requested bins

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Drift detection toolkit ready for integration with Phase 4 agent loop
- DriftObserver.get_full_report() provides single-call comprehensive drift assessment
- needs_diagnosis flag on DriftReport is the trigger for autonomous diagnostic cycles
- All streaming detectors auto-create for new metric names (lazy initialization)

## Self-Check: PASSED

All 8 created files verified present. Both task commits (ab8f72b, 4696d4e) confirmed in git log.

---
*Phase: 03-sandbox-experiment-infrastructure*
*Completed: 2026-02-19*
