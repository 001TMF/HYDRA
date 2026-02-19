---
phase: 02-signal-layer-baseline-model
plan: 01
subsystem: signals
tags: [cot, sentiment, percentile-rank, scipy, dataclass]

# Dependency graph
requires:
  - phase: 01-data-infrastructure
    provides: "feature_store with COT features (cot_managed_money_net, cot_total_oi)"
provides:
  - "SentimentScore dataclass with score [-1,+1], confidence [0,1], components dict"
  - "compute_cot_sentiment function for normalized crowd positioning"
  - "sentiment submodule under hydra.signals"
affects: [02-03-divergence-detector, 02-04-feature-engineering]

# Tech tracking
tech-stack:
  added: [scipy.stats.percentileofscore]
  patterns: [percentile-rank normalization, confidence weighting via OI + concentration]

key-files:
  created:
    - src/hydra/signals/sentiment/__init__.py
    - src/hydra/signals/sentiment/cot_scoring.py
    - tests/test_cot_scoring.py
  modified: []

key-decisions:
  - "Confidence formula: 0.6*oi_rank + 0.4*min(concentration*5, 1.0) -- weights OI magnitude over concentration"
  - "percentileofscore with default 'rank' method for percentile computation"
  - "Minimum 4 weeks history threshold for neutral fallback"

patterns-established:
  - "SentimentScore dataclass pattern: score + confidence + components dict for signal decomposition"
  - "History-based normalization: raw values mapped to [-1,+1] via percentile rank"

requirements-completed: [SGNL-01]

# Metrics
duration: 2min
completed: 2026-02-19
---

# Phase 2 Plan 01: COT Sentiment Scoring Summary

**Percentile-rank-based COT sentiment scoring with confidence weighting from OI magnitude and positioning concentration**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-19T09:28:11Z
- **Completed:** 2026-02-19T09:30:26Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- SentimentScore dataclass providing normalized score [-1,+1], confidence [0,1], and component breakdown
- compute_cot_sentiment function using 52-week percentile rank normalization
- Full TDD cycle: 8 failing tests -> implementation -> all passing
- Confidence weighting blends OI magnitude (60%) and positioning concentration (40%)

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Write failing tests** - `c403268` (test)
2. **Task 2: GREEN + REFACTOR -- Implement COT sentiment scoring** - `5ae745c` (feat)

## Files Created/Modified
- `src/hydra/signals/sentiment/__init__.py` - Public re-exports for SentimentScore and compute_cot_sentiment
- `src/hydra/signals/sentiment/cot_scoring.py` - Core scoring: SentimentScore dataclass + compute_cot_sentiment function
- `tests/test_cot_scoring.py` - 8 TDD tests covering edge cases, bounds, and component validation

## Decisions Made
- Confidence formula: `0.6 * oi_rank + 0.4 * min(concentration * 5.0, 1.0)` -- OI magnitude weighted higher because it reflects market participation quality
- `percentileofscore` default method (rank) used for consistent percentile computation
- Minimum 4-week history threshold returns neutral (0.0, 0.0) to avoid unreliable percentile estimates

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test isolation for concentration scaling test**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** test_confidence_scales_with_concentration used different total_oi values (150k vs 10k), causing oi_rank to dominate and mask the concentration effect being tested
- **Fix:** Held total_oi constant at 100k (median) for both scenarios so only concentration varies
- **Files modified:** tests/test_cot_scoring.py
- **Verification:** All 8 tests pass
- **Committed in:** 5ae745c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test fix necessary for correct isolation of confidence components. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Sentiment scoring module ready for divergence detector (Plan 02-03)
- SentimentScore provides score + confidence + components for downstream signal combination
- Feature store integration will be wired in Plan 02-03 when reading historical COT data

## Self-Check: PASSED

- [x] src/hydra/signals/sentiment/__init__.py -- FOUND
- [x] src/hydra/signals/sentiment/cot_scoring.py -- FOUND
- [x] tests/test_cot_scoring.py -- FOUND
- [x] Commit c403268 (test) -- FOUND
- [x] Commit 5ae745c (feat) -- FOUND

---
*Phase: 02-signal-layer-baseline-model*
*Completed: 2026-02-19*
