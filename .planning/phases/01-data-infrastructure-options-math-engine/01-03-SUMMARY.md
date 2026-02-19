---
phase: 01-data-infrastructure-options-math-engine
plan: 03
subsystem: options-math
tags: [svi, volatility-surface, scipy, numpy, black-76, options-pricing, tdd]

# Dependency graph
requires:
  - phase: 01-data-infrastructure-options-math-engine
    provides: "Project scaffold with numpy/scipy dependencies (Plan 01)"
provides:
  - "SVI volatility surface calibration (calibrate_svi)"
  - "SVI total variance formula (svi_total_variance)"
  - "SVI-to-call-prices via Black-76 (svi_to_call_prices)"
  - "Butterfly arbitrage detection"
  - "SVICalibrationResult dataclass"
affects: [01-04-density-extraction, 01-05-greeks-flow, 01-06-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SVI 5-parameter fit via scipy.optimize.minimize with L-BFGS-B"
    - "Butterfly arbitrage detection via d2w/dk2 convexity check"
    - "Black-76 call pricing for smooth B-L input"
    - "Graceful warnings for sparse data (< 8 strikes)"

key-files:
  created:
    - "src/hydra/signals/options_math/surface.py"
    - "tests/test_svi_surface.py"
  modified: []

key-decisions:
  - "L-BFGS-B with ftol=1e-14 and maxiter=1000 for robust convergence on sparse data"
  - "Convexity check on fine grid (500 points) with threshold -1e-10 for arbitrage detection"
  - "Sparse data warning at < 8 strikes rather than hard failure, per OPTS-05 graceful degradation"

patterns-established:
  - "SVI fit-then-price pipeline: calibrate_svi -> svi_to_call_prices -> (B-L in Plan 04)"
  - "All options math uses NumPy arrays for compute, scipy for optimization"

requirements-completed: [OPTS-03]

# Metrics
duration: 3min
completed: 2026-02-19
---

# Plan 03: SVI Volatility Surface Calibration Summary

**SVI 5-parameter volatility surface calibration with L-BFGS-B optimization, butterfly arbitrage detection, and Black-76 call pricing for Breeden-Litzenberger input**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T08:09:16Z
- **Completed:** 2026-02-19T08:12:39Z
- **Tasks:** 3 (TDD RED-GREEN-REFACTOR cycle)
- **Files modified:** 2

## Accomplishments
- SVI calibration fits flat vol with RMSE < 0.001 and skewed smile with RMSE < 0.02
- Handles sparse data (8 strikes with 5% noise, 5 strikes with warning) without NaN/Inf
- Butterfly arbitrage detection via second derivative convexity check on fine grid
- svi_to_call_prices produces smooth, monotonically decreasing, convex call prices suitable for B-L differentiation
- All 16 TDD tests pass

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests for SVI calibration** - `5b0b877` (test)
2. **GREEN: SVI implementation passing all tests** - `6b46bce` (feat)
3. **REFACTOR: No changes needed** - (skipped: code was clean)

_TDD RED-GREEN-REFACTOR cycle. Refactor skipped as implementation was already minimal and well-structured._

## Files Created/Modified
- `src/hydra/signals/options_math/surface.py` - SVI calibration: svi_total_variance, calibrate_svi, svi_to_call_prices, SVICalibrationResult, butterfly arbitrage detection (264 lines)
- `tests/test_svi_surface.py` - 16 tests covering flat vol, skewed smile, sparse data, arbitrage detection, call pricing (311 lines)

## Decisions Made
- L-BFGS-B with ftol=1e-14 for tight convergence even on sparse data
- Convexity check uses 500-point fine grid with -1e-10 threshold (conservative enough to catch real concavity, tolerant of numerical noise)
- Sparse data produces warnings rather than errors, following OPTS-05 graceful degradation requirement
- No QuantLib needed: pure NumPy/SciPy handles SVI calibration as recommended in research

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SVI surface calibration ready as input to Plan 04 (Breeden-Litzenberger density extraction)
- svi_to_call_prices provides the smooth call price curve that B-L needs for stable second derivatives
- SVICalibrationResult.has_butterfly_arbitrage flag available for data quality monitoring in Plan 06

## Self-Check: PASSED

- FOUND: src/hydra/signals/options_math/surface.py
- FOUND: tests/test_svi_surface.py
- FOUND: 5b0b877 (RED commit)
- FOUND: 6b46bce (GREEN commit)

---
*Phase: 01-data-infrastructure-options-math-engine*
*Completed: 2026-02-19*
