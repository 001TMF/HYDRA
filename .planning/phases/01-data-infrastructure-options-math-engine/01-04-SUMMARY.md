---
phase: 01-data-infrastructure-options-math-engine
plan: 04
subsystem: options-math
tags: [breeden-litzenberger, density-extraction, implied-moments, scipy, numpy, risk-neutral, tdd]

# Dependency graph
requires:
  - phase: 01-data-infrastructure-options-math-engine
    provides: "SVI volatility surface calibration (Plan 03)"
provides:
  - "Breeden-Litzenberger risk-neutral density extraction (extract_density)"
  - "Implied moments computation (compute_moments)"
  - "DataQuality enum canonical location (density.py)"
  - "ImpliedDensityResult dataclass"
  - "ImpliedMoments dataclass"
  - "Graceful degradation to ATM IV (OPTS-05)"
affects: [01-06-integration, phase-02-divergence-signal]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "B-L density via double np.gradient on SVI-smoothed call prices"
    - "Brentq IV inversion from Black-76 call prices"
    - "Liquidity filtering by OI and bid-ask spread before density extraction"
    - "Normalization after negative density clipping"

key-files:
  created:
    - "src/hydra/signals/options_math/density.py"
    - "src/hydra/signals/options_math/moments.py"
    - "tests/test_bl_density.py"
    - "tests/test_implied_moments.py"
  modified:
    - "src/hydra/signals/options_math/greeks.py"

key-decisions:
  - "DataQuality enum consolidated into density.py as canonical location; greeks.py now imports from density.py"
  - "Brentq with xtol=1e-10 and maxiter=200 for IV inversion; returns None on failure"
  - "200-point fine grid with 5% margin for B-L density extraction avoids edge effects"
  - "Negative density clipping followed by normalization; warning issued when clipping occurs"
  - "scipy.stats.norm imported at module level in density.py for performance in IV inversion loop"

patterns-established:
  - "Pipeline: liquid filter -> IV inversion -> SVI calibration -> smooth call prices -> d2C/dK2 -> density"
  - "Moments computed via np.trapezoid numerical integration over density"
  - "DEGRADED quality returns atm_iv only; None for all computed features"

requirements-completed: [OPTS-01, OPTS-02, OPTS-05]

# Metrics
duration: 7min
completed: 2026-02-19
---

# Plan 04: Breeden-Litzenberger Density Extraction + Implied Moments Summary

**B-L risk-neutral density extraction from SVI-smoothed call prices with brentq IV inversion, negative density clipping, and implied moments (mean, variance, skew, kurtosis) via numerical integration**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-19T08:21:43Z
- **Completed:** 2026-02-19T08:28:55Z
- **Tasks:** 3 (TDD RED-GREEN-REFACTOR cycle)
- **Files modified:** 5

## Accomplishments
- B-L density from 20-strike clean data integrates to ~1.0 and has mean within 2% of forward price
- Graceful degradation to ATM IV only when fewer than 8 liquid strikes pass OI and spread filters
- Negative density regions clipped to zero with warning, then renormalized
- Log-normal benchmark: density from flat-vol Black-76 prices recovers unimodal shape peaked near forward
- Implied moments (variance, skew) match analytic log-normal values within 5%
- DataQuality enum consolidated from greeks.py into density.py as canonical location
- All 27 new tests pass, all 41 existing tests pass (68 total)

## Task Commits

Each task was committed atomically:

1. **RED: Failing tests for density and moments** - `a200832` (test)
2. **GREEN: Implementation passing all tests** - `966e0e1` (feat)
3. **REFACTOR: Module-level scipy import** - `86442f4` (refactor)

_TDD RED-GREEN-REFACTOR cycle completed._

## Files Created/Modified
- `src/hydra/signals/options_math/density.py` - B-L density extraction: extract_density, DataQuality, ImpliedDensityResult, ATM IV inversion (319 lines)
- `src/hydra/signals/options_math/moments.py` - Implied moments: compute_moments, ImpliedMoments (116 lines)
- `src/hydra/signals/options_math/greeks.py` - Consolidated DataQuality import from density.py (removed local enum definition)
- `tests/test_bl_density.py` - 16 tests: integration, mean, quality, degradation, negative clipping, log-normal benchmark, liquidity filtering (450 lines)
- `tests/test_implied_moments.py` - 11 tests: dataclass fields, full/degraded quality, log-normal benchmark, stability (287 lines)

## Decisions Made
- DataQuality enum consolidated into density.py as canonical location; greeks.py now imports from there (per STATE.md decision to consolidate when 01-04 executes)
- Brentq IV inversion with xtol=1e-10, maxiter=200; returns None on failure (safe fallback)
- 200-point fine grid with 5% margin for stable B-L second derivatives
- Negative density values clipped to zero with Python warnings.warn and list-based warning tracking
- Forward price approximated as spot price (sufficient for Phase 1; can refine with cost-of-carry later)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Density extraction and moments computation ready for Plan 06 integration testing
- ImpliedDensityResult feeds directly into compute_moments for the full pipeline
- DataQuality enum now has single canonical location for all options math modules
- Implied moments (mean, variance, skew, kurtosis) are the features that feed Phase 2 divergence signal

## Self-Check: PASSED

- FOUND: src/hydra/signals/options_math/density.py
- FOUND: src/hydra/signals/options_math/moments.py
- FOUND: tests/test_bl_density.py
- FOUND: tests/test_implied_moments.py
- FOUND: a200832 (RED commit)
- FOUND: 966e0e1 (GREEN commit)
- FOUND: 86442f4 (REFACTOR commit)

---
*Phase: 01-data-infrastructure-options-math-engine*
*Completed: 2026-02-19*
