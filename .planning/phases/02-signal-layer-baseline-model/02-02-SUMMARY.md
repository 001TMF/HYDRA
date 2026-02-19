---
phase: 02-signal-layer-baseline-model
plan: 02
subsystem: risk
tags: [slippage, kelly-criterion, circuit-breakers, position-sizing, risk-management]

# Dependency graph
requires:
  - phase: 01-data-infrastructure-options-math
    provides: "Data quality patterns, project structure conventions"
provides:
  - "estimate_slippage: volume-adaptive slippage using square-root impact model"
  - "fractional_kelly: half-Kelly position sizing with configurable cap"
  - "volume_capped_position: integer contract sizing with ADV volume cap"
  - "CircuitBreaker: state machine (ACTIVE->TRIGGERED->COOLDOWN->ACTIVE)"
  - "CircuitBreakerManager: 4 independent breakers checked pre-trade"
affects: [02-05-walk-forward-backtesting, phase-05-paper-trading]

# Tech tracking
tech-stack:
  added: []
  patterns: [square-root-impact-slippage, fractional-kelly, circuit-breaker-state-machine]

key-files:
  created:
    - src/hydra/risk/__init__.py
    - src/hydra/risk/slippage.py
    - src/hydra/risk/position_sizing.py
    - src/hydra/risk/circuit_breakers.py
    - tests/test_slippage.py
    - tests/test_position_sizing.py
    - tests/test_circuit_breakers.py
  modified: []

key-decisions:
  - "upper_bound flag on CircuitBreaker for position_size breaker (triggers on value > threshold, unlike loss breakers which trigger on value < threshold)"
  - "math.sqrt used instead of numpy for slippage -- no numpy dependency needed for simple scalar math"

patterns-established:
  - "Circuit breaker state machine: ACTIVE->TRIGGERED->COOLDOWN->ACTIVE with configurable cooldown_periods"
  - "Risk module public API re-exported via __init__.py for clean imports"

requirements-completed: [MODL-03, MODL-04, MODL-05]

# Metrics
duration: 3min
completed: 2026-02-19
---

# Phase 2 Plan 2: Risk Infrastructure Summary

**Volume-adaptive slippage (square-root impact), fractional Kelly sizing with volume cap, and 4-breaker circuit breaker state machine with cooldown**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T09:28:16Z
- **Completed:** 2026-02-19T09:31:58Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Square-root impact slippage model: spread/2 + k * sigma * sqrt(V_order/V_daily) with configurable k (default 0.1)
- Fractional Kelly position sizing: (p*b - q)/b formula, negative edge returns 0, capped at max_position_pct
- Volume-capped integer contract sizing: min(kelly_contracts, 2% of ADV)
- Circuit breaker state machine with 4 independent breakers (daily loss, drawdown, position size, single trade loss)
- 27 tests covering all edge cases and formula verification

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Write failing tests** - `31c3613` (test)
2. **Task 2: GREEN + REFACTOR -- Implement all three modules** - `98eb0ff` (feat)

_Note: TDD plan -- test-first then implementation._

## Files Created/Modified
- `src/hydra/risk/__init__.py` - Public re-exports for risk module
- `src/hydra/risk/slippage.py` - Volume-adaptive slippage with square-root impact model
- `src/hydra/risk/position_sizing.py` - Fractional Kelly + volume-capped position sizing
- `src/hydra/risk/circuit_breakers.py` - CircuitBreaker state machine + CircuitBreakerManager
- `tests/test_slippage.py` - 6 tests: zero order, monotonicity, volatility scaling, custom k, exact formula, zero volume guard
- `tests/test_position_sizing.py` - 9 tests: negative Kelly, half/full Kelly, cap, volume cap, zero edge cases
- `tests/test_circuit_breakers.py` - 12 tests: state machine transitions, all 4 breaker types, manager deny logic, cooldown reset

## Decisions Made
- Used `upper_bound=True` flag on CircuitBreaker to distinguish position_size breaker (triggers on value > threshold) from loss breakers (trigger on value < threshold)
- Used `math.sqrt` instead of `numpy.sqrt` in slippage.py -- pure Python is sufficient for scalar operations, avoids unnecessary numpy import
- CircuitBreaker `check()` returns False for already-triggered/cooldown breakers, preventing trades during recovery period
- Drawdown computed as `(current_equity - peak_equity) / peak_equity` in manager for consistent percentage-based thresholds

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test expectations for max_position_pct cap interaction**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** Tests `test_half_kelly_default` and `test_full_kelly` expected uncapped Kelly values (0.2, 0.25) but the function's default max_position_pct=0.10 was capping the output
- **Fix:** Added `max_position_pct=1.0` to those test calls so they validate the Kelly formula itself; the capping behavior is tested separately in `test_capped_at_max_position_pct`
- **Files modified:** tests/test_position_sizing.py
- **Verification:** All 27 tests pass
- **Committed in:** 98eb0ff (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in test expectations)
**Impact on plan:** Auto-fix corrected test expectations to properly separate concerns (formula testing vs cap testing). No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Risk module complete and ready for walk-forward backtester (Plan 02-05)
- `estimate_slippage` will be called for every simulated trade in backtest loop
- `CircuitBreakerManager.check_trade` will be called pre-trade in backtest loop
- `fractional_kelly` + `volume_capped_position` will determine position sizes

## Self-Check: PASSED

All 8 files verified present. Both task commits (31c3613, 98eb0ff) verified in git log.

---
*Phase: 02-signal-layer-baseline-model*
*Completed: 2026-02-19*
