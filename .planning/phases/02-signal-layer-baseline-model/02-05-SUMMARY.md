---
phase: 02-signal-layer-baseline-model
plan: 05
subsystem: model
tags: [walk-forward, backtesting, purged-cv, embargo, backtest-metrics, equity-curve, slippage-adjusted]

# Dependency graph
requires:
  - phase: 02-signal-layer-baseline-model
    provides: "BaselineModel LightGBM classifier (02-04), slippage + Kelly sizing + circuit breakers (02-02)"
provides:
  - "PurgedWalkForwardSplit: expanding-window cross-validator with configurable embargo gap"
  - "WalkForwardEngine: full walk-forward backtest integrating model + risk stack per fold"
  - "BacktestResult: dataclass with Sharpe, drawdown, hit rate, equity curve, trade log, per-fold Sharpes"
  - "compute_backtest_metrics: slippage-adjusted metric computation from daily returns"
affects: [phase-03-hyperparameter-tuning, phase-05-paper-trading]

# Tech tracking
tech-stack:
  added: []
  patterns: [purged-walk-forward-cv, embargo-gap-leakage-prevention, expanding-window-backtest, risk-stack-integration-per-fold]

key-files:
  created:
    - src/hydra/model/walk_forward.py
    - src/hydra/model/evaluation.py
    - tests/test_walk_forward.py
    - tests/test_evaluation.py
  modified:
    - src/hydra/model/__init__.py

key-decisions:
  - "Single-period returns for PnL computation (price[t]-price[t-1])/price[t-1] -- simple, avoids lookahead"
  - "Circuit breakers reset per fold to prevent carry-over bias between OOS periods"
  - "Running win/loss statistics for Kelly sizing -- adapts position size as fold progresses"
  - "Trade log passed through compute_backtest_metrics for full audit trail"

patterns-established:
  - "Walk-forward engine pattern: splitter produces folds, engine trains/predicts/applies-risk per fold, evaluation aggregates"
  - "Backtest result pattern: BacktestResult dataclass captures all metrics + equity curve + trade log for downstream analysis"

requirements-completed: [MODL-02]

# Metrics
duration: 5min
completed: 2026-02-19
---

# Phase 2 Plan 05: Walk-Forward Backtesting Engine Summary

**Purged walk-forward backtest engine with embargo gap, per-fold LightGBM training, volume-adaptive slippage, fractional Kelly sizing, and circuit breaker integration producing slippage-adjusted Sharpe/drawdown/hit-rate metrics**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-19T10:01:25Z
- **Completed:** 2026-02-19T10:06:55Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- PurgedWalkForwardSplit with expanding window, configurable embargo gap, and graceful degradation when data is insufficient
- WalkForwardEngine integrates full risk stack per fold: BaselineModel training, probability predictions, circuit breaker checks, fractional Kelly position sizing with volume cap, and volume-adaptive slippage
- BacktestResult dataclass captures Sharpe ratio, total return, max drawdown, hit rate, per-fold Sharpes, equity curve, and complete trade log
- 19 tests covering splitter properties (no overlap, expanding window, embargo gap, edge cases) and evaluation metrics (Sharpe sign, drawdown, hit rate, fold Sharpes)
- End-to-end smoke test on 500-sample synthetic data: 3 folds, 78 trades, produces valid BacktestResult

## Task Commits

Each task was committed atomically:

1. **Task 1: PurgedWalkForwardSplit + evaluation metrics + tests** - `3f5cfc7` (feat)
2. **Task 2: WalkForwardEngine integrating model + risk stack** - `5c80d6f` (feat)

## Files Created/Modified
- `src/hydra/model/walk_forward.py` - PurgedWalkForwardSplit (expanding-window CV with embargo) and WalkForwardEngine (full backtest loop with risk stack)
- `src/hydra/model/evaluation.py` - BacktestResult dataclass and compute_backtest_metrics (Sharpe, drawdown, hit rate, equity curve)
- `tests/test_walk_forward.py` - 9 tests: split count, no overlap, expanding window, embargo gap, min train size, insufficient data, zero samples, test size consistency, get_n_splits
- `tests/test_evaluation.py` - 10 tests: Sharpe sign (positive/negative), max drawdown, hit rate (perfect/random), empty returns, total return, n_trades, fold Sharpes, equity curve length
- `src/hydra/model/__init__.py` - Added re-exports for WalkForwardEngine, PurgedWalkForwardSplit, BacktestResult, compute_backtest_metrics

## Decisions Made
- Single-period returns `(price[t]-price[t-1])/price[t-1]` for PnL computation -- simple, avoids any lookahead within the test window
- Circuit breakers reset per fold to prevent carry-over bias between out-of-sample periods; each fold starts with clean risk state
- Running win/loss statistics for Kelly sizing within each fold -- position size adapts as the fold progresses based on realized performance
- Trade log collected in engine and passed through `compute_backtest_metrics` via new `trade_log` parameter for full audit trail

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Trade log not propagated to BacktestResult**
- **Found during:** Task 2 (smoke test verification)
- **Issue:** `compute_backtest_metrics` always returned empty `trade_log=[]` even though WalkForwardEngine collected a full trade log
- **Fix:** Added `trade_log` parameter to `compute_backtest_metrics` and passed the engine's trade_log through to BacktestResult
- **Files modified:** src/hydra/model/evaluation.py, src/hydra/model/walk_forward.py
- **Verification:** Smoke test confirms `len(result.trade_log) == result.n_trades` (78 entries)
- **Committed in:** 5c80d6f (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in trade log propagation)
**Impact on plan:** Auto-fix ensured trade log audit trail works end-to-end. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Walk-forward backtest engine complete -- Phase 2 validation gate can now be evaluated
- Run `WalkForwardEngine().run(X, y, prices)` with real feature data to get OOS Sharpe ratio
- If OOS Sharpe > 0 after slippage: proceed to Phase 3 (hyperparameter tuning)
- If OOS Sharpe <= 0: re-examine divergence signal thesis before proceeding
- Engine ready for Phase 3 integration (hyperparameter tuning will modify model params, engine remains stable)
- Risk stack fully integrated: slippage, Kelly sizing, circuit breakers all applied per trade

## Self-Check: PASSED

All 5 files verified present. Both task commits (3f5cfc7, 5c80d6f) verified in git log.

---
*Phase: 02-signal-layer-baseline-model*
*Completed: 2026-02-19*
