# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** The agent loop must reliably detect model degradation, diagnose root causes, generate and test improvement hypotheses, and promote better models -- all without human intervention.
**Current focus:** Phase 2: Signal Layer + Baseline Model

## Current Position

Phase: 2 of 5 (Signal Layer + Baseline Model)
Plan: 5 of 5 in current phase
Status: Phase Complete
Last activity: 2026-02-19 -- Completed 02-05-PLAN.md

Progress: [##########] 100% (Phase 2)

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 5min
- Total execution time: 1.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 6 | 36min | 6min |
| 02 | 5 | 23min | 4.6min |

**Recent Trend:**
- Last 5 plans: 02-01 (2min), 02-02 (3min), 02-03 (7min), 02-04 (6min), 02-05 (5min)
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Five-phase structure following strict data-flow dependencies; two hard validation gates after Phase 1 (options math stability) and Phase 2 (signal predictive power)
- [Roadmap]: Phase 4 flagged for deeper research before planning (LLM structured output reliability, prompt engineering, mutation playbook design)
- [01-01]: SQLite with WAL mode for Phase 1 feature store; schema designed for TimescaleDB migration
- [01-01]: Hive partitioning by data_type/market/year/month for Parquet lake
- [01-01]: UUID-based unique file naming for append-only Parquet semantics
- [01-03]: L-BFGS-B with ftol=1e-14 and maxiter=1000 for robust SVI convergence on sparse data
- [01-03]: Sparse data produces warnings rather than errors (< 8 strikes), per OPTS-05 graceful degradation
- [01-03]: Pure NumPy/SciPy for SVI -- no QuantLib needed
- [01-04]: DataQuality enum consolidated into density.py as canonical location; greeks.py imports from density.py
- [01-04]: Brentq IV inversion with xtol=1e-10, maxiter=200; returns None on failure for safe degradation
- [01-04]: 200-point fine grid with 5% margin for stable B-L second derivatives
- [01-04]: Negative density clipped to zero with warning, then renormalized to integrate to 1.0
- [01-05]: Scalar math module for per-option Black-76 computation; scipy.stats.norm for PDF/CDF
- [01-05]: Liquidity filter uses OR logic (strike liquid if either call or put OI meets threshold)
- [Phase 01]: [01-02]: COT available_at uses fixed EST offset (UTC-5); production should use proper DST handling
- [Phase 01]: [01-02]: Options chain joined from three Databento schemas (mbp-1, definition, statistics) on instrument_id
- [01-06]: Weekend-only heuristic for trading day detection; no holiday calendar in Phase 1
- [01-06]: Three-tier quality status: healthy/degraded/stale maps to operational decisions
- [01-06]: Configurable thresholds via config dict for per-environment tuning
- [02-01]: Confidence formula: 0.6*oi_rank + 0.4*min(concentration*5, 1.0) -- weights OI magnitude over concentration
- [02-01]: percentileofscore with default 'rank' method for percentile computation
- [02-01]: Minimum 4 weeks history threshold for neutral fallback
- [02-02]: upper_bound flag on CircuitBreaker distinguishes position_size breaker (>threshold) from loss breakers (<threshold)
- [02-02]: math.sqrt for slippage scalar ops -- no numpy dependency needed
- [02-03]: Priority-ordered classification: vol_play > bullish/bearish divergence > overreaction > early_signal > trend_follow > neutral
- [02-03]: Configurable module-level threshold constants for all classification rules
- [02-03]: Z-scoring requires minimum 10 historical divergence values; falls back to raw magnitude
- [02-03]: Degraded quality applies 0.5 confidence penalty factor; None implied_mean returns neutral
- [02-04]: Divergence + sentiment features computed live in assemble_at() rather than pre-stored in feature store
- [02-04]: LightGBM conservative defaults: num_leaves=31, lr=0.1, n_estimators=100 -- no tuning in Phase 2
- [02-04]: NaN preservation for LightGBM native NaN handling -- missing features passed as NaN, not imputed
- [02-05]: Single-period returns for PnL: (price[t]-price[t-1])/price[t-1] -- simple, avoids lookahead
- [02-05]: Circuit breakers reset per fold to prevent carry-over bias between OOS periods
- [02-05]: Running win/loss statistics for Kelly sizing within each fold -- adapts position size based on realized performance

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: Data vendor selection not yet finalized -- Databento vs. CME DataMine vs. IB historical needs hands-on evaluation
- [Phase 1]: Target market selection (oats vs. lean hogs vs. ethanol) depends on which has sufficient options chain data

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 02-05-PLAN.md (Walk-forward backtesting engine) -- Phase 2 COMPLETE
Resume file: None
Next: Phase 3 planning (pending Phase 2 validation gate: OOS Sharpe > 0)
