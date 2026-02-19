# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** The agent loop must reliably detect model degradation, diagnose root causes, generate and test improvement hypotheses, and promote better models -- all without human intervention.
**Current focus:** Phase 6 added -- Dashboard + monitoring + containerisation

## Current Position

Phase: 6 of 6 (Dashboard + Monitoring + Containerisation)
Plan: 2 of 4 in current phase
Status: Executing Phase 6 -- Docker containerisation complete
Last activity: 2026-02-19 -- Docker Compose stack created (Dockerfile, compose, .env.example, .dockerignore)

Progress: [#####-----] 50% (Phase 6: 2/4 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 28
- Average duration: 4min
- Total execution time: 1.79 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 6 | 36min | 6min |
| 02 | 5 | 23min | 4.6min |
| 03 | 6 | 18min | 3min |

**Recent Trend:**
- Last 5 plans: 02-04 (6min), 02-05 (5min), 03-05 (2min), 03-06 (8min)
- Trend: Stable

*Updated after each plan completion*
| Phase 03 P03 | 3min | 2 tasks | 3 files |
| Phase 03 P04 | 3min | 2 tasks | 10 files |
| Phase 03 P01 | 5min | 2 tasks | 3 files |
| Phase 03 P06 | 8min | 2 tasks | 7 files |
| Phase 04 P02 | 4min | 2 tasks | 7 files |
| Phase 04 P01 | 4min | 2 tasks | 11 files |
| Phase 04 P03 | 5min | 2 tasks | 5 files |
| Phase 04 P04 | 8min | 2 tasks | 7 files |
| Phase 04 P05 | 8min | 2 tasks | 4 files |
| Phase 05 P01 | 3min | 2 tasks | 6 files |
| Phase 05 P02 | 3min | 1 tasks | 3 files |
| Phase 05 P03 | 3min | 2 tasks | 5 files |
| Phase 05 P04 | 5min | 2 tasks | 5 files |
| Phase 05 P05 | 8min | 3 tasks | 2 files |
| Phase 06 P02 | 4min | 2 tasks | 5 files |

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
- [03-05]: Inverted normalization via _INVERTED_METRICS frozenset -- extensible to future metrics where lower is better
- [03-05]: Duck-typed BacktestResult access via attribute access -- no import coupling to model/evaluation.py
- [Phase 03]: [03-03]: LIKE-based tag querying on JSON array column for SQLite compatibility
- [Phase 03]: [03-03]: Dynamic WHERE clause builder with parameterized AND-combined filters for journal queries
- [Phase 03]: [03-03]: ExperimentRecord required positional fields for core data, optional defaults for metadata extensibility
- [03-02]: Absolute file:// tracking URI default prevents MLflow relative-path confusion
- [03-02]: Alias-based lifecycle (champion/archived) instead of deprecated MLflow stages
- [03-02]: Explicit logging (not autolog) for full control over what gets tracked
- [Phase 03]: [03-04]: River library (v0.23) for ADWIN adaptive windowing -- avoids reimplementing complex algorithm
- [Phase 03]: [03-04]: ADWIN + CUSUM paired per streaming metric for complementary drift detection
- [Phase 03]: [03-04]: Epsilon smoothing (1e-4) with re-normalization for PSI zero-bin handling
- [Phase 03]: [03-01]: Self-contained metrics in replay.py to avoid circular import with evaluation.py
- [Phase 03]: [03-01]: Observer callback pattern (add_callback) for non-intrusive drift monitoring hooks
- [Phase 03]: [03-01]: Rolling volatility 20-bar lookback, 0.02 default for insufficient data
- [03-06]: Rich formatters return renderables (Table/Panel) rather than printing -- keeps formatters testable
- [03-06]: AgentState defaults to PAUSED when state file missing -- safe default prevents unintended agent execution
- [03-06]: Diagnose uses synthetic data in Phase 3; live data integration deferred to Phase 4
- [03-06]: Journal defaults to ~/.hydra/experiment_journal.db when no path provided
- [Phase 04]: Minimum-level comparison (level >= required) for permission gating -- simpler than nested dict from research
- [Phase 04]: Strict inequality for degradation threshold -- exactly at threshold is NOT degraded (conservative)
- [Phase 04]: Strict inequality for promotion wins -- tied fitness does NOT count as candidate win (conservative)
- [Phase 04]: Canonical types.py created as stub with DriftCategory/MutationType enums and DiagnosisResult/Hypothesis dataclasses -- 04-03 will extend
- [Phase 04]: instructor.from_openai wraps OpenAI clients for Together AI and DeepSeek providers with Pydantic structured output
- [Phase 04]: LLM schemas (Pydantic BaseModel) kept separate from core types (dataclasses) to avoid coupling agent to Pydantic
- [Phase 04]: Token estimation uses 4 chars/token heuristic for LLM cost tracking without requiring API usage metadata
- [04-03]: Priority-ordered classification: feature_drift > performance > regime_change > overfitting > default
- [04-03]: Config diff resolution via regex patterns (multiply, floor_div, max) -- no eval() for security
- [04-03]: propose_multiple caps at playbook size to avoid duplicate hypotheses
- [04-03]: LLM enhancement threshold: confidence < 0.6 triggers optional LLM call
- [04-04]: Defensive promotion_decision filtering in load_cooldowns_from_journal -- checks field on each record, not just query filter
- [04-04]: Config expression resolution via regex for +, -, *, / operators with "current" token (no eval())
- [04-04]: sentence-transformers installed as blocking dependency fix when 04-01 not yet executed
- [04-05]: Dual promotion path: 3-of-5 PromotionEvaluator when window scores provided, single-score comparison as Honda default
- [04-05]: check_permission returns graceful skip (not PermissionDeniedError) at each loop step for testability
- [04-05]: Diagnosis inconclusive threshold: confidence < 0.3 AND no evidence causes early exit
- [05-01]: Paper port 4002 as default with is_paper property for safe mode detection
- [05-01]: Client ID allocation: 1=trading, 2=diagnostic, 3=CLI
- [05-01]: Exponential backoff reconnection: 1s->2s->4s->8s max 30s, 10 attempts max
- [05-01]: RiskGate has no submit_order passthrough -- only submit() with mandatory risk check
- [05-02]: Module-level LimitOrder import from ib_async for testability via patch
- [05-02]: Three-stage patience escalation: mid-price -> step toward market -> cross spread with shrinking timeouts
- [05-02]: TWAP remainder distribution: first N slices get +1 contract (13/5 = [3,3,3,2,2])
- [05-02]: 10x price_step_pct for spread-crossing approximation
- [05-03]: Minimum 10 fills required for reconciliation -- returns None for insufficient data
- [05-03]: Constant-array correlation handled gracefully (returns 0.0 instead of NaN)
- [05-03]: Pessimism multiplier uses epsilon floor (1e-10) to avoid division by zero
- [05-04]: APScheduler AsyncIOScheduler with CronTrigger for daily cycle -- avoids blocking event loop
- [05-04]: Live mode requires HYDRA_LIVE_CONFIRMED=true env var -- double safety with CLI --yes-i-mean-live flag
- [05-04]: Agent loop and model.predict() are independent calls in daily cycle -- agent maintains quality, model produces signal
- [05-04]: CLI paper-trade displays config and exits -- long-running process uses python -m hydra.execution.runner
- [05-05]: Integration tests use mocked ib_async.IB for CI; skipif IB_GATEWAY_HOST for real IB testing
- [05-05]: Port safety verified in integration: port 4002 = paper (True), port 4001 = paper (False)
- [05-05]: IB Gateway setup deferred to when ready -- 4-week paper trading begins upon configuration
- [05-05]: PAPER_TRADING_PLAN.md requires ALL six gate conditions met before any live capital
- [06-02]: Single container for runner + dashboard (SQLite WAL safety)
- [06-02]: gnzsnz/ib-gateway:stable for headless IB Gateway containerisation
- [06-02]: Internal port 4004 for paper trading between Docker containers
- [06-02]: Named volumes for data persistence; dashboard bound to 127.0.0.1 only

### Roadmap Evolution

- Phase 6 added: Dashboard + monitoring for paper trading and full lightweight containerisation

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: Data vendor selection not yet finalized -- Databento vs. CME DataMine vs. IB historical needs hands-on evaluation
- [Phase 1]: Target market selection (oats vs. lean hogs vs. ethanol) depends on which has sufficient options chain data

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 06-02-PLAN.md
Resume file: None
Next: Continue Phase 6 execution (06-03, 06-04)

## Milestone v1 Summary

All 5 phases complete. 26 plans executed across 5 phases in ~1.66 hours.

**IB Gateway Dry Run Results (2026-02-19):**
- BrokerGateway: Connected, Paper mode, Port 4002
- Contract qualification: ZOK6 (ZO May '26), conId=703249611
- Historical bars: 20 bars, ADV=145, Volatility=0.0128, close=$322.0
- Market data: Delayed data (free) via reqMarketDataType(3), last=$322.0
- RiskGate: Safe params allowed, risky params blocked (max_daily_loss)
- FillJournal + Reconciler: 15 fills, bias=0.044, RMSE=0.050, corr=0.982
- 547 unit/integration tests passing
