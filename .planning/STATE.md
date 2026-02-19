# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-18)

**Core value:** The agent loop must reliably detect model degradation, diagnose root causes, generate and test improvement hypotheses, and promote better models -- all without human intervention.
**Current focus:** Phase 5: Execution + Hardening

## Current Position

Phase: 5 of 5 (Execution + Hardening)
Plan: 2 of 5 in current phase
Status: Executing Phase 5
Last activity: 2026-02-19 -- Completed 05-01-PLAN.md (Broker Abstraction + Risk Gate)

Progress: [##--------] 20% (Phase 5: 1/5 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 23
- Average duration: 4min
- Total execution time: 1.45 hours

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

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 1]: Data vendor selection not yet finalized -- Databento vs. CME DataMine vs. IB historical needs hands-on evaluation
- [Phase 1]: Target market selection (oats vs. lean hogs vs. ethanol) depends on which has sufficient options chain data

## Session Continuity

Last session: 2026-02-19
Stopped at: Completed 05-01-PLAN.md (Broker Abstraction + Risk Gate)
Resume file: None
Next: 05-02-PLAN.md (Order Management)
