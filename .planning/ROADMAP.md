# Roadmap: HYDRA

## Overview

HYDRA builds from raw data to autonomous self-healing through a strict five-phase dependency chain. Data infrastructure and options math come first because every downstream component depends on computable features from market data. Signal construction and thesis validation come second because building self-healing infrastructure around a worthless signal is the most expensive mistake possible. Sandbox and experiment infrastructure come third because the agent loop needs a safe environment before it exists. The agent core with LLM integration comes fourth as the primary competitive moat. Execution and paper trading come last because real money is the highest-risk concern and depends on everything else working. Two hard validation gates (after Phase 1 and Phase 2) force early confrontation with existential uncertainties before committing to downstream infrastructure.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Infrastructure + Options Math Engine** - Ingestion pipelines, feature store, and options math that extract computable signals from thin-market data
- [x] **Phase 2: Signal Layer + Baseline Model** - Divergence signal construction, baseline ML model, and walk-forward thesis validation
- [x] **Phase 3: Sandbox + Experiment Infrastructure** - Market replay, model registry, experiment journal, observer, evaluator, and CLI for safe experimentation
- [x] **Phase 4: Agent Core + LLM Integration** - Full autonomous agent loop with LLM-powered diagnosis, hypothesis generation, and self-healing capability
- [ ] **Phase 5: Execution + Hardening** - IB paper trading pipeline, order management, risk middleware, and 4+ week live validation

## Phase Details

### Phase 1: Data Infrastructure + Options Math Engine
**Goal**: Raw market data flows into a feature store and options math engine that produces stable implied distributions, moments, volatility surfaces, and Greeks from thin-market options chains
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, OPTS-01, OPTS-02, OPTS-03, OPTS-04, OPTS-05
**Success Criteria** (what must be TRUE):
  1. Futures OHLCV bars and full options chain data for the target market are ingested daily and persisted in Parquet with append-only semantics
  2. CFTC COT reports are ingested with correct as-of/release date handling (Tuesday data not available until Friday)
  3. Feature store answers point-in-time queries ("what was available at time T?") without lookahead bias
  4. Breeden-Litzenberger produces stable implied probability distributions from the target market's options chains (plot implied vs. realized distributions to verify)
  5. Options math gracefully degrades to ATM implied vol when fewer than 8 liquid strikes are available
**Validation Gate**: Can B-L produce stable implied distributions from thin-market options? If data is too sparse or noisy for stable signal extraction, the entire thesis needs re-examination before proceeding.
**Research Flag**: Data vendor selection (Databento vs. CME DataMine vs. IB historical) and target market selection (oats vs. lean hogs vs. ethanol) need hands-on evaluation of API quality, cost, and thin-market options chain completeness.
**Plans**: 6 plans in 4 waves

Plans:
- [x] 01-01-PLAN.md -- Project scaffold, Parquet lake, feature store with lookahead prevention
- [x] 01-02-PLAN.md -- Futures, options, and COT ingestion pipelines
- [x] 01-03-PLAN.md -- SVI volatility surface calibration (TDD)
- [x] 01-04-PLAN.md -- Breeden-Litzenberger density extraction + implied moments (TDD)
- [x] 01-05-PLAN.md -- Greeks flow aggregation: GEX, vanna, charm (TDD)
- [x] 01-06-PLAN.md -- Data quality monitoring + Phase 1 integration checkpoint

### Phase 2: Signal Layer + Baseline Model
**Goal**: The divergence between options-implied expectations and sentiment signals demonstrates out-of-sample predictive power, validated through walk-forward backtesting with realistic slippage
**Depends on**: Phase 1
**Requirements**: SGNL-01, SGNL-02, SGNL-03, MODL-01, MODL-02, MODL-03, MODL-04, MODL-05
**Success Criteria** (what must be TRUE):
  1. COT data produces a normalized sentiment score in [-1, +1] with confidence weight
  2. Divergence detector classifies options-implied vs. sentiment divergence into the 6-type taxonomy with direction, magnitude, confidence, and suggested bias
  3. LightGBM baseline model trained on divergence + feature store features produces directional predictions with walk-forward OOS Sharpe > 0 after slippage
  4. Circuit breakers halt trading when max daily loss, max drawdown, max position size, or max single-trade loss thresholds are breached
  5. All backtest and evaluation metrics use volume-adaptive slippage model (not fixed slippage)
**Validation Gate**: Does the divergence signal predict future price movement out-of-sample with Sharpe > 0 after slippage? If NO, stop. Re-examine the thesis, target market, or signal construction before proceeding to Phase 3.
**Plans**: 5 plans in 4 waves

Plans:
- [x] 02-01-PLAN.md -- COT sentiment scoring: normalized [-1,+1] score with confidence (TDD)
- [x] 02-02-PLAN.md -- Risk infrastructure: slippage, position sizing, circuit breakers (TDD)
- [x] 02-03-PLAN.md -- Divergence detector: 6-type taxonomy classification (TDD)
- [x] 02-04-PLAN.md -- Feature matrix assembler + LightGBM baseline model
- [x] 02-05-PLAN.md -- Walk-forward backtesting engine + evaluation metrics

### Phase 3: Sandbox + Experiment Infrastructure
**Goal**: A safe experimentation environment exists where models can be trained, evaluated, versioned, and compared through replay of historical data with realistic slippage, and an operator can inspect and control the system via CLI
**Depends on**: Phase 2
**Requirements**: SBOX-01, SBOX-02, SBOX-03, SBOX-04, SBOX-05, SBOX-06, CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06
**Success Criteria** (what must be TRUE):
  1. Market replay engine replays historical data with volume-adaptive thin-market slippage and produces results that correlate with real market behavior
  2. MLflow model registry tracks all trained models with full config snapshot, metrics, and champion/candidate/archived lifecycle
  3. Experiment journal logs every experiment with hypothesis, config diff, results, and promotion decision -- and is queryable by tag, date range, mutation type, and outcome
  4. Observer detects model drift via rolling performance metrics (Sharpe, drawdown, hit rate, calibration) and feature distribution drift (PSI, KS, ADWIN/CUSUM)
  5. CLI commands (`status`, `diagnose`, `rollback`, `pause`/`run`, `journal query`) work with Rich-formatted terminal output
**Plans**: 6 plans in 2 waves

Plans:
- [x] 03-01-PLAN.md -- Market replay engine with volume-adaptive slippage
- [x] 03-02-PLAN.md -- MLflow model registry with champion/candidate/archived lifecycle
- [x] 03-03-PLAN.md -- Experiment journal with SQLite storage and query layer
- [x] 03-04-PLAN.md -- Drift detectors (PSI, KS, CUSUM, ADWIN) + DriftObserver
- [x] 03-05-PLAN.md -- Composite fitness evaluator with 6-metric weighted scoring
- [x] 03-06-PLAN.md -- Typer CLI with Rich formatting (status, diagnose, rollback, pause/run, journal)

### Phase 4: Agent Core + LLM Integration (Single-Head "Honda" Version)
**Goal**: A single-head autonomous agent loop that detects model drift, diagnoses root causes via rule-based triage, proposes mutations from a curated playbook, tests them in sandbox isolation, and promotes winners -- with optional LLM enhancement for ambiguous cases. All guardrails (autonomy, rollback, promotion, dedup) are in place. The system runs at zero token cost by default. Multi-head architecture (AGNT-11-18) deferred to future "turbocharge" phase.
**Depends on**: Phase 3
**Requirements**: AGNT-01, AGNT-02, AGNT-03, AGNT-04, AGNT-05, AGNT-06, AGNT-07, AGNT-08, AGNT-09, AGNT-10
**Deferred**: AGNT-11, AGNT-12, AGNT-13, AGNT-14, AGNT-15, AGNT-16, AGNT-17, AGNT-18 (multi-head architecture -- future phase)
**Success Criteria** (what must be TRUE):
  1. Single-head agent loop works end-to-end: observe-diagnose-hypothesize-experiment-evaluate with each step logged to the experiment journal
  2. LLM client calls DeepSeek-R1 with Pydantic-validated structured output and falls back through the chain (DeepSeek-R1 -> Qwen2.5 -> rule-based) when parsing fails, with parse failure rate below 5%
  3. Autonomy levels (supervised, semi-auto, autonomous, lockdown) gate agent actions appropriately -- supervised requires human approval for promotion, autonomous does not
  4. Automatic rollback triggers on sustained performance degradation with hysteresis to prevent flapping, and candidate promotion requires beating champion on composite fitness across 3 of 5 independent evaluation windows
  5. Mutation budgets, semantic deduplication (embedding similarity > 0.85 = reject), and cooldowns prevent degenerate experiment loops
  6. System operates entirely rule-based by default with zero LLM API calls; LLM is optional enhancement
**Research Flag**: Research complete (04-RESEARCH.md). Multi-head sections archived in `.planning/phases/04-agent-core-llm-integration/archived-multihead/`.
**Plans**: 5 plans in 2 waves

Plans:
- [x] 04-01-PLAN.md -- LLM client with fallback chain, task-type router, Pydantic schemas, cost tracking
- [x] 04-02-PLAN.md -- Autonomy gating, hysteresis rollback, 3-of-5 promotion evaluation
- [x] 04-03-PLAN.md -- Diagnostician (rule-based structured triage), hypothesis engine (mutation playbook)
- [x] 04-04-PLAN.md -- Experiment runner (subprocess isolation), semantic dedup, mutation budgets + cooldowns
- [x] 04-05-PLAN.md -- Agent loop state machine wiring all modules into observe-diagnose-hypothesize-experiment-evaluate cycle

### Phase 5: Execution + Hardening
**Goal**: The system trades on paper through Interactive Brokers with the same execution path as live, validates that simulated performance matches real fills, and demonstrates at least one successful autonomous self-healing cycle over 4+ weeks
**Depends on**: Phase 4
**Requirements**: EXEC-01, EXEC-02, EXEC-03, EXEC-04, EXEC-05
**Success Criteria** (what must be TRUE):
  1. Paper trading pipeline uses IB paper account with the same order management and risk management path that live trading would use
  2. All paper trading fills are logged with timestamps and actual slippage, and simulated slippage model is validated against real fill data
  3. Order management implements smart order routing (limit orders with patience, TWAP for larger positions)
  4. Risk management runs as mandatory middleware in the execution path, not an optional check
  5. System completes 4+ weeks of stable paper trading with at least one successful self-healing cycle (agent detects degradation, diagnoses, proposes fix, tests, promotes improvement that lasts 30+ days)
**Validation Gate**: 4+ weeks paper trading with stable or improving performance AND at least one successful autonomous self-healing cycle before any live capital.
**Research Flag**: ib_insync maintenance status must be verified before committing. If unmaintained, evaluate ib-async (community fork) or plan a custom wrapper around ibapi.
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Infrastructure + Options Math Engine | 6/6 | Complete    | 2026-02-19 |
| 2. Signal Layer + Baseline Model | 5/5 | Complete    | 2026-02-19 |
| 3. Sandbox + Experiment Infrastructure | 0/6 | Complete    | 2026-02-19 |
| 4. Agent Core + LLM Integration (Single-Head) | 5/5 | Complete    | 2026-02-19 |
| 5. Execution + Hardening | 0/TBD | Not started | - |
