# Requirements: HYDRA

**Defined:** 2026-02-19
**Core Value:** The agent loop must reliably detect model degradation, diagnose root causes, generate and test improvement hypotheses, and promote better models — all without human intervention.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Infrastructure

- [x] **DATA-01**: System ingests futures OHLCV price bars (EOD + intraday) for the target market from a data vendor
- [x] **DATA-02**: System ingests full options chain data (all strikes, bids, asks, OI, volume, expiry) for the target market
- [x] **DATA-03**: System ingests CFTC Commitments of Traders (COT) reports with correct as-of/release date handling
- [x] **DATA-04**: Feature store provides point-in-time correct queries ("what features were available at time T?") preventing lookahead bias
- [x] **DATA-05**: Raw data persisted in Parquet format with append-only semantics
- [x] **DATA-06**: Data quality monitoring detects staleness, missing strikes, and anomalous values with configurable alerts

### Options Math Engine

- [x] **OPTS-01**: Breeden-Litzenberger extracts risk-neutral implied probability distribution from options chain
- [x] **OPTS-02**: Implied moments computed from distribution (mean, variance, skew, kurtosis)
- [x] **OPTS-03**: Volatility surface constructed across strike x expiry grid with SVI smoothing for sparse data
- [x] **OPTS-04**: Greeks flow aggregation computed (GEX, vanna, charm) from options chain open interest and volume
- [x] **OPTS-05**: Options math gracefully degrades when data quality is insufficient (< 8 liquid strikes falls back to ATM implied vol)

### Signal Layer

- [x] **SGNL-01**: COT data produces normalized sentiment score in [-1, +1] with confidence weight
- [x] **SGNL-02**: Divergence detector classifies options-implied vs. sentiment divergence into 6 types per PRD taxonomy
- [x] **SGNL-03**: Divergence output includes direction, magnitude, type, confidence, and suggested bias

### Model Layer

- [x] **MODL-01**: LightGBM baseline model trained on divergence + feature store features produces directional predictions
- [x] **MODL-02**: Walk-forward backtesting with expanding/rolling window and embargo gaps validates model out-of-sample
- [x] **MODL-03**: Fractional Kelly position sizing caps positions at configurable fraction of average daily volume
- [x] **MODL-04**: Circuit breakers halt trading on max daily loss, max drawdown, max position size, or max single-trade loss thresholds
- [x] **MODL-05**: All backtest and evaluation metrics are slippage-adjusted using volume-adaptive slippage model

### Sandbox & Monitoring

- [x] **SBOX-01**: Market replay engine replays historical data with volume-adaptive thin-market slippage model
- [x] **SBOX-02**: Model registry (MLflow) tracks all trained models with full config snapshot, metrics, and champion/candidate/archived lifecycle
- [x] **SBOX-03**: Experiment journal logs every experiment with hypothesis, config diff, results, promotion decision, and tags
- [x] **SBOX-04**: Observer detects model drift via rolling performance metrics (Sharpe, drawdown, hit rate, calibration) and feature distribution drift (PSI, KS, ADWIN/CUSUM)
- [x] **SBOX-05**: Evaluator scores candidate models on 6-metric composite fitness (Sharpe 0.25, drawdown 0.20, calibration 0.15, robustness 0.15, slippage-adjusted return 0.15, simplicity 0.10)
- [x] **SBOX-06**: Journal is queryable by tag, date range, mutation type, and outcome

### Agent Core

- [ ] **AGNT-01**: Agent runs full observe -> diagnose -> hypothesize -> experiment -> evaluate loop autonomously
- [ ] **AGNT-02**: Diagnostician performs structured triage (data audit, SHAP attribution, regime check, overfitting test) before proposing fixes
- [ ] **AGNT-03**: Hypothesis engine proposes mutations from curated playbook matched to diagnosed root causes
- [ ] **AGNT-04**: Experiment runner executes candidate training in isolation (subprocess) with configurable timeout
- [ ] **AGNT-05**: LLM client calls DeepSeek-R1 with fallback chain (DeepSeek-R1 -> Qwen2.5 -> rule-based) and Pydantic validation on all outputs
- [ ] **AGNT-06**: Autonomy levels (supervised, semi-auto, autonomous, lockdown) are configurable and gate agent actions appropriately
- [ ] **AGNT-07**: Automatic rollback triggers on sustained performance degradation with hysteresis to prevent flapping
- [ ] **AGNT-08**: Candidate must beat champion on composite fitness across 3 of 5 independent evaluation windows for promotion
- [ ] **AGNT-09**: Mutation budgets, semantic deduplication, and cooldowns prevent degenerate experiment loops
- [ ] **AGNT-10**: System operates in degraded mode (rule-based fallbacks) when LLM is unavailable

### CLI Interface

- [ ] **CLI-01**: `status` command shows model health, active experiments, alerts, current autonomy level
- [ ] **CLI-02**: `diagnose` command forces a diagnostic cycle on the current champion model
- [ ] **CLI-03**: `rollback` command reverts to previous champion model
- [ ] **CLI-04**: `pause` / `run` commands halt and resume the agent loop
- [ ] **CLI-05**: `journal query` command searches experiment history by tag, date, mutation type
- [ ] **CLI-06**: CLI uses Rich-formatted terminal output with tables, colored alerts, and progress indicators

### Execution & Paper Trading

- [ ] **EXEC-01**: Paper trading pipeline uses Interactive Brokers paper account with same execution path as live
- [ ] **EXEC-02**: All paper trading fills logged with timestamps and actual slippage for model validation
- [ ] **EXEC-03**: Order management implements smart order routing (limit orders with patience, TWAP for larger positions)
- [ ] **EXEC-04**: Risk management runs as middleware in execution path (not optional check)
- [ ] **EXEC-05**: 4+ weeks of stable paper trading with at least one successful self-healing cycle before any live capital

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Enhanced Signals

- **ESIG-01**: FinBERT NLP sentiment scoring on financial news headlines
- **ESIG-02**: Semantic journal query via LLM-powered natural language search
- **ESIG-03**: Social sentiment integration (Twitter/Reddit) with noise filtering

### Advanced Sandbox

- **ASBX-01**: Synthetic data generator (copula-based with parameterized stress scenarios)
- **ASBX-02**: A/B testing framework running champion and candidate side-by-side on live data feed
- **ASBX-03**: Regime-aware model selection with HMM or online change-point detection

### Scaling

- **SCAL-01**: Multi-market expansion with market-agnostic abstractions
- **SCAL-02**: Tick-level order flow features for markets where marginal value is demonstrated
- **SCAL-03**: Conditional GAN synthetic data generation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Web/Mobile Dashboard | Single operator — CLI provides everything needed. UI development effort yields zero alpha. |
| Reinforcement Learning Agent | No proven production solutions; terrible sample efficiency; the mutation-based agent loop is more practical and debuggable |
| Custom Execution Engine | IB API is battle-tested; HYDRA's edge is signal quality and self-healing, not execution speed |
| Blockchain/DeFi Integration | Entirely different infrastructure; focus on CME/ICE futures through IB |
| Social/Copy Trading | Personal quant tool; adds regulatory burden and zero alpha |
| Complex Options Strategies | HYDRA reads options for signals; trades futures for positions |
| GPU-Dependent Infrastructure | LightGBM is CPU-optimized; single-market data volume doesn't warrant GPU |
| Kubernetes/Cloud | Single developer, single market, single machine; Docker Compose for DB only |
| Automated Strategy Discovery | The divergence thesis IS the strategy; unbounded search produces spurious results |
| LangChain/LangGraph | Agent loop is a fixed state machine; direct API calls via openai package |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Complete |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| OPTS-01 | Phase 1 | Complete |
| OPTS-02 | Phase 1 | Complete |
| OPTS-03 | Phase 1 | Complete |
| OPTS-04 | Phase 1 | Complete |
| OPTS-05 | Phase 1 | Complete |
| SGNL-01 | Phase 2 | Complete |
| SGNL-02 | Phase 2 | Complete |
| SGNL-03 | Phase 2 | Complete |
| MODL-01 | Phase 2 | Complete |
| MODL-02 | Phase 2 | Complete |
| MODL-03 | Phase 2 | Complete |
| MODL-04 | Phase 2 | Complete |
| MODL-05 | Phase 2 | Complete |
| SBOX-01 | Phase 3 | Complete |
| SBOX-02 | Phase 3 | Complete |
| SBOX-03 | Phase 3 | Complete |
| SBOX-04 | Phase 3 | Complete |
| SBOX-05 | Phase 3 | Complete |
| SBOX-06 | Phase 3 | Complete |
| AGNT-01 | Phase 4 | Pending |
| AGNT-02 | Phase 4 | Pending |
| AGNT-03 | Phase 4 | Pending |
| AGNT-04 | Phase 4 | Pending |
| AGNT-05 | Phase 4 | Pending |
| AGNT-06 | Phase 4 | Pending |
| AGNT-07 | Phase 4 | Pending |
| AGNT-08 | Phase 4 | Pending |
| AGNT-09 | Phase 4 | Pending |
| AGNT-10 | Phase 4 | Pending |
| CLI-01 | Phase 3 | Pending |
| CLI-02 | Phase 3 | Pending |
| CLI-03 | Phase 3 | Pending |
| CLI-04 | Phase 3 | Pending |
| CLI-05 | Phase 3 | Pending |
| CLI-06 | Phase 3 | Pending |
| EXEC-01 | Phase 5 | Pending |
| EXEC-02 | Phase 5 | Pending |
| EXEC-03 | Phase 5 | Pending |
| EXEC-04 | Phase 5 | Pending |
| EXEC-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 46 total
- Mapped to phases: 46
- Unmapped: 0

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-19 after initial definition*
