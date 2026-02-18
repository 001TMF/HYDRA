# Project Research Summary

**Project:** HYDRA
**Domain:** Autonomous self-healing ML trading agent for low-volume futures markets
**Researched:** 2026-02-19
**Confidence:** MEDIUM

## Executive Summary

HYDRA is a layered system where deterministic computation (data ingestion, options math, ML training) flows upward through well-defined layers, and an LLM-powered agent loop sits on top as a meta-controller -- deciding what to improve, not what to trade. Experts build systems like this by separating the signal pipeline (options-implied probability distributions, sentiment scoring, divergence detection) from the model pipeline (training, evaluation, promotion) and from the agent pipeline (observation, diagnosis, hypothesis, experimentation). The critical architectural insight from research is that the LLM must never touch the prediction hot path. It proposes; deterministic code disposes. This pattern eliminates an entire class of catastrophic failures where LLM hallucinations become trading losses.

The recommended approach is a strict five-phase build that follows data flow dependencies: data infrastructure and options math first (Phase 1), signal layer and thesis validation second (Phase 2), sandbox and experiment infrastructure third (Phase 3), agent core with LLM integration fourth (Phase 4), and execution/hardening last (Phase 5). Two hard validation gates punctuate this sequence. After Phase 1, the options chain data quality must be verified -- can Breeden-Litzenberger produce stable implied distributions from thin-market data? After Phase 2, the divergence signal must demonstrate out-of-sample predictive power (Sharpe > 0 after slippage). If either gate fails, work stops and the upstream problem is fixed before building further. This is non-negotiable. Building self-healing infrastructure around a worthless signal is the most expensive mistake possible.

The top risks are (1) options math numerical instability in thin markets with sparse, wide-spread strike ladders, (2) lookahead bias silently inflating every backtest the agent runs, (3) unrealistic slippage models that make sandbox results diverge from reality, (4) agent degenerate experiment loops that burn budget without improvement, and (5) cheap LLM structured output failures that cause the agent to act on hallucinated reasoning. All five have concrete mitigations identified in the pitfalls research, and each maps to a specific phase where it must be addressed. The stack is Python 3.11, LightGBM for baseline modeling, QuantLib + NumPy/SciPy for options math, TimescaleDB for feature storage, MLflow for experiment tracking, DeepSeek-R1 (distilled 70B) via Together AI for agent reasoning, and Interactive Brokers for execution in Phase 5+.

## Key Findings

### Recommended Stack

The stack centers on Python 3.11 with a deliberate bias toward lightweight, focused tools over heavyweight frameworks. The most consequential technology decisions are: DeepSeek-R1 distilled 70B via Together AI for agent reasoning (best cost/quality ratio for analytical tasks at ~$6/month projected), LightGBM as the baseline model (fast iteration, interpretable via SHAP, start simple), TimescaleDB as the unified feature store and performance database (PostgreSQL with time-series superpowers, replaces both a dedicated time-series DB and a separate relational DB), and MLflow for self-hosted experiment tracking (trading IP stays local). See `.planning/research/STACK.md` for full details.

**Core technologies:**
- **Python 3.11 + uv:** Runtime and package management -- 3.11 is the sweet spot for ML library compatibility; uv replaces pip+virtualenv with 10-100x faster resolution
- **DeepSeek-R1-Distill-Qwen-70B (Together AI):** Agent reasoning layer -- strongest chain-of-thought reasoning among cheap Chinese models; ~$0.20/day at projected usage; Qwen2.5-72B as fallback if structured output proves unreliable
- **LightGBM (>=4.3):** Baseline ML model -- best tree-based model for tabular data; fast training; SHAP feature importances power the diagnostician
- **QuantLib + NumPy/SciPy:** Options math -- QuantLib for vol surface construction (SABR), NumPy/SciPy for Breeden-Litzenberger and Greeks flow aggregation; use both, each for what it does best
- **TimescaleDB (>=2.14):** Feature store, performance DB, experiment journal -- single PostgreSQL instance with time-series extensions; continuous aggregates for point-in-time correctness; replaces Feast/InfluxDB
- **MLflow (>=2.12):** Experiment tracking and model registry -- self-hosted (trading IP stays local); champion/candidate/archived lifecycle
- **APScheduler:** Agent loop scheduling -- in-process, lightweight; replaces Celery/Airflow which are overkill for single-process agent
- **Typer + Rich:** CLI framework -- better DX than Click; formatted terminal output for operator dashboards

**Critical "do NOT use" decisions:**
- No LangChain/LangGraph -- the agent loop is a fixed state machine, not a dynamic graph; direct API calls via `openai` package
- No Pandas on hot path -- NumPy for computation, Polars for transforms, DuckDB for analytics
- No Docker for experiment isolation initially -- subprocess isolation sufficient; add Docker Phase 4+ if needed
- No Feast -- overkill for single-developer, single-market; TimescaleDB continuous aggregates provide the same guarantees
- No Airflow/Prefect -- DAG schedulers for batch pipelines; HYDRA's agent is a continuous loop

### Expected Features

The feature landscape divides cleanly into three tiers with hard dependencies between them. See `.planning/research/FEATURES.md` for the complete analysis.

**Must have (table stakes -- system does not function without these):**
- **Data Pipeline** (futures OHLCV, options chains, COT) -- the root dependency; nothing works without data flowing
- **Options Math Engine** (Breeden-Litzenberger, implied moments, vol surface, Greeks flows) -- core signal extraction; the entire trading thesis depends on this
- **Sentiment Engine** (COT scoring minimum, FinBERT NLP recommended) -- the counterparty view for divergence detection
- **Divergence Detector** -- the alpha signal itself; options-implied vs. sentiment divergence
- **Feature Store with point-in-time correctness** -- prevents look-ahead bias; the #1 source of backtest fraud
- **Baseline LightGBM model + walk-forward backtesting** -- converts features into predictions with proper temporal validation
- **Circuit breakers + position sizing (fractional Kelly)** -- non-negotiable safety for anything touching capital
- **CLI interface** -- operator must be able to inspect, control, and override the system
- **Autonomy levels** -- graduated trust from supervised to autonomous; start locked down

**Should have (differentiators -- what makes HYDRA genuinely novel):**
- **Full agent loop** (observe -> diagnose -> hypothesize -> experiment -> evaluate) -- the primary competitive moat; no open-source system does this
- **Diagnostician with SHAP attribution** -- diagnoses WHY models fail, not just THAT they fail
- **Hypothesis Engine with mutation playbook** -- targeted evolution, not brute force search
- **LLM-powered reasoning** (DeepSeek-R1) -- natural language diagnosis and hypothesis generation
- **Champion/candidate promotion protocol** -- multi-objective fitness evaluation prevents single-metric overfitting
- **Market replay engine with thin-market slippage** -- no off-the-shelf backtester handles this correctly
- **Experiment journal with semantic query** -- long-term institutional memory; becomes the most valuable asset over time

**Defer (v2+):**
- Synthetic data generation (conditional GAN) -- copula-based is sufficient initially
- Regime-aware model selection -- need multiple regime transitions in data first
- Social sentiment (Twitter/Reddit) -- noisy, expensive; COT + NLP likely sufficient
- Multi-market expansion -- prove thesis on one market first
- A/B testing framework -- requires double paper trading infrastructure

### Architecture Approach

The architecture is a strict five-layer system (Data Infrastructure -> Signal -> Sandbox -> Agent -> Execution) with one additional CLI layer that touches everything. Each layer depends only on the layer below it. No upward dependencies. The agent layer is explicitly a meta-controller: it does not make trading predictions; it decides how to improve the thing that makes predictions. The LLM has exactly three touchpoints (Diagnostician, Hypothesis Engine, Journal query), each wrapped in a deterministic envelope of Pydantic validation, playbook constraints, and journal deduplication. The system MUST be able to run in degraded mode without the LLM -- rule-based fallbacks at every LLM touchpoint. See `.planning/research/ARCHITECTURE.md` for full details.

**Major components:**
1. **Data Infrastructure Layer** -- Ingestion pipelines (one per source), Feature Store (TimescaleDB), Performance DB, Raw Data Lake (Parquet)
2. **Signal Layer** -- Options Math Engine (stateless compute: B-L density, moments, surface, Greeks), Sentiment Engine (COT + NLP scoring), Divergence Detector (deterministic classification)
3. **Sandbox Layer** -- Market Replay Engine (custom thin-market slippage), Model Registry (MLflow), Evaluator/Judge (multi-objective fitness), Synthetic Data Generator
4. **Agent Layer** -- Observer (drift detection via ADWIN/CUSUM), Diagnostician (LLM-powered triage), Hypothesis Engine (LLM-powered mutation proposal), Experiment Runner (subprocess isolation), Experiment Journal (structured memory)
5. **Execution Layer** (Phase 5+) -- Order Manager (Kelly sizing), Risk Manager (circuit breakers), IB Gateway (ib_insync)

**Key architectural patterns:**
- Deterministic envelope around LLM reasoning -- LLM proposes, code validates and executes
- Event bus for cross-layer communication -- components publish/subscribe rather than direct calls
- Feature Store as single source of truth -- all models read from the store, never from raw data
- Abstract model interface -- architecture swaps via config change, not code change
- Experiment isolation via process forking -- no shared mutable state between champion and candidates
- Circuit breaker state machine -- explicit operational states (RUNNING -> DEGRADED -> HALTED -> LOCKDOWN)

### Critical Pitfalls

The research identified 18 pitfalls including 4 compound risks unique to HYDRA's specific combination of cheap LLM + thin markets + autonomous operation. The top 5 are summarized here; see `.planning/research/PITFALLS.md` for the complete analysis with prevention strategies and warning signs.

1. **Lookahead bias in backtesting** (CATASTROPHIC, Phase 1-3) -- Implement strict `as_of` timestamps on every data point from day one. COT data collected Tuesday must not be available until Friday release. Build a "time traveler" detector that flags features with suspiciously high correlation to the target. Every experiment the agent runs compounds any systematic leakage.

2. **Options math numerical instability in thin markets** (CATASTROPHIC, Phase 1) -- Thin markets have 5-15 liquid strikes with 10-30% bid-ask spreads. Breeden-Litzenberger's second derivatives amplify noise quadratically. Pre-smooth with SVI parametric surface fitting. Require minimum 8 liquid strikes before computing a distribution. Fall back to ATM implied vol when data quality is insufficient.

3. **Unrealistic slippage models** (CATASTROPHIC, Phase 3) -- Fixed slippage is catastrophically wrong for thin markets. Build a volume-adaptive model: `slippage = base_spread + impact * (order_size / volume)^power`. Continuously validate by comparing simulated fills vs. paper trading fills. All evaluation metrics must be slippage-adjusted from day one.

4. **Agent degenerate experiment loops** (HIGH, Phase 4) -- The agent can enter oscillating loops of promote-regress-diagnose-repeat. Implement mutation budgets per category per time window, semantic deduplication (embedding similarity > 0.85 = reject), experiment cooldowns after consecutive failures, and minimum champion hold periods.

5. **Cheap LLM structured output failures** (HIGH, Phase 4) -- DeepSeek-R1 produces chain-of-thought tags that break JSON parsing. Implement Pydantic validation on every LLM output, robust JSON extraction that strips non-JSON content, retry with reformulation, and a fallback chain: DeepSeek-R1 -> Qwen2.5 -> rule-based heuristics. Parse failure rate must stay below 5%.

## Implications for Roadmap

Based on research, the following five-phase structure is recommended. It follows strict data-flow dependencies: each phase produces the inputs the next phase needs.

### Phase 1: Data Infrastructure + Options Math Engine
**Rationale:** Everything downstream depends on data flowing and features being computable. Options chain data quality in thin markets is the first existential uncertainty -- validating this early prevents building on a broken foundation.
**Delivers:** Ingestion pipelines for futures OHLCV, options chains, and COT data; Feature Store (start with SQLite, designed for TimescaleDB migration); Raw Data Lake (Parquet); Options Math Engine (Breeden-Litzenberger density, implied moments, vol surface construction, Greeks flow aggregation); data quality monitoring with staleness checks.
**Addresses features:** Data Pipeline, Options Math Engine, Feature Store (point-in-time)
**Avoids pitfalls:** Lookahead bias (#1 -- `as_of` timestamps baked in from day one), Options math instability (#2 -- SVI smoothing, data quality thresholds), Silent data pipeline failures (#8 -- staleness checks, validators), QuantLib binding complexity (#10 -- NumPy/SciPy first, QuantLib optional)
**Stack elements:** Python 3.11, uv, NumPy, SciPy, QuantLib (optional), pyarrow, structlog, APScheduler
**VALIDATION GATE:** Can Breeden-Litzenberger produce stable implied distributions from the target market's options chains? Plot implied vs. realized distributions. If the data is too sparse or noisy for stable signal extraction, the entire thesis needs re-examination before proceeding.

### Phase 2: Signal Layer + Baseline Model
**Rationale:** The core trading thesis must be validated before building infrastructure to optimize it. If options-sentiment divergence has no predictive power out-of-sample, everything after this phase is wasted effort. This is the existential risk gate.
**Delivers:** Sentiment Engine (COT scoring, optionally FinBERT NLP); Divergence Detector (6-type taxonomy); baseline LightGBM model consuming divergence features; walk-forward backtesting with embargo gaps; circuit breakers (hard-coded limits); fractional Kelly position sizing.
**Addresses features:** Sentiment Engine (COT minimum), Divergence Detector, Baseline ML Model, Walk-Forward Backtesting, Circuit Breakers, Position Sizing
**Avoids pitfalls:** Sentiment overfitting (#6 -- start COT-only, gate NLP behind significance testing), COT timing trap (#13 -- as-of vs. release date handling), Walk-forward window bias (#11 -- expanding windows, embargo gaps)
**Stack elements:** LightGBM, scikit-learn, FinBERT/transformers (optional), Optuna
**VALIDATION GATE:** Does the divergence signal predict future price movement out-of-sample with Sharpe > 0 after slippage? If NO, stop. Re-examine the thesis, the target market, or the signal construction. Do not proceed to Phase 3 with a non-predictive signal.

### Phase 3: Sandbox + Experiment Infrastructure
**Rationale:** The agent loop (Phase 4) needs a safe environment to run experiments. The sandbox must exist before the agent does, or experiments have nowhere to run safely. The market replay engine with realistic thin-market slippage is the linchpin -- if the sandbox produces results that do not correlate with reality, every promotion decision will be tainted.
**Delivers:** Market Replay Engine with volume-adaptive slippage model; Model Versioning via MLflow (model registry, champion/candidate/archived lifecycle); Experiment Journal (structured logging, tag-based query); basic Observer Module (rolling performance metrics, drift detection via ADWIN/CUSUM); Evaluator/Judge (multi-objective fitness function); CLI core (status, diagnose, rollback, pause/run, journal query).
**Addresses features:** Market Replay Engine, Model Versioning, Experiment Journal, Observer Module, Evaluator/Judge, CLI Interface
**Avoids pitfalls:** Unrealistic slippage (#5 -- volume-adaptive model, validated against paper fills), MLflow state explosion (#12 -- PostgreSQL backend, retention policies from day one), Walk-forward window bias (#11 -- expanding windows, minimum test size)
**Stack elements:** MLflow, TimescaleDB (migrate from SQLite), river (ADWIN/CUSUM), Typer, Rich, SHAP
**Implements architecture:** Sandbox Layer, partial Agent Layer (Observer), CLI Layer

### Phase 4: Agent Core + LLM Integration
**Rationale:** This is the hardest phase and the primary competitive moat. Every component is individually complex (Diagnostician, Hypothesis Engine, Experiment Runner), and their interaction creates emergent complexity (degenerate loops, journal poisoning, false promotions). The LLM integration adds another dimension of uncertainty -- structured output reliability with DeepSeek-R1 needs hands-on validation. This phase should start in `supervised` autonomy and graduate to `semi-auto` only after 20+ experiments with human review.
**Delivers:** Diagnostician (LLM-powered triage with deterministic data audit first); Hypothesis Engine (LLM-powered mutation proposal constrained by playbook); Experiment Runner (subprocess isolation, 1-hour timeout); LLM client with fallback chain (DeepSeek-R1 -> Qwen2.5 -> rules); Semantic Journal Query; Autonomy Levels (config-driven); Automatic Rollback with hysteresis; full agent loop orchestration.
**Addresses features:** Full Agent Loop, Diagnostician Module, Hypothesis Engine, Experiment Runner, LLM Agent Reasoning, Autonomy Levels, Automatic Rollback, Semantic Journal Query
**Avoids pitfalls:** Agent degenerate loops (#3 -- mutation budgets, cooldowns, semantic deduplication), Cheap LLM structured output (#4 -- Pydantic validation, retry, fallback chain), LLM reasoning degradation (#9 -- RAG over journal, context budget), False positive promotions (#7 -- regime-diverse evaluation windows), Compound risks (CR1-CR4 -- deterministic checks before LLM reasoning, market activity detector, coupled fitness function, note verification)
**Stack elements:** openai (for Together AI), Pydantic, sentence-transformers (for semantic similarity), asyncio
**Research flag:** This phase NEEDS `/gsd:research-phase` before planning. LLM structured output reliability with DeepSeek-R1 and Qwen2.5, prompt engineering for financial domain reasoning, and mutation playbook design all require hands-on experimentation that could not be verified during project research.

### Phase 5: Execution + Hardening
**Rationale:** Execution is last because it carries the highest risk (real money) and depends on everything else working. The 4+ week paper trading requirement with at least one successful self-healing cycle is a hard gate before any live capital deployment. IB API integration is a separate concern from signal quality and agent intelligence.
**Delivers:** Order Manager (fractional Kelly sizing, smart order routing); Risk Manager (position limits, circuit breakers as middleware); IB Gateway (ib_insync or fallback); Paper Trading Pipeline (same execution path as live); alerting and monitoring; production hardening (reconnection logic, heartbeat monitoring, order state machine).
**Addresses features:** Paper Trading Pipeline, IB Integration, A/B Testing Framework (optional)
**Avoids pitfalls:** IB API instability (#14 -- reconnection, heartbeat, order state machine), Compound: optimization vs. illiquidity (#CR3 -- coupled fitness function carries forward)
**Stack elements:** ib_insync (verify maintenance status) or ibapi, Prometheus client (optional)
**VALIDATION GATE:** 4+ weeks paper trading with stable or improving performance AND at least one successful autonomous self-healing cycle (agent detects degradation, diagnoses cause, proposes fix, tests it, promotes improvement that lasts 30+ days).

### Phase Ordering Rationale

- **Data first, agent last:** The dependency chain is absolute. You cannot validate signals without data, cannot train models without signals, cannot run experiments without a sandbox, cannot self-heal without an agent, and cannot trade live without everything working. Skipping ahead in this sequence guarantees rework.
- **Two validation gates prevent wasted effort:** Phase 1 gate (can we extract stable signals from thin-market options?) and Phase 2 gate (does the divergence signal predict?) are the two points where the entire project direction could change. Validating early saves months.
- **Agent before execution:** The agent IS the product. A working agent with paper trading validation is infinitely more valuable than a live execution layer with a static model. Prove the self-healing thesis; execution is "just" broker API integration.
- **LLM integration deferred to Phase 4:** The agent loop's skeleton can run with rule-based fallbacks. Adding the LLM adds intelligence but also adds unreliability. Building the deterministic skeleton first means the system works (in degraded mode) even if the LLM proves problematic.
- **Circuit breakers in Phase 2, not Phase 5:** Safety is not a production concern -- it is a fundamental design constraint from the moment the first model produces a prediction.

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 1:** Data vendor selection (Databento vs. CME DataMine vs. IB historical) needs hands-on evaluation of API quality, cost, and thin-market options chain completeness. Target market selection (oats vs. lean hogs vs. ethanol) depends on which has sufficient options chain data.
- **Phase 4:** LLM structured output reliability with DeepSeek-R1 and Qwen2.5 needs hands-on validation. Prompt engineering for financial domain reasoning, mutation playbook design, and degenerate loop prevention strategies all require experimentation. This is the highest-uncertainty phase.
- **Phase 5:** ib_insync maintenance status must be verified. If the library is unmaintained, evaluate ib-async (community fork) or plan a custom wrapper around ibapi.

**Phases with standard patterns (likely skip research-phase):**
- **Phase 2:** Walk-forward validation, LightGBM training, COT data parsing, and sentiment scoring are well-documented patterns with established best practices. FinBERT is a standard Hugging Face model. The novel piece is the divergence detector, which is deterministic classification logic.
- **Phase 3:** MLflow model registry, SQLite/PostgreSQL experiment journal, ADWIN/CUSUM drift detection (via river library), and CLI via Typer are all standard tooling with good documentation.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | All versions from training data (mid-2025 cutoff); must validate with `pip index versions` before pinning. LLM API pricing changes frequently. ib_insync maintenance status unknown. |
| Features | MEDIUM-HIGH | Table stakes and differentiators are well-categorized based on established quant practice. Dependency graph is deterministic. Competitor analysis based on training data. |
| Architecture | MEDIUM | Layer separation, deterministic envelope around LLM, and event bus patterns are established. Specific library APIs (TimescaleDB hypertables, MLflow Model Registry) need version verification. |
| Pitfalls | MEDIUM | Quant trading pitfalls (#1, #2, #5, #6, #7) from well-established literature (HIGH confidence). LLM-specific pitfalls (#4, #9, CR1, CR4) evolve rapidly (MEDIUM-LOW confidence). Compound risks are original analysis. |

**Overall confidence:** MEDIUM

The core architecture and feature landscape are well-understood -- this is a mature domain (quantitative trading) with a novel twist (LLM-powered self-healing). The highest uncertainty is concentrated in two areas: (1) whether thin-market options data is sufficient quality for the signal extraction thesis, and (2) whether cheap LLMs produce reliable enough structured output for the agent loop. Both uncertainties are addressed by validation gates that force early confrontation.

### Gaps to Address

- **Data vendor selection and cost:** Databento, CME DataMine, and IB historical data were compared in research but actual API quality, thin-market options chain completeness, and pricing need hands-on evaluation in Phase 1. This is the single biggest practical uncertainty in Phase 1.
- **ib_insync maintenance status:** The original author passed away. Community fork status needs verification before committing. Fallback plan (ib-async or custom ibapi wrapper) exists but adds Phase 5 scope.
- **DeepSeek-R1 structured output reliability:** Training data reports chain-of-thought tags breaking JSON parsing. Actual failure rates with current model versions need hands-on measurement during Phase 4.
- **Target market selection:** Oats vs. lean hogs vs. ethanol depends on options chain data quality, which cannot be assessed until Phase 1 data pipelines are operational. The architecture is market-agnostic, so this decision can be deferred to Phase 1 execution.
- **LLM API provider reliability:** Together AI is recommended but actual uptime, latency from US, and rate limits need validation. Fireworks AI is the planned failover.
- **QuantLib Python 3.12 compatibility:** SWIG bindings may lag. Stick with Python 3.11 to be safe; verify QuantLib compatibility before any Python version upgrade.
- **X/Twitter API pricing:** Has changed multiple times; current tier structure unknown. Social sentiment is v2+ anyway, so this is low priority.

## Sources

### Primary (HIGH confidence)
- HYDRA PRD (`prd-get-shit-done.md`) -- comprehensive system specification covering architecture, modules, phases, and risk register
- HYDRA PROJECT.md (`.planning/PROJECT.md`) -- project constraints, key decisions, and scope boundaries
- John Hull, "Options, Futures, and Other Derivatives" -- options pricing theory, put-call parity
- Sheldon Natenberg, "Option Volatility and Pricing" -- vol surface construction
- Jim Gatheral, "The Volatility Surface" -- SVI parameterization, arbitrage-free constraints

### Secondary (MEDIUM confidence)
- Marcos Lopez de Prado, "Advances in Financial Machine Learning" -- backtesting pitfalls, walk-forward validation
- Robert Carver, "Systematic Trading" -- position sizing, slippage modeling, thin market considerations
- Ernie Chan, "Quantitative Trading" -- backtesting biases, data quality
- MLflow, TimescaleDB, LightGBM, QuantLib documentation (training data versions)
- LLM agent system design patterns (deterministic envelope, structured output validation, fallback chains)

### Tertiary (LOW confidence -- needs validation)
- DeepSeek-R1 / Qwen2.5 structured output reliability -- rapidly evolving; test during Phase 4
- ib_insync library status -- maintenance uncertain; verify before Phase 5
- LLM API provider pricing (Together AI, Fireworks AI) -- changes frequently
- Package version numbers -- all from training data; validate with `pip index versions`

---
*Research completed: 2026-02-19*
*Ready for roadmap: yes*
