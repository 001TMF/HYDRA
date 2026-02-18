# Feature Landscape

**Domain:** Autonomous self-healing ML trading agent for low-volume futures markets
**Researched:** 2026-02-18
**Confidence:** MEDIUM (training data only -- no web verification available; however, quantitative trading, options mathematics, and MLOps are deeply covered in training data, and the PRD provides strong specification)

---

## Table Stakes (System Does Not Function Without These)

Without any one of these, the system either cannot trade, cannot self-heal, or cannot be safely operated. These are the minimum viable autonomous trading agent.

### Signal Extraction Layer

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Data Pipeline (Futures + Options + COT)** | No data = no system. These three feeds are the minimum: futures prices for execution, options chains for implied signals, COT for positioning sentiment. | HIGH | Hardest practical challenge of Phase 1. Data vendor selection (Databento, CME DataMine, or Polygon), API integration, cleaning, gap handling, storage. Thin markets have missing strikes, stale quotes, and irregular updates. Parquet for raw data, time-series DB for computed features. |
| **Options Math Engine** | The entire trading thesis depends on extracting the market's probabilistic view from options chains. No options math = no signal. | HIGH | Breeden-Litzenberger risk-neutral density extraction (numerical 2nd derivative of call prices w.r.t. strike), implied moments (mean, variance, skew, kurtosis), volatility surface construction (strike x expiry grid), Greeks flow aggregation (GEX, vanna, charm). Cubic spline interpolation critical for thin markets with gapped strikes. Must use finite differences with step size tuned to actual strike spacing. |
| **Sentiment Engine (COT minimum, NLP recommended)** | The divergence signal requires a counterparty view to compare against options-implied expectations. COT is the highest-signal, lowest-noise source for futures positioning. | MEDIUM | Start with COT (free from CFTC, weekly, high signal-to-noise for commodity futures). Normalize to [-1, +1] with confidence weight. Add FinBERT headline NLP as second source for higher-frequency signal. Social and order flow are enhancements, NOT table stakes. |
| **Divergence Detector** | This IS the alpha signal. Options-implied vs. sentiment divergence is the core trading thesis. Without it you have two unconnected signal streams producing no actionable output. | MEDIUM | Taxonomy of divergence types per PRD: bullish-options/bearish-sentiment, bearish-options/bullish-sentiment, flat-options/strong-sentiment (fade), early-signal (options leading), aligned (momentum), high-kurtosis (vol play). Output: direction, magnitude, type, confidence, suggested bias. |
| **Feature Store with Point-in-Time Correctness** | Computed features must be stored, versioned, and queryable by timestamp. The agent needs historical features to detect drift, and the model needs training data. Point-in-time correctness prevents look-ahead bias, which silently inflates backtest results. | HIGH | Time-series storage keyed by (market, timestamp). MUST support as-of queries: "what features were available at time T?" Never use future data for normalization or imputation. This is subtle and the #1 source of backtest fraud in quant. TimescaleDB continuous aggregates or SQLite with careful schema design. |

### Model Layer

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Baseline ML Model (LightGBM)** | You need at least one model that consumes features and produces predictions. Without it, signals are just analytics that generate no P&L. | MEDIUM | LightGBM first -- fast to train, handles mixed feature types, naturally interpretable via SHAP feature importances. Walk-forward validation mandatory. No neural nets until tree-based baseline is beaten. The baseline also serves as the first "champion" for the agent loop. |
| **Walk-Forward Backtesting** | Without proper backtesting, you cannot validate any model. Walk-forward (expanding or rolling window, NOT simple random train/test split) is required to avoid look-ahead bias in time-series data. | MEDIUM | Expanding or rolling window with embargo periods between train and test sets to prevent leakage. Must account for thin-market slippage in P&L calculations. Must NOT use future data for feature normalization. Anchored walk-forward is safer for initial validation. |
| **Model Versioning** | Cannot promote, rollback, or compare models without versioning. Foundational to the agent loop -- without it, the system has no concept of "champion" or "candidate." | MEDIUM | Every trained model gets an immutable ID, full config snapshot (hyperparameters, feature list, training data window), and performance metrics. MLflow model registry is the standard tool and provides champion/staging/archived lifecycle. A simpler filesystem + JSON manifest works for v1 if MLflow setup is deferred. |

### Safety Layer

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Circuit Breakers** | Without automatic safety stops, a degrading model or rogue agent can cause unbounded losses. Non-negotiable for ANY system touching real capital, even in paper trading. | MEDIUM | Hard-coded limits (not overridable by agent): max daily loss, max drawdown (trailing), max position size, max single-trade loss, max position as fraction of average daily volume. Must halt trading AND alert operator. Implement as middleware in the execution path, not as an optional check. |
| **Position Sizing (Fractional Kelly)** | A model that predicts direction but sizes positions randomly will blow up on an unlucky streak. Kelly criterion (fractional) is the standard mathematical approach to sizing bets given estimated edge and variance. | LOW | Modified Kelly: f* = (p*b - q) / b, then use f*/N where N = 2-4 for conservative sizing. Must cap at fraction of average daily volume to avoid moving the thin market. Simple to implement, critical to get right. |
| **Paper Trading Pipeline** | PRD mandates 4+ weeks paper trading before live capital. Without this, there is no safe path to production. Must use the same execution path as live trading. | MEDIUM | Interactive Brokers paper account uses the same API as live. Must log all fills with realistic timestamps and slippage. Must feed into performance monitoring so the observer tracks paper results the same way it would track live. |
| **Autonomy Levels** | The operator must constrain the agent from day 1. Shipping a fully autonomous agent with no guardrails is reckless. Different stages of system maturity warrant different levels of trust. | LOW | Four levels per PRD: supervised (agent proposes, human approves), semi-auto (agent experiments freely, human approves promotions), autonomous (full autonomy, alerts only), lockdown (monitoring only, no experiments). Start in supervised mode. Config-driven switch. |

### Operator Interface

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **CLI Interface (Core Commands)** | The operator needs to inspect, control, and override the system. Without CLI, the agent is an uncontrollable black box. | MEDIUM | Minimum commands: `status` (model health, active experiments, alerts), `diagnose` (force diagnostic), `rollback` (revert to previous champion), `pause/run` (halt/start agent loop), `journal query` (search experiment history). Use Typer (modern, type-hinted Click alternative) with Rich for formatted terminal output. |

### Monitoring Layer

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **Observer Module (Drift Detection)** | Without monitoring, the agent does not know when the model is degrading. The observer is the trigger for the entire self-healing loop. Without it, the agent is blind. | HIGH | Rolling performance metrics (Sharpe, drawdown, hit rate, PnL curve, prediction calibration). Feature distribution drift (PSI scores, KS tests per feature). Data freshness monitoring (staleness checks). Alert taxonomy: INFO (log only), WARNING (schedule diagnostic), CRITICAL (immediate rollback + emergency diagnostic). Change-point detection via ADWIN or CUSUM. |
| **Experiment Journal (Basic Logging)** | The agent needs memory to avoid repeating failed experiments. Without it, the agent is amnesiac and will waste cycles retrying things that already failed. | MEDIUM | Structured log of every experiment: hypothesis, config diff, training data window, results (all metrics), promotion decision, agent notes, tags. Must be queryable by tag, date range, mutation type, and outcome. SQLite + JSON columns is sufficient for v1. |

---

## Differentiators (Competitive Advantage Over Static ML Trading Systems)

These features are what make HYDRA genuinely different from a standard quant model that gets trained once and deployed. Most quant systems -- open source and institutional -- lack ALL of these. Having even a few working puts HYDRA in an entirely different category.

### The Self-Healing Agent Loop (Primary Differentiator)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Full Agent Loop (Observe -> Diagnose -> Hypothesize -> Experiment -> Evaluate)** | Static models decay. An agent that detects degradation, diagnoses root causes, proposes targeted fixes, tests them in sandbox, and promotes improvements is the entire competitive moat. This is what "self-healing" means and it is what no open-source trading system does. | VERY HIGH | The single hardest feature to build AND the single most valuable. Each sub-module is substantial: observer (drift detection), diagnostician (root cause analysis), hypothesis engine (mutation proposal), experiment runner (isolated testing), evaluator (promotion decision). Build incrementally -- observer first, then diagnostician, then the rest. |
| **Diagnostician Module** | Most MLOps systems detect drift but do NOT diagnose root cause. A diagnostician that runs structured triage (data audit, SHAP-based feature attribution, regime check, overfitting decay test, external event scan) before proposing fixes is rare even in institutional settings. | HIGH | Powers targeted mutations instead of blind hyperparameter search. SHAP decomposition on LightGBM is fast (TreeSHAP). The LLM reasoning layer adds natural language diagnosis that pure statistical systems lack -- "charm flow feature degraded because options liquidity dried up in the front-month contract." |
| **Hypothesis Engine with Mutation Playbook** | Instead of random search, the agent draws from a curated playbook of mutations matched to diagnosed root causes. This is targeted evolution, not brute force optimization. | HIGH | Two categories: self-healing mutations (retrain on recent window, quarantine degraded feature, switch architecture, adjust retraining frequency, rollback, replace stale data source) and self-optimization mutations (new derived features, hyperparameter perturbation via Bayesian optimization, architecture swap, loss function modification, ensemble reweighting, lookback window adjustment). Constrained creativity: max 3 concurrent experiments, max 2 variables changed per experiment, must check journal for prior similar experiments. |
| **LLM-Powered Agent Reasoning** | Using an LLM for hypothesis generation, diagnosis narration, and natural language journal queries is a genuine differentiator. Statistical systems cannot reason about WHY a model failed or generate novel hypotheses in natural language. | HIGH | Cheap Chinese LLM (DeepSeek-R1 or Qwen2.5 via API provider) keeps costs manageable for continuous loop operation. Must handle structured output (JSON) reliably -- this needs validation early. Critical distinction: the LLM does NOT make trading decisions. It reasons about model health and proposes experiments. The trading decisions remain with the ML model. |
| **Champion/Candidate Promotion Protocol** | Multi-objective fitness evaluation prevents promoting models that overfit a lucky backtest window. Most systems use a single metric (Sharpe) and get fooled by variance. | MEDIUM | Six-metric fitness function per PRD: out-of-sample Sharpe (0.25 weight), max drawdown (0.20), prediction calibration/Brier (0.15), robustness/rolling Sharpe variance (0.15), slippage-adjusted return (0.15), simplicity/parameter count (0.10). Must beat champion on composite across 3 of 5 independent eval windows. Mandatory paper trading validation period before live promotion. |
| **Automatic Rollback** | When a promoted model fails in production, automatic rollback to the previous champion preserves capital without requiring human intervention. Most systems require manual response. | LOW | Mechanically simple (model swap since all models are versioned), but the hard part is reliable failure detection (observer) and avoiding flapping (rollback -> re-promote -> rollback cycles). Hysteresis threshold: only rollback if degradation persists for N periods, not on a single bad day. |

### Enhanced Signal Capabilities

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Experiment Journal with Semantic Query** | Long-term institutional memory that the agent and operator can query in natural language. "What experiments improved Sharpe during high-vol regimes?" "Have we tried removing charm flow before?" Prevents wasted cycles, accumulates knowledge, and becomes more valuable over time than any single model. | MEDIUM | Natural language queries via LLM. Tag-based search for structured queries. Temporal decay weighting on old entries (recent experiments are more relevant). Over time, this journal becomes the most valuable asset in the entire system. |
| **Market Replay Engine with Thin-Market Slippage Model** | Most backtesting frameworks assume liquid markets with instant fills at mid-price. A replay engine that models slippage as a function of order size, average daily volume, bid-ask spread, and time-of-day is critical for thin markets and rare in any system. | HIGH | Standard tools (backtrader, zipline, vectorbt) use fixed slippage or fill-at-mid. Thin markets need: partial fill simulation, wide/variable spreads, queue priority modeling, market impact estimation. Slippage model: slippage = base_spread/2 + impact * (order_size / ADV)^power. Must be parameterized per market. Custom implementation required. |
| **Synthetic Data Generator** | Stress-testing models against scenarios that have not happened yet (flash crash, liquidity drought, vol explosion, contract rollover stress) finds fragility before the market does. Especially critical in thin markets where historical data is sparse. | HIGH | Start with copula-based simulation preserving cross-asset correlation structure. Parameterized scenario library: flash crash (sudden -5 sigma move), liquidity drought (ADV drops 80%), vol explosion (implied vol doubles in a day), trend reversal, data outage. Conditional GAN is ambitious -- defer to v2+. Fat-tail injection via Student-t or stable distributions. |
| **Regime-Aware Model Selection** | Most ML trading systems use one model for all market conditions. Markets behave fundamentally differently in trending vs. mean-reverting vs. crisis regimes. Detecting regime changes and switching models or adjusting features per regime is a significant edge. | HIGH | Hidden Markov Models or online change-point detection for regime identification. Regime-conditional feature selection (some features are predictive only in certain regimes -- charm flow matters during trending, kurtosis matters during transitions). The agent's architecture-swap mutation provides a crude version; explicit regime detection is the refined version. |
| **A/B Testing Framework** | Running champion and candidate side-by-side on live data feed (paper mode) provides real-world validation that no backtest can match. Especially important given thin-market slippage uncertainty. | MEDIUM | Split incoming signals to both models, paper-trade both, compare with statistical significance testing (paired t-test or bootstrap). Automated kill switch if candidate diverges badly from champion during the test period. Requires paper trading infrastructure to be operational first. |
| **SHAP-Based Feature Attribution** | Understanding which features drive model predictions -- and which features are driving ERRORS -- enables targeted improvement instead of blind experimentation. Powers the diagnostician module. | MEDIUM | TreeSHAP on LightGBM is fast and exact. For neural models, use KernelSHAP (slower). Track feature importance stability over time -- unstable importances suggest overfitting. |

---

## Anti-Features (Deliberately NOT Building)

These features seem valuable on the surface but will destroy focus, add complexity without proportional benefit, or are premature for a single-market single-operator system. Each entry includes what to do instead.

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Web/Mobile Dashboard** | Visual monitoring feels safer and more professional. | Massive UI development effort for a single operator. CLI provides everything needed. Dashboards become maintenance burdens. You will spend more time building and fixing the dashboard than trading. Every hour spent on CSS is an hour not spent on signal quality. | CLI with rich terminal output via Typer + Rich (tables, sparklines, colored alerts). Log files. Ad-hoc Jupyter notebooks for deep visual analysis when needed. Grafana only if Prometheus monitoring is added, and even then it is optional. |
| **Multi-Market from Day 1** | More markets = more diversified, more opportunities. | Each market has unique options chain characteristics (strike spacing, liquidity, expiry cycles), slippage profiles, data sources, and sentiment dynamics. Multi-market on day 1 means nothing works well instead of one thing working perfectly. The slippage model alone needs per-market calibration. | Design abstractions (Market, DataSource, Model interfaces) that are market-agnostic from the start. Implement concrete classes for ONE market. Scale to additional markets after the system is profitable and self-healing on the first. |
| **Real-Time Tick-Level Order Flow (Day 1)** | Highest-resolution data = best signals. Order flow is the "ground truth." | Tick data is expensive (Databento tick-level is significantly more than bar data), massive in storage volume, and requires event-driven processing infrastructure. For markets trading 2,000-10,000 contracts/day, EOD and intraday bars capture most of the information content. The marginal value of ticks in thin markets is low -- there simply are not that many ticks. | Start with EOD options chains + intraday futures bars. Add tick data only AFTER: (a) the system works at lower frequency, AND (b) feature attribution analysis shows order flow features are the performance bottleneck. |
| **Custom Execution Engine** | Better fills, lower latency, more control. | Building a custom execution engine is a multi-month project with exchange connectivity, order management, regulatory implications, and FIX protocol complexity. Interactive Brokers API is battle-tested, supports futures and options, handles order routing, and provides paper trading. HYDRA's edge is signal quality and self-healing, NOT execution speed. In thin markets, execution speed is largely irrelevant. | Interactive Brokers API via ib_insync (or ibapi). Implement simple smart order routing (limit orders with patience, TWAP for larger positions) as thin wrappers around IB orders. |
| **Reinforcement Learning Agent** | RL is "the future" of autonomous decision-making. | RL for trading is an active research problem with no proven production-quality solutions at any scale. Sample efficiency is terrible (needs millions of episodes), reward shaping is fragile (what reward? P&L? Sharpe? Risk-adjusted?), and sim-to-real transfer fails spectacularly in financial markets where the environment is non-stationary and adversarial. The agent loop in the PRD is more practical, more debuggable, and more controllable. | Supervised learning (LightGBM baseline, then neural if warranted) with the mutation-based agent loop. The agent loop provides adaptation without RL's instabilities. The hypothesis engine + promotion protocol is effectively a more controlled form of "learning from experience." |
| **Automated Strategy Discovery** | Let the agent invent entirely new trading strategies beyond the divergence thesis. | The options-sentiment divergence thesis IS the strategy. Automated strategy discovery (genetic programming, neural architecture search for alpha, automated feature synthesis) is a PhD-level research project. Unbounded search spaces explode combinatorially and produce spurious results. The agent should optimize execution of the thesis, not reinvent the thesis. | Agent mutates features, hyperparameters, architectures, and ensemble compositions WITHIN the bounds of the core divergence thesis. This is already a rich search space. If the thesis fails validation (Phase 2 gate), rethink the thesis manually, do not automate thesis generation. |
| **Blockchain/DeFi Integration** | Niche crypto derivatives are mentioned as a potential market. | Entirely different infrastructure: Web3 libraries, DEX APIs, on-chain data, gas optimization, smart contract interaction. CME/ICE futures are the target. Mixing DeFi infrastructure with traditional futures trading adds massive complexity for a speculative market expansion. | Focus on CME/ICE futures through Interactive Brokers. If niche crypto derivatives are eventually targeted, use CENTRALIZED exchange APIs (Deribit, Binance Futures) that look like traditional broker APIs, not DeFi protocols. |
| **Social Trading / Copy Trading** | Monetization path if the system works. | This is a personal quant tool. Social/copy trading adds: regulatory burden (investment advisor registration, SEC/CFTC compliance), liability exposure, user management, authentication, payments, and support. All of this generates zero alpha. | If the system is profitable, the monetization is the P&L. If capital scaling is desired, a separate fund vehicle with proper legal structure is the correct path, not a social trading platform. |
| **Complex Options Strategies (Multi-Leg Execution)** | Options are a core signal source, why not trade them directly? | HYDRA uses options chains for SIGNAL EXTRACTION -- reading what the options market implies about futures prices. Trading options directly in thin markets means: terrible fills on multi-leg strategies, pin risk, assignment risk, and execution complexity that dwarfs the signal extraction. The alpha is in the futures trade informed by options intelligence, not in options trading itself. | Read options chains for implied information. Trade futures for the actual position. If options trading is eventually desired (selling vol, hedging), it is a separate phase built on top of a working and profitable futures system. |
| **GPU-Dependent Infrastructure** | Neural nets need GPUs for training. | For a single-market system with LightGBM as the baseline, GPU is unnecessary overhead. LightGBM is CPU-optimized. Neural architectures (if promoted by the agent) should be small enough to train on CPU in reasonable time for a single market's data volume. Adding GPU dependency means: cloud costs (or local GPU purchase), CUDA version management, driver compatibility, and infrastructure complexity. | CPU-first for everything. LightGBM is fastest on CPU. If neural models are tested via the agent's architecture-swap mutation, keep them small (2-3 layer MLP, not transformers). If a neural architecture proves clearly superior and training time is the bottleneck, rent spot GPU instances for training runs only -- do not architect the system around GPU availability. |
| **Kubernetes / Cloud Infrastructure** | Production systems "should" be in the cloud with orchestration. | Single developer, single market, single machine. Kubernetes adds operational complexity (cluster management, networking, persistent volumes, monitoring) that provides zero benefit at this scale. Cloud adds recurring costs when the system is designed to run continuously. | Run everything locally. Docker Compose for database services (TimescaleDB/PostgreSQL) only. The system is a single Python process with a database -- it does not need container orchestration. Cloud deployment is a scaling concern for multi-market v2+, not a v1 concern. |

---

## Feature Dependencies

```
[Data Pipeline (Futures OHLCV + Options Chains + COT)]
    |
    +---> [Options Math Engine (B-L, moments, surface, Greeks)]
    |         |
    |         +---> [Feature Store (point-in-time correct)] <---+
    |                    |                                       |
    +---> [Sentiment Engine (COT scoring + NLP)] ---------------+
    |         |
    |         +---> [Divergence Detector]
    |                    |
    |                    +---> [Baseline ML Model (LightGBM)]
    |                              |
    |                              +---> [Walk-Forward Backtesting]
    |                              |         |
    |                              |         +---> **VALIDATION GATE: Does the signal predict?**
    |                              |                    |
    |                              |                    +---> [Model Versioning (MLflow)]
    |                              |                              |
    |                              +---> [Position Sizing]        +---> [Champion/Candidate Lifecycle]
    |                              |                              |         |
    |                              +---> [Circuit Breakers]       |         +---> [Promotion Protocol]
    |                                                             |         |
    |                                                             |         +---> [Automatic Rollback]
    |                                                             |
    +---> [Observer Module (drift detection, performance monitoring)]
    |         |
    |         +---> [Diagnostician (SHAP attribution, regime check)]
    |         |         |
    |         |         +---> [Hypothesis Engine + Mutation Playbook]
    |         |                    |
    |         |                    +---> [Experiment Runner (isolated sandbox)]
    |         |                              |
    |         |                              +---> [Evaluator/Judge]
    |         |                                        |
    |         |                                        +---> [Promotion Protocol] (loops back)
    |         |
    |         +---> [Circuit Breakers] (observer triggers breakers on CRITICAL alerts)
    |
    +---> [Experiment Journal (logging + query)]
    |         |
    |         +---> [Semantic Journal Query (LLM-powered)]
    |
    +---> [Market Replay Engine]
    |         |
    |         +---> [Synthetic Data Generator]
    |                    |
    |                    +---> [Stress Testing]
    |
    +---> [Paper Trading Pipeline (IB paper account)]
    |         |
    |         +---> [A/B Testing Framework]
    |         |
    |         +---> [Live Execution (IB live account)]
    |
    +---> [CLI Interface] ---> touches all components

[Autonomy Levels] ---> gates [Hypothesis Engine], [Experiment Runner], [Promotion Protocol]

[LLM Agent Reasoning (DeepSeek/Qwen)] ---> powers [Hypothesis Engine], [Diagnostician], [Semantic Query]
```

### Dependency Notes

- **Data Pipeline is the root dependency:** Absolutely nothing works until futures prices, options chains, and COT data are flowing and stored. This must be Phase 1 priority zero.
- **Options Math Engine requires full options chains:** Specifically, the full chain (all available strikes, bids, asks, OI, volume, expiry dates). OHLCV price bars alone are insufficient -- you need the chain.
- **Divergence Detector requires BOTH Options Math AND Sentiment:** These two signal streams must both produce valid output before divergence detection makes sense. Building one without the other produces zero signal.
- **VALIDATION GATE between Phase 2 and Phase 3:** Does the divergence signal predict future price movement out-of-sample with Sharpe > 0? If NO, stop and re-examine the thesis before building the agent loop. Building self-healing infrastructure around a model with no edge is wasted effort.
- **Agent Loop requires Model Versioning + Experiment Journal:** You cannot run experiments or promote models without tracking what you have, what you have tried, and what happened.
- **Observer requires Feature Store + Live Performance Data:** Drift detection needs both feature distribution history and model prediction/outcome history.
- **Market Replay requires Historical Data Accumulation:** The data pipeline must run long enough to accumulate meaningful history, OR historical data must be backfilled from vendor.
- **Paper Trading requires IB Integration:** A separate integration point from the data pipelines. IB API for execution, separate vendors for market data.
- **LLM Agent Reasoning requires reliable structured output:** The cheap Chinese LLM must produce valid JSON for diagnoses and experiment proposals. This MUST be validated early -- if structured output is unreliable, the agent loop cannot function. Test with DeepSeek-R1 and Qwen2.5 before committing.
- **A/B Testing requires Paper Trading x2:** Running champion and candidate simultaneously needs two parallel paper trading instances through IB.
- **Autonomy Levels GATE the agent loop:** Must be implemented BEFORE the agent loop runs experiments to prevent uncontrolled behavior. Default to `supervised` or `lockdown`.

---

## MVP Definition

### Launch With (v1) -- "Can the system generate a signal and trade it safely?"

The minimum to validate the core thesis: options-sentiment divergence produces alpha in the target market.

- [ ] **Data Pipeline** (futures OHLCV, options chains, COT reports for one market) -- without data, nothing exists
- [ ] **Options Math Engine** (implied distribution via B-L, implied moments, basic vol surface) -- core signal extraction
- [ ] **Sentiment Engine** (COT scoring only, normalized to [-1, +1]) -- simplest, highest-signal sentiment source
- [ ] **Divergence Detector** (options-implied vs. COT sentiment) -- the alpha signal
- [ ] **Feature Store** (time-series storage with point-in-time queries) -- features accessible for training and monitoring
- [ ] **Baseline ML Model** (LightGBM on divergence features) -- consumes features, produces predictions
- [ ] **Walk-Forward Backtesting** (expanding window, embargo periods) -- validates model out-of-sample
- [ ] **Model Versioning** (filesystem + JSON manifest minimum, MLflow preferred) -- track what you have
- [ ] **Circuit Breakers** (drawdown limits, position limits, hard-coded) -- prevent catastrophe
- [ ] **Position Sizing** (fractional Kelly, capped at fraction of ADV) -- size trades rationally
- [ ] **CLI** (status, diagnose, rollback, pause, run) -- operator control
- [ ] **Autonomy Levels** (start in lockdown/supervised) -- safety constraint from day 1

**Validation gate:** Run walk-forward backtest. If out-of-sample Sharpe <= 0 after slippage, the divergence thesis needs re-examination before building further.

### Add After Validation (v1.x) -- "Can the system heal itself?"

Once the core system produces validated predictions, add self-healing and production capabilities.

- [ ] **Observer Module** (rolling performance, drift detection via PSI/KS/ADWIN) -- trigger: need automated degradation detection
- [ ] **Experiment Journal** (structured logging with tag-based query) -- trigger: agent loop needs memory
- [ ] **Market Replay Engine** (historical data replay with thin-market slippage) -- trigger: experiments need realistic evaluation environment
- [ ] **Diagnostician Module** (SHAP attribution, regime check, data audit) -- trigger: observer produces alerts that need root cause analysis
- [ ] **Hypothesis Engine + Mutation Playbook** -- trigger: diagnoses need to map to actionable experiment proposals
- [ ] **Experiment Runner with Isolation** -- trigger: hypotheses need testing without affecting champion
- [ ] **Evaluator/Judge + Promotion Protocol** -- trigger: experiments need objective multi-metric evaluation
- [ ] **LLM Agent Reasoning Integration** (DeepSeek-R1 or Qwen2.5) -- trigger: agent loop functional, needs reasoning layer for hypothesis generation and diagnosis narration
- [ ] **NLP Sentiment** (FinBERT on news headlines) -- trigger: COT-only sentiment proves too low-frequency (weekly updates)
- [ ] **Paper Trading Pipeline** (IB paper account) -- trigger: model performance in backtest warrants forward-testing with real market data
- [ ] **Semantic Journal Query** (LLM-powered natural language search) -- trigger: journal has enough entries that tag-based search is insufficient
- [ ] **Automatic Rollback** -- trigger: agent loop can promote models, need safety net for bad promotions

### Future Consideration (v2+) -- "Can the system evolve and scale?"

Defer until the single-market system is profitable and self-healing has been demonstrated with at least 3 successful autonomous improvement cycles.

- [ ] **Synthetic Data Generator** (copula-based + parameterized stress scenarios) -- why defer: complex, and market replay covers most evaluation needs initially
- [ ] **A/B Testing Framework** -- why defer: requires double paper trading infrastructure; champion/candidate offline comparison is sufficient for initial experiments
- [ ] **Regime-Aware Model Selection** -- why defer: need enough data across multiple regime changes to build and validate regime detector; the agent's architecture-swap mutation provides a crude version in the meantime
- [ ] **Social Sentiment** (Twitter/X, Reddit) -- why defer: noisy, API costs ($100+/mo for X), and COT + NLP likely provides sufficient sentiment signal for commodity futures
- [ ] **Order Flow Features** (tick-level) -- why defer: expensive data, infrastructure-heavy, and marginal value in thin markets is unproven
- [ ] **Multi-Market Expansion** -- why defer: per PRD, prove thesis on one market first; architecture supports it via market-agnostic abstractions
- [ ] **Advanced Synthetic Data** (Conditional GAN) -- why defer: copula-based is simpler and likely sufficient; GANs add research-grade complexity and training instability
- [ ] **Neural Architecture Search** (agent proposes novel architectures) -- why defer: LightGBM + simple MLP covers the useful architecture space; NAS is research, not production

---

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority | Phase |
|---------|------------|---------------------|----------|-------|
| Data Pipeline (Futures + Options + COT) | HIGH | HIGH | P1 | 1 |
| Options Math Engine | HIGH | HIGH | P1 | 1 |
| Feature Store (point-in-time) | HIGH | MEDIUM | P1 | 1 |
| Sentiment Engine (COT) | HIGH | LOW | P1 | 1-2 |
| Divergence Detector | HIGH | MEDIUM | P1 | 2 |
| Baseline ML Model (LightGBM) | HIGH | MEDIUM | P1 | 2 |
| Walk-Forward Backtesting | HIGH | MEDIUM | P1 | 2 |
| Circuit Breakers | HIGH | LOW | P1 | 2 |
| Position Sizing (Fractional Kelly) | HIGH | LOW | P1 | 2 |
| Model Versioning | HIGH | LOW-MEDIUM | P1 | 2-3 |
| CLI (Core Commands) | HIGH | MEDIUM | P1 | 2-3 |
| Autonomy Levels | MEDIUM | LOW | P1 | 3 |
| Observer Module (Drift Detection) | HIGH | HIGH | P2 | 3 |
| Experiment Journal (Basic) | HIGH | MEDIUM | P2 | 3 |
| Market Replay Engine | HIGH | HIGH | P2 | 3 |
| NLP Sentiment (FinBERT) | MEDIUM | MEDIUM | P2 | 2-3 |
| Diagnostician Module | HIGH | HIGH | P2 | 4 |
| Hypothesis Engine + Playbook | HIGH | HIGH | P2 | 4 |
| Experiment Runner (Isolated) | HIGH | MEDIUM | P2 | 4 |
| Evaluator/Judge | HIGH | MEDIUM | P2 | 4 |
| Champion/Candidate Promotion | HIGH | MEDIUM | P2 | 4 |
| LLM Agent Reasoning | HIGH | HIGH | P2 | 4 |
| Automatic Rollback | HIGH | LOW | P2 | 4-5 |
| Paper Trading (IB Integration) | HIGH | MEDIUM | P2 | 5 |
| Semantic Journal Query | MEDIUM | MEDIUM | P2 | 4-5 |
| SHAP Feature Attribution | MEDIUM | LOW | P2 | 3-4 |
| A/B Testing Framework | MEDIUM | MEDIUM | P3 | 5 |
| Synthetic Data Generator (Copula) | MEDIUM | HIGH | P3 | 5+ |
| Regime-Aware Model Selection | MEDIUM | HIGH | P3 | v2 |
| Social Sentiment (Twitter/Reddit) | LOW | MEDIUM | P3 | v2 |
| Order Flow Features (Tick-Level) | LOW | HIGH | P3 | v2 |
| Conditional GAN Synthetic Data | LOW | VERY HIGH | P3 | v2+ |

**Priority key:**
- P1: Must have for launch -- system does not function as a trading agent without it
- P2: Should have -- enables self-healing capability and production readiness
- P3: Nice to have -- enhances capability but system works without it

---

## Competitor Feature Analysis

Note: This analysis is based on training data knowledge. Confidence is MEDIUM -- web search was unavailable to verify current feature sets.

| Feature | Zipline / QuantConnect (Backtesting Platforms) | Freqtrade / Jesse (Trading Bots) | Institutional Quant Desks | HYDRA (Our Approach) |
|---------|-----------------------------------------------|----------------------------------|--------------------------|---------------------|
| Options math signal extraction | Not built-in; user must implement everything | Not supported (crypto/forex focused) | Custom proprietary systems, often in C++ | First-class: B-L density, vol surface, Greeks flows, implied moments |
| Sentiment integration | Not built-in | Basic crypto sentiment plugins | Custom NLP pipelines, Bloomberg terminal | COT + FinBERT NLP, phased addition of social |
| Divergence detection | N/A (user implements strategies) | N/A (rule-based strategies) | Common concept ("cross-asset signal") but proprietary | Core alpha signal with 6-type divergence taxonomy |
| Self-healing / autonomous improvement | None -- models are static | None -- rule-based, no ML | Some have automated retraining; very few have diagnosis or hypothesis generation | Full agent loop with LLM-powered reasoning |
| Experiment tracking | None built-in (user adds MLflow) | None | MLflow/W&B is standard practice | MLflow-backed journal with semantic query |
| Model versioning | None | None | Standard practice (internal registries) | Integrated with promotion protocol and rollback |
| Backtesting fidelity | Good (Zipline); cloud-based (QC) | Built-in for rule-based strategies | Custom internal systems, high fidelity | Custom replay with thin-market slippage model |
| Live execution | QC cloud; Zipline deprecated for live | Multiple exchange integrations | Custom FIX/proprietary, co-located | Interactive Brokers API |
| Risk management | Basic (user-implemented) | Stop-loss, trailing stop, position sizing | Comprehensive (regulatory requirement) | Circuit breakers, fractional Kelly, automatic rollback, autonomy levels |
| Thin-market specialization | Not addressed (assumes liquid markets) | Not addressed (crypto is liquid) | Sometimes (prop trading shops in niche markets) | Core design focus: slippage model calibrated to thin markets |
| Autonomy levels | N/A (manual operation) | Runs unattended but no graduated autonomy | Human oversight is standard; graduated autonomy is rare | Four levels: supervised -> semi-auto -> autonomous -> lockdown |
| CLI / operator interface | Script-based / web IDE (QC) | Web dashboard | Bloomberg terminal / custom internal tools | Typer-based power-user CLI with Rich formatting |
| LLM integration | None | None | Emerging (GPT for research, not for trading loop) | DeepSeek/Qwen for hypothesis generation, diagnosis, journal query |

**Key insight:** No existing open-source system combines options math signal extraction with an autonomous self-healing agent loop. Institutional desks may have components (automated retraining, drift detection) but they are proprietary, lack LLM reasoning, and are not designed for thin markets. HYDRA's specific combination -- options-derived signals in thin markets + LLM-powered self-healing agent -- occupies an empty niche.

---

## Complexity Estimates by Phase

| Phase | Key Features | Estimated Complexity | Biggest Risk |
|-------|-------------|---------------------|-------------|
| Phase 1: Foundation | Data pipeline, options math engine, feature store | HIGH | Data vendor selection and API reliability; options chain data quality/completeness in thin markets; getting B-L to work with gapped strike ladders |
| Phase 2: Signal Layer | Sentiment (COT + NLP), divergence detector, baseline model, backtesting, circuit breakers, position sizing | MEDIUM-HIGH | **Thesis validation:** does the divergence signal actually predict? If Sharpe <= 0 OOS, everything built after this is wasted. This is the existential risk gate. |
| Phase 3: Sandbox + Monitoring | Market replay, model versioning, experiment journal, observer module, CLI, autonomy levels | HIGH | Building realistic thin-market replay engine (no off-the-shelf solution); getting drift detection thresholds right (too sensitive = false alarms, too loose = missed degradation) |
| Phase 4: Agent Core | Diagnostician, hypothesis engine, experiment runner, evaluator, LLM integration, promotion protocol | VERY HIGH | Making the agent loop reliable: LLM structured output consistency, preventing degenerate experiment loops, ensuring promotion protocol doesn't promote overfit models, experiment isolation correctness |
| Phase 5: Hardening + Live | Paper trading, IB integration, A/B testing, stress testing, automatic rollback, alerting | MEDIUM-HIGH | Sim-to-real gap: paper trading may reveal slippage assumptions are wrong; IB API quirks; first live capital is highest-anxiety moment |

---

## Sources

- HYDRA PRD (`/Users/tristanfarmer/Documents/HYDRA/prd-get-shit-done.md`) -- primary specification, extremely detailed
- HYDRA PROJECT.md (`/Users/tristanfarmer/Documents/HYDRA/.planning/PROJECT.md`) -- project constraints and key decisions
- Training data knowledge of: quantitative trading systems, options mathematics (Black-Scholes framework, Breeden-Litzenberger theorem, Greeks), ML operations (MLflow, experiment tracking, model registries), algorithmic trading frameworks (Zipline, QuantConnect, Freqtrade, backtrader, vectorbt), risk management (Kelly criterion, circuit breakers, VaR), LLM agent architectures, CFTC COT reports, implied volatility surface construction, thin-market microstructure -- MEDIUM confidence (no web verification available, but these are deeply documented domains in training data)

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Table Stakes categorization | HIGH | These features are well-established requirements for any ML trading system. The specific implementations (B-L, fractional Kelly, walk-forward validation) are standard quant practice documented in textbooks and papers. |
| Differentiator categorization | HIGH | The self-healing agent loop with LLM reasoning IS genuinely novel in the open-source and publicly documented quant space. Whether it WORKS is separate from whether it DIFFERENTIATES. |
| Anti-feature decisions | MEDIUM | Decisions to exclude RL, multi-market, and GUI are well-reasoned for v1 but could be revisited if project scope changes. RL exclusion is the most debatable -- the field is advancing quickly. |
| Dependency graph | HIGH | Dependencies follow from the logical structure of the system (data -> features -> model -> agent). This is deterministic, not speculative. |
| Competitor analysis | MEDIUM | Based on training data knowledge of open-source ecosystem as of early 2025. Individual project features may have changed. Institutional systems are proprietary and opaque. |
| Complexity estimates | MEDIUM | Actual complexity depends on data vendor APIs, LLM structured output reliability, and thin-market data quality -- none of which can be verified without hands-on experimentation. Phase 4 (Agent Core) is almost certainly underestimated -- agent systems are notoriously harder than they seem. |

---
*Feature research for: HYDRA -- Autonomous Self-Healing ML Agent for Quantitative Trading*
*Researched: 2026-02-18*
