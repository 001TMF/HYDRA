# Stack Research

**Domain:** Autonomous self-healing ML trading agent for low-volume futures markets
**Researched:** 2026-02-18
**Confidence:** MEDIUM (versions based on training data through mid-2025; could not verify via PyPI/official docs due to tool restrictions -- all version numbers should be validated before pinning)

---

## IMPORTANT: Verification Notice

WebSearch, WebFetch, and Bash were all unavailable during this research session. All version numbers and pricing data are sourced from training data (cutoff: mid-2025). Before pinning any dependency version, run `pip index versions <package>` to confirm the latest stable release. Confidence levels reflect this limitation.

---

## Recommended Stack

### Core Runtime

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Python | 3.11 or 3.12 | Runtime | 3.11 is the sweet spot: all ML libraries fully support it, excellent performance. 3.12 is fine too but some C-extension libraries may lag. Do NOT use 3.13 yet -- too new for SWIG-based libraries like QuantLib. | HIGH |
| uv | latest | Package management | 10-100x faster than pip, reliable lockfiles via `uv.lock`, replaces pip+pip-tools+virtualenv. The Python packaging world has consolidated around uv in 2025. | MEDIUM |

### Agent LLM Layer (Critical Decision)

**Recommendation: DeepSeek-R1 (distilled 70B) via Together AI or Fireworks AI**

| Model | Structured Output | Tool/Function Calling | Reasoning Quality | Cost (per 1M tokens, approx) | Verdict | Confidence |
|-------|-------------------|----------------------|-------------------|-------------------------------|---------|------------|
| DeepSeek-R1-Distill-Qwen-70B | Good -- follows JSON schemas reliably with system prompt enforcement | Supported via Together/Fireworks function calling API | Excellent chain-of-thought, strong at multi-step reasoning and analysis | ~$0.50-0.90 input / ~$0.90-1.50 output (Together AI) | **RECOMMENDED** | MEDIUM |
| Qwen2.5-72B-Instruct | Very good -- native structured output support, reliable JSON mode | Good -- purpose-built tool calling in Qwen2.5 | Strong general reasoning, slightly less depth than DeepSeek-R1 on complex analysis | ~$0.50-0.90 input / ~$0.90-1.50 output | Strong fallback | MEDIUM |
| DeepSeek-R1 (full 671B MoE) | Good | Supported | Best reasoning of the three, but overkill for most agent tasks | ~$2-4 input / ~$8-12 output | Too expensive for continuous agent loop | LOW |
| GLM-4-9B | Adequate | Basic | Weakest reasoning of the three, significantly smaller model | Cheapest (~$0.10-0.20/M tokens) | Too weak for hypothesis generation and diagnosis | MEDIUM |

**Rationale for DeepSeek-R1 distilled 70B:**

1. **Reasoning quality matters most for this use case.** The agent needs to analyze performance metrics, diagnose drift root causes, propose plausible hypotheses, and evaluate experiment results. DeepSeek-R1's chain-of-thought training gives it the strongest analytical reasoning of the Chinese open-source models.
2. **The 70B distilled version is the cost/quality sweet spot.** The full 671B MoE is too expensive for a continuous agent loop (potentially thousands of calls/day). The 70B distilled variant retains most of the reasoning capability at 5-10x lower cost.
3. **Together AI is the recommended provider** because they have the best function calling implementation on top of open-source models, competitive pricing, and reliable uptime. Fireworks AI is a close second.
4. **Qwen2.5-72B is the fallback.** If DeepSeek-R1's structured output proves unreliable in practice (which is possible -- R1 models sometimes produce verbose chain-of-thought that breaks JSON parsing), switch to Qwen2.5-72B which has more predictable output formatting.

**API Integration Pattern:**

```python
# Use the OpenAI-compatible API -- all providers support this
from openai import OpenAI

client = OpenAI(
    api_key="your-together-api-key",
    base_url="https://api.together.xyz/v1",
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-70B",
    messages=[...],
    response_format={"type": "json_object"},  # Enforce JSON output
    temperature=0.3,  # Low temperature for analytical tasks
)
```

**Cost Projection (MEDIUM confidence):**
- Agent loop at ~100 calls/day, ~2K tokens avg per call = ~200K tokens/day
- At ~$1/M tokens blended: ~$0.20/day = ~$6/month
- Even at 10x that volume: ~$60/month -- well within budget

### ML / Modeling

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| LightGBM | >=4.3 | Baseline model, fast iteration | Best tree-based model for tabular data. Fast training, excellent for feature-engineered financial data. Start here -- you will learn more from LightGBM feature importances than from a neural network. | MEDIUM |
| PyTorch | >=2.2 | Neural architectures (later phases) | Standard for custom neural nets. Use for temporal fusion transformers, attention-based architectures, or conditional GANs for synthetic data. Do NOT reach for this in Phase 1-2. | MEDIUM |
| scikit-learn | >=1.4 | Utilities, preprocessing, metrics | StandardScaler, train_test_split, classification_report, calibration curves. The glue of ML pipelines. | HIGH |
| Optuna | >=3.6 | Hyperparameter optimization | Better than GridSearch/RandomSearch. Bayesian optimization with pruning. Built-in LightGBM integration. Replaces manual hyperparameter tuning in the agent's experiment loop. | MEDIUM |
| SHAP | >=0.44 | Feature attribution, diagnostics | The agent's diagnostician needs to know WHICH features contributed to errors. SHAP values on LightGBM are fast (Tree SHAP is O(n) per prediction). | MEDIUM |

### Options Math

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| QuantLib (QuantLib-Python) | >=1.33 | Volatility surface construction, Black-Scholes, Greeks | Industrial-strength options math. The SWIG bindings are clunky but QuantLib is the only open-source library that handles the full surface construction pipeline (calibration, interpolation, extrapolation). No substitute. | MEDIUM |
| NumPy | >=1.26 | Numerical computation, Breeden-Litzenberger | Finite differences for B-L density extraction, moment computation, matrix operations. The backbone. | HIGH |
| SciPy | >=1.12 | Interpolation, optimization, statistics | `scipy.interpolate.CubicSpline` for strike ladder smoothing, `scipy.optimize` for implied vol inversion, `scipy.stats` for distribution tests. | HIGH |

**QuantLib vs. custom NumPy/SciPy: Use both.**

- QuantLib for: volatility surface construction (SABR calibration, strike/expiry interpolation), term structure building, Greek computation
- Custom NumPy/SciPy for: Breeden-Litzenberger (simpler to implement directly than wrapping QuantLib), GEX/vanna/charm flow aggregation, implied moment extraction

The PRD says "QuantLib or custom NumPy/SciPy" -- the answer is both, each for what they do best.

### NLP / Sentiment

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| transformers (Hugging Face) | >=4.40 | Model loading, inference pipeline | Standard interface for all pretrained NLP models. | MEDIUM |
| FinBERT (ProsusAI/finbert) | - | Financial sentiment classification | Purpose-trained on financial text. Significantly better than general-purpose BERT for financial news headlines. Three-class output (positive/negative/neutral) maps directly to the [-1, +1] sentiment score the PRD requires. | HIGH |
| sentence-transformers | >=3.0 | Semantic similarity, topic clustering | For clustering news headlines by topic and detecting novel vs. recurring themes. Useful for the "has the news already been priced in?" question. | MEDIUM |
| praw | >=7.7 | Reddit API access | For r/futures, r/commodities, commodity-specific subreddits. Simple, well-maintained. | HIGH |
| tweepy | >=4.14 | Twitter/X API access | For social sentiment. Note: X API pricing has changed significantly -- may need to evaluate cost. | LOW |

**NLP Architecture Decision:** Do NOT fine-tune FinBERT initially. Use it zero-shot. Fine-tuning on commodity-specific text is a Phase 3+ optimization experiment that the agent can propose. Starting with zero-shot FinBERT gives you a working baseline faster.

### Data Infrastructure

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| TimescaleDB | >=2.14 | Time-series feature store | PostgreSQL extension, so you get full SQL + time-series optimizations (continuous aggregates, compression, retention policies). Better than InfluxDB for this use case because your features are relational (join options data with sentiment data by timestamp). Feast adds unnecessary complexity on top. | MEDIUM |
| PostgreSQL | >=16 | Experiment journal, metadata | TimescaleDB IS PostgreSQL, so you get this for free. Store experiment journal entries as JSONB columns. No need for a separate database. | HIGH |
| Parquet (via pyarrow) | >=15.0 | Raw data storage | Columnar, compressed, fast reads. Partition by date and market. The standard for financial data lakes. | HIGH |
| DuckDB | >=0.10 | Ad-hoc analytics on Parquet | In-process analytical SQL engine. Query Parquet files directly without loading into memory. Perfect for "let me quickly check what happened on date X" analysis. | MEDIUM |

**Why NOT Feast:** The PRD mentions Feast, but Feast is overkill for a single-developer project targeting one market. Feast's value is in team environments where feature consistency across training and serving is hard to manage. For HYDRA, TimescaleDB continuous aggregates give you the same point-in-time correctness guarantees with far less operational overhead. If you ever need Feast-style feature serving at scale, it can be added later on top of TimescaleDB.

### Experiment Tracking

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| MLflow | >=2.12 | Experiment tracking, model registry, artifact storage | The PRD's choice is correct. MLflow is the standard for experiment tracking. The model registry provides the champion/candidate/archived model lifecycle the PRD describes. Self-hostable (important for trading IP). W&B is better UI but SaaS-only -- your experiment data should stay local. | MEDIUM |

### Execution / Broker

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| ib_insync | >=0.9.86 | Interactive Brokers API wrapper | Asyncio-based, Pythonic wrapper around IB's Java TWS API. Much better DX than the raw ibapi package. **WARNING:** ib_insync's maintenance status has been uncertain -- the original author (Ewald de Wit) passed away. Check if the community fork is active before committing. | LOW |
| ibapi (official IB API) | latest | Fallback IB connection | Official but clunky. Use only if ib_insync is unmaintained. | MEDIUM |

**Alternative to ib_insync:** If ib_insync is dead, consider `ib-async` (a community continuation) or wrapping the official `ibapi` with your own async layer. This needs validation at project start.

### Backtesting / Market Replay

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| vectorbt | >=0.26 | Quick strategy prototyping, vectorized backtests | Fast for initial signal validation. NOT sufficient for the full market replay engine -- it does not model thin-market microstructure (partial fills, queue priority, wide spreads). | MEDIUM |
| Custom replay engine | - | Production backtesting with realistic slippage | Must be custom. No off-the-shelf backtester handles thin-market microstructure correctly. Build on top of your TimescaleDB historical data. This is the PRD's explicit recommendation and it is correct. | HIGH |

**Why NOT Zipline/Backtrader:** Both are designed for equity markets with deep liquidity. Their slippage models assume you can fill at the quoted price, which is catastrophically wrong for oat futures trading 2,000 contracts/day. A custom engine that models slippage as f(order_size, ADV, spread, time_of_day) is non-negotiable.

### Scheduling / Orchestration

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| APScheduler | >=3.10 | Agent loop timing, cron-like scheduling | Lightweight, in-process. Perfect for "run the observer every 5 minutes, run diagnostics hourly, check for experiment completion every 30 seconds." Celery is overkill -- you don't need distributed task queues for a single-process agent. | MEDIUM |
| asyncio | stdlib | Async I/O, concurrent operations | The agent loop is inherently async (waiting on LLM calls, waiting on IB data, waiting on model training). Use asyncio as the concurrency backbone, not threading. | HIGH |

### CLI / Interface

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| Typer | >=0.12 | CLI framework | Better than Click for this use case -- automatic help generation, type hints as validation, rich output support. The PRD says "Click or Typer" -- use Typer. | MEDIUM |
| Rich | >=13.7 | Terminal formatting, tables, progress bars | Beautiful terminal output for status dashboards, experiment results tables, model comparison. | MEDIUM |

### Monitoring / Observability

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| structlog | >=24.1 | Structured logging | JSON-structured logs that can be queried, filtered, and piped to any backend. Better than stdlib logging for ML systems where you need to log metrics alongside messages. | MEDIUM |
| Prometheus client | >=0.20 | Metrics export | Lightweight metrics (model performance, feature staleness, agent loop timing). Grafana dashboard optional but valuable for visual monitoring. | LOW |

### Drift Detection / Statistical Testing

| Technology | Version | Purpose | Why Recommended | Confidence |
|------------|---------|---------|-----------------|------------|
| river | >=0.21 | Online drift detection (ADWIN, CUSUM) | The PRD specifies ADWIN and CUSUM -- river implements both for streaming data. Lightweight, no heavy dependencies. | MEDIUM |
| alibi-detect | >=0.12 | Advanced drift detection (MMD, KS tests) | For batch-mode distribution shift detection on feature columns. More comprehensive than river for periodic diagnostic checks. | LOW |

### Data Sources

| Source | Library/Method | Cost | Priority | Confidence |
|--------|---------------|------|----------|------------|
| CFTC COT Data | Custom scraper (CFTC website, free) | Free | P0 | HIGH |
| CME Futures OHLCV | Databento or IB historical data API | Databento: ~$100-500/mo; IB: included with commissions | P0 | MEDIUM |
| CME Options Chains | Databento (OPRA feed) or IB | Databento: additional cost; IB: included but delayed | P0 | LOW |
| Financial News | newsapi.org (free tier: 100 req/day) or Benzinga | NewsAPI free tier may suffice initially | P1 | MEDIUM |
| Cross-market data (VIX, etc.) | yfinance (free, unofficial Yahoo API) | Free | P1 | HIGH |
| Social media (Reddit) | PRAW + Reddit API | Free (within rate limits) | P2 | HIGH |
| Social media (Twitter/X) | Tweepy + X API | $100/mo minimum for useful access | P2 | LOW |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Feature Store | TimescaleDB | Feast | Feast adds operational overhead without proportional benefit for single-developer, single-market project. TimescaleDB continuous aggregates provide point-in-time correctness. |
| Feature Store | TimescaleDB | InfluxDB | InfluxDB's query language (Flux) is inferior to SQL for relational joins needed when combining options + sentiment features. |
| Experiment Tracking | MLflow | Weights & Biases | W&B is SaaS-only. Trading system experiment data (strategies, performance, model weights) should not leave your infrastructure. MLflow is self-hosted. |
| Experiment Tracking | MLflow | Neptune.ai | Same SaaS concern as W&B. |
| Hyperparameter Tuning | Optuna | Ray Tune | Ray Tune pulls in the entire Ray ecosystem. Optuna is focused, lightweight, and has first-class LightGBM integration. |
| Scheduling | APScheduler | Celery | Celery requires Redis/RabbitMQ broker. Massive overhead for a single-process agent loop. APScheduler runs in-process. |
| Scheduling | APScheduler | Prefect / Airflow | DAG-based orchestrators are for batch data pipelines. The agent loop is a continuous process, not a DAG. |
| CLI | Typer | Click | Typer is built on Click but adds type-hint-based validation and auto-generated help. Strictly better DX. |
| Agent LLM | DeepSeek-R1-70B | GPT-4o-mini | GPT-4o-mini is cheap but reasoning quality is significantly below DeepSeek-R1 for analytical tasks. |
| Agent LLM | DeepSeek-R1-70B | Llama 3.1 70B | Llama 3.1 is good but DeepSeek-R1's chain-of-thought training gives it an edge on the diagnostic/analytical reasoning this agent needs. |
| Backtesting | Custom engine | Backtrader/Zipline | Cannot model thin-market microstructure. Their slippage models assume liquid markets. |
| Time series DB | TimescaleDB | QuestDB | QuestDB is faster for pure append-only ingestion but weaker for the relational queries needed to join options + sentiment + model performance data. |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| LangChain | Massive abstraction layer with constant breaking changes. Your agent loop is deterministic with LLM calls at specific decision points -- you don't need a framework for this. | Direct OpenAI-compatible API calls via the `openai` Python package. |
| LangGraph | The PRD mentions LangGraph but it adds complexity without proportional value. Your agent loop is a fixed state machine (observe -> diagnose -> hypothesize -> experiment -> evaluate), not a dynamic graph. | Implement the state machine directly with asyncio + APScheduler. |
| Pandas for hot-path computation | Pandas is slow for row-by-row operations in live signal computation. | NumPy for numerical computation, Polars for data transformation, DuckDB for analytics. |
| TensorFlow | PyTorch has won the research/custom model war. TensorFlow adds a second ML framework with no benefit. | PyTorch for all neural architectures. |
| Airflow / Prefect | DAG schedulers for batch pipelines. The agent is a continuous loop, not a batch job. | APScheduler for periodic tasks within the agent process. |
| MongoDB | Schema-less sounds appealing for experiment journals but you will regret it when querying. | PostgreSQL (via TimescaleDB) with JSONB columns -- flexible schema with SQL queryability. |
| Docker for experiment isolation (initially) | The PRD mentions "full containerization of each experiment run." This is premature in Phase 1-3. Process-level isolation (separate Python processes) is sufficient until you have evidence that resource contention is a problem. | Subprocess-based isolation with separate MLflow run IDs. Add Docker in Phase 4+ if needed. |
| InfluxDB | Flux query language is a dead end. Telegraf/InfluxDB ecosystem is designed for DevOps metrics, not financial data. | TimescaleDB -- it's PostgreSQL with time-series superpowers. |
| Feast (initially) | Adds Kubernetes-oriented complexity. Designed for team environments where feature consistency across many services matters. | TimescaleDB continuous aggregates for point-in-time feature computation. |

---

## Stack Patterns by Variant

**If options data is EOD only (likely for thin markets):**
- Skip real-time streaming infrastructure entirely
- Use APScheduler to run signal computation once daily after market close
- Store features in TimescaleDB with daily granularity
- This dramatically simplifies the architecture

**If you get intraday options data:**
- Add a streaming layer: consider ZeroMQ for internal pub/sub between data ingestion and signal computation
- Increase feature store granularity to 1-minute or 5-minute bars
- Agent loop frequency increases -- ensure LLM API calls are rate-limited to avoid cost spikes

**If ib_insync is unmaintained:**
- Fall back to official `ibapi` with a custom async wrapper
- Consider `ib-async` (community fork) -- needs validation
- Worst case: use IB's REST API (Client Portal) which is newer but less feature-complete

**If DeepSeek-R1 structured output proves unreliable:**
- Switch to Qwen2.5-72B-Instruct which has more predictable JSON output
- Alternatively, use DeepSeek-R1 for reasoning but add a lightweight output parser/validator that retries on malformed JSON
- Consider instructor library (structured output extraction) as middleware

---

## Version Compatibility Notes

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| QuantLib-Python >=1.33 | Python 3.11 only (verify 3.12 support) | SWIG bindings sometimes lag behind Python releases. Test before committing to 3.12. |
| PyTorch >=2.2 | Python 3.11-3.12, CUDA 12.x | CPU-only is fine for LightGBM. GPU needed only for PyTorch neural models and FinBERT inference. |
| LightGBM >=4.3 | Python 3.8-3.12 | Broadly compatible. No GPU needed for tree models. |
| TimescaleDB >=2.14 | PostgreSQL 15-16 | Install as PostgreSQL extension. Docker image simplest for dev. |
| MLflow >=2.12 | Python 3.8-3.12, SQLite or PostgreSQL backend | Use PostgreSQL backend (same TimescaleDB instance) to avoid SQLite locking issues. |

---

## Installation

```bash
# Create virtual environment with uv
uv venv --python 3.11
source .venv/bin/activate

# Core ML
uv pip install lightgbm scikit-learn optuna shap

# PyTorch (CPU for now, add CUDA later if needed)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Options math
uv pip install QuantLib numpy scipy

# NLP / Sentiment
uv pip install transformers sentence-transformers
# FinBERT will download on first use via transformers

# Data infrastructure
uv pip install psycopg2-binary sqlalchemy pyarrow duckdb

# Agent LLM
uv pip install openai  # OpenAI-compatible client for Together/Fireworks

# Experiment tracking
uv pip install mlflow

# Broker
uv pip install ib_insync  # Verify maintenance status first

# Scheduling & async
uv pip install apscheduler

# CLI & formatting
uv pip install typer rich

# Monitoring & logging
uv pip install structlog

# Drift detection
uv pip install river

# Data sources
uv pip install praw yfinance requests

# Dev dependencies
uv pip install pytest pytest-asyncio ruff mypy pre-commit
```

---

## LLM Provider Comparison (for Agent Layer)

| Provider | Models Available | Function Calling | Pricing Tier | Reliability | Confidence |
|----------|-----------------|------------------|-------------|-------------|------------|
| Together AI | DeepSeek-R1 (full + distilled), Qwen2.5, Llama | Yes, OpenAI-compatible | Competitive (~$0.50-1.50/M tokens for 70B) | Good uptime, established | MEDIUM |
| Fireworks AI | DeepSeek-R1, Qwen2.5, Llama | Yes, OpenAI-compatible | Slightly cheaper than Together | Good, fast inference | MEDIUM |
| SiliconFlow | DeepSeek, Qwen, GLM | Yes | Cheapest (China-based infra) | Latency from US may be high, possible regulatory concerns with China-based provider | LOW |
| DeepSeek API (official) | DeepSeek-R1, DeepSeek-V2.5 | Yes | Very cheap ($0.14/M input, $0.28/M output for R1) | Has had capacity issues, China-based | LOW |
| Groq | Llama, Mixtral (limited DeepSeek) | Yes | Free tier available, paid is cheap | Fastest inference but limited model selection | MEDIUM |

**Recommendation:** Start with Together AI for the best balance of model selection, API quality, and reliability. Keep Fireworks AI as a failover. Do NOT depend on China-based providers (SiliconFlow, DeepSeek API) as primary -- latency and reliability from the US are concerns, plus potential future regulatory restrictions.

---

## Sources

All recommendations are based on training data through mid-2025. The following should be validated against current sources before implementation:

- **Version numbers:** Run `pip index versions <package>` for all packages before pinning
- **LLM API pricing:** Check Together AI, Fireworks AI, and DeepSeek pricing pages -- these change frequently
- **ib_insync maintenance status:** Check https://github.com/erdewit/ib_insync for recent commits and community fork status
- **QuantLib Python 3.12 compatibility:** Check https://github.com/lballabio/QuantLib-SWIG/releases
- **TimescaleDB version:** Check https://docs.timescale.com/about/latest/timescaledb-releases/
- **Feast current status:** Check if Feast has simplified its architecture since 2024 (it was trending toward complexity)
- **X/Twitter API pricing:** Has changed multiple times; verify current tier structure

---
*Stack research for: HYDRA -- Autonomous ML Trading Agent*
*Researched: 2026-02-18*
*Confidence: MEDIUM overall (training data only, no live verification possible)*
