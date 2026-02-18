# Pitfalls Research

**Domain:** Autonomous self-healing ML trading agent for low-volume futures markets
**Researched:** 2026-02-18
**Confidence:** MEDIUM (based on training data -- domain is well-documented in quant literature but web verification was unavailable)

**Note on sources:** WebSearch and WebFetch were unavailable during this research. Findings are drawn from training data covering quantitative finance literature (Marcos Lopez de Prado, Ernie Chan, Robert Carver), options pricing theory (Hull, Natenberg, Gatheral), LLM agent system design patterns, and practitioner post-mortems. These domains are mature and well-documented, so confidence remains MEDIUM rather than LOW. The cheap-LLM-specific pitfalls (DeepSeek-R1, Qwen2.5) are MEDIUM-LOW confidence because behavior may have changed since training cutoff. Validate through hands-on testing during Phase 4.

---

## Critical Pitfalls

Mistakes that cause full rewrites, blown accounts, or systems that silently produce garbage outputs while appearing healthy.

### Pitfall 1: Lookahead Bias in Backtesting (The Silent Account Killer)

**What goes wrong:**
The backtest shows beautiful returns, the model gets promoted, and it immediately fails in live trading. The root cause is information leakage from the future into historical training/evaluation. In HYDRA's case, this is especially dangerous because the agent loop will run hundreds of backtests autonomously -- any systematic lookahead bias gets compounded across every experiment the agent runs. The agent literally optimizes for exploiting the leakage.

**Why it happens:**
- Using COT data based on report date rather than release date (COT data is collected Tuesday, released Friday -- 3 days of lookahead).
- Computing options-implied features using end-of-day settlement prices that include after-hours adjustments not available at the time.
- Normalizing features using statistics computed over the full dataset rather than expanding windows.
- Using `fillna(method='ffill')` on features that have genuine gaps, making it appear that information was available before it actually was.
- Walk-forward validation that inadvertently leaks test period data through feature engineering pipelines (e.g., fitting a scaler on the full dataset before splitting).
- The agent's journal queries referencing future experiment outcomes when generating hypotheses for current experiments.
- `pandas.shift(-1)` instead of `shift(1)` -- accidentally using tomorrow's data.

**How to avoid:**
- Implement a strict `as_of` timestamp on every data point. No data point should be queryable before its availability timestamp. For COT: availability = Friday release, not Tuesday collection.
- Build the feature pipeline as a pure function of `(raw_data, as_of_timestamp)` -- never `(raw_data)`. This forces every computation to respect temporal boundaries.
- Use `sklearn.model_selection.TimeSeriesSplit` or a custom walk-forward splitter with mandatory embargo gaps (1-5 trading days) between train and test.
- Add a "time traveler" detector: a simple automated test that checks if any feature at time T has correlation > 0.5 with the target at time T. Suspiciously high correlation = probable leakage.
- Validate the full pipeline end-to-end: compute features in "streaming mode" (one bar at a time) and compare against batch-computed features. Any discrepancy reveals leakage.
- Never shuffle time-series data in train/test splits.

**Warning signs:**
- Backtest Sharpe ratio above 3.0 on any market (almost certainly leakage in thin markets).
- Dramatic performance cliff between walk-forward backtest and paper trading.
- Model performance that improves when you add more features (in thin markets, more features usually means more noise -- improvement signals leakage).
- Out-of-sample performance dramatically worse than in-sample.

**Phase to address:** Phase 1 (Data Infrastructure) and Phase 3 (Sandbox). The `as_of` timestamp system must be baked into the data layer from day one. The sandbox must enforce temporal integrity in every replay.

**Severity:** CATASTROPHIC. Every model the agent produces will be tainted. You will not know until live money is at risk. Months of work invalidated.

---

### Pitfall 2: Options Math Numerical Instability in Thin Markets

**What goes wrong:**
The Breeden-Litzenberger method requires computing second derivatives of call prices across the strike ladder. In thin markets with wide bid-ask spreads and sparse strike coverage, this produces wildly unstable implied distributions -- negative probability densities, implied distributions with multiple spurious modes, or implied moments (skew, kurtosis) that jump around chaotically between days with no real change in market conditions. The entire signal layer is built on top of this computation, so garbage here means garbage everywhere downstream.

**Why it happens:**
- Thin markets have 5-15 liquid strikes vs. 50+ in liquid markets like SPX. Cubic spline interpolation on 5 points is unreliable.
- Bid-ask spreads of 10-30% of option value mean that using mid-prices introduces massive noise. The "true" price could be anywhere in that spread.
- Finite difference second derivatives amplify noise quadratically. A 1% error in option price becomes a 100%+ error in the second derivative at typical strike spacings.
- Some strikes have zero open interest or zero volume -- they are listed but effectively fictional prices.
- Near-expiry options have time decay artifacts that distort the surface.
- Stale quotes: many strikes have not traded in days; displayed price is yesterday's close, not current market.
- End-of-day vs. settlement: options settlement prices may differ from last trade prices.

**How to avoid:**
- Do NOT blindly apply Breeden-Litzenberger to raw thin-market option prices. Pre-smooth the volatility surface first: fit a parametric model like SVI (Stochastic Volatility Inspired, from Jim Gatheral) to the sparse implied vols, then compute the density from the smooth parametric surface. SVI has 5 parameters, works well with sparse data, and guarantees no butterfly arbitrage in the fitted surface.
- Implement arbitrage-free constraints: call prices must be decreasing and convex in strike. If your fitted surface violates this, the density will go negative. Reject and re-fit.
- Use option mid-prices only when spread < 20% of mid. Otherwise, use a volume-weighted or open-interest-weighted estimate.
- Set minimum data quality thresholds: require at least 8 liquid strikes (OI > 50 contracts) across at least 2 expiries before computing a distribution. Below this threshold, emit a "low confidence" flag and fall back to simpler metrics (ATM implied vol only).
- Filter: maximum bid-ask spread threshold (< 10% of mid for inclusion in surface calibration).
- Confidence-weight each strike by liquidity (OI * volume) in surface calibration.
- Validate: the extracted density should integrate to approximately 1.0 (+/- 0.05). Clip negative density values to zero.
- Compare your implied vol against IB's quoted implied vol for sanity checks.
- Version and cache every computed surface with its data quality score. The agent needs to know when it is making decisions based on high-quality vs. degraded signal.
- Consider starting with pure NumPy/SciPy for Breeden-Litzenberger and moments; add QuantLib for vol surface only. Have a NumPy fallback for surface construction to avoid QuantLib SWIG binding issues (see Pitfall 10).

**Warning signs:**
- Implied kurtosis values jumping by 50%+ day-to-day with stable underlying price.
- Negative probability density regions in the fitted distribution.
- Implied mean that diverges significantly from the futures price (should be close for short-dated options).
- GEX values that flip sign on consecutive days without corresponding open interest changes.
- Implied vol surface has sudden spikes or negative values.
- Percentage of strikes passing quality filters drops below 50%.

**Phase to address:** Phase 1 (Options Math Engine). This is the foundation of the entire signal layer. Getting it wrong here poisons everything downstream.

**Severity:** CATASTROPHIC. Garbage features in, garbage predictions out, and the agent will spend its entire mutation budget trying to fix model performance when the real problem is upstream data quality.

---

### Pitfall 3: Agent Degenerate Experiment Loops

**What goes wrong:**
The self-healing agent enters a loop where it keeps proposing similar experiments, getting marginal improvements in backtest, promoting candidates that fail in practice, diagnosing the failure, and proposing the same type of experiment again. The mutation budget burns through while the system oscillates rather than improving. In the worst case, the agent "discovers" that overfitting to the evaluation window is the most reliable way to get promoted. The agent loop itself becomes an overfitting engine.

**Why it happens:**
- The hypothesis engine draws from a fixed mutation playbook. After exhausting the obviously useful mutations, it starts recombining them or tweaking hyperparameters in diminishing-returns territory.
- Short evaluation windows for candidate models. The agent finds a configuration that performs well on a specific evaluation window, gets promoted, then regresses to the mean on new data. The agent diagnoses the regression, proposes a tweak, which performs well on the NEW evaluation window, gets promoted, regresses... ad infinitum.
- No cooldown between promotions -- agent can promote a new champion every day.
- The journal deduplication check is too narrow. "Add vanna flow feature" and "add vanna flow feature with 3-day lag" are treated as different experiments, but they test the same underlying hypothesis.
- The agent lacks a concept of "this avenue of investigation is exhausted." It will keep returning to feature engineering when the fundamental issue is model architecture or market regime change.
- Fitness function does not penalize complexity, so agent keeps adding features.
- Agent "learns" from experiment journal that certain mutations work, but the journal reflects overfitted history.
- The cheap LLM used for reasoning may have insufficient judgment to distinguish genuinely novel hypotheses from rephrased versions of failed ones.

**How to avoid:**
- Implement a **mutation budget per category per time window**: maximum 5 feature engineering experiments per month, maximum 3 architecture swaps per month, etc. When a category is exhausted, the agent must try a different category or wait.
- Use **semantic similarity** (embedding-based) on experiment hypotheses to detect near-duplicates, not just string matching. If a new hypothesis has > 0.85 cosine similarity with a failed experiment in the last 90 days, require explicit justification or reject.
- Implement **experiment cooldown**: after 3 consecutive failures in a category, that category enters a 2-week cooldown.
- Require **monotonically increasing evaluation window diversity**: each promotion must pass on at least one evaluation window that was NOT used for the previous promotion.
- Add a **meta-diagnostic**: every 20 experiments, the agent must produce a summary of what it has learned, which avenues are exhausted, and what unexplored directions remain.
- Implement an **experiment velocity alarm**: if the agent is running experiments faster than once per 6 hours consistently, something is likely wrong.
- **Minimum champion hold period** (48 hours, configurable). **Minimum evaluation window** (20+ trading days).
- **Simplicity penalty** in fitness function (fewer features = higher score at equal performance).
- **Decay weighting** on experiment journal (recent experiments weighted higher, but not exclusively).
- Start at `supervised` autonomy. Graduate to `semi-auto` only after 20+ experiments with human review.

**Warning signs:**
- More than 5 experiment cycles without a durable promotion (one that lasts > 2 weeks).
- Rising experiment frequency (agent is trying harder but not getting smarter).
- Champion model version numbers incrementing rapidly but composite fitness staying flat.
- Experiment journal showing clusters of very similar hypotheses.
- Champion model churn rate exceeds once per week.

**Phase to address:** Phase 4 (Agent Core). The mutation budget system, cooldowns, and meta-diagnostics must be designed into the agent architecture, not bolted on later.

**Severity:** HIGH. The system does not blow up, but it consumes compute/API budget while producing no improvement. Over months, this erodes confidence in the entire approach. Each bad promotion briefly exposes capital to a worse model.

---

### Pitfall 4: Cheap LLM Structured Output Failures

**What goes wrong:**
The agent reasoning layer (DeepSeek/Qwen/GLM) is asked to produce structured outputs (JSON diagnosis objects, experiment proposals, journal queries). The model intermittently produces malformed JSON, hallucinates field names, omits required fields, or generates values outside expected ranges. This causes the deterministic execution layer to crash, silently ignore malformed fields, or take bizarre actions based on hallucinated data.

DeepSeek-R1 specifically produces chain-of-thought reasoning (including `<think>` tags) interspersed with the JSON output, breaking structured output parsing. Other models have their own idiosyncrasies.

**Why it happens:**
- Cheaper/smaller LLMs have significantly worse instruction following than frontier models, especially for complex structured output.
- Some models are trained primarily on Chinese-language data; structured output reliability may degrade on English-language prompts with domain-specific terminology (quantitative finance jargon).
- Temperature and sampling settings interact unpredictably with structured output tasks. What works for reasoning quality may be terrible for format compliance.
- Context window utilization: as the agent sends longer context (journal history, diagnostic data), structured output quality degrades, especially near the model's context limit.
- Models may "reason" correctly but express the answer in a format that breaks the parser -- wrapping JSON in markdown code blocks, using single quotes instead of double quotes, trailing commas, comments in JSON.

**How to avoid:**
- **Never trust raw LLM output.** Always parse through a strict validation layer (Pydantic models with `model_validate_json()`). Every structured output must have a schema, and every response must be validated against it before any downstream action.
- **Use `response_format={"type": "json_object"}`** in the API call if supported. Use constrained decoding / JSON mode / function calling if the provider supports it.
- **Add a robust JSON extraction layer** that strips non-JSON content (think tags, markdown code fences, preamble text) before parsing.
- **Implement retry with reformulation**: on parse failure, retry up to 3 times with increasingly explicit formatting instructions. Track failure rates per model and per output type.
- **Keep structured output schemas simple.** Flatten nested objects. Use enums for categorical fields (e.g., `mutation_type` should be one of a fixed set, not free-text).
- **Separate reasoning from formatting.** Use a two-stage prompt: first ask the model to reason in free text, then ask it to fill in a specific template. This produces much better results than asking for reasoning and formatting simultaneously.
- **Build a fallback chain**: DeepSeek-R1 (primary) -> Qwen2.5 (fallback) -> hardcoded heuristic rules (emergency fallback). If the LLM fails to produce valid structured output after retries, fall back to simpler rule-based diagnostics rather than crashing.
- **Consider the `instructor` library** for structured output extraction with Pydantic validation.
- **Log every LLM interaction** with input/output and parse success/failure. Monitor structured output failure rate over time.

**Warning signs:**
- Parse failure rate above 5% on any output type.
- Increasing parse failures over time (context getting longer, model struggling more).
- Agent actions that seem random or uncorrelated with diagnostic data (likely caused by silent field misinterpretation).
- LLM producing experiment proposals that reference features or model types not in the mutation playbook (hallucination).

**Phase to address:** Phase 4 (Agent Core) with validation framework designed in Phase 1 (project scaffolding). The Pydantic validation layer and fallback chain must be first-class architectural components, not afterthoughts.

**Severity:** HIGH. Silent failures here can cause the agent to take actions based on hallucinated reasoning. A misclassified diagnosis severity could skip a critical rollback. A hallucinated mutation type could crash the experiment runner.

---

### Pitfall 5: Unrealistic Slippage Models for Thin Markets

**What goes wrong:**
The backtest uses a fixed slippage model (e.g., "0.5 ticks per trade") that drastically underestimates actual execution costs in thin markets. The model appears profitable in backtest but loses money live because every trade moves the market against you. The agent keeps producing candidates that look great in the sandbox but fail to translate -- and it cannot diagnose why because the sandbox's slippage model is wrong. This is especially deadly in oat futures and similar low-volume markets.

**Why it happens:**
- Default backtester slippage models assume continuous liquidity. In oat futures or lean hog options, the order book may have 5-20 contracts at the best bid/ask, with the next level 2-3 ticks away.
- Time-of-day effects are massive in thin markets. Liquidity is concentrated in the first and last 30 minutes of the session. Mid-day orders face dramatically worse fills.
- Market impact is non-linear. 1 contract might fill at the bid, but 5 contracts might move the market 3 ticks.
- Partial fills are the norm, not the exception. An order for 10 contracts might fill 3 immediately and the rest over 20 minutes at progressively worse prices.
- Thin markets have "ghost liquidity" -- orders that appear in the book but are pulled when you try to hit them.
- Using the quoted mid price as the fill price ignores the bid-ask spread entirely.
- Queue priority means your limit order may never fill if you are behind existing orders.

**How to avoid:**
- Build a **volume-adaptive slippage model**: `slippage = base_spread + impact_coefficient * (order_size / trailing_5min_volume)^impact_exponent`. Calibrate the coefficients from actual fill data (start with paper trading fills, then real fills).
- Model slippage as a **distribution, not a point estimate**. For each simulated trade, sample slippage from a calibrated distribution. This captures the fat-tailed nature of thin-market execution.
- Implement **time-of-day liquidity curves**: different slippage parameters for market open, mid-session, and close.
- Set a **hard rule**: never simulate orders larger than 10% of trailing 20-minute volume, and never more than 2% of ADV.
- Include **failed trade costs**: in thin markets, sometimes you simply cannot execute. Model the probability of non-execution and include the opportunity cost.
- **Validate the slippage model continuously**: compare simulated fills from the sandbox against actual fills from paper/live trading. If the simulation is systematically optimistic, recalibrate immediately.
- All evaluation metrics must be slippage-adjusted from day one. Test with 2x your expected slippage to build margin of safety.
- Make slippage a **first-class feature** in the evaluation fitness function. The PRD has "slippage-adjusted return" at 0.15 weight -- this is good, but the slippage model must be realistic for the weight to matter.

**Warning signs:**
- Paper trading results consistently 30%+ worse than backtest for the same model.
- Slippage model producing the same slippage regardless of time of day or order size.
- Model preferring to trade during low-liquidity periods (it has learned to exploit the unrealistic slippage model).
- Backtest results improving when position sizes are increased (in reality, larger positions = worse fills in thin markets).
- Slippage eats more than 30% of gross return.

**Phase to address:** Phase 3 (Sandbox). The slippage model is part of the market replay engine. It must be calibrated before the agent starts running experiments, or every experiment result will be misleading.

**Severity:** CATASTROPHIC. The entire agent loop depends on the sandbox producing results that correlate with reality. If the sandbox is systematically wrong about execution costs, every model promotion decision is tainted.

---

### Pitfall 6: Sentiment Signal Overfitting and Noise Dominance

**What goes wrong:**
The sentiment engine produces scores that appear predictive in backtest but are actually fitting to noise. NLP features from financial news are particularly dangerous because they contain a high ratio of noise to signal in thin markets. The model overfits to coincidental correlations between news sentiment and price movements during the training window.

**Why it happens:**
- Thin markets have very little dedicated news coverage. The NLP sentiment score for lean hogs might be dominated by general "agriculture" or "commodities" sentiment, which has low relevance.
- FinBERT and similar models are trained on broad financial text about equities and earnings. They do not understand the specific drivers of niche futures markets. "Grain prices rally" might be miscategorized as relevant to oat futures when the rally is in corn. Commodity-specific language ("crop report," "export inspections," "cattle on feed") may not score correctly.
- Social media sentiment for thin markets is near-zero volume. A single Reddit post or tweet can swing the score, making it pure noise.
- COT data is weekly, creating a temptation to interpolate or lag-fill daily sentiment scores from weekly data. This introduces artifacts.
- Backtest overfitting: with limited data history in thin markets (maybe 3-5 years of clean data), the model has very few independent data points to learn from. Sentiment features add dimensionality faster than they add signal.

**How to avoid:**
- **Start with COT only.** It is the highest signal-to-noise sentiment source for thin markets. It is institutional, structured, and relevant. Add NLP and social sentiment only after COT-based models demonstrate baseline predictive power. Weight COT heavily (0.6+) and NLP lightly (0.2) initially.
- **Implement a feature importance gate**: any sentiment feature must demonstrate statistical significance (p < 0.05 with Bonferroni correction for multiple testing) in walk-forward validation before being included in the production model.
- **Use sentiment-momentum rather than sentiment-level.** The change in COT positioning (are commercials adding to longs?) is typically more predictive than the absolute level.
- **Domain-filter NLP carefully.** Build a keyword/topic filter specific to the target market. Reject articles that do not contain at least one domain keyword in the first 200 words.
- **Cap the dimensionality contribution of sentiment features.** If the total feature vector is 20 features, no more than 5 should be sentiment-derived.
- **Implement a "sentiment blackout" test**: periodically evaluate the champion model with all sentiment features zeroed out. If performance drops by less than 5%, sentiment is not adding real value and may be adding overfitting risk.
- Validate FinBERT on a hand-labeled sample of commodity headlines before trusting it. Log sentiment scores alongside headlines so you can audit calibration.

**Warning signs:**
- Sentiment features ranking in the top 5 by SHAP importance (in thin markets, options math features should dominate if the thesis is correct).
- Model performance improving dramatically when social media sentiment is added (almost certainly noise fitting with sparse social data).
- Sentiment scores showing high autocorrelation (> 0.8 at lag-1) -- indicates interpolation artifacts or stale source data.
- Walk-forward performance variance increasing after adding sentiment features.

**Phase to address:** Phase 2 (Signal Layer). Start with COT only. Gate NLP/social behind statistical validation. Do not add complexity before proving the base signal works.

**Severity:** HIGH. Overfitting to sentiment is the most likely source of false confidence in backtest results. The model looks great, passes promotion, then decays because the sentiment correlations were spurious.

---

### Pitfall 7: False Positive Model Promotions from Short Evaluation Windows

**What goes wrong:**
A candidate model outperforms the champion over the evaluation windows and passes all promotion criteria, but the evaluation period happens to coincide with a market regime that favors the candidate. Once the regime changes, the new champion performs worse than the model it replaced. The system has effectively regressed.

**Why it happens:**
- The PRD specifies "3 of 5 independent evaluation windows" for promotion. If these windows are all from recent history, they likely share the same regime. Passing 3 of 5 is not meaningfully different from passing 1 of 1 if all windows are in the same regime.
- Thin markets have fewer regime transitions per year than liquid markets (less information flow = more persistent regimes). A 6-month evaluation dataset might contain only 1-2 regime transitions.
- The 48-hour paper trading validation is far too short. In thin markets with daily-frequency signals, 48 hours is 2 data points. This catches catastrophic failures but not systematic underperformance.
- Sharpe ratio is noisy at short horizons. The standard error of a Sharpe ratio estimate from N independent observations is approximately `sqrt((1 + 0.5 * SR^2) / N)`. For a true Sharpe of 1.0 measured over 60 daily observations, the standard error is ~0.14 -- meaning a candidate with true Sharpe of 0.7 could appear to have Sharpe 1.1 through pure noise.

**How to avoid:**
- **Extend the paper trading validation to at least 2 weeks**, preferably 4 weeks, before live promotion. The PRD says "4 weeks before live capital" but the 48-hour window for champion promotion is dangerous even in paper mode.
- **Require regime diversity in evaluation windows.** Define 2-3 regime types (trending, mean-reverting, high-vol, low-vol) using a regime detection model. At least 2 of the 5 evaluation windows must come from different regimes.
- **Implement a promotion decay period.** After promotion, the new champion must outperform the previous champion's last 30 days within its own first 30 days, or automatic rollback occurs.
- **Use a Bayesian model comparison** instead of point estimates. Rather than "candidate Sharpe > champion Sharpe + 0.1," compute the posterior probability that the candidate is truly better. Require P(candidate > champion) > 0.80.
- **Track the "promotion success rate"**: what fraction of promotions are still the champion after 30 days? If this falls below 50%, the evaluation criteria are too lenient.
- Validate across multiple non-overlapping test windows (the PRD requires 3 of 5, which is good, but window selection matters more than window count).

**Warning signs:**
- Rapid champion turnover (more than one promotion per month in steady-state operation).
- Promoted models being rolled back within a week.
- Evaluation window performance showing high variance across windows (some great, some bad).
- All 5 evaluation windows coming from the same 6-month period.

**Phase to address:** Phase 4 (Agent Core, Evaluator/Judge). The promotion protocol must be designed with regime diversity and Bayesian comparison from the start. Phase 5 (Hardening) should validate with extended paper trading.

**Severity:** HIGH. Each false promotion briefly exposes capital to a worse model. Frequent false promotions erode the agent's journal quality (successful promotions that were actually luck get logged as evidence for similar mutations).

---

### Pitfall 8: Data Pipeline Silent Failures

**What goes wrong:**
A data source stops updating, changes format, or degrades quality, but the system continues operating on stale or corrupted data without raising an alert. The options math engine computes features from yesterday's options chain, the sentiment engine uses week-old news scores, and the model makes predictions based on a reality that no longer exists.

**Why it happens:**
- Data sources for thin markets are less reliable than for liquid markets. CME options data for oats may have delayed updates, missing strikes, or occasional formatting changes.
- Free data sources (CFTC COT, Yahoo Finance) have no SLA. They break without warning and without notification.
- Data pipeline code often uses `try/except: pass` patterns that silently swallow errors.
- Feature stores that use `ffill` (forward-fill) make stale data look fresh. The feature store shows a value for today, but it is actually last Thursday's value.
- Upstream API changes (new authentication requirements, endpoint URL changes, response format changes) break ingestion with no error -- just empty responses.

**How to avoid:**
- **Implement staleness checks on every data source.** Each source has a maximum expected gap (futures prices: 1 trading day; options chains: 1 trading day; COT: 1 week; news: 4 hours during market hours). If a source exceeds its maximum gap, raise WARNING. If it exceeds 2x, raise CRITICAL.
- **Never use `ffill` without a staleness limit.** `ffill(limit=N)` where N is the maximum acceptable forward-fill periods. After N periods, the value should become NaN, forcing downstream consumers to handle the gap explicitly.
- **Compute and log data quality metrics daily**: completeness (% of expected records received), freshness (age of most recent record), consistency (does the data pass basic sanity checks).
- **Implement source-specific validators**: options chains must have monotonically decreasing call prices with strike (no-arbitrage), COT positions must sum correctly, futures prices must be within N standard deviations of the previous close.
- **Design for graceful degradation**: when a source is stale/missing, reduce confidence in affected features, shrink position sizing proportionally, and log the degradation. Do NOT continue trading at full size with stale data.
- **Test data pipeline resilience explicitly**: inject simulated outages (zero-length API responses, malformed JSON, HTTP 503) and verify the system detects and handles them correctly.
- Handle weekend/holiday staleness correctly: alerts should not fire when markets are closed.

**Warning signs:**
- Feature store showing identical values on consecutive trading days (especially for volatile features like implied vol).
- Data pipeline logs showing no errors but also no new records for a source.
- Model performance degrading without any drift detection alerts (the drift detector uses the same stale features, so it sees no change).
- Options math engine computing surfaces from chains with fewer strikes than expected.

**Phase to address:** Phase 1 (Data Infrastructure). Staleness checks and data quality monitoring must be built into the ingestion layer from the beginning, not added after a failure.

**Severity:** HIGH. Silent data failures are the most common cause of unexplained model degradation in production ML systems. The agent will waste its mutation budget trying to fix the model when the real problem is upstream data quality.

---

## Moderate Pitfalls

### Pitfall 9: Cheap LLM Reasoning Quality Degradation Under Load

**What goes wrong:**
The cheap LLM (DeepSeek/Qwen) produces acceptable reasoning quality during initial testing with short contexts and simple queries. As the system matures and the context grows (longer journal history, more complex diagnostic data, more nuanced hypotheses), reasoning quality silently degrades. The model starts producing shallow diagnoses ("retrain the model" for every problem), repetitive hypotheses, and diagnoses that miss the actual root cause.

**Why it happens:**
- Smaller models have a steeper quality degradation curve as context length increases compared to frontier models.
- The agent's prompts get longer over time as more history and context is included.
- Domain-specific reasoning (quantitative finance) is likely underrepresented in the training data of general-purpose Chinese LLMs.
- The model may "satisfice" -- produce outputs that pass format validation but contain shallow or template-like reasoning that does not actually address the specific situation.

**How to avoid:**
- Implement **reasoning quality metrics**: have the LLM explain its reasoning chain, and check for specificity (does it reference actual feature names, actual metric values, actual experiment IDs?) vs. generic statements.
- Use **RAG over the journal** rather than dumping the full journal into context. Retrieve the 5-10 most relevant experiments for the current situation, not all 500.
- Enforce a **context budget**: maximum 4,000 tokens of context per LLM call. Summarize and compress aggressively.
- Run periodic **reasoning audits**: have a human review the last 10 LLM diagnoses and hypotheses for quality. If quality is declining, consider upgrading the model or simplifying the prompts.
- Build **deterministic guardrails** around LLM reasoning: the LLM proposes, but deterministic code validates that the proposal is physically possible, references real features, and is within the mutation playbook.

**Warning signs:**
- LLM diagnoses becoming shorter and more generic over time.
- LLM proposing mutations for features that do not exist in the current feature set.
- LLM repeatedly suggesting "retrain on recent data" regardless of the diagnosed problem.
- Experiment proposals that are syntactically valid but semantically identical to previous failures.

**Phase to address:** Phase 4 (Agent Core). Build reasoning quality checks into the agent from the start.

---

### Pitfall 10: QuantLib SWIG Binding Complexity

**What goes wrong:**
QuantLib's Python bindings are auto-generated by SWIG from C++ code. The API is not Pythonic, documentation is sparse, and errors are cryptic. This can block progress for days during Phase 1.

**Why it happens:**
- SWIG wraps C++ classes directly -- Python developers must understand the C++ API.
- Memory management gotchas (Python objects holding references to deleted C++ objects).
- Version mismatches between QuantLib C++ and the SWIG bindings.
- Building from source is painful on macOS (requires Boost headers).

**How to avoid:**
- Install from PyPI (`pip install QuantLib`) -- do NOT build from source.
- Pin the exact version and do not upgrade casually.
- Wrap all QuantLib calls in your own Pythonic interface layer. Never expose QuantLib types outside the options math module.
- Write extensive tests against known-good values (Black-Scholes closed-form for verification).
- Start with pure NumPy/SciPy for Breeden-Litzenberger and moments; add QuantLib for vol surface construction only if needed.
- Have a pure NumPy fallback for surface construction.

**Warning signs:** Segfaults, mysterious `RuntimeError: unable to convert` messages, memory leaks.

**Phase to address:** Phase 1 (Options Math Engine).

---

### Pitfall 11: Walk-Forward Validation Window Selection Bias

**What goes wrong:**
Choosing the wrong training/test window sizes. Too short = noisy estimates. Too long = includes irrelevant regime data. No single "correct" choice, and the agent may inadvertently overfit to the window configuration itself.

**How to avoid:**
- Use expanding window (not sliding) for training: always train on ALL available history.
- Test window: minimum 20 trading days to get statistically meaningful estimates.
- Embargo: 1-5 days between train end and test start (prevents autocorrelation leakage).
- The agent should be able to experiment with window parameters, but constrain minimum test window size.
- Validate across multiple non-overlapping test windows.

**Phase to address:** Phase 3 (Sandbox/Backtesting).

---

### Pitfall 12: MLflow / Experiment State Explosion

**What goes wrong:**
The agent generates hundreds of experiments, each with full model artifacts. The artifact store fills up disk, the UI becomes unusable, queries slow down, and SQLite locks under concurrent writes from the agent loop.

**How to avoid:**
- Retention policy: auto-archive experiments older than 90 days, delete artifacts for non-promoted experiments older than 30 days.
- Log lightweight metrics, not full model weights, for failed experiments.
- Use PostgreSQL backend (not SQLite) for the tracking store.
- Regular cleanup job as part of the agent's maintenance cycle.

**Phase to address:** Phase 3 (Experiment Tracking).

---

### Pitfall 13: COT Data Timing and Revision Trap

**What goes wrong:**
COT data reflects Tuesday positions but is released Friday after close. Using Friday's release as if it represents Friday's positioning introduces a 3-day stale signal. Also, the CFTC revises historical COT reports, and failing to re-download recent weeks means your feature store has outdated values.

**How to avoid:**
- Timestamp COT data as Tuesday (the as-of date) for feature computation, but do not make it available in the pipeline until Friday (the release date).
- Use COT positioning CHANGES (week-over-week deltas) rather than absolute levels.
- Re-download the last 4 weeks each cycle to catch revisions.
- Use the Disaggregated Futures-Only reports (they separate Producer/Merchant, Swap Dealer, Managed Money, Other) -- not the legacy format.

**Phase to address:** Phase 2 (Sentiment Engine, COT integration).

---

### Pitfall 14: IB API Connection Instability

**What goes wrong:**
IB's TWS/Gateway disconnects periodically (daily restart at ~midnight ET, weekend shutdowns, random disconnects under load). The 50 messages/second rate limit is easy to hit during active querying.

**How to avoid:**
- Implement automatic reconnection with exponential backoff.
- Use IB Gateway (headless) instead of TWS for production.
- Handle the daily restart gracefully -- save state before midnight, reconnect after.
- Implement heartbeat monitoring -- if no heartbeat for 30 seconds, trigger reconnection.
- Queue orders during disconnection and replay them on reconnection (with staleness check).
- Implement order state machine (submitted -> partially_filled -> filled/cancelled).
- Respect rate limits with request queuing.

**Phase to address:** Phase 5 (Live Trading).

---

## Minor Pitfalls

### Pitfall 15: Timezone Chaos

**What goes wrong:** Market data in exchange time (CT for CME), API timestamps in UTC, local machine in a third timezone. Feature computation produces wrong results silently.

**Prevention:** Normalize everything to UTC immediately at ingestion. Store all timestamps as UTC. Convert to exchange time only for display.

---

### Pitfall 16: yfinance API Breakage

**What goes wrong:** yfinance is an unofficial scraper of Yahoo Finance. It breaks periodically when Yahoo changes their frontend.

**Prevention:** Only use yfinance for cross-market reference data (VIX, correlated commodities) that is not latency-sensitive. Have a fallback. Do not use it for your primary data pipeline.

---

### Pitfall 17: Agent LLM Cost Creep

**What goes wrong:** Debug logging of full LLM prompts/responses, retries on failures, verbose chain-of-thought reasoning all increase token usage beyond projections. A degenerate loop can burn through an entire month's budget in hours.

**Prevention:**
- Track token usage per agent cycle in metrics.
- Set a daily token budget with hard cap.
- Compress historical context sent to the LLM.
- Log LLM interactions at DEBUG level, not INFO.

---

### Pitfall 18: NumPy Floating Point Precision in Options Math

**What goes wrong:** Breeden-Litzenberger's second derivative amplifies noise. Finite differences on noisy, unevenly-spaced strike prices produce wild oscillations. Using `float32` instead of `float64` introduces additional numerical error that compounds through the computation chain.

**Prevention:**
- Use `float64` everywhere in the options math engine.
- Smooth the call price curve (SVI or cubic spline) before computing derivatives.
- Use Savitzky-Golay filter or Gaussian smoothing as alternative.
- Validate: the extracted density should integrate to approximately 1.0.
- Clip negative density values to zero.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using mid-price for all options instead of volume-weighted | Simpler code, no volume data needed | 10-30% error in implied distributions for wide-spread options | Never in thin markets -- spread is too wide |
| Fixed slippage (e.g., 1 tick) in backtests | Fast to implement | Every backtest result is wrong; agent learns to exploit the gap | Only for initial smoke tests; must be replaced before agent runs experiments |
| Storing features as CSVs instead of proper time-series DB | No database setup needed | Slow queries, no `as_of` semantics, no concurrent access, feature staleness invisible | First 2 weeks of prototyping, then migrate |
| Single training/test split instead of walk-forward | Faster iteration | Massive overfitting risk, especially with thin market data volumes | Never for promotion decisions; acceptable for rapid hypothesis screening only |
| Hardcoding the risk-free rate | Avoids data pipeline for treasury rates | Miscalculated implied forwards and put-call parity violations | Acceptable for prototyping if rate is updated weekly manually |
| Skipping experiment isolation (running candidate in same process as champion) | Simpler deployment | Candidate crash can take down champion; shared state can cause data leakage between experiments | Never -- this is a hard requirement |
| Using the LLM for deterministic logic (e.g., computing metrics) | Fewer code paths | Non-deterministic outputs, occasionally wrong arithmetic, expensive for simple math | Never -- LLM for reasoning, code for computation |
| Forward-filling COT data daily | Makes daily feature vector complete | Creates illusion of daily sentiment updates when data is weekly; model may overfit to interpolation artifacts | Only if explicitly marked as "stale_fill" with staleness counter feature included |
| Skipping the embargo gap in walk-forward validation | More training/test data available | Autocorrelation leaks future information into training set | Never |

## Integration Gotchas

Common mistakes when connecting to external services specific to this domain.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| CME/OPRA Options Data | Assuming all listed strikes have meaningful prices; using strikes with OI < 10 | Filter to strikes with OI > 50 and non-zero volume in last 5 sessions; mark remaining as "illiquid" |
| CFTC COT Reports | Parsing the legacy format and missing disaggregated report data; not handling report revisions | Use Disaggregated Futures-Only reports; re-download last 4 weeks each cycle to catch revisions |
| Interactive Brokers API | Assuming synchronous order fills; not handling partial fills; ignoring the 50 msg/sec rate limit | Use `reqExecutions` for fill reports; implement order state machine; respect rate limits with request queuing |
| News APIs (Benzinga, NewsAPI) | Treating all articles equally; not deduplicating syndicated content | Deduplicate on headline similarity (> 0.9 cosine = same story); weight by source credibility; filter to market-relevant articles |
| DeepSeek/Qwen API | Assuming consistent response times; not handling rate limiting or regional outages | Implement exponential backoff; maintain response time SLA (> 30s = timeout and fallback); cache recent reasoning for replay during outages |
| Yahoo Finance (cross-market data) | Relying on it as primary data source; not handling frequent API changes | Use as supplementary source only; implement scraping fallback; cache with 15-minute TTL |

## Performance Traps

Patterns that work at small scale but fail as the system grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Computing full volatility surface from scratch every cycle | Works fine for 1 market at daily frequency | Cache previous surface; only recompute changed strikes; use incremental updates | When adding intraday frequency or multiple markets |
| Storing experiment artifacts as files on disk | Simple, works for first 50 experiments | Use MLflow or similar artifact store with metadata indexing | After ~200 experiments; journal queries take minutes; disk space grows unboundedly |
| Loading full feature history for each training run | Fine with 1 year of daily data (252 rows) | Implement feature cache with lazy loading; only load the training window needed | When history exceeds 5 years or frequency goes intraday |
| Running all candidate experiments sequentially | Acceptable with 1 experiment every few days | Implement experiment queue with parallel sandbox instances | When agent queues 5+ experiments; sequential execution creates multi-day backlogs |
| LLM context growing unboundedly | Agent sends full journal to LLM for each reasoning step | Implement RAG over the journal; summarize old entries; enforce context budget | When journal exceeds 500 entries; context window fills, quality degrades, costs spike |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Storing API keys (broker, data provider, LLM) in code or config files | Key theft leads to unauthorized trading, data access, or API cost accumulation | Use environment variables or secrets manager; never commit keys to git; rotate broker API keys quarterly |
| Agent having unrestricted access to broker API | Autonomous agent malfunction could place unlimited orders | Implement broker API wrapper with hard limits: max order size, max daily orders, max position size, max daily loss; wrapper enforces these regardless of agent requests |
| Not rate-limiting the agent's LLM API calls | Degenerate agent loop burns through API budget; potential runaway costs | Hard cap on LLM API calls per hour (e.g., 60); per-day budget cap; alert at 80% of budget |
| Running the agent with root/admin privileges | Agent process compromise = full system compromise | Run in limited-privilege container; no sudo; read-only code access; write only to designated data directories |
| Logging raw LLM reasoning that includes API keys or account details | Credentials in logs can be exfiltrated | Sanitize all logs; never pass credentials through LLM context; audit log outputs for credential patterns |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Options Math Engine:** Often missing negative density rejection -- verify that `implied_distribution` never contains negative probabilities and that the integral is approximately 1.0 (+/- 0.05)
- [ ] **Backtesting:** Often missing embargo gap between train/test splits -- verify that the gap is at least 5 trading days to prevent feature leakage from autocorrelated time series
- [ ] **Walk-Forward Validation:** Often missing data pipeline replay -- verify that features are recomputed in each window using only data available at that point, not pre-computed features from the full dataset
- [ ] **Slippage Model:** Often missing time-of-day variation -- verify that slippage at 12:00pm is materially higher than slippage at 9:35am in the simulation
- [ ] **COT Data Integration:** Often missing the release-date vs. collection-date distinction -- verify that COT data for Tuesday is not available until Friday in the feature pipeline
- [ ] **Sentiment Engine:** Often missing deduplication -- verify that syndicated news articles are not counted multiple times, inflating sentiment scores
- [ ] **Experiment Journal:** Often missing failed experiment cleanup -- verify that failed experiment artifacts (model checkpoints, intermediate data) are cleaned up and not accumulating on disk
- [ ] **Model Promotion:** Often missing rollback validation -- verify that rolling back from champion v0.3.8 to v0.3.7 actually restores v0.3.7's full configuration (features, hyperparameters, preprocessing pipeline, not just model weights)
- [ ] **Agent Reasoning:** Often missing input validation on LLM outputs -- verify that the agent never acts on LLM output that fails schema validation, even partially
- [ ] **Circuit Breakers:** Often missing the "resume" logic -- verify that after a circuit breaker triggers and the agent rolls back, the system can actually resume trading (not stuck in lockdown with no manual intervention path)
- [ ] **Data Quality Monitoring:** Often missing weekend/holiday handling -- verify that staleness alerts do not fire on weekends and holidays when markets are closed
- [ ] **Feature Pipeline:** Often missing the `as_of` enforcement -- verify that no feature computed at time T uses any data point with availability timestamp after T

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Lookahead bias discovered in production model | HIGH | 1. Immediately switch to lockdown mode. 2. Identify the leakage source. 3. Fix the data pipeline. 4. Invalidate ALL experiment journal entries that used leaked data. 5. Retrain champion from scratch on clean data. 6. Re-run promotion evaluation. |
| Options math producing garbage distributions | MEDIUM | 1. Fall back to ATM implied vol only (simple, robust feature). 2. Diagnose the surface fitting failure. 3. Implement SVI parameterization if not already done. 4. Rebuild feature history from raw options data with the fixed pipeline. |
| Agent in degenerate experiment loop | LOW | 1. Pause the agent (`gsd pause`). 2. Review the last 20 experiments in the journal. 3. Identify the loop pattern. 4. Add the pattern to the deduplication blacklist. 5. Reset the mutation budget. 6. Resume with human oversight at `supervised` level for 1 week. |
| Cheap LLM producing unreliable outputs | MEDIUM | 1. Switch to `supervised` autonomy level. 2. Evaluate alternative model (switch from DeepSeek to Qwen or vice versa). 3. Simplify output schemas. 4. If all models fail, fall back to rule-based diagnostics and hypothesis generation (less creative but reliable). |
| Unrealistic slippage causing false promotions | HIGH | 1. Lockdown mode. 2. Audit the last 3 promotions by comparing backtest fills vs. paper trading fills. 3. Recalibrate the slippage model using actual fill data. 4. Re-run all candidate evaluations with the corrected model. 5. Rollback champion if current one was falsely promoted. |
| Silent data pipeline failure | MEDIUM | 1. Identify which sources failed and when. 2. Determine if any model predictions were based on stale data. 3. Backfill missing data if possible. 4. Re-run any experiments that used stale data. 5. Implement the missing staleness checks. 6. Reduce position sizing until data quality is confirmed restored. |
| Sentiment overfitting discovered | MEDIUM | 1. Retrain champion with sentiment features removed. 2. Compare performance with and without sentiment on truly out-of-sample data. 3. If sentiment adds no value, remove it. 4. If marginal value, add regularization and re-gate behind statistical significance testing. |
| False positive promotion | LOW | 1. Rollback to previous champion. 2. Review evaluation window composition for regime diversity. 3. Tighten promotion criteria (require P(candidate > champion) > 0.80). 4. Extend paper trading validation period. |

## HYDRA-Specific Compound Risks

These are risks unique to HYDRA's specific design -- the combination of cheap LLM reasoning + thin markets + autonomous operation creates failure modes that would not exist in any one of these domains alone.

### Compound Risk 1: Cheap LLM Misdiagnosis Leading to Wrong Healing Action

The agent observes model degradation. The cheap LLM diagnoses "feature drift in sentiment scores" when the actual cause is a data pipeline outage. The agent quarantines the sentiment features and retrains without them, appearing to "fix" the issue (because the model now uses fewer stale features). When the data pipeline recovers, the agent has no signal to re-include sentiment. The system has permanently amputated a useful feature based on a wrong diagnosis.

**Prevention:** Always run the data audit (staleness check) BEFORE the LLM-based diagnosis. If any data quality issues are found, address those first and re-evaluate before asking the LLM to reason about deeper causes. Make data quality checks deterministic code, not LLM reasoning.

**Phase to address:** Phase 4 (Agent Core). The diagnostician's triage order must be: data audit (deterministic) -> feature attribution (deterministic) -> regime check (deterministic) -> LLM reasoning (for novel situations only).

---

### Compound Risk 2: Thin Market Regime Changes Misinterpreted as Model Failure

Thin markets can go through extended periods of zero volatility (no price movement for days) followed by sharp, illiquid moves. The observer detects "model producing degenerate outputs" (constant predictions during the flat period). The agent diagnoses model failure and initiates healing. But the model is correct -- there is genuinely nothing happening. The healing mutations add noise to try to generate alpha that does not exist in a flat market.

**Prevention:** Build a "market activity detector" that classifies the current state as active/dormant. During dormant periods, suppress the agent loop. Do not try to heal a model that is correctly predicting "nothing will happen."

**Phase to address:** Phase 4 (Observer Module). The observer needs to distinguish "model is broken" from "market is dead."

---

### Compound Risk 3: Agent Optimization Pressure Against Illiquidity Constraints

The agent discovers that the model performs best (in backtest) with larger position sizes during specific conditions. But in thin markets, larger positions are precisely when slippage is worst. The agent keeps optimizing the model toward higher conviction signals (which suggest larger positions) while the execution environment silently degrades the fills. The metrics look great in the sandbox; the actual PnL is negative.

**Prevention:** Make position sizing part of the sandbox's evaluation, not separate from it. The fitness function must evaluate the model + position sizing + slippage as a single unit. The agent should never be able to increase conviction/position size without the slippage impact being reflected in the fitness score.

**Phase to address:** Phase 3 (Sandbox) and Phase 4 (Evaluator). The fitness function design must couple model performance with execution cost.

---

### Compound Risk 4: Journal Poisoning Through LLM Hallucination

The cheap LLM writes experiment notes and tags to the journal. Over time, hallucinated observations ("charm flow showed strong signal in trending regimes" when the data actually showed no significance) accumulate in the journal. Future LLM reasoning retrieves these hallucinated notes and treats them as established knowledge, building hypotheses on a foundation of fabricated evidence. The journal, intended to be the system's institutional knowledge, becomes a source of misinformation.

**Prevention:** Agent notes in the journal must be grounded in verifiable metrics. Every claim in the notes must reference a specific metric value that can be independently computed. Implement a "note verification" step that spot-checks 10% of LLM-generated notes against actual experiment metrics. If a note's claims do not match the recorded metrics, flag it and mark the entry as "unverified."

**Phase to address:** Phase 4 (Experiment Journal). The journal schema must distinguish between computed metrics (deterministic, trustworthy) and LLM-generated notes (potentially hallucinated, requires verification).

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Lookahead bias (#1) | Phase 1: Build `as_of` timestamp into data layer | Run "time traveler" detector on every feature; compare streaming vs. batch feature computation |
| Options math instability (#2) | Phase 1: SVI surface fitting with arbitrage-free constraints | Check no negative densities; implied mean within 2% of futures price for short-dated options |
| Agent degenerate loops (#3) | Phase 4: Mutation budgets, cooldowns, semantic deduplication | Experiment-to-durable-promotion ratio > 1:10 |
| Cheap LLM structured output (#4) | Phase 4: Pydantic validation, retry logic, fallback chain | Parse failure rate < 5% across all output types |
| Unrealistic slippage (#5) | Phase 3: Volume-adaptive slippage model | Compare simulated fills vs. paper trading fills; correlation > 0.8 |
| Sentiment overfitting (#6) | Phase 2: Start COT-only; gate NLP/social behind significance test | "Sentiment blackout" test shows < 5% performance drop |
| False positive promotions (#7) | Phase 4: Regime-diverse evaluation windows; Bayesian comparison | Promotion success rate (still champion after 30 days) > 70% |
| Silent data pipeline failures (#8) | Phase 1: Staleness checks, data quality metrics, `ffill(limit=N)` | Zero undetected outages lasting more than 1 max-expected-gap |
| LLM reasoning degradation (#9) | Phase 4: RAG over journal, context budget, quality checks | Human audit of LLM reasoning quality quarterly |
| QuantLib binding issues (#10) | Phase 1: NumPy/SciPy first, QuantLib as optional enhancement | All options math functions have pure Python fallback |
| Walk-forward window bias (#11) | Phase 3: Expanding windows, minimum 20-day test, embargo gaps | Walk-forward results stable across window parameter variations |
| MLflow state explosion (#12) | Phase 3: PostgreSQL backend, retention policies from day one | Journal query latency < 5 seconds; disk usage under monitoring |
| COT timing trap (#13) | Phase 2: Proper as-of vs. release date handling | COT feature unavailable before Friday in pipeline tests |
| IB API instability (#14) | Phase 5: Reconnection logic, heartbeat monitoring, order state machine | Zero dropped orders during daily restart; 100% partial fill handling |
| Compound: LLM misdiagnosis (#CR1) | Phase 4: Deterministic checks before LLM reasoning | Data audit runs before every diagnostic cycle |
| Compound: False dormancy detection (#CR2) | Phase 4: Market activity detector | Agent suppresses healing during objectively dormant markets |
| Compound: Optimization vs. illiquidity (#CR3) | Phase 3+4: Coupled fitness function | Fitness function penalizes large positions in low-volume conditions |
| Compound: Journal poisoning (#CR4) | Phase 4: Note verification, metric grounding | 10% of LLM notes spot-checked against computed metrics |

---

## Sources

- Marcos Lopez de Prado: "Advances in Financial Machine Learning" -- backtesting pitfalls, walk-forward validation, combinatorial purged cross-validation (MEDIUM confidence, well-established)
- Robert Carver: "Systematic Trading" -- position sizing, slippage modeling, thin market considerations (MEDIUM confidence)
- Ernie Chan: "Quantitative Trading" -- backtesting biases, data quality issues (MEDIUM confidence)
- John Hull: "Options, Futures, and Other Derivatives" -- options pricing theory, put-call parity (HIGH confidence, foundational)
- Sheldon Natenberg: "Option Volatility and Pricing" -- vol surface construction (HIGH confidence)
- Jim Gatheral: "The Volatility Surface" -- SVI parameterization, arbitrage-free surface constraints (HIGH confidence, seminal work)
- Breeden-Litzenberger theorem -- numerical implementation considerations from academic finance literature (HIGH confidence for theory, MEDIUM for implementation details)
- LLM agent system design patterns from training data -- multi-agent failure modes, structured output reliability (MEDIUM-LOW confidence, rapidly evolving field)
- QuantLib, MLflow, Interactive Brokers API operational experience from practitioner community (MEDIUM confidence)
- PRD risk register (Section 10) -- aligned with and significantly expanded upon

**Confidence note:** The quantitative trading pitfalls (#1, #2, #5, #6, #7) are drawn from well-established literature and have HIGH to MEDIUM confidence. The LLM-specific pitfalls (#4, #9, CR1, CR4) are MEDIUM-LOW confidence because the specific models (DeepSeek-R1, Qwen2.5) evolve rapidly. The compound risks (CR1-CR4) are original analysis based on domain knowledge combination -- MEDIUM confidence on the failure modes, but specific mitigations should be validated during implementation.

---
*Pitfalls research for: Autonomous self-healing ML trading agent (HYDRA)*
*Researched: 2026-02-18*
