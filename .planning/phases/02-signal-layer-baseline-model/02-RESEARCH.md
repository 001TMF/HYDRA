# Phase 2: Signal Layer + Baseline Model - Research

**Researched:** 2026-02-19
**Domain:** COT sentiment scoring, options-sentiment divergence detection, LightGBM classification, walk-forward backtesting, position sizing, risk management, slippage modeling
**Confidence:** HIGH

## Summary

Phase 2 transforms the raw data and options math from Phase 1 into tradeable signals and validates the core thesis: that divergence between options-implied expectations and COT-derived sentiment predicts future price movement. The work divides into three distinct layers: (1) sentiment signal construction from COT positioning data, (2) a divergence detector that classifies the relationship between options-implied signals and sentiment into a 6-type taxonomy, and (3) a LightGBM baseline model with walk-forward backtesting, position sizing, circuit breakers, and volume-adaptive slippage.

The existing Phase 1 codebase provides a clean foundation. The COT ingestion pipeline (`src/hydra/data/ingestion/cot.py`) already fetches managed money, producer, and swap positions with correct as_of/available_at timing. The options math modules (`density.py`, `moments.py`, `greeks.py`) produce implied distributions, moments (mean, variance, skew, kurtosis), and Greeks flows (GEX, vanna, charm). The feature store (`feature_store.py`) provides point-in-time correct queries. Phase 2 reads these outputs and builds new modules on top, requiring no modifications to Phase 1 code.

The validation gate for this phase is existential: if the divergence signal produces Sharpe <= 0 out-of-sample after slippage, the entire project thesis needs re-examination before committing to Phases 3-5. This means the backtesting framework must be rigorous -- walk-forward with embargo gaps to prevent information leakage, and volume-adaptive slippage rather than optimistic fixed slippage.

**Primary recommendation:** Build in strict dependency order -- sentiment scoring first, divergence detector second, then model + backtest infrastructure -- because each layer depends on the previous. The LightGBM model should use `objective='binary'` for directional prediction (up/down) with probability calibration, not regression. Walk-forward backtesting must use purged/embargoed splits, not random cross-validation.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SGNL-01 | COT data produces normalized sentiment score in [-1, +1] with confidence weight | COT index via 52-week percentile ranking (standard industry approach), confidence from OI magnitude and velocity. Existing COT ingestion provides managed_money_net, producer_net, swap_net, total_oi. |
| SGNL-02 | Divergence detector classifies options-implied vs. sentiment divergence into 6 types per PRD taxonomy | Taxonomy from PRD Section 5.3. Inputs: implied moments (mean vs spot, skew, kurtosis) from `moments.py` + sentiment score from SGNL-01. Classification via threshold-based rules, not ML. |
| SGNL-03 | Divergence output includes direction, magnitude, type, confidence, and suggested bias | Dataclass with all fields. Direction from sign of (implied_mean - spot) vs sentiment_score. Magnitude from z-scored difference. Confidence from data quality flags (FULL/DEGRADED) and sentiment confidence. |
| MODL-01 | LightGBM baseline model trained on divergence + feature store features produces directional predictions | LightGBM 4.6.0 with `objective='binary'`, `metric='binary_logloss'`. Features: divergence fields + implied moments + Greeks flows + COT features from feature store. Target: binary (price up/down over N-day horizon). |
| MODL-02 | Walk-forward backtesting with expanding/rolling window and embargo gaps validates model out-of-sample | Custom `PurgedWalkForwardSplit` with configurable embargo gap (default 5 trading days). Expanding window preferred for thin-market data scarcity. Compute OOS Sharpe per fold, aggregate. |
| MODL-03 | Fractional Kelly position sizing caps positions at configurable fraction of average daily volume | Half-Kelly default (fraction=0.5). Volume cap at configurable % of 20-day average daily volume (default 2%). Position = min(kelly_size, volume_cap). |
| MODL-04 | Circuit breakers halt trading on max daily loss, max drawdown, max position size, or max single-trade loss thresholds | Four independent breakers with configurable thresholds. State machine: ACTIVE -> TRIGGERED -> COOLDOWN -> ACTIVE. All checked pre-trade in backtest loop. |
| MODL-05 | All backtest and evaluation metrics are slippage-adjusted using volume-adaptive slippage model | Square-root impact model: slippage = spread/2 + k * sigma * sqrt(V_order / V_daily). Calibrate k from thin-market data. Conservative default k=0.1 for thin markets. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LightGBM | 4.6.0 | Gradient boosted tree classifier for directional prediction | M5 competition winner. Handles missing values natively, fast training, built-in categorical feature support. CPU-optimized per project constraints. |
| NumPy | >=1.26 | Numerical computation (already in project) | Foundation for all numerical work |
| SciPy | >=1.12 | Statistical functions, z-scores (already in project) | `scipy.stats.percentileofscore`, `scipy.stats.zscore` for normalization |
| structlog | >=24.1 | Structured logging (already in project) | Consistent with Phase 1 logging approach |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | >=2.1 | DataFrame operations for feature matrix assembly and backtest results | Assembling feature matrices for LightGBM training, backtest result analysis |
| scikit-learn | >=1.4 | Metrics (log_loss, accuracy_score, roc_auc_score), TimeSeriesSplit base | Evaluation metrics and cross-validation scaffold |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| LightGBM | XGBoost/CatBoost | LightGBM faster on smaller datasets, less memory; XGBoost slightly better on some tabular benchmarks but difference marginal |
| Custom walk-forward | skfolio CombinatorialPurgedCV | CPCV is statistically superior but complex; walk-forward is industry standard and simpler to reason about for a baseline |
| Custom slippage model | QuantConnect VolumeShareSlippage | External dependency; our thin-market needs require custom calibration anyway |
| pandas | Polars | Polars faster but LightGBM/scikit-learn ecosystem expects pandas; avoid friction |

**Installation:**
```bash
uv add lightgbm pandas scikit-learn
```

## Architecture Patterns

### Recommended Project Structure
```
src/hydra/
├── data/                         # Phase 1 (existing)
│   ├── ingestion/                # COT, futures, options pipelines
│   ├── store/                    # feature_store.py, parquet_lake.py
│   └── quality.py                # Data quality monitoring
├── signals/
│   ├── options_math/             # Phase 1 (existing): surface, density, moments, greeks
│   ├── sentiment/                # Phase 2 NEW
│   │   ├── __init__.py
│   │   └── cot_scoring.py        # SGNL-01: COT sentiment score [-1, +1]
│   └── divergence/               # Phase 2 NEW
│       ├── __init__.py
│       └── detector.py           # SGNL-02, SGNL-03: Divergence detection + classification
├── model/                        # Phase 2 NEW
│   ├── __init__.py
│   ├── features.py               # Feature matrix assembly from feature store
│   ├── baseline.py               # MODL-01: LightGBM wrapper with train/predict
│   ├── walk_forward.py           # MODL-02: Walk-forward backtesting engine
│   └── evaluation.py             # Backtest metrics computation
├── risk/                         # Phase 2 NEW
│   ├── __init__.py
│   ├── position_sizing.py        # MODL-03: Fractional Kelly + volume cap
│   ├── circuit_breakers.py       # MODL-04: Trading halt conditions
│   └── slippage.py               # MODL-05: Volume-adaptive slippage model
```

### Pattern 1: Sentiment Scoring Pipeline
**What:** Transform raw COT positioning data into a normalized sentiment score in [-1, +1].
**When to use:** Every time new COT data is available (weekly, after Friday release).
**Implementation approach:**

The standard COT index approach uses a percentile ranking over a lookback window:

```python
# COT Index: where does current net positioning fall in its historical range?
# Uses 52-week (1 year) lookback as industry standard
def cot_index(net_position: float, history: np.ndarray) -> float:
    """Compute COT index as percentile rank in [0, 1] range."""
    if len(history) < 2:
        return 0.5  # neutral when insufficient history
    rank = scipy.stats.percentileofscore(history, net_position) / 100.0
    return rank  # 0.0 = most bearish, 1.0 = most bullish

# Normalize to [-1, +1]: score = 2 * cot_index - 1
# Confidence weight from: OI magnitude relative to history + positioning velocity
```

Key insight: Use managed money net positioning as the primary signal (speculators drive price), with producer/commercial net as a contrarian confirmation signal. When commercials are extremely short and speculators extremely long, the market is likely overextended.

### Pattern 2: Divergence Classification (Rule-Based)
**What:** Classify the relationship between options-implied signals and sentiment into 6 types.
**When to use:** After computing both implied moments and sentiment score for a given timestamp.
**Implementation approach:**

```python
@dataclass
class DivergenceSignal:
    direction: int          # +1 long, -1 short, 0 neutral
    magnitude: float        # z-scored magnitude of divergence
    divergence_type: str    # one of 6 taxonomy types
    confidence: float       # [0, 1] composite confidence
    suggested_bias: str     # "long", "short", "fade_sentiment", "early_entry", "trend_follow", "vol_play"

def classify_divergence(
    implied_mean: float,
    spot: float,
    implied_skew: float,
    implied_kurtosis: float,
    sentiment_score: float,  # [-1, +1]
    options_quality: DataQuality,
    sentiment_confidence: float,
) -> DivergenceSignal:
    """Classify divergence per PRD taxonomy (Section 5.3)."""
    # Threshold-based classification, NOT ML
    # Options bias: (implied_mean - spot) / spot, z-scored
    # Compare sign of options bias vs sentiment score sign
```

The 6-type taxonomy from the PRD maps to these conditions:
1. **Options bullish + Sentiment bearish** -> Long (trust options)
2. **Options bearish + Sentiment bullish** -> Short (trust options)
3. **Options neutral + Sentiment strong** -> Fade sentiment
4. **Options directional + skew shift + Sentiment hasn't moved** -> Early entry
5. **Both aligned** -> Trend follow
6. **High kurtosis + flat mean** -> Volatility play

### Pattern 3: Walk-Forward Backtesting
**What:** Train model on expanding window, predict on out-of-sample fold, with embargo gap between train and test.
**When to use:** Validating any model against historical data.
**Implementation approach:**

```python
class PurgedWalkForwardSplit:
    """Walk-forward cross-validator with embargo gap.

    Timeline: [====TRAIN====][--EMBARGO--][==TEST==]

    - Expanding window: train set grows each fold
    - Embargo gap: N trading days removed between train end and test start
    - No future data leaks into training
    """
    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        min_train_size: int = 252,  # ~1 year of trading days
        test_size: int = 63,        # ~3 months
    ):
        ...

    def split(self, X, y=None, groups=None):
        """Yield (train_idx, test_idx) tuples."""
        ...
```

### Pattern 4: Volume-Adaptive Slippage
**What:** Model slippage as a function of order size, daily volume, and volatility.
**When to use:** Every simulated trade in backtesting and paper trading.
**Implementation approach:**

The square-root impact model is the standard for market microstructure:

```python
def estimate_slippage(
    order_size: float,          # number of contracts
    daily_volume: float,        # average daily volume
    spread: float,              # bid-ask spread
    daily_volatility: float,    # annualized vol / sqrt(252)
    impact_coefficient: float = 0.1,  # calibrated per market
) -> float:
    """Volume-adaptive slippage per contract.

    slippage = spread/2 + k * sigma_daily * sqrt(V_order / V_daily)

    Components:
    - spread/2: half the bid-ask spread (crossing cost)
    - k * sigma * sqrt(participation_rate): market impact
    """
    participation_rate = order_size / max(daily_volume, 1)
    market_impact = impact_coefficient * daily_volatility * np.sqrt(participation_rate)
    return spread / 2.0 + market_impact
```

### Anti-Patterns to Avoid
- **Random cross-validation on time series:** Guarantees information leakage. Always use temporal splits.
- **Fixed slippage in thin markets:** 1 tick slippage is wildly optimistic for oats/lean hogs. Volume-adaptive is mandatory.
- **Training on divergence type as a feature:** The divergence type is a derived label, not a raw feature. Use the underlying numerical components (magnitude, direction, skew delta) as features, not the categorical classification.
- **Optimizing backtest parameters to pass the Sharpe > 0 gate:** The gate is meant to validate the thesis, not to be gamed. Use reasonable defaults, not grid-searched parameters.
- **Mixing options math quality levels:** When density quality is DEGRADED, the implied moments are None. The feature matrix must handle this gracefully (NaN features that LightGBM handles natively).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gradient boosted trees | Custom tree ensemble | LightGBM | Heavily optimized C++ core, handles NaN natively, proven in competitions |
| Classification metrics | Custom accuracy/AUC | scikit-learn metrics | `accuracy_score`, `roc_auc_score`, `log_loss`, `brier_score_loss` -- battle-tested |
| Percentile ranking | Custom sort + rank | `scipy.stats.percentileofscore` | Handles edge cases (ties, empty arrays) correctly |
| DataFrame operations | Custom dict-of-arrays wrangling | pandas DataFrame | Feature matrix assembly, groupby, merge-asof for temporal alignment |
| Statistical tests | Custom significance tests | `scipy.stats.ttest_1samp` | Testing if mean OOS Sharpe > 0 with proper confidence intervals |

**Key insight:** The domain complexity is in the signal construction and backtesting methodology, not in the ML infrastructure. LightGBM and scikit-learn handle the ML plumbing; the hard part is correct temporal handling and avoiding lookahead bias.

## Common Pitfalls

### Pitfall 1: Lookahead Bias in Feature Construction
**What goes wrong:** Using features that wouldn't have been available at prediction time. COT data from Tuesday used before Friday release. Options chain data from today used for yesterday's prediction.
**Why it happens:** The feature store prevents this for point-in-time queries, but feature engineering code can accidentally bypass it by computing derived features (like rolling averages of COT data) without respecting available_at timestamps.
**How to avoid:** ALL feature computation must go through `feature_store.get_features_at(market, query_time)`. Never compute features directly from raw data in the model training loop. The feature store's `available_at <= query_time` filter is the single source of truth.
**Warning signs:** OOS performance suspiciously close to in-sample. Sharpe ratio much higher than expected for weekly COT signals.

### Pitfall 2: Survivorship Bias in Walk-Forward Windows
**What goes wrong:** Starting the backtest at a point where data happens to be unusually clean or the signal happens to work well.
**Why it happens:** Thin markets have structural breaks (contract specification changes, exchange rule changes, seasonal patterns). Starting after a favorable regime biases results.
**How to avoid:** Use the maximum available data history. Report performance across all folds, not just the aggregate. Flag any fold with extreme performance (good or bad) for manual inspection.
**Warning signs:** High variance across walk-forward folds. One fold dominates the aggregate Sharpe.

### Pitfall 3: Overfitting LightGBM Hyperparameters
**What goes wrong:** Grid-searching hyperparameters on the walk-forward test set, then reporting the best result as "out-of-sample" performance.
**Why it happens:** Natural temptation to tune the model to pass the Sharpe > 0 gate.
**How to avoid:** Use conservative default hyperparameters for the baseline: `num_leaves=31`, `learning_rate=0.1`, `n_estimators=100`, `min_child_samples=20`. NO hyperparameter tuning in Phase 2. Tuning belongs in Phase 3 (sandbox) where it can be properly tracked in the experiment journal.
**Warning signs:** Model with 500+ trees on a dataset with < 500 training examples. Hyperparameters that were clearly grid-searched.

### Pitfall 4: Unrealistic Slippage in Thin Markets
**What goes wrong:** Using fixed slippage (e.g., 1 tick) that dramatically underestimates actual execution costs in thin markets.
**Why it happens:** Most backtesting tutorials assume liquid equity markets. Lean hogs and oats have wide spreads and low volume.
**How to avoid:** Volume-adaptive slippage model with conservative impact coefficient. Validate slippage assumptions against actual bid-ask spreads in the data. Double the estimated slippage as a robustness check.
**Warning signs:** Sharpe drops below 0 when slippage is doubled. Net returns are only marginally positive after slippage (signal-to-noise ratio too low).

### Pitfall 5: Ignoring COT Data Limitations
**What goes wrong:** Treating COT positioning as a real-time signal when it's actually weekly with a 3-day publication lag.
**Why it happens:** COT data is collected Tuesday, released Friday. Between releases, the signal is stale. Over-weighting COT in a daily prediction model inflates its apparent value.
**How to avoid:** COT sentiment should decay toward neutral between releases (exponential decay or step function). The confidence weight should decrease as days since last release increase. The prediction horizon should be weekly or longer to match the signal frequency.
**Warning signs:** Model performance drops significantly when COT features are lagged by an additional week. COT features dominate SHAP importance despite being updated only weekly.

### Pitfall 6: Circuit Breakers That Don't Reset
**What goes wrong:** Circuit breaker triggers and permanently halts trading, making the backtest useless.
**Why it happens:** No cooldown/reset mechanism. A single bad day triggers the breaker and all subsequent folds show zero returns.
**How to avoid:** Circuit breakers should have a cooldown period (e.g., 1 trading day for daily loss, 5 days for drawdown). After cooldown, trading resumes with reduced position sizing.
**Warning signs:** Backtest shows long flat periods after a drawdown. Circuit breaker triggers on > 50% of folds.

## Code Examples

### COT Sentiment Score Computation
```python
import numpy as np
from scipy.stats import percentileofscore
from dataclasses import dataclass

@dataclass
class SentimentScore:
    """Normalized sentiment score with confidence weight."""
    score: float       # [-1, +1]: -1 bearish, +1 bullish
    confidence: float  # [0, 1]: confidence in the score
    components: dict   # breakdown by trader category

def compute_cot_sentiment(
    managed_money_net: float,
    producer_net: float,
    total_oi: float,
    history_managed: np.ndarray,  # last 52 weeks of managed_money_net
    history_oi: np.ndarray,       # last 52 weeks of total_oi
) -> SentimentScore:
    """Compute normalized COT sentiment score.

    Primary signal: managed money (speculators) net positioning
    Confirmation: producer/commercial positioning (contrarian)
    Confidence: OI magnitude + positioning concentration
    """
    # Percentile rank of current speculator positioning
    if len(history_managed) < 4:
        return SentimentScore(score=0.0, confidence=0.0, components={})

    pct_rank = percentileofscore(history_managed, managed_money_net) / 100.0
    # Normalize to [-1, +1]
    score = 2.0 * pct_rank - 1.0

    # Confidence from OI magnitude (higher OI = more conviction)
    oi_rank = percentileofscore(history_oi, total_oi) / 100.0

    # Confidence from positioning concentration
    if total_oi > 0:
        concentration = abs(managed_money_net) / total_oi
    else:
        concentration = 0.0

    confidence = 0.6 * oi_rank + 0.4 * min(concentration * 5.0, 1.0)
    confidence = np.clip(confidence, 0.0, 1.0)

    return SentimentScore(
        score=float(np.clip(score, -1.0, 1.0)),
        confidence=float(confidence),
        components={
            "managed_money_pct_rank": pct_rank,
            "oi_rank": oi_rank,
            "concentration": concentration,
        },
    )
```

### LightGBM Baseline Model
```python
import lightgbm as lgb
import numpy as np

class BaselineModel:
    """LightGBM binary classifier for directional prediction.

    Conservative defaults -- NO hyperparameter tuning in Phase 2.
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None):
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = lgb.LGBMClassifier(**merged)
        self._is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on feature matrix X and binary labels y."""
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of positive class (price up)."""
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        """Return feature importances for diagnostics."""
        return dict(zip(
            self.model.feature_name_,
            self.model.feature_importances_,
        ))
```

### Fractional Kelly Position Sizing
```python
def fractional_kelly(
    win_prob: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,  # half-Kelly default
    max_position_pct: float = 0.10,  # max 10% of capital
) -> float:
    """Compute fractional Kelly position size as fraction of capital.

    f* = (p * b - q) / b where b = avg_win / avg_loss
    Returns fraction * f*, capped at max_position_pct.
    """
    if avg_loss <= 0 or win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = avg_win / avg_loss  # win/loss ratio
    q = 1.0 - win_prob
    kelly = (win_prob * b - q) / b

    # Negative Kelly means expected value is negative -- don't trade
    if kelly <= 0:
        return 0.0

    sized = fraction * kelly
    return min(sized, max_position_pct)

def volume_capped_position(
    kelly_pct: float,
    capital: float,
    contract_value: float,
    avg_daily_volume: float,
    max_volume_pct: float = 0.02,  # max 2% of ADV
) -> int:
    """Convert Kelly fraction to integer contracts, capped by volume.

    Returns number of contracts to trade (integer, rounded down).
    """
    kelly_contracts = int(kelly_pct * capital / contract_value)
    volume_cap = int(max_volume_pct * avg_daily_volume)
    return max(0, min(kelly_contracts, volume_cap))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed slippage (1 tick) | Volume-adaptive square-root impact | Standard since Almgren & Chriss (2001), widely adopted by 2015 | Prevents overly optimistic backtests in thin markets |
| Random K-fold CV for time series | Purged walk-forward with embargo | Lopez de Prado (2018), now standard in financial ML | Eliminates temporal information leakage |
| Manual COT analysis | Normalized COT index with statistical ranking | Industry standard practice | Systematic, comparable across markets and time |
| Full Kelly position sizing | Fractional Kelly (half or quarter) | Thorp (2006), widely adopted | Dramatically smoother equity curve, lower drawdowns |
| Single backtest window | Multiple walk-forward folds with statistical testing | Standard practice since 2020+ | Detects overfitting, provides confidence intervals |

**Deprecated/outdated:**
- **Fixed slippage models:** Do not use for thin markets. Volume-adaptive is mandatory per MODL-05.
- **Standard K-fold cross-validation:** Never appropriate for financial time series. Use temporal splits only.
- **Full Kelly position sizing:** Too aggressive for practical use. Always use fractional Kelly.

## Open Questions

1. **Prediction horizon for binary target**
   - What we know: COT data updates weekly. Options data can be daily. The divergence signal likely operates on a weekly-to-monthly timescale.
   - What's unclear: Optimal prediction horizon (5-day? 10-day? 20-day?) for the binary up/down classification target.
   - Recommendation: Start with 5 trading days (1 week) to match COT frequency. Test 10 and 20 as robustness checks. The planner should make this configurable.

2. **COT lookback window for percentile ranking**
   - What we know: 52-week (1 year) is the industry standard. Some practitioners use 26 weeks or 3 years.
   - What's unclear: Whether thin commodity markets have structural breaks that make longer lookbacks unreliable.
   - Recommendation: Default to 52 weeks. Make configurable. The walk-forward backtest will reveal if the lookback matters.

3. **Divergence magnitude scaling**
   - What we know: The divergence between implied_mean and sentiment has arbitrary units. Z-scoring against recent history normalizes it.
   - What's unclear: What lookback to use for z-scoring. How to handle the early period when history is insufficient.
   - Recommendation: Use expanding-window z-score (all available history up to that point). Minimum 26 weeks of history before generating signals.

4. **Impact coefficient calibration for slippage**
   - What we know: The square-root impact model uses a coefficient k that varies by market. Published values range from 0.05 (liquid equities) to 0.3+ (illiquid markets).
   - What's unclear: The correct k for lean hogs and oats futures specifically.
   - Recommendation: Start with k=0.1 (conservative). Run robustness check at k=0.2 (very conservative). If signal survives k=0.2, it's robust. Calibrate with real fill data in Phase 5.

5. **Minimum data requirement for meaningful backtest**
   - What we know: With weekly COT signals and 5-day prediction horizon, one year gives ~52 independent predictions. Walk-forward with 5 folds needs at least 3-4 years of data.
   - What's unclear: How much historical COT + options chain data is actually available for the target market.
   - Recommendation: The planner should include a data availability check as the first task. If < 3 years of overlapping COT + options data exist, the backtest will be statistically weak and the validation gate result should be interpreted cautiously.

## Sources

### Primary (HIGH confidence)
- LightGBM 4.6.0 official documentation -- Parameters, Python API, installation. Verified: `objective='binary'`, `metric='binary_logloss'`, native NaN handling, `LGBMClassifier` API.
- scikit-learn 1.8.0 documentation -- `TimeSeriesSplit` with `gap` parameter for embargo-like behavior.
- SciPy documentation -- `scipy.stats.percentileofscore` for COT index calculation.
- CFTC official COT reports page -- Disaggregated report structure, Tuesday as-of / Friday release timing.
- Existing Phase 1 codebase (direct inspection) -- `cot.py`, `density.py`, `moments.py`, `greeks.py`, `feature_store.py`.

### Secondary (MEDIUM confidence)
- Lopez de Prado, "Advances in Financial Machine Learning" (2018) -- Purged cross-validation, embargo gaps, walk-forward methodology. Widely cited, consistent across multiple sources.
- Almgren & Chriss (2001) square-root impact model -- Standard market microstructure model for slippage estimation. Verified across QuantStart, IBKR Campus, and academic literature.
- Kelly criterion / fractional Kelly -- Wikipedia, QuantInsti, Frontiers in Applied Mathematics. Consistent across all sources.
- COT index percentile ranking methodology -- InsiderWeek, Tradingster, MacroMicro, Forecaster.biz. Standard industry practice confirmed across multiple platforms.

### Tertiary (LOW confidence)
- Optimal impact coefficient k for thin commodity futures -- No published calibration found for oats/lean hogs specifically. The k=0.1 default is a conservative estimate based on general illiquid market literature. Needs validation with real fill data.
- Optimal prediction horizon for COT-derived signals -- No rigorous study found specifically for thin commodity markets. 5-day horizon is a reasonable starting point based on COT update frequency.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - LightGBM, scikit-learn, pandas are well-documented and verified via official docs
- Architecture: HIGH - Module structure follows Phase 1 patterns, clean dependency chain from existing code
- Signal construction (SGNL-01/02/03): HIGH - COT index is standard industry practice; divergence taxonomy is explicitly defined in PRD
- Model + Backtest (MODL-01/02): HIGH - LightGBM binary classification and walk-forward are well-documented
- Position sizing (MODL-03): MEDIUM - Fractional Kelly is standard but volume cap calibration is market-specific
- Circuit breakers (MODL-04): HIGH - Straightforward threshold-based state machine
- Slippage model (MODL-05): MEDIUM - Square-root impact model is standard but coefficient calibration for thin markets is uncertain
- Pitfalls: HIGH - Well-documented in financial ML literature

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (30 days -- stable domain, no fast-moving dependencies)
