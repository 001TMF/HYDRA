# Phase 1: Data Infrastructure + Options Math Engine - Research

**Researched:** 2026-02-19
**Domain:** Market data ingestion, feature store design, options mathematics (Breeden-Litzenberger, SVI, Greeks flows)
**Confidence:** MEDIUM-HIGH

## Summary

Phase 1 lays the foundation for the entire HYDRA system: raw market data flows in, gets persisted immutably, and an options math engine extracts stable implied distributions, moments, volatility surfaces, and Greeks flows from thin-market options chains. This phase has zero external dependencies and must deliver a point-in-time correct feature store that prevents lookahead bias from contaminating every downstream experiment.

The primary technical risk is that thin-market options chains (5-15 liquid strikes, 10-30% bid-ask spreads) may not produce stable Breeden-Litzenberger implied densities. The SVI parametric surface fitting approach is the established solution for sparse data, and the graceful degradation to ATM implied vol when fewer than 8 liquid strikes are available is a critical safety valve. The data vendor decision (Databento vs. CME DataMine vs. IB historical) has significant implications for data quality and cost; Databento is the recommended primary vendor based on API quality, Python client support, and CME options coverage, with IB historical as a free supplement.

**Primary recommendation:** Use Databento ($179/month Standard plan) for CME futures and options chain data, the `cot_reports` Python library for free CFTC COT data, Parquet with hive-style partitioning for the raw data lake, SQLite with `as_of` timestamp semantics for the Phase 1 feature store (designed for TimescaleDB migration), and NumPy/SciPy for all options math with QuantLib as an optional enhancement for SABR surface fitting.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | System ingests futures OHLCV price bars (EOD + intraday) for the target market from a data vendor | Databento `GLBX.MDP3` dataset with `ohlcv-1d` and `ohlcv-1m` schemas; parent symbology (`HE.FUT`, `ZO.FUT`) fetches all expirations; `get_cost` estimates cost before download |
| DATA-02 | System ingests full options chain data (all strikes, bids, asks, OI, volume, expiry) for the target market | Databento parent symbology (`HE.OPT`, `ZO.OPT`) fetches all options for a root; `mbp-1` schema provides bid/ask; `definition` schema provides strike/expiry metadata; OI available via `statistics` schema |
| DATA-03 | System ingests CFTC COT reports with correct as-of/release date handling | `cot_reports` Python library (`cot_year()`, `cot_all()`) downloads Disaggregated Futures-Only reports; as-of date = Tuesday (collection), availability date = Friday (release); re-download last 4 weeks to catch revisions |
| DATA-04 | Feature store provides point-in-time correct queries preventing lookahead bias | SQLite schema with `(market, feature_name, as_of_timestamp, available_at_timestamp)` keying; queries filter `WHERE available_at <= :query_time`; COT available_at = Friday, not Tuesday |
| DATA-05 | Raw data persisted in Parquet format with append-only semantics | pyarrow `write_to_dataset` with hive partitioning (`market=X/year=Y/month=Z/day=D/`), `existing_data_behavior='overwrite_or_ignore'` + unique `basename_template` for append workflow |
| DATA-06 | Data quality monitoring detects staleness, missing strikes, and anomalous values | Per-source staleness thresholds (futures: 1 trading day, options: 1 trading day, COT: 1 week), `ffill(limit=N)` with staleness limit, strike count validation, no-arbitrage checks on call prices |
| OPTS-01 | Breeden-Litzenberger extracts risk-neutral implied probability distribution from options chain | SVI-smoothed vol surface -> call prices via Black-76 -> numerical second derivative via central finite differences -> density; integrate to ~1.0, clip negatives to zero |
| OPTS-02 | Implied moments computed from distribution (mean, variance, skew, kurtosis) | Numerical integration of density * K^n over strike range; `scipy.integrate.trapezoid` on the extracted density; four moments directly from the B-L density |
| OPTS-03 | Volatility surface constructed across strike x expiry grid with SVI smoothing | SVI 5-parameter fit per expiry slice via `scipy.optimize.minimize`; no-arbitrage constraints (convexity check); QuantLib `SabrSmileSection` as optional alternative for SABR calibration |
| OPTS-04 | Greeks flow aggregation (GEX, vanna, charm) from options chain OI and volume | Per-strike: gamma/vanna/charm via Black-76 formulas; aggregate = sum(greek_i * OI_i * contract_multiplier * spot); net dealer exposure assumes dealer is short options to customers |
| OPTS-05 | Options math gracefully degrades when < 8 liquid strikes | Liquid strike = OI > 50 AND bid-ask spread < 20% of mid; if count < 8, skip B-L/surface/moments, emit ATM implied vol only + `data_quality: "degraded"` flag |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.11 | Runtime | Best ML library compatibility; QuantLib SWIG wheels available; 3.12 risky for SWIG bindings |
| uv | latest | Package management | 10-100x faster than pip; reliable lockfiles; replaces pip+virtualenv |
| NumPy | >=1.26 | Numerical backbone | float64 arithmetic for all options math; finite differences; array operations |
| SciPy | >=1.12 | Optimization, interpolation, integration | `optimize.minimize` for SVI calibration; `interpolate.CubicSpline` for strike smoothing; `integrate.trapezoid` for moments; `stats` for distribution tests |
| pyarrow | >=15.0 | Parquet I/O, raw data lake | `write_to_dataset` with hive partitioning; columnar compression; the standard for financial data lakes |
| structlog | >=24.1 | Structured logging | JSON-structured logs with context binding; processor pipeline architecture; separate dev (pretty) vs. production (JSON) output |
| APScheduler | >=3.10 | Ingestion scheduling | In-process cron-like scheduling; triggers daily ingestion after market close; lightweight alternative to Celery/Airflow |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| QuantLib | >=1.36 | SABR surface fitting (optional) | Only if SVI via SciPy proves insufficient for surface construction; `SabrSmileSection`, `sabrVolatility` functions; install from PyPI wheels, do NOT build from source |
| databento | latest | Databento Python client | Primary data vendor; `Historical.timeseries.get_range()` for futures/options; `metadata.get_cost()` for cost estimation |
| cot_reports | latest | CFTC COT data download | `cot_year(year, 'disaggregated_fut')` for Disaggregated Futures-Only reports; free, MIT license |
| DuckDB | >=0.10 | Ad-hoc Parquet analytics | Query raw Parquet files directly with SQL; useful for data exploration and validation; not on hot path |
| Polars | >=0.20 | Data transforms | Faster than Pandas for ETL; use for data cleaning and feature computation pipelines; NOT on hot path |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Databento ($179/mo) | CME DataMine | CME DataMine is authoritative source but API is clunkier (FTP-based downloads, less Python-friendly); Databento wraps CME data with modern REST API + Python client |
| Databento ($179/mo) | IB Historical Data (free with account) | IB data is free but limited: 15-min delayed options, no bulk historical download, rate-limited to 50 msg/sec, data gaps in thin markets; use as supplement, not primary |
| SVI (NumPy/SciPy) | QuantLib SABR | QuantLib SABR is more powerful but SWIG bindings are clunky and error-prone; SVI is simpler (5 params vs. 4 for SABR), well-suited to sparse data, and easier to debug |
| SQLite (Phase 1 feature store) | TimescaleDB | TimescaleDB is the target for Phase 3+; SQLite is zero-setup for Phase 1 development and forces good schema design; migrate when concurrent access is needed |
| cot_reports library | Custom CFTC scraper | Library handles URL construction, format parsing, and historical archive; only write custom code if library breaks |
| pyarrow write_to_dataset | DeltaLake / Iceberg | ACID transactions and schema evolution are overkill for single-writer append-only; pyarrow is simpler and sufficient |

**Installation:**

```bash
# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Core dependencies
uv pip install numpy scipy pyarrow structlog apscheduler

# Data sources
uv pip install databento cot-reports

# Optional (add when needed)
uv pip install QuantLib duckdb polars

# Dev dependencies
uv pip install pytest pytest-asyncio ruff mypy
```

## Architecture Patterns

### Recommended Project Structure (Phase 1 scope only)

```
src/hydra/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base.py           # Abstract IngestPipeline: fetch() -> validate() -> store()
│   │   ├── futures.py        # Databento futures OHLCV ingestion
│   │   ├── options.py        # Databento options chain ingestion
│   │   └── cot.py            # CFTC COT data ingestion
│   ├── store/
│   │   ├── __init__.py
│   │   ├── parquet_lake.py   # Parquet write/read with hive partitioning
│   │   └── feature_store.py  # SQLite feature store with as_of semantics
│   └── quality.py            # Data quality monitoring (staleness, validators)
├── signals/
│   └── options_math/
│       ├── __init__.py
│       ├── surface.py        # SVI vol surface construction
│       ├── density.py        # Breeden-Litzenberger density extraction
│       ├── moments.py        # Implied moments from density
│       └── greeks.py         # GEX, vanna, charm flow aggregation
└── config/
    ├── default.yaml          # All configurable thresholds
    └── markets/
        └── lean_hogs.yaml    # Market-specific: strike spacing, contract multiplier, etc.
```

### Pattern 1: Abstract Ingestion Pipeline

**What:** Every data source implements a common interface: `fetch() -> validate() -> persist()`. Each pipeline is a separate module. The abstract base class enforces the contract.

**When to use:** Every data source (futures, options, COT).

**Example:**

```python
# src/hydra/data/ingestion/base.py
from abc import ABC, abstractmethod
from datetime import datetime
import structlog

logger = structlog.get_logger()

class IngestPipeline(ABC):
    """Abstract base for all data ingestion pipelines."""

    @abstractmethod
    def fetch(self, market: str, date: datetime) -> dict:
        """Fetch raw data from external source. Returns raw payload."""
        ...

    @abstractmethod
    def validate(self, raw_data: dict) -> tuple[dict, list[str]]:
        """Validate raw data. Returns (cleaned_data, list_of_warnings)."""
        ...

    @abstractmethod
    def persist(self, data: dict, market: str, date: datetime) -> None:
        """Write validated data to Parquet lake and feature store."""
        ...

    def run(self, market: str, date: datetime) -> bool:
        """Execute full pipeline: fetch -> validate -> persist."""
        log = logger.bind(pipeline=self.__class__.__name__, market=market, date=str(date))
        try:
            raw = self.fetch(market, date)
            cleaned, warnings = self.validate(raw)
            for w in warnings:
                log.warning("validation_warning", detail=w)
            self.persist(cleaned, market, date)
            log.info("ingestion_complete", record_count=len(cleaned.get("records", [])))
            return True
        except Exception as e:
            log.error("ingestion_failed", error=str(e), exc_info=True)
            return False
```

### Pattern 2: Point-in-Time Feature Store with as_of Semantics

**What:** Every feature row has two timestamps: `as_of` (when the data represents) and `available_at` (when it became available to the system). Queries always filter by `available_at <= query_time`.

**When to use:** Every feature read operation. This is the foundation that prevents lookahead bias.

**Example:**

```python
# src/hydra/data/store/feature_store.py
import sqlite3
from datetime import datetime

class FeatureStore:
    """Point-in-time correct feature store.

    Schema: (market TEXT, feature_name TEXT, as_of TEXT, available_at TEXT, value REAL)
    - as_of: the date the data represents (e.g., Tuesday for COT)
    - available_at: when the data became queryable (e.g., Friday for COT)
    """

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                market TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                as_of TEXT NOT NULL,
                available_at TEXT NOT NULL,
                value REAL,
                quality TEXT DEFAULT 'normal',
                PRIMARY KEY (market, feature_name, as_of)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_pit
            ON features (market, feature_name, available_at)
        """)
        self.conn.commit()

    def write_feature(self, market: str, feature_name: str,
                      as_of: datetime, available_at: datetime,
                      value: float, quality: str = "normal"):
        self.conn.execute("""
            INSERT OR REPLACE INTO features
            (market, feature_name, as_of, available_at, value, quality)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (market, feature_name, as_of.isoformat(), available_at.isoformat(),
              value, quality))
        self.conn.commit()

    def get_features_at(self, market: str, query_time: datetime) -> dict[str, float]:
        """Get the latest available value for each feature as of query_time.
        This is the point-in-time correct query that prevents lookahead."""
        rows = self.conn.execute("""
            SELECT feature_name, value, quality FROM features
            WHERE market = ?
              AND available_at <= ?
              AND as_of = (
                  SELECT MAX(f2.as_of) FROM features f2
                  WHERE f2.market = features.market
                    AND f2.feature_name = features.feature_name
                    AND f2.available_at <= ?
              )
        """, (market, query_time.isoformat(), query_time.isoformat())).fetchall()
        return {name: val for name, val, qual in rows}
```

### Pattern 3: Parquet Lake with Append-Only Semantics

**What:** Raw data is written to a hive-partitioned Parquet dataset. Each ingestion run creates new files with unique names. Old files are never modified or deleted.

**When to use:** All raw data persistence (futures bars, options chains, COT reports).

**Example:**

```python
# src/hydra/data/store/parquet_lake.py
import pyarrow as pa
import pyarrow.dataset as ds
from datetime import datetime
from pathlib import Path
import uuid

class ParquetLake:
    """Append-only Parquet data lake with hive partitioning."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def write(self, table: pa.Table, data_type: str, market: str,
              date: datetime) -> Path:
        """Write a batch to the lake. Each write creates a unique file."""
        # Hive partitioning: data_type=futures/market=HE/year=2026/month=02/
        partition_cols = ["data_type", "market", "year", "month"]

        # Add partition columns to the table
        table = table.append_column("data_type", pa.array([data_type] * len(table)))
        table = table.append_column("market", pa.array([market] * len(table)))
        table = table.append_column("year", pa.array([str(date.year)] * len(table)))
        table = table.append_column("month", pa.array([f"{date.month:02d}"] * len(table)))

        # Unique basename prevents overwriting
        basename = f"batch_{date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}.parquet"

        ds.write_dataset(
            table,
            base_dir=str(self.base_path),
            format="parquet",
            partitioning=partition_cols,
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
            basename_template=basename,
        )
        return self.base_path

    def read(self, data_type: str, market: str,
             start: datetime = None, end: datetime = None) -> pa.Table:
        """Read from the lake with optional date filtering."""
        dataset = ds.dataset(
            str(self.base_path),
            format="parquet",
            partitioning="hive",
        )
        filters = [
            ("data_type", "=", data_type),
            ("market", "=", market),
        ]
        return dataset.to_table(filter=ds.field("data_type") == data_type)
```

### Pattern 4: Options Math with Graceful Degradation

**What:** Every options math function accepts a quality context and returns results tagged with a quality level. When data is insufficient, functions degrade to simpler computations rather than producing garbage.

**When to use:** Every function in `signals/options_math/`.

**Example:**

```python
# src/hydra/signals/options_math/density.py
import numpy as np
from dataclasses import dataclass
from enum import Enum

class DataQuality(Enum):
    FULL = "full"         # >= 8 liquid strikes, spread < 20%
    DEGRADED = "degraded" # < 8 liquid strikes, ATM vol only
    STALE = "stale"       # data older than staleness threshold
    MISSING = "missing"   # no data available

@dataclass
class ImpliedDensityResult:
    strikes: np.ndarray       # strike prices
    density: np.ndarray       # probability density
    quality: DataQuality
    liquid_strike_count: int
    atm_iv: float             # always available as fallback
    warnings: list[str]

def extract_density(strikes: np.ndarray, call_prices: np.ndarray,
                    oi: np.ndarray, bid_ask_spread_pct: np.ndarray,
                    spot: float, r: float, T: float,
                    min_liquid_strikes: int = 8,
                    max_spread_pct: float = 0.20,
                    min_oi: int = 50) -> ImpliedDensityResult:
    """Extract risk-neutral density via Breeden-Litzenberger.

    Gracefully degrades when data quality is insufficient.
    """
    # Step 1: Filter to liquid strikes
    liquid_mask = (oi >= min_oi) & (bid_ask_spread_pct <= max_spread_pct)
    liquid_count = np.sum(liquid_mask)

    # Always compute ATM IV as fallback
    atm_idx = np.argmin(np.abs(strikes - spot))
    atm_iv = _implied_vol_from_price(call_prices[atm_idx], spot, strikes[atm_idx], r, T)

    if liquid_count < min_liquid_strikes:
        return ImpliedDensityResult(
            strikes=np.array([spot]),
            density=np.array([1.0]),
            quality=DataQuality.DEGRADED,
            liquid_strike_count=int(liquid_count),
            atm_iv=atm_iv,
            warnings=[f"Only {liquid_count} liquid strikes (need {min_liquid_strikes}); "
                      "falling back to ATM implied vol"]
        )

    # Step 2: Use only liquid strikes
    K = strikes[liquid_mask]
    C = call_prices[liquid_mask]

    # Step 3: Smooth call prices via SVI surface (see surface.py)
    # or cubic spline as minimum
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(K, C)

    # Step 4: Second derivative via central finite differences
    dK = np.diff(K).mean() * 0.5  # half the average strike spacing
    K_fine = np.linspace(K.min(), K.max(), 200)
    C_fine = cs(K_fine)
    d2C = np.gradient(np.gradient(C_fine, K_fine), K_fine)

    # Step 5: Density = e^(rT) * d2C/dK2
    density = np.exp(r * T) * d2C

    # Step 6: Clip negative values, normalize
    warnings = []
    neg_count = np.sum(density < 0)
    if neg_count > 0:
        warnings.append(f"Clipped {neg_count} negative density values to zero")
        density = np.maximum(density, 0.0)

    integral = np.trapezoid(density, K_fine)
    if abs(integral - 1.0) > 0.10:
        warnings.append(f"Density integral = {integral:.4f} (expected ~1.0)")
    if integral > 0:
        density = density / integral  # normalize

    return ImpliedDensityResult(
        strikes=K_fine,
        density=density,
        quality=DataQuality.FULL,
        liquid_strike_count=int(liquid_count),
        atm_iv=atm_iv,
        warnings=warnings
    )
```

### Anti-Patterns to Avoid

- **Using mid-prices without spread filtering:** In thin markets, mid-price can be 15%+ away from "true" value. Always filter by spread < 20% of mid, or use OI-weighted price estimates.
- **Computing B-L density directly from raw option prices:** Raw prices in thin markets are too noisy. Always smooth the vol surface first (SVI or cubic spline on implied vols), then compute call prices from the smooth surface, then differentiate.
- **Forward-filling COT data without staleness tracking:** COT data is weekly. Using `ffill()` without a limit makes stale data look fresh. Always use `ffill(limit=N)` and include a `staleness_days` feature.
- **Storing timestamps without timezone:** All timestamps must be UTC from ingestion. Convert to exchange time (CT for CME) only for display.
- **Using Pandas on the hot path:** NumPy arrays for all computation in `options_math/`. Polars only for ETL. Pandas only in notebooks.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Options chain download from CME | Custom HTTP scraper for CME | Databento Python client | Exchange symbology, contract specifications, and data format handling is complex; Databento normalizes this |
| COT report parsing | Custom CFTC FTP scraper | `cot_reports` library | Handles URL construction, format parsing, historical archive; MIT license; well-maintained |
| Implied volatility inversion | Newton-Raphson solver from scratch | `scipy.optimize.brentq` with Black-76 formula | Convergence edge cases (deep ITM/OTM) are tricky; Brent's method is robust |
| Parquet partitioning logic | Custom directory management | pyarrow `write_to_dataset` with hive partitioning | Partition discovery, schema evolution, and compression handled by pyarrow |
| JSON structured logging | Custom log formatter | structlog with JSONRenderer | Processor pipeline, context binding, dev/prod config switching are production-hardened |
| Cron-like scheduling | Custom sleep loops or crontab | APScheduler | Handles missed jobs, timezone-aware scheduling, persistent job stores |

**Key insight:** Phase 1 is about getting data flowing and math working, not about building infrastructure from scratch. Every hour spent on custom infrastructure is an hour not spent validating whether B-L produces stable distributions from your target market's options chains.

## Common Pitfalls

### Pitfall 1: Breeden-Litzenberger Numerical Instability in Thin Markets

**What goes wrong:** The second derivative of call prices across the strike ladder produces wildly unstable implied distributions -- negative densities, spurious modes, or chaotically jumping moments.

**Why it happens:** Thin markets have 5-15 liquid strikes vs. 50+ in SPX. Bid-ask spreads of 10-30% introduce massive noise. Finite difference second derivatives amplify noise quadratically. A 1% error in option price becomes 100%+ error in the second derivative.

**How to avoid:**
1. Pre-smooth the volatility surface with SVI parameterization (5 parameters, works well with sparse data)
2. Compute call prices from the smooth SVI surface, not raw market prices
3. Apply the second derivative to the smooth call price curve
4. Require minimum 8 liquid strikes (OI > 50, spread < 20% of mid)
5. Clip negative density values to zero
6. Validate: density integrates to ~1.0 (+/- 0.05)
7. Validate: implied mean is within 2% of futures price for short-dated options
8. Fall back to ATM implied vol when quality thresholds are not met

**Warning signs:** Implied kurtosis jumping 50%+ day-to-day with stable underlying price. Negative density regions. GEX flipping sign without corresponding OI changes.

### Pitfall 2: Lookahead Bias via COT Timing

**What goes wrong:** COT data collected Tuesday is used as if available Tuesday, but it is not released until Friday at 3:30pm ET. This gives 3 days of lookahead in every backtest.

**Why it happens:** The data itself is timestamped Tuesday. Without explicit release-date tracking, it is natural to make it available immediately.

**How to avoid:**
- Feature store has separate `as_of` (Tuesday) and `available_at` (Friday 3:30pm ET) timestamps
- All queries filter by `available_at`, never `as_of` alone
- Write a specific test: "COT data for any Tuesday T is NOT returned by `get_features_at(T + 2 days)`"
- Re-download last 4 weeks each cycle to catch CFTC revisions

**Warning signs:** Backtest Sharpe > 3.0 (almost certainly leakage in thin markets). Performance cliff between backtest and paper trading.

### Pitfall 3: Data Pipeline Silent Failures

**What goes wrong:** A data source stops updating, changes format, or degrades quality, but the system continues on stale data.

**Why it happens:** Data sources for thin markets are less reliable. Free sources (CFTC) have no SLA. `try/except: pass` patterns silently swallow errors. Feature stores with unlimited `ffill` make stale data look fresh.

**How to avoid:**
- Staleness thresholds per source: futures 1 trading day, options 1 trading day, COT 1 week
- `ffill(limit=N)` with configurable N, after which value becomes NaN
- Daily data quality log: completeness, freshness, record count
- Source-specific validators: call prices monotonically decreasing with strike, COT positions sum correctly
- Weekend/holiday-aware staleness: do not alert when markets are closed

**Warning signs:** Feature store showing identical values on consecutive trading days. Pipeline logs showing no errors but also no new records.

### Pitfall 4: QuantLib SWIG Binding Complexity

**What goes wrong:** QuantLib's Python bindings are auto-generated from C++ via SWIG. The API is not Pythonic, errors are cryptic (segfaults, `RuntimeError: unable to convert`), and memory management has gotchas.

**Why it happens:** SWIG wraps C++ classes directly. Python developers must understand the C++ API. Version mismatches between QuantLib C++ and SWIG bindings cause silent failures.

**How to avoid:**
- Start with pure NumPy/SciPy for ALL options math (B-L, moments, Greeks)
- Add QuantLib only for SABR surface fitting if SVI proves insufficient
- Install from PyPI wheels (`pip install QuantLib`), never build from source
- Pin exact version (>=1.36)
- Wrap all QuantLib calls in a Pythonic interface; never expose QuantLib types outside `options_math/`
- Write tests against known-good Black-Scholes values for verification

**Warning signs:** Segfaults, memory leaks, mysterious `RuntimeError` messages.

### Pitfall 5: Timezone Chaos

**What goes wrong:** Market data in exchange time (CT for CME), API timestamps in UTC, local machine in a third timezone. Feature computation produces wrong results silently.

**Why it happens:** CME uses CT (Central Time), Databento returns UTC, CFTC reports use ET, and local dev machines vary.

**How to avoid:**
- Normalize everything to UTC immediately at ingestion. Store all timestamps as UTC in both Parquet and feature store.
- Convert to exchange time only for display or market-hours logic.
- Use `datetime.timezone.utc` everywhere, never naive datetimes.

## Code Examples

### Databento Futures OHLCV Ingestion

```python
# Source: Databento official docs + API reference
import databento as db
from datetime import date

client = db.Historical("YOUR_API_KEY")

# Estimate cost before downloading
cost = client.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["HE.FUT"],          # Parent symbol: all Lean Hog futures
    stype_in="parent",
    schema="ohlcv-1d",           # Daily OHLCV bars
    start="2024-01-01",
    end="2024-12-31",
)
print(f"Estimated cost: ${cost:.4f}")

# Download if cost is acceptable
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["HE.FUT"],
    stype_in="parent",
    schema="ohlcv-1d",
    start="2024-01-01",
    end="2024-12-31",
)

# Convert to DataFrame or numpy
df = data.to_df()
```

### Databento Options Chain Ingestion

```python
# Source: Databento options on futures examples
import databento as db

client = db.Historical("YOUR_API_KEY")

# Fetch all options for Lean Hogs
data = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["HE.OPT"],          # Parent: all HE options (all strikes, expiries)
    stype_in="parent",
    schema="mbp-1",              # Top-of-book: bid, ask, bid_size, ask_size
    start="2024-06-01",
    end="2024-06-01T23:59:59",
)

# Also fetch definitions for strike/expiry metadata
definitions = client.timeseries.get_range(
    dataset="GLBX.MDP3",
    symbols=["HE.OPT"],
    stype_in="parent",
    schema="definition",
    start="2024-06-01",
    end="2024-06-01T23:59:59",
)
```

### CFTC COT Data Ingestion

```python
# Source: cot_reports library (GitHub: NDelventhal/cot_reports)
import cot_reports as cot
from datetime import datetime, timedelta

# Download Disaggregated Futures-Only for a specific year
df = cot.cot_year(year=2025, cot_report_type='disaggregated_fut')

# Filter to target market (e.g., Lean Hogs = CFTC code 054642)
hogs_df = df[df['CFTC_Contract_Market_Code'] == '054642']

# Key columns for sentiment:
# - Prod_Merc_Positions_Long_All / Short_All (producers/hedgers)
# - M_Money_Positions_Long_All / Short_All (managed money/speculators)
# - Swap_Positions_Long_All / Short_All (swap dealers)

# CRITICAL: as_of = Tuesday (report_date), available_at = Friday (release)
for _, row in hogs_df.iterrows():
    report_date = row['As_of_Date_In_Form_YYMMDD']  # Tuesday
    # Release date = next Friday after report_date
    release_date = _next_friday(report_date)
    feature_store.write_feature(
        market="HE",
        feature_name="cot_managed_money_net",
        as_of=report_date,
        available_at=release_date,
        value=row['M_Money_Positions_Long_All'] - row['M_Money_Positions_Short_All'],
    )
```

### SVI Volatility Surface Calibration

```python
# Source: Gatheral "The Volatility Surface" (2006); SciPy optimize
import numpy as np
from scipy.optimize import minimize

def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float,
                        m: float, sigma: float) -> np.ndarray:
    """SVI parameterization of total implied variance w(k).

    Parameters:
        k: log-moneyness = ln(K/F) where F = forward price
        a, b, rho, m, sigma: SVI parameters
    Returns:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def calibrate_svi(strikes: np.ndarray, market_ivs: np.ndarray,
                  forward: float, T: float) -> dict:
    """Fit SVI to market implied volatilities for one expiry slice.

    Returns dict with SVI parameters and fit quality metrics.
    """
    k = np.log(strikes / forward)  # log-moneyness
    market_w = market_ivs**2 * T    # total variance

    def objective(params):
        a, b, rho, m, sigma = params
        model_w = svi_total_variance(k, a, b, rho, m, sigma)
        return np.sum((model_w - market_w)**2)

    # Constraints: b >= 0, -1 < rho < 1, sigma > 0
    bounds = [(-1, 1), (0, 5), (-0.99, 0.99), (-2, 2), (0.01, 5)]
    # Initial guess from market data
    x0 = [np.mean(market_w), 0.1, -0.3, 0.0, 0.5]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    a, b, rho, m, sigma = result.x
    fitted_w = svi_total_variance(k, a, b, rho, m, sigma)
    fitted_iv = np.sqrt(fitted_w / T)
    rmse = np.sqrt(np.mean((fitted_iv - market_ivs)**2))

    # Convexity check: d2w/dk2 >= 0 for no butterfly arbitrage
    k_fine = np.linspace(k.min() - 0.5, k.max() + 0.5, 500)
    w_fine = svi_total_variance(k_fine, a, b, rho, m, sigma)
    d2w = np.gradient(np.gradient(w_fine, k_fine), k_fine)
    has_arbitrage = np.any(d2w < -1e-10)

    return {
        "params": {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma},
        "rmse": rmse,
        "has_butterfly_arbitrage": has_arbitrage,
        "fitted_iv": fitted_iv,
    }
```

### Greeks Flow Aggregation (GEX, Vanna, Charm)

```python
# Source: Black-76 Greeks formulas; dealer exposure convention
import numpy as np
from scipy.stats import norm

def black76_greeks(F: float, K: float, r: float, T: float,
                   sigma: float, is_call: bool) -> dict:
    """Compute Greeks for a single option under Black-76 model.

    F: futures/forward price
    K: strike price
    r: risk-free rate
    T: time to expiry (years)
    sigma: implied volatility
    """
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    gamma = discount * norm.pdf(d1) / (F * sigma * np.sqrt(T))
    vanna = -discount * norm.pdf(d1) * d2 / sigma
    charm = -discount * norm.pdf(d1) * (
        2 * r * T - d2 * sigma * np.sqrt(T)
    ) / (2 * T * sigma * np.sqrt(T))

    return {"gamma": gamma, "vanna": vanna, "charm": charm}


def compute_gex_vanna_charm(chain: dict, spot: float, r: float,
                             contract_multiplier: float) -> dict:
    """Aggregate Greeks flows across the full options chain.

    Assumes dealer is net short options (sold to customers).
    GEX = sum(gamma_i * OI_i * contract_multiplier * spot^2 / 100)

    chain: dict with keys strikes, call_ivs, put_ivs, call_oi, put_oi,
           expiries_T (time to expiry in years for each option)
    """
    gex_total = 0.0
    vanna_total = 0.0
    charm_total = 0.0

    for i in range(len(chain["strikes"])):
        K = chain["strikes"][i]
        T = chain["expiries_T"][i]

        if T <= 0 or T > 2.0:  # skip expired or too far out
            continue

        # Call contribution
        if chain["call_ivs"][i] > 0 and chain["call_oi"][i] > 0:
            g = black76_greeks(spot, K, r, T, chain["call_ivs"][i], True)
            sign = 1  # dealer short calls -> positive gamma exposure
            gex_total += sign * g["gamma"] * chain["call_oi"][i] * contract_multiplier * spot**2 / 100
            vanna_total += sign * g["vanna"] * chain["call_oi"][i] * contract_multiplier * spot
            charm_total += sign * g["charm"] * chain["call_oi"][i] * contract_multiplier

        # Put contribution
        if chain["put_ivs"][i] > 0 and chain["put_oi"][i] > 0:
            g = black76_greeks(spot, K, r, T, chain["put_ivs"][i], False)
            sign = -1  # dealer short puts -> negative gamma exposure
            gex_total += sign * g["gamma"] * chain["put_oi"][i] * contract_multiplier * spot**2 / 100
            vanna_total += sign * g["vanna"] * chain["put_oi"][i] * contract_multiplier * spot
            charm_total += sign * g["charm"] * chain["put_oi"][i] * contract_multiplier

    return {"gex": gex_total, "vanna_flow": vanna_total, "charm_flow": charm_total}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed slippage in backtests | Volume-adaptive slippage: `slippage = base + impact * (size/volume)^power` | Standard practice since ~2018 (Carver, Lopez de Prado) | Phase 3 concern but slippage model parameters should be tracked from Phase 1 data |
| Raw B-L on noisy data | SVI-smoothed surface then B-L | Gatheral SVI (2004), widespread adoption 2015+ | Critical for Phase 1 -- direct B-L on thin markets fails |
| QuantLib-only for all options math | QuantLib for surfaces + NumPy/SciPy for B-L and flows | Current best practice | Avoids SWIG binding complexity for simple math |
| Pandas everywhere | NumPy for compute, Polars for ETL, DuckDB for analytics | Polars mainstream 2023+; DuckDB mainstream 2024+ | Pandas still fine for Phase 1 prototyping; switch hot paths later |
| CME DataMine FTP downloads | Databento REST API with Python client | Databento launched CME coverage ~2023 | Much better DX; programmatic cost estimation; normalized schemas |
| QuantLib SWIG 4.1.x | QuantLib SWIG 4.2.x (limited API) | QuantLib 1.36 (Oct 2024) | Fewer wheels needed; broader Python version coverage |

**Deprecated/outdated:**
- Feast for single-developer feature stores: overkill; TimescaleDB or SQLite with proper schema is sufficient
- Pandas for hot-path computation: use NumPy arrays directly
- Building QuantLib from source: install from PyPI wheels

## Open Questions

1. **Target market selection: Lean Hogs vs. Oats vs. Ethanol**
   - What we know: Lean Hogs have ~138K futures OI (moderate liquidity); Oats have ~3.5K futures OI (very thin); CME just launched physically-delivered Ethanol futures/options in Feb 2025 (too new, no historical data)
   - What's unclear: How many liquid option strikes are typically available for each market? This cannot be determined without downloading actual chain data.
   - Recommendation: Start with Lean Hogs (HE) as the primary target -- most liquid of the thin-market candidates. Download a sample of oat options (ZO) to compare. Ethanol is too new. Use Databento's `get_cost` to estimate the data cost for each before committing.
   - **This is a hands-on evaluation that must happen in the first week of Phase 1 implementation.**

2. **Databento Standard plan scope for options data**
   - What we know: Standard plan is $179/month, includes unlimited live data and some free historical data per schema tier
   - What's unclear: Exactly how much historical options chain data is included in the Standard plan vs. pay-as-you-go. The free historical data varies by schema tier.
   - Recommendation: Sign up for $125 free credits, estimate cost with `get_cost()` for 1 year of HE.OPT data, then decide on plan. Historical pay-as-you-go may be cheaper than $179/month if only downloading historical data periodically.

3. **SVI vs. SABR for thin-market surface fitting**
   - What we know: SVI has 5 parameters (a, b, rho, m, sigma) and is designed for implied vol interpolation. SABR has 4 parameters (alpha, beta, rho, nu) and is designed for stochastic vol dynamics. Both work for sparse data.
   - What's unclear: Which produces more stable fits on 8-15 data points from thin-market options? SVI is generally more robust for static snapshots; SABR is better for dynamics.
   - Recommendation: Implement SVI first (simpler, pure SciPy). Add SABR via QuantLib only if SVI fails. Test both on the same thin-market data during Phase 1 validation.

4. **SQLite performance for feature store at daily frequency**
   - What we know: Phase 1 will have ~20-30 features per market per day. At 252 trading days/year * 5 years of history = ~1,260 rows * 30 features = ~37,800 rows. SQLite handles this trivially.
   - What's unclear: When concurrent access becomes necessary (Phase 3+ with agent loop), SQLite's single-writer limitation will be a problem.
   - Recommendation: SQLite is fine for Phase 1. Design the schema to be compatible with TimescaleDB migration. Add a `FeatureStore` abstraction layer so the migration is a backend swap, not a rewrite.

5. **Dealer position assumption for GEX/vanna/charm**
   - What we know: Standard GEX computation assumes dealers are net short options (customers buy, dealers sell). This is well-established for equity options where most volume is retail/institutional buying.
   - What's unclear: In commodity futures options, the customer base includes producers hedging (selling calls, buying puts) which reverses the typical flow. Lean Hog options may have producers as the primary options sellers, not buyers.
   - Recommendation: Compute flows under both assumptions (dealer-short and dealer-long). Use COT data to determine the predominant positioning. This is a research question that Phase 1 data will help answer.

## Sources

### Primary (HIGH confidence)
- [Databento CME Futures Data](https://databento.com/futures) -- Coverage, API, Python client
- [Databento Options Data](https://databento.com/options) -- Options on futures coverage for CME/ICE
- [Databento Pricing](https://databento.com/pricing) -- Standard plan $179/month, $125 free credits
- [Databento Python Client](https://github.com/databento/databento-python) -- Official Python client
- [Databento Historical API: get_cost](https://databento.com/docs/api-reference-historical/metadata/metadata-get-cost) -- Cost estimation before download
- [cot_reports Python Library](https://github.com/NDelventhal/cot_reports) -- CFTC COT data download; supports disaggregated reports
- [CFTC Disaggregated Reports](https://publicreporting.cftc.gov/Commitments-of-Traders/Disaggregated-Futures-Only/72hh-3qpy) -- Official data source
- [Apache Arrow PyArrow Documentation](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html) -- write_dataset, hive partitioning, append behavior
- [structlog Documentation](https://www.structlog.org/en/stable/logging-best-practices.html) -- Logging best practices, JSONRenderer

### Secondary (MEDIUM confidence)
- [Gatheral "The Volatility Surface" (2006)](https://mfe.baruch.cuny.edu/wp-content/uploads/2013/01/OsakaSVI2012.pdf) -- SVI parameterization, arbitrage-free constraints
- [Breeden-Litzenberger GitHub implementation](https://github.com/PavanAnanthSharma/Breeden-Litzenberger-formula-for-risk-neutral-densities) -- Reference Python implementation
- [Finance Halo: Market-Implied PDF from Options](https://blog.financehalo.com/market-implied-pdf-options) -- B-L implementation walkthrough (2025)
- [QuantLib-Python Documentation v1.40](https://quantlib-python-docs.readthedocs.io/en/latest/termstructures/volatility.html) -- Volatility term structures, SABR
- [QuantLib SWIG Releases](https://github.com/lballabio/QuantLib-SWIG/releases) -- v1.36 (Oct 2024), SWIG 4.2.x upgrade
- [GEX/Vanna/Charm Computation (Medium)](https://medium.com/option-screener/so-youve-heard-about-gamma-exposure-gex-but-what-about-vanna-and-charm-exposures-47ed9109d26a) -- Flow exposure formulas and dealer assumption
- [gflows GitHub](https://github.com/aaguiar10/gflows) -- Open-source delta/gamma/vanna/charm exposure viewer
- [CME Lean Hogs](https://www.cmegroup.com/markets/agriculture/livestock/lean-hogs.html) -- Futures OI ~138K
- [CME Oats Volume & OI](https://www.cmegroup.com/markets/agriculture/grains/oats.volume.html) -- Futures OI ~3.5K
- [CME Ethanol Futures Launch (Feb 2025)](https://www.cmegroup.com/media-room/press-releases/2025/2/10/cme_group_announcesfirsttradesofphysically-deliveredethanolfutur.html) -- Too new for historical data

### Tertiary (LOW confidence -- needs validation)
- Databento Standard plan exact historical data inclusion per schema tier -- verify at signup
- QuantLib 1.36 Python 3.11 wheel availability on macOS ARM -- test at `pip install` time
- Lean Hog options typical liquid strike count -- requires downloading actual chain data
- Oat options existence and liquidity -- requires hands-on verification with Databento
- Dealer position assumption correctness for commodity futures options -- requires COT analysis

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified via official docs and PyPI; versions confirmed for 2024-2025
- Architecture: MEDIUM-HIGH -- patterns are established (B-L, SVI, point-in-time stores); specific thin-market adaptations need validation
- Data vendor: MEDIUM -- Databento pricing and API confirmed; exact historical data inclusion and thin-market options completeness need hands-on testing
- Options math: MEDIUM -- theory is well-established (Gatheral, Hull, Natenberg); numerical stability with real thin-market data is the open question that Phase 1 validation gate addresses
- Pitfalls: HIGH -- drawn from established quantitative finance literature and directly applicable

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (stable domain; data vendor pricing may change)
