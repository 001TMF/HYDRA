# Phase 3: Sandbox + Experiment Infrastructure - Research

**Researched:** 2026-02-19
**Domain:** ML experiment infrastructure, model lifecycle management, drift detection, CLI tooling
**Confidence:** HIGH

## Summary

Phase 3 builds the experimentation environment that Phase 4's autonomous agent loop will operate within. It has six major subsystems: (1) a market replay engine that replays historical data with volume-adaptive slippage, (2) an MLflow-backed model registry with champion/candidate/archived lifecycle, (3) an experiment journal stored in SQLite for queryable experiment history, (4) a drift observer that monitors both performance metrics and feature distributions, (5) a composite fitness evaluator that scores candidate models on 6 weighted metrics, and (6) a Typer+Rich CLI for operator control.

The codebase already has the core building blocks: `WalkForwardEngine` runs backtests with slippage, `BaselineModel` wraps LightGBM, `BacktestResult` computes metrics, and `FeatureStore` provides point-in-time queries. Phase 3 wraps these existing components in a replay/registry/observation framework rather than rewriting them.

**Primary recommendation:** Use MLflow (v3.9+) with local file backend for model registry, SQLite for the experiment journal (consistent with existing `FeatureStore` pattern), River library for ADWIN drift detection, hand-roll PSI/KS/CUSUM from NumPy/SciPy (simple formulas, no library needed), and Typer+Rich for CLI.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SBOX-01 | Market replay engine replays historical data with volume-adaptive thin-market slippage model | Extend existing `WalkForwardEngine` + `estimate_slippage` into an event-driven replay engine. Core slippage math already implemented in `src/hydra/risk/slippage.py`. |
| SBOX-02 | Model registry (MLflow) tracks all trained models with full config snapshot, metrics, and champion/candidate/archived lifecycle | MLflow 3.9+ with `mlflow.lightgbm` flavor, aliases API for champion/candidate/archived. Local file backend, no server needed. |
| SBOX-03 | Experiment journal logs every experiment with hypothesis, config diff, results, promotion decision, and tags | SQLite journal table with JSON columns for config_diff and results. Matches existing `FeatureStore` SQLite/WAL pattern. |
| SBOX-04 | Observer detects model drift via rolling performance metrics (Sharpe, drawdown, hit rate, calibration) and feature distribution drift (PSI, KS, ADWIN/CUSUM) | PSI/KS from NumPy/SciPy (trivial formulas), ADWIN from River library, CUSUM hand-rolled (~20 lines). Rolling Sharpe/drawdown/hit rate reuse `evaluation.py` helpers. |
| SBOX-05 | Evaluator scores candidate models on 6-metric composite fitness (Sharpe 0.25, drawdown 0.20, calibration 0.15, robustness 0.15, slippage-adjusted return 0.15, simplicity 0.10) | Weighted composite score from existing `BacktestResult` metrics + new calibration/robustness/simplicity metrics. Pure Python computation. |
| SBOX-06 | Journal is queryable by tag, date range, mutation type, and outcome | SQLite query layer with indexed columns + JSON tag storage. Full-text search optional (SQLite FTS5). |
| CLI-01 | `status` command shows model health, active experiments, alerts, current autonomy level | Typer command + Rich tables/panels pulling from observer + registry + journal. |
| CLI-02 | `diagnose` command forces a diagnostic cycle on the current champion model | Typer command that triggers observer health check + drift detectors on champion model. |
| CLI-03 | `rollback` command reverts to previous champion model | Typer command that swaps MLflow "champion" alias to previous version. |
| CLI-04 | `pause` / `run` commands halt and resume the agent loop | Typer commands setting a state flag (file-based or in-memory) that the Phase 4 agent loop checks. |
| CLI-05 | `journal query` command searches experiment history by tag, date, mutation type | Typer command wrapping SQLite queries with Rich-formatted table output. |
| CLI-06 | CLI uses Rich-formatted terminal output with tables, colored alerts, and progress indicators | Rich Console, Table, Panel, and Progress from the Rich library (v14.2+). |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| MLflow | >=3.9 | Model registry, experiment tracking, artifact storage | Industry standard ML lifecycle platform. Native LightGBM flavor via `mlflow.lightgbm`. Aliases replaced deprecated stages for champion/candidate management. |
| Typer | >=0.22 | CLI framework | FastAPI-style CLI with type hints. Built-in Rich integration. HYDRA tech stack mandates Typer. |
| Rich | >=14.0 | Terminal formatting (tables, panels, colors, progress bars) | HYDRA tech stack mandates Rich. Tables, colored alerts, progress indicators for CLI-06. |
| River | >=0.22 | ADWIN drift detection | Production-quality online ML library. ADWIN with mathematical guarantees for concept drift. Avoids hand-rolling a complex adaptive windowing algorithm. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | (already in deps) | KS test for distribution comparison | `ks_2samp()` for feature distribution drift detection. Already a dependency. |
| numpy | (already in deps) | PSI computation, CUSUM implementation | PSI is a ~15-line formula. CUSUM is ~20 lines. Both trivial with NumPy. |
| sqlite3 | (stdlib) | Experiment journal storage | Consistent with existing FeatureStore pattern. WAL mode. No new dependency. |
| structlog | (already in deps) | Structured logging throughout new modules | Already used across the codebase. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| MLflow local file backend | MLflow with SQLite backend | SQLite backend enables concurrent reads, but file backend is simpler and sufficient for single-operator use. Start with file, switch to SQLite backend if needed. |
| River for ADWIN | Hand-rolled ADWIN | ADWIN's adaptive bucketing is non-trivial (~200 lines). River provides a battle-tested implementation. Worth the dependency. |
| SQLite for journal | MLflow experiment tracking for journal | MLflow tracks runs/metrics well, but the experiment journal needs custom fields (hypothesis, config_diff, promotion_decision, mutation_type). SQLite gives full schema control. Use MLflow for model artifacts, SQLite for journal metadata. |
| Evidently AI for drift | Hand-rolled PSI/KS + River ADWIN | Evidently is a heavyweight monitoring platform. HYDRA only needs 4 drift detectors (PSI, KS, ADWIN, CUSUM). Hand-rolling PSI/KS/CUSUM + River ADWIN is lighter and avoids a large transitive dependency tree. |

**Installation:**
```bash
uv add mlflow "typer[all]>=0.22" "rich>=14.0" "river>=0.22"
```

Note: `typer[all]` includes Rich and Shellingham. Rich is also a standalone dependency for direct Console/Table usage beyond Typer's built-in formatting.

## Architecture Patterns

### Recommended Project Structure

```
src/hydra/
├── sandbox/
│   ├── __init__.py
│   ├── replay.py           # Market replay engine (SBOX-01)
│   ├── registry.py         # MLflow model registry wrapper (SBOX-02)
│   ├── journal.py          # Experiment journal (SBOX-03, SBOX-06)
│   ├── observer.py         # Drift + performance monitoring (SBOX-04)
│   ├── evaluator.py        # Composite fitness scorer (SBOX-05)
│   └── drift/
│       ├── __init__.py
│       ├── psi.py          # Population Stability Index
│       ├── ks.py           # Kolmogorov-Smirnov test wrapper
│       ├── adwin.py        # ADWIN wrapper around River
│       └── cusum.py        # CUSUM change-point detector
├── cli/
│   ├── __init__.py
│   ├── app.py              # Typer app + subcommands (CLI-01 through CLI-06)
│   ├── formatters.py       # Rich table/panel formatters
│   └── state.py            # Agent loop state (pause/run flag)
```

### Pattern 1: Market Replay Engine (Extends WalkForwardEngine)

**What:** An event-driven replay that feeds historical bars through the existing pipeline one step at a time, computing slippage via the existing `estimate_slippage()` function.

**When to use:** Any time a model needs evaluation against historical data with realistic execution simulation.

**Key design decision:** The replay engine is NOT a full rewrite of `WalkForwardEngine`. Instead, it wraps the existing walk-forward logic and adds:
1. Configurable data source (replay from Parquet lake or feature store)
2. Per-bar event emission (for observer to hook into)
3. Slippage parameter variation (thin-market conditions vary over time)
4. Trade execution logging compatible with the experiment journal

```python
# Pattern: Replay engine wrapping existing components
class MarketReplayEngine:
    """Replays historical data with volume-adaptive slippage.

    Unlike WalkForwardEngine (which does train/test splits), the replay
    engine takes a pre-trained model and replays a date range bar-by-bar,
    simulating execution with realistic slippage.
    """

    def __init__(self, config: dict):
        self.slippage_config = config.get("slippage", {})
        self.callbacks: list[Callable] = []  # observer hooks

    def replay(
        self,
        model,  # trained BaselineModel or loaded MLflow model
        features: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        spreads: np.ndarray,
    ) -> ReplayResult:
        """Replay bar-by-bar with volume-adaptive slippage."""
        for i in range(len(features)):
            # Use existing slippage model
            slippage = estimate_slippage(
                order_size=position_size,
                daily_volume=volumes[i],
                spread=spreads[i],
                daily_volatility=self._rolling_vol(prices, i),
            )
            # ... execute trade logic, emit events
            for cb in self.callbacks:
                cb(trade_event)
```

### Pattern 2: MLflow Registry Wrapper (Thin Abstraction)

**What:** A thin wrapper around MLflow's Python client that enforces HYDRA's champion/candidate/archived lifecycle conventions.

**When to use:** Every model training, evaluation, and promotion operation.

**Key design decision:** The wrapper enforces naming conventions and alias semantics, but delegates all storage to MLflow. This keeps the MLflow upgrade path clean.

```python
# Pattern: Thin MLflow wrapper enforcing lifecycle conventions
import mlflow
from mlflow import MlflowClient

class ModelRegistry:
    REGISTERED_MODEL_NAME = "hydra-baseline"
    ALIASES = ("champion", "candidate", "archived")

    def __init__(self, tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def log_candidate(self, model, metrics: dict, config: dict, tags: dict) -> str:
        """Log a new candidate model with full config snapshot."""
        with mlflow.start_run() as run:
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(
                model.model,  # underlying LGBMClassifier
                artifact_path="model",
                registered_model_name=self.REGISTERED_MODEL_NAME,
            )
            for k, v in tags.items():
                mlflow.set_tag(k, v)
            return run.info.run_id

    def promote_to_champion(self, version: int) -> None:
        """Promote a model version to champion, archiving the current one."""
        try:
            current = self.client.get_model_version_by_alias(
                self.REGISTERED_MODEL_NAME, "champion"
            )
            self.client.set_registered_model_alias(
                self.REGISTERED_MODEL_NAME, "archived", int(current.version)
            )
        except Exception:
            pass  # no current champion
        self.client.set_registered_model_alias(
            self.REGISTERED_MODEL_NAME, "champion", version
        )

    def load_champion(self):
        """Load the current champion model."""
        return mlflow.lightgbm.load_model(
            f"models:/{self.REGISTERED_MODEL_NAME}@champion"
        )

    def rollback(self) -> None:
        """Revert champion to the previous (archived) version."""
        archived = self.client.get_model_version_by_alias(
            self.REGISTERED_MODEL_NAME, "archived"
        )
        self.promote_to_champion(int(archived.version))
```

### Pattern 3: SQLite Experiment Journal

**What:** A SQLite-backed journal that records every experiment with structured fields + JSON blobs for flexible querying.

**When to use:** Logging experiments, querying history for the agent's memory.

**Schema design:**

```sql
CREATE TABLE experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,        -- ISO 8601
    hypothesis TEXT NOT NULL,        -- what we're testing
    mutation_type TEXT NOT NULL,     -- e.g., "hyperparameter", "feature_add", "feature_remove"
    config_diff TEXT NOT NULL,       -- JSON: what changed from champion config
    results TEXT NOT NULL,           -- JSON: all metrics from evaluation
    champion_metrics TEXT,           -- JSON: champion metrics at time of experiment
    promotion_decision TEXT NOT NULL, -- "promoted", "rejected", "pending"
    promotion_reason TEXT,           -- why promoted/rejected
    run_id TEXT,                     -- MLflow run ID
    model_version INTEGER,          -- MLflow model version
    tags TEXT DEFAULT '[]'           -- JSON array of string tags
);

CREATE INDEX idx_experiments_created ON experiments(created_at);
CREATE INDEX idx_experiments_mutation ON experiments(mutation_type);
CREATE INDEX idx_experiments_decision ON experiments(promotion_decision);
```

```python
# Pattern: Journal with JSON columns for flexible querying
class ExperimentJournal:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def log_experiment(self, experiment: ExperimentRecord) -> int:
        """Log an experiment, return its ID."""
        ...

    def query(
        self,
        tags: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        mutation_type: str | None = None,
        outcome: str | None = None,
    ) -> list[ExperimentRecord]:
        """Query experiments with filters. All filters are AND-combined."""
        ...
```

### Pattern 4: Composite Observer (Performance + Distribution Drift)

**What:** Monitors both output drift (rolling Sharpe, drawdown, hit rate, calibration) and input drift (PSI, KS on feature distributions, ADWIN/CUSUM on streaming metrics).

**When to use:** Continuously while the model is active. Observer alerts trigger diagnostic cycles.

```python
class DriftObserver:
    """Monitors model drift via performance and feature distribution metrics."""

    def __init__(self, config: dict):
        self.psi_threshold = config.get("psi_threshold", 0.25)
        self.ks_alpha = config.get("ks_alpha", 0.05)
        self.adwin_delta = config.get("adwin_delta", 0.002)
        self.cusum_threshold = config.get("cusum_threshold", 5.0)
        self.adwin_detectors: dict[str, ADWIN] = {}

    def check_performance_drift(
        self, recent_returns: np.ndarray, baseline_returns: np.ndarray
    ) -> PerformanceDriftReport:
        """Compare rolling performance metrics against baseline."""
        ...

    def check_feature_drift(
        self, current_features: np.ndarray, baseline_features: np.ndarray,
        feature_names: list[str],
    ) -> FeatureDriftReport:
        """Check PSI and KS for each feature column."""
        ...

    def update_streaming(self, metric_name: str, value: float) -> bool:
        """Feed a single observation to ADWIN/CUSUM. Returns True if drift detected."""
        ...
```

### Pattern 5: Typer CLI with Rich Formatting

**What:** Typer app with subcommands, each producing Rich-formatted output.

**When to use:** All operator-facing commands.

```python
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(rich_markup_mode="rich")
console = Console()

@app.command()
def status():
    """Show model health, active experiments, alerts, autonomy level."""
    # Pull from registry, observer, journal
    table = Table(title="Model Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    ...
    console.print(table)

@app.command()
def journal(
    tag: str = typer.Option(None, help="Filter by tag"),
    since: str = typer.Option(None, help="Filter by date (YYYY-MM-DD)"),
    mutation: str = typer.Option(None, help="Filter by mutation type"),
):
    """Search experiment history."""
    ...
```

### Anti-Patterns to Avoid

- **Rewriting WalkForwardEngine for replay:** The replay engine should WRAP, not replace, the existing walk-forward infrastructure. The slippage model, position sizing, and circuit breakers already work. Replay adds bar-by-bar simulation, not new math.
- **Using MLflow stages instead of aliases:** Model stages (`Staging`, `Production`, `Archived`) are deprecated since MLflow 2.9. Use aliases (`champion`, `candidate`, `archived`) exclusively.
- **Storing experiment journal in MLflow:** MLflow's run/metric model does not support custom queryable fields like `hypothesis`, `mutation_type`, or `promotion_decision`. Use SQLite for the journal, MLflow for model artifacts. They reference each other via `run_id`.
- **Heavyweight drift monitoring library:** Libraries like Evidently AI, NannyML, or Deepchecks bring massive dependency trees. The 4 drift detectors needed (PSI, KS, ADWIN, CUSUM) are each under 50 lines of code (except ADWIN, which River provides).
- **CLI polling for state:** The `pause`/`run` commands should write a state file or set an in-process flag. The agent loop (Phase 4) checks this flag at the top of each cycle. Do NOT implement a long-polling or socket-based IPC system.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ADWIN adaptive windowing | Custom sliding window with bucket merging | `river.drift.ADWIN` | The algorithm has subtlety in bucket compression and memory efficiency. ~200 lines to implement correctly. River's version has mathematical guarantees and is battle-tested. |
| LightGBM model serialization | Custom pickle/JSON save/load | `mlflow.lightgbm.log_model()` / `load_model()` | MLflow handles conda environments, pip requirements, model signatures, and version tracking. Custom serialization breaks when LightGBM updates internal format. |
| CLI argument parsing | argparse or raw sys.argv | Typer | Type hint-driven, auto-generates help text, built-in Rich integration. Already in tech stack. |
| Terminal tables/colors | ANSI escape codes or tabulate | Rich | Rich handles terminal width, Unicode, color support detection, and dozens of output formats. |

**Key insight:** Phase 3 is infrastructure glue, not novel algorithms. The novel parts (slippage model, LightGBM training, walk-forward validation) are already implemented in Phase 1-2. Phase 3 wraps these in lifecycle management, monitoring, and operator tooling. The risk is over-engineering the glue.

## Common Pitfalls

### Pitfall 1: MLflow Tracking URI Confusion
**What goes wrong:** MLflow defaults to `./mlruns` in the current working directory. If the CLI is invoked from different directories, each invocation creates a separate `mlruns` directory with its own registry.
**Why it happens:** MLflow's default tracking URI is relative, not absolute.
**How to avoid:** Always set `MLFLOW_TRACKING_URI` to an absolute path (e.g., `file:///Users/tristanfarmer/Documents/HYDRA/mlruns`) or configure it in a central config file. The `ModelRegistry` wrapper should enforce this.
**Warning signs:** "Model not found" errors, missing champion alias, empty model list.

### Pitfall 2: PSI Zero-Bin Division
**What goes wrong:** PSI formula has `ln(actual% / expected%)`. If any bin has zero observations in either distribution, you get `log(0)` or division by zero.
**Why it happens:** Sparse features (common in thin markets) produce empty histogram bins.
**How to avoid:** Add epsilon smoothing: replace zero proportions with `1e-4` before computing PSI. Use quantile-based binning instead of equal-width to ensure non-empty bins.
**Warning signs:** NaN or Inf PSI values.

### Pitfall 3: ADWIN False Positives on Regime Changes
**What goes wrong:** ADWIN detects drift on legitimate market regime changes (e.g., seasonal patterns in agricultural futures), causing unnecessary diagnostic cycles.
**Why it happens:** ADWIN cannot distinguish between harmful concept drift and expected regime shifts.
**How to avoid:** Use ADWIN with a grace period and require multiple corroborating signals before triggering action. The observer should report drift, not act on it directly -- Phase 4's diagnostician decides whether drift warrants intervention.
**Warning signs:** High `n_detections` on ADWIN with no corresponding model performance degradation.

### Pitfall 4: MLflow Model Version Numbering
**What goes wrong:** MLflow auto-increments model versions. If you delete a version, the number is not reused. Version numbers can become non-contiguous and confusing.
**Why it happens:** MLflow's versioning is append-only by design.
**How to avoid:** Never rely on version number arithmetic. Always use aliases (`champion`, `candidate`, `archived`) to reference models. Store the `run_id` in the experiment journal for traceability.
**Warning signs:** Code that does `version - 1` to find the previous model.

### Pitfall 5: Experiment Journal Schema Rigidity
**What goes wrong:** The journal schema gets locked in too early, and Phase 4 needs fields that were not anticipated (e.g., `agent_autonomy_level`, `parent_experiment_id` for chained experiments).
**Why it happens:** Under-estimating the information Phase 4's agent loop will need.
**How to avoid:** Use JSON columns for extensible data (`config_diff`, `results`, `tags`). Keep the SQL-indexed columns to a small set of known query patterns. Add a `metadata TEXT` JSON column as an escape hatch for Phase 4.
**Warning signs:** Adding columns to a populated journal table (requires migration).

### Pitfall 6: Composite Fitness Normalization
**What goes wrong:** The 6-metric composite fitness score (Sharpe 0.25, drawdown 0.20, etc.) produces meaningless numbers because the individual metrics are on different scales.
**Why it happens:** Sharpe might be [-2, 3], drawdown [-0.50, 0], hit rate [0, 1]. Raw weighted sum is dominated by whichever metric has the largest absolute values.
**How to avoid:** Normalize each metric to [0, 1] using min-max scaling against historical baselines BEFORE computing the weighted sum. Define what "good" and "bad" look like for each metric.
**Warning signs:** Fitness score dominated by a single metric; tiny improvements in Sharpe swamp large improvements in drawdown.

## Code Examples

### PSI Computation (NumPy)

```python
# Source: Standard PSI formula with epsilon smoothing
import numpy as np

def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.10: No significant shift
    PSI 0.10-0.25: Moderate shift
    PSI >= 0.25: Significant shift

    Uses quantile-based binning from baseline to avoid empty bins.
    """
    # Quantile-based bin edges from baseline
    bin_edges = np.quantile(baseline, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
    current_counts = np.histogram(current, bins=bin_edges)[0]

    # Proportions with epsilon smoothing
    baseline_pct = baseline_counts / len(baseline) + epsilon
    current_pct = current_counts / len(current) + epsilon

    # Re-normalize after epsilon
    baseline_pct /= baseline_pct.sum()
    current_pct /= current_pct.sum()

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)
```

### KS Test Wrapper (SciPy)

```python
# Source: scipy.stats.ks_2samp
from scipy.stats import ks_2samp

def check_ks_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """Two-sample KS test for distribution drift.

    Returns (is_drifted, statistic, p_value).
    """
    stat, pvalue = ks_2samp(baseline, current)
    return pvalue < alpha, float(stat), float(pvalue)
```

### CUSUM Detector (NumPy)

```python
# Source: Standard CUSUM algorithm
import numpy as np

class CUSUMDetector:
    """Cumulative sum change-point detector.

    Tracks cumulative positive and negative deviations from a target.
    Signals drift when either cumulative sum exceeds the threshold.
    """

    def __init__(self, target: float = 0.0, threshold: float = 5.0, drift: float = 0.5):
        self.target = target
        self.threshold = threshold
        self.drift = drift  # allowance parameter
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.drift_detected = False

    def update(self, x: float) -> bool:
        """Process one observation. Returns True if drift detected."""
        self.s_pos = max(0, self.s_pos + x - self.target - self.drift)
        self.s_neg = max(0, self.s_neg - x + self.target - self.drift)
        self.drift_detected = self.s_pos > self.threshold or self.s_neg > self.threshold
        return self.drift_detected

    def reset(self) -> None:
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.drift_detected = False
```

### ADWIN Wrapper (River)

```python
# Source: https://riverml.xyz/dev/api/drift/ADWIN/
from river.drift import ADWIN

class ADWINDetector:
    """Wrapper around River's ADWIN for streaming drift detection.

    Parameters
    ----------
    delta : float
        Confidence parameter. Smaller delta = fewer false positives,
        slower detection. Default 0.002.
    grace_period : int
        Minimum observations before checking for drift. Default 30.
    """

    def __init__(self, delta: float = 0.002, grace_period: int = 30):
        self.detector = ADWIN(delta=delta, grace_period=grace_period)

    def update(self, value: float) -> bool:
        """Feed one observation. Returns True if drift detected."""
        self.detector.update(value)
        return self.detector.drift_detected

    @property
    def estimation(self) -> float:
        """Current mean estimate within the window."""
        return self.detector.estimation

    @property
    def n_detections(self) -> int:
        return self.detector.n_detections
```

### MLflow LightGBM Logging

```python
# Source: https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html
import mlflow
import mlflow.lightgbm

# Set absolute tracking URI
mlflow.set_tracking_uri("file:///path/to/hydra/mlruns")

with mlflow.start_run(run_name="experiment-42") as run:
    # Log parameters (config snapshot)
    mlflow.log_params({
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
    })

    # Log metrics
    mlflow.log_metrics({
        "sharpe_ratio": 0.85,
        "max_drawdown": -0.12,
        "hit_rate": 0.56,
    })

    # Log model with registration
    mlflow.lightgbm.log_model(
        lgb_model=model.model,  # the underlying LGBMClassifier
        artifact_path="model",
        registered_model_name="hydra-baseline",
    )

    # Tag the run
    mlflow.set_tag("mutation_type", "hyperparameter")
    mlflow.set_tag("hypothesis", "Reduce num_leaves to prevent overfitting")

# Promote to champion via alias
client = mlflow.MlflowClient()
client.set_registered_model_alias("hydra-baseline", "champion", version_number)
```

### Typer + Rich CLI

```python
# Source: https://typer.tiangolo.com/ + https://rich.readthedocs.io/
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

app = typer.Typer(
    name="hydra",
    help="HYDRA autonomous trading system CLI",
    rich_markup_mode="rich",
)
console = Console()

@app.command()
def status():
    """Show model health, active experiments, alerts, current autonomy level."""
    # Build status table
    table = Table(title="Champion Model Status", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Status", style="bold")

    table.add_row("Sharpe Ratio", "0.85", "[green]OK[/green]")
    table.add_row("Max Drawdown", "-12.3%", "[yellow]WATCH[/yellow]")
    table.add_row("Hit Rate", "56.2%", "[green]OK[/green]")
    table.add_row("Feature Drift (PSI)", "0.08", "[green]OK[/green]")

    console.print(table)

    # Alerts panel
    console.print(Panel(
        "[yellow]1 active experiment[/yellow]: Testing lr=0.05 (run abc-123)",
        title="Active Experiments",
        border_style="blue",
    ))

@app.command()
def rollback():
    """Revert to previous champion model."""
    console.print("[yellow]Rolling back champion model...[/yellow]")
    # ... call registry.rollback()
    console.print("[green]Rollback complete. Champion is now version N.[/green]")

if __name__ == "__main__":
    app()
```

### Composite Fitness Evaluator

```python
# Pattern: Normalized weighted composite score
import numpy as np

# Metric weights from SBOX-05 requirement
WEIGHTS = {
    "sharpe": 0.25,
    "drawdown": 0.20,
    "calibration": 0.15,
    "robustness": 0.15,
    "slippage_adjusted_return": 0.15,
    "simplicity": 0.10,
}

# Normalization ranges (what "0" and "1" look like for each metric)
RANGES = {
    "sharpe": (-1.0, 3.0),           # -1 is terrible, 3 is excellent
    "drawdown": (-0.50, 0.0),         # -50% is terrible, 0% is perfect
    "calibration": (0.0, 1.0),         # 0 is uncalibrated, 1 is perfect
    "robustness": (0.0, 1.0),         # fraction of folds with Sharpe > 0
    "slippage_adjusted_return": (-0.5, 1.0),  # annual return after slippage
    "simplicity": (0.0, 1.0),         # 1 / log(n_features + n_estimators)
}

def compute_fitness(metrics: dict[str, float]) -> float:
    """Compute normalized weighted composite fitness in [0, 1]."""
    score = 0.0
    for metric, weight in WEIGHTS.items():
        raw = metrics[metric]
        lo, hi = RANGES[metric]
        normalized = np.clip((raw - lo) / (hi - lo), 0, 1)
        score += weight * normalized
    return float(score)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MLflow Model Stages (Staging/Production/Archived) | MLflow Model Aliases (champion/candidate/any-string) | MLflow 2.9 (2023) | Use `set_registered_model_alias()` not `transition_model_version_stage()`. Stages will be removed in a future major release. |
| scikit-multiflow for ADWIN | River library | 2021 merger | scikit-multiflow is deprecated. River is the successor. Import from `river.drift.ADWIN`. |
| typer-slim (no Rich dependency) | typer[all] includes Rich | Typer 0.22 (2025) | `typer-slim` no longer supported. All Typer installs include Rich. |

**Deprecated/outdated:**
- `mlflow.register_model()` with stage transitions: Use aliases instead
- `scikit-multiflow`: Merged into River. Do not depend on `skmultiflow`.
- `mlflow.tracking.MlflowClient()`: Still works but `mlflow.MlflowClient()` is the current import path

## Open Questions

1. **MLflow Autolog vs. Explicit Logging**
   - What we know: `mlflow.lightgbm.autolog()` automatically logs parameters, metrics, and models during `model.fit()`. Explicit logging gives full control.
   - What's unclear: Whether autolog captures everything HYDRA needs (config snapshot, custom tags, custom metrics like composite fitness).
   - Recommendation: Use explicit logging. Autolog is convenient for notebooks but lacks the control needed for automated experiment pipelines. Log config snapshots as JSON artifacts, not just params.

2. **Calibration Metric Definition**
   - What we know: SBOX-05 lists "calibration" as 15% of composite fitness. Calibration typically means predicted probabilities match observed frequencies.
   - What's unclear: Exact calibration metric -- Brier score? Expected Calibration Error (ECE)? Log loss is already common but different from calibration.
   - Recommendation: Use Brier score (lower is better, measures calibration + discrimination). It decomposes into reliability (calibration) + resolution + uncertainty, giving diagnostic value beyond a single number.

3. **Simplicity Metric Definition**
   - What we know: SBOX-05 lists "simplicity" as 10% of composite fitness. This is an Occam's razor penalty.
   - What's unclear: How to quantify model simplicity for LightGBM. Number of trees? Number of leaves? Feature count?
   - Recommendation: `simplicity = 1 / log2(n_features * n_estimators + 1)`. This penalizes models that use many features or many trees. Normalize to [0, 1] against historical range.

4. **Robustness Metric Definition**
   - What we know: SBOX-05 lists "robustness" as 15% of composite fitness.
   - What's unclear: Exact definition. Cross-fold stability? Performance under perturbation?
   - Recommendation: `robustness = fraction of walk-forward folds where fold_sharpe > 0`. A model with Sharpe > 0 in 4/5 folds is more robust than one with Sharpe > 0 in 2/5 folds, even if the aggregate Sharpe is similar.

5. **Agent Loop State Persistence (pause/run)**
   - What we know: CLI-04 needs pause/run commands. The agent loop (Phase 4) is not built yet.
   - What's unclear: Whether the pause/run state should be a file on disk, a database flag, or an in-process signal.
   - Recommendation: Use a JSON state file at a known path (e.g., `~/.hydra/agent_state.json`). This survives process restarts, is human-readable, and Phase 4 can read it at the top of each cycle. Keep it simple.

## Sources

### Primary (HIGH confidence)
- [MLflow Model Registry docs](https://mlflow.org/docs/latest/model-registry/) - Aliases API, lifecycle management, deprecation of stages
- [mlflow.lightgbm API](https://mlflow.org/docs/latest/python_api/mlflow.lightgbm.html) - LightGBM model logging, saving, loading
- [MLflow Model Registry Workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/) - Registration, aliases, tags, deployment patterns
- [River ADWIN API](https://riverml.xyz/dev/api/drift/ADWIN/) - Constructor parameters, methods, usage example
- [MLflow PyPI](https://pypi.org/project/mlflow/) - Version 3.9.0, Python >=3.10 requirement
- [Typer docs](https://typer.tiangolo.com/) - CLI framework, Rich integration, argument/option patterns
- [Rich docs](https://rich.readthedocs.io/en/stable/) - Tables, panels, progress, console

### Secondary (MEDIUM confidence)
- [NannyML PSI Guide](https://www.nannyml.com/blog/population-stability-index-psi) - PSI formula, binning strategies, interpretation thresholds
- [DataCamp Drift Detection](https://www.datacamp.com/tutorial/understanding-data-drift-model-drift) - Comprehensive PSI/KS/ADWIN overview
- [QuantStart Event-Driven Backtesting](https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/) - Event-driven replay architecture patterns
- [GitHub: population-stability-index](https://github.com/mwburke/population-stability-index) - Reference PSI implementation in Python
- [MLflow Tracking with Local Database](https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/) - Local file/SQLite backend setup

### Tertiary (LOW confidence)
- River library version on PyPI: searched for latest version but results were ambiguous (v0.10.1 vs v0.22). Verify with `pip install river` before planning.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - MLflow, Typer, Rich are well-documented, actively maintained, and confirmed via official docs
- Architecture: HIGH - Patterns derived from existing codebase analysis (WalkForwardEngine, FeatureStore, BaselineModel) and established MLOps patterns
- Drift detection: HIGH - PSI/KS are textbook formulas, CUSUM is well-documented, ADWIN via River has official API docs
- Pitfalls: MEDIUM - Based on community patterns and known MLflow gotchas; some may not apply to HYDRA's specific use case
- Composite fitness: MEDIUM - Metric definitions (calibration, simplicity, robustness) need design decisions during planning

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (30 days - stable libraries, no fast-moving ecosystem changes)
