# Architecture Research: HYDRA

**Domain:** Autonomous self-healing ML trading agent for low-volume futures markets
**Researched:** 2026-02-18
**Confidence:** MEDIUM (PRD is authoritative primary source; architectural patterns from training data; web verification unavailable)

---

## Standard Architecture

### System Overview

HYDRA is a layered system where deterministic computation (data, signals, ML training) flows upward, and an LLM-powered agent loop sits on top making strategic decisions about *what* to compute and *when* to change the model. The critical architectural insight: **the agent is a meta-controller, not an inline component.** It does not make trading predictions -- it decides how to improve the thing that makes predictions.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI / OPERATOR LAYER                         │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐    │
│  │  Typer   │  │  Status      │  │  Human Override /           │    │
│  │  CLI     │→ │  Dashboard   │  │  Autonomy Controls          │    │
│  └──────────┘  └──────────────┘  └────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                     AGENT LAYER (meta-controller)                    │
│                                                                      │
│  ┌──────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │
│  │ Observer │→ │Diagnostician│→ │ Hypothesis │→ │  Experiment  │   │
│  │          │  │  (+ LLM)   │  │ Engine     │  │  Runner      │   │
│  └──────────┘  └────────────┘  │  (+ LLM)   │  └──────────────┘   │
│       ↑                         └────────────┘         │            │
│       └──────────────── Evaluator/Judge ────────────────┘            │
│                              ↕                                       │
│                    ┌────────────────────┐                            │
│                    │ Experiment Journal │                            │
│                    │ (Memory Layer)     │                            │
│                    └────────────────────┘                            │
├─────────────────────────────────────────────────────────────────────┤
│                      SANDBOX LAYER                                   │
│                                                                      │
│  ┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐    │
│  │ Market Replay │  │ Synthetic Data  │  │ A/B Paper Trading  │    │
│  │ Engine        │  │ Generator       │  │ Framework          │    │
│  └───────────────┘  └─────────────────┘  └────────────────────┘    │
│                                                                      │
│  ┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐    │
│  │ Candidate     │  │ Champion        │  │ Graveyard          │    │
│  │ Models        │  │ Model           │  │ (failed models)    │    │
│  └───────────────┘  └─────────────────┘  └────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   MLflow Model Registry                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                      SIGNAL LAYER                                    │
│                                                                      │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ Options Math     │  │ Sentiment       │  │ Divergence       │   │
│  │ Engine           │  │ Engine          │  │ Detector         │   │
│  │                   │  │                  │  │                   │   │
│  │ - Implied Dist.  │  │ - COT Scorer    │  │ - Signal Fusion  │   │
│  │ - Greeks/Flows   │  │ - NLP Scorer    │  │ - Divergence     │   │
│  │ - Vol Surface    │  │ - Social Scorer │  │   Classification │   │
│  │ - GEX/Vanna      │  │ - Flow Scorer   │  │ - Confidence     │   │
│  └────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘   │
│           │                      │                     │             │
├───────────┴──────────────────────┴─────────────────────┴─────────────┤
│                    DATA INFRASTRUCTURE LAYER                         │
│                                                                      │
│  ┌────────────┐  ┌───────────────┐  ┌──────────────┐               │
│  │ Ingestion  │  │ Feature Store │  │ Performance  │               │
│  │ Pipelines  │→ │ (TimescaleDB) │→ │ Database     │               │
│  └────────────┘  └───────────────┘  └──────────────┘               │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Raw Data Lake: Parquet files partitioned by date + market     │  │
│  └────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                    EXECUTION LAYER (Phase 5+)                        │
│                                                                      │
│  ┌──────────────────┐  ┌───────────────┐  ┌──────────────────┐     │
│  │ Order Manager    │  │ Risk Manager  │  │ IB Gateway       │     │
│  │ (Position sizing,│  │ (Limits,      │  │ (ib_insync)      │     │
│  │  order routing)  │  │  circuit       │  │                   │     │
│  │                   │  │  breakers)    │  │                   │     │
│  └──────────────────┘  └───────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer dependency rule:** Each layer depends only on the layer below it. No upward dependencies. This enforces clean separation and allows testing each layer independently.

---

### Component Responsibilities

#### Layer 1: Data Infrastructure

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Ingestion Pipelines** | Fetch raw market data (futures OHLCV, options chains, COT, news, social) on schedule or trigger. One pipeline class per data source. | Python with APScheduler; abstract base class per source type. Each pipeline: fetch, validate, normalize, store. |
| **Feature Store** | Store computed time-series features with versioning; serve features for training and inference with point-in-time correctness. | TimescaleDB hypertables keyed by (market, timestamp, feature_name). SQLite acceptable for Phase 1 development. |
| **Performance Database** | Track champion model predictions, actual outcomes, PnL, all evaluation metrics. Append-only. | TimescaleDB or PostgreSQL; one row per prediction with full input context (feature vector, model version, confidence). |
| **Raw Data Lake** | Immutable archive of all raw ingested data for replay and retraining. Never modified after write. | Parquet files on local filesystem, partitioned by `market/year/month/`. |

**Boundary rule:** Nothing above this layer touches raw data directly. All consumers read from the Feature Store or Raw Data Lake via defined interfaces.

#### Layer 2: Signal Layer

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Options Math Engine** | Compute implied distributions (Breeden-Litzenberger), Greeks flows (GEX, vanna, charm), volatility surfaces from raw options chains. | Python module: NumPy/SciPy for numerical differentiation and spline interpolation. QuantLib optional for SABR surface fitting. |
| **Sentiment Engine** | Produce normalized sentiment scores [-1, +1] from COT positioning, news NLP, social media, order flow. Confidence-weighted composite. | Python module: FinBERT for NLP scoring, custom parsers for COT CSV, configurable source weights. |
| **Divergence Detector** | Compare options-implied view vs. sentiment to classify divergence type (from PRD taxonomy) and produce trading signals with confidence. | Python module: pure classification logic. No ML -- deterministic rules mapping divergence patterns to signal types. |

**Boundary rule:** Signal components are stateless compute functions. They read features, compute outputs, write results back to the Feature Store. No state, no side effects, no awareness of models or agents. This makes them independently testable and replaceable.

#### Layer 3: Sandbox Layer

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Model Registry** | Version, store, and manage all model artifacts (configs, weights, metrics). Track champion/candidate/graveyard status. | MLflow Model Registry with local file backend. Each model has: version ID, parent ID, config, metrics, promotion status. |
| **Market Replay Engine** | Replay historical data with realistic order book simulation. Model thin-market slippage as function of order size, time-of-day, volume. | Custom Python. Feeds bar data through model inference, simulates fills with configurable slippage model. Supports variable speed (fast-forward for backtests, real-time for paper trading). |
| **Synthetic Data Generator** | Generate plausible unseen market scenarios for stress testing. Scenario library: flash crash, liquidity drought, vol explosion, regime change. | Copula-based simulator preserving cross-asset correlations (Phase 3). Conditional GAN for more realistic generation (Phase 5+). |
| **A/B Testing Framework** | Run champion and candidate side-by-side on live paper data with statistical significance testing. | Custom Python. Parallel predictions on same live feed, track divergence, automated kill switch if candidate underperforms threshold. |
| **Evaluator/Judge** | Multi-objective fitness evaluation. Decide promotion. | Deterministic Python. Weighted composite: Sharpe (0.25) + drawdown (0.20) + calibration (0.15) + robustness (0.15) + slippage-adjusted return (0.15) + simplicity (0.10). |

**Boundary rule:** The sandbox is hermetically sealed from production. Candidate models inside the sandbox never touch the execution layer. Only a promoted champion -- after passing the Evaluator -- gets deployed.

#### Layer 4: Agent Layer (Meta-Controller)

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Observer** | Monitor champion health (rolling Sharpe, drawdown, hit rate, calibration), data health (feature staleness, distribution shifts via PSI), market regime (vol regime, trend state, liquidity). | Python module. ADWIN for concept drift, CUSUM for changepoints, KS tests for feature distributions. Runs on configurable interval (default: hourly). |
| **Diagnostician** | When Observer raises alert: structured triage. Data audit, feature attribution (SHAP), regime check, overfitting decay test, external event scan. | **LLM-powered:** structured diagnosis via cheap Chinese LLM. Deterministic data audit subroutines feed context to LLM. LLM produces typed Diagnosis object. Rule-based fallback if LLM fails. |
| **Hypothesis Engine** | Generate candidate improvements from diagnosis or ongoing optimization. Must check journal for prior similar experiments before proposing. | **LLM-powered:** LLM proposes mutations constrained by Mutation Playbook. Deterministic validation rejects invalid/duplicate proposals. Max 3 concurrent experiments. |
| **Experiment Runner** | Fork champion config, apply mutation, train in isolated subprocess, evaluate via walk-forward + stress tests, log everything. | Deterministic Python. Process-level isolation (subprocess, not threads). 1-hour timeout per experiment. |
| **Experiment Journal** | Long-term memory. Every experiment, diagnosis, and decision logged with full context. Queryable by agent and human. | SQLite with JSON columns. Natural language query via LLM translating to SQL. Structured query API for programmatic access. |

**Critical boundary rule:** The LLM (Qwen/DeepSeek/GLM) is used ONLY in the Diagnostician and Hypothesis Engine for reasoning, and in Journal for natural language queries. Every other component is deterministic Python. **The LLM proposes; deterministic code disposes.** This is the most important architectural decision in the entire system.

#### Layer 5: Execution Layer (Phase 5+)

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Order Manager** | Convert model signals into sized, routed orders. Implements fractional Kelly sizing. | Python module. Reads champion prediction + confidence, computes position size, constructs order. |
| **Risk Manager** | Enforce position limits (max 2% of ADV), drawdown circuit breakers, exposure limits. | Python module. Stateful -- tracks cumulative exposure. Defaults to "do nothing" on ambiguity. Circuit breakers cut to zero, not reduced. |
| **IB Gateway** | Interface with Interactive Brokers TWS/Gateway for order submission and fill reporting. | ib_insync library wrapping IB API. Persistent TCP connection with reconnection logic. |

**Boundary rule:** Execution layer defaults to "reject" on any ambiguity. Circuit breakers are hard stops, not soft suggestions.

#### Layer 6: CLI Layer

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **CLI Interface** | Human operator commands matching PRD Section 6.1. Status, diagnose, experiment, journal, sandbox, config, run/pause/rollback. | Typer CLI framework. Each command group is a separate module. Zero business logic in CLI -- delegates to service classes. |

**Boundary rule:** CLI is a skin, not a brain. Every command delegates to a service class that can also be called programmatically by the agent.

---

## Recommended Project Structure

```
hydra/
├── pyproject.toml                # Single project config (dependencies, linting, build)
├── config/
│   ├── default.yaml              # All default parameters (thresholds, intervals, weights)
│   ├── markets/
│   │   └── lean_hogs.yaml        # Market-specific overrides (strike spacing, slippage params)
│   └── agent/
│       └── mutation_playbook.yaml   # Allowed mutation types + constraints
│
├── src/
│   └── hydra/
│       ├── __init__.py
│       │
│       ├── data/                 # DATA INFRASTRUCTURE LAYER
│       │   ├── __init__.py
│       │   ├── ingestion/
│       │   │   ├── __init__.py
│       │   │   ├── base.py       # Abstract pipeline class
│       │   │   ├── futures.py    # Futures OHLCV ingestion
│       │   │   ├── options.py    # Options chains ingestion
│       │   │   ├── cot.py        # CFTC COT data
│       │   │   ├── news.py       # News API ingestion
│       │   │   └── social.py     # Social media ingestion
│       │   ├── store/
│       │   │   ├── __init__.py
│       │   │   ├── timescale.py  # TimescaleDB connection + feature read/write
│       │   │   └── parquet.py    # Parquet read/write for raw data lake
│       │   └── performance_db.py # Model performance tracking
│       │
│       ├── signals/              # SIGNAL LAYER
│       │   ├── __init__.py
│       │   ├── options_math/
│       │   │   ├── __init__.py
│       │   │   ├── surface.py    # Vol surface construction
│       │   │   ├── density.py    # Breeden-Litzenberger density extraction
│       │   │   ├── moments.py    # Implied moments computation
│       │   │   └── greeks.py     # GEX, vanna, charm flow aggregation
│       │   ├── sentiment/
│       │   │   ├── __init__.py
│       │   │   ├── cot.py        # COT positioning scorer
│       │   │   ├── nlp.py        # FinBERT headline scorer
│       │   │   └── composite.py  # Confidence-weighted sentiment blend
│       │   ├── divergence.py     # Divergence detector + classification
│       │   └── features.py       # Feature vector assembly (point-in-time)
│       │
│       ├── sandbox/              # SANDBOX LAYER
│       │   ├── __init__.py
│       │   ├── replay.py         # Market replay engine with slippage
│       │   ├── synthetic.py      # Synthetic data generation
│       │   ├── slippage.py       # Thin-market slippage model
│       │   ├── evaluator.py      # Multi-objective fitness + promotion protocol
│       │   ├── ab_test.py        # A/B paper trading framework
│       │   └── models/
│       │       ├── __init__.py
│       │       ├── base.py       # Abstract TradingModel interface
│       │       ├── lightgbm_model.py  # LightGBM baseline
│       │       ├── torch_model.py     # PyTorch neural architectures
│       │       └── ensemble.py        # Ensemble composition
│       │
│       ├── agent/                # AGENT LAYER
│       │   ├── __init__.py
│       │   ├── loop.py           # Main agent loop orchestration (asyncio)
│       │   ├── observer.py       # Health monitoring + drift detection
│       │   ├── diagnostician.py  # LLM-powered triage
│       │   ├── hypothesis.py     # LLM-powered mutation proposal
│       │   ├── experiment.py     # Experiment runner (deterministic, subprocess)
│       │   ├── journal.py        # Experiment journal (SQLite/PostgreSQL)
│       │   └── llm/
│       │       ├── __init__.py
│       │       ├── client.py     # LLM API client (Qwen/DeepSeek/GLM)
│       │       ├── prompts.py    # Prompt templates for diagnosis + hypothesis
│       │       └── parsers.py    # Structured output parsing + Pydantic validation
│       │
│       ├── execution/            # EXECUTION LAYER (Phase 5+)
│       │   ├── __init__.py
│       │   ├── order_manager.py  # Order construction + Kelly sizing
│       │   ├── risk_manager.py   # Position limits, circuit breakers
│       │   └── ib_gateway.py     # Interactive Brokers interface (ib_insync)
│       │
│       └── cli/                  # CLI LAYER
│           ├── __init__.py
│           ├── app.py            # Typer app definition + top-level commands
│           ├── status.py         # gsd status commands
│           ├── experiment.py     # gsd experiment commands
│           ├── journal.py        # gsd journal commands
│           ├── sandbox.py        # gsd sandbox commands
│           └── config_cmd.py     # gsd config commands
│
├── tests/
│   ├── unit/
│   │   ├── test_options_math.py
│   │   ├── test_sentiment.py
│   │   ├── test_divergence.py
│   │   ├── test_evaluator.py
│   │   ├── test_observer.py
│   │   └── test_journal.py
│   ├── integration/
│   │   ├── test_data_pipeline.py
│   │   ├── test_signal_pipeline.py
│   │   ├── test_experiment_cycle.py
│   │   └── test_agent_loop.py
│   └── fixtures/
│       ├── sample_options_chain.parquet
│       ├── sample_cot_report.csv
│       └── sample_features.parquet
│
├── notebooks/                    # Exploratory (not production code)
│   ├── 01_implied_distribution_validation.ipynb
│   └── 02_signal_quality_analysis.ipynb
│
└── data/                         # Local data directory (gitignored)
    ├── raw/                      # Parquet files by market/year/month/
    ├── models/                   # MLflow artifact store
    └── journal/                  # SQLite database
```

### Structure Rationale

- **`src/hydra/` layout mirrors the architecture layers exactly.** Each subdirectory is one layer. No cross-layer imports except through defined interfaces.
- **`data/ingestion/` has one module per source.** Prevents monolithic ingestion code. Each source has different APIs, rates, formats -- they should not share a module.
- **`agent/llm/` is isolated.** All LLM interaction in one subdirectory. If the LLM provider changes (Qwen to DeepSeek, API to local model), only `client.py` changes. Prompts and parsers are separate because they evolve independently of the client.
- **`sandbox/models/` uses an abstract base class.** Every model type implements the same `TradingModel` interface. The agent swaps architectures by instantiating a different class, not by rewriting code.
- **`cli/` is thin.** Each CLI module maps to a Typer command group. All logic lives in the service layer.
- **`signals/options_math/` is split by computation type.** Surface construction, density extraction, moments, and Greeks are distinct computations with different test strategies. Keeping them in one file would create a 1000+ line module.

---

## Architectural Patterns

### Pattern 1: Deterministic Envelope Around LLM Reasoning

**What:** The LLM generates structured proposals (diagnosis, hypothesis). Deterministic code validates, constrains, and executes those proposals. The LLM never directly modifies state, triggers actions, or touches data.

**When to use:** Every point where the LLM interfaces with the system (exactly 3 touchpoints -- see LLM Integration section below).

**Trade-offs:** More boilerplate validation code, but eliminates an entire class of catastrophic failures. LLM hallucinations become log entries, not trading losses.

**Example:**

```python
# agent/hypothesis.py

@dataclass
class MutationProposal:
    """Deterministic schema the LLM must fill."""
    mutation_type: Literal["retrain", "feature_add", "feature_remove",
                           "hyperparameter", "architecture_swap",
                           "loss_function", "ensemble_change",
                           "window_adjust"]
    target_component: str
    parameters: dict
    hypothesis_text: str
    expected_improvement: str

class HypothesisEngine:
    def __init__(self, llm_client: LLMClient, journal: ExperimentJournal,
                 playbook: MutationPlaybook):
        self.llm = llm_client
        self.journal = journal
        self.playbook = playbook

    def propose(self, diagnosis: Diagnosis) -> Optional[MutationProposal]:
        # 1. LLM generates a proposal
        raw_response = self.llm.generate(
            prompt=HYPOTHESIS_PROMPT.format(
                diagnosis=diagnosis,
                recent_experiments=self.journal.recent(n=20),
                playbook=self.playbook.describe()
            )
        )

        # 2. Parse into typed schema (rejects malformed output)
        proposal = parse_structured(raw_response, MutationProposal)
        if proposal is None:
            logger.warning("LLM produced unparseable proposal, skipping")
            return None

        # 3. Deterministic validation gates
        if proposal.mutation_type not in self.playbook.allowed_mutations:
            logger.warning(f"Rejected: {proposal.mutation_type} not in playbook")
            return None

        if self.journal.has_similar(proposal, similarity_threshold=0.85):
            logger.info("Rejected: too similar to recent experiment")
            return None

        if self.journal.active_experiment_count() >= 3:
            logger.info("Rejected: max concurrent experiments reached")
            return None

        return proposal
```

### Pattern 2: Event Bus for Cross-Layer Communication

**What:** Components communicate through a lightweight in-process event system rather than direct function calls across layer boundaries. Observer emits alerts; Diagnostician subscribes. Evaluator emits promotion events; Execution Layer subscribes.

**When to use:** All cross-layer communication. Within a layer, direct function calls are fine.

**Trade-offs:** Slight indirection, but essential for a phased build -- you can add layers without modifying existing ones. The Data Layer does not need to know the Agent Layer exists.

**Example:**

```python
# Simple in-process event bus -- no Kafka/Redis needed at this scale

from collections import defaultdict
from typing import Callable

class EventBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable):
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, payload: dict):
        for handler in self._subscribers[event_type]:
            handler(payload)

# Wire up during initialization:
bus = EventBus()
bus.subscribe("alert.warning", diagnostician.handle_alert)
bus.subscribe("alert.critical", diagnostician.handle_critical)
bus.subscribe("model.promoted", execution_layer.update_champion)  # Added Phase 5

# In Observer:
bus.publish("alert.warning", {
    "type": "drift_detected",
    "feature": "implied_skew",
    "psi_score": 0.34,
    "timestamp": now()
})
```

### Pattern 3: Feature Store as Single Source of Truth

**What:** All computed features flow through TimescaleDB. Models read from the feature store, never from raw data directly.

**When to use:** Always. This is the architectural backbone.

**Trade-offs:** Requires discipline -- every new feature must be registered in the store. But enables point-in-time correctness (features at time T use only data available at time T) and eliminates training/serving skew.

**Example:**

```python
# Feature computation writes to the store (Signal Layer)
async def compute_and_store_features(timestamp: datetime, market: str):
    raw_chain = await feature_store.get_raw_options(market, timestamp)
    features = options_engine.compute_features(raw_chain)
    await feature_store.write_features(market, timestamp, features)

# Model reads from the store (Sandbox Layer) -- never computes features itself
async def predict(timestamp: datetime, market: str):
    features = await feature_store.get_features(market, as_of=timestamp)
    return champion_model.predict(features)
```

### Pattern 4: Abstract Model Interface for Architecture Swapping

**What:** All ML models implement an identical interface. The agent swaps architectures by changing a config parameter, not code.

**When to use:** Every model implementation.

**Trade-offs:** Some models have unique capabilities (LightGBM feature importance vs. PyTorch gradient-based attribution). The base interface covers common operations; model-specific capabilities are exposed through optional methods.

**Example:**

```python
# sandbox/models/base.py

from abc import ABC, abstractmethod
import numpy as np

class TradingModel(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, config: dict) -> "TradingModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return dict of metric_name -> metric_value."""
        ...

    @abstractmethod
    def serialize(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def deserialize(cls, path: str) -> "TradingModel":
        ...

    def feature_importance(self) -> Optional[dict[str, float]]:
        """Optional: return feature importance scores for SHAP analysis."""
        return None
```

### Pattern 5: Experiment Isolation via Process Forking

**What:** Every experiment starts by deep-copying the champion model's entire configuration. The mutation is applied to the copy. Training runs in an isolated subprocess. No shared mutable state.

**When to use:** Every experiment the agent runs.

**Trade-offs:** Subprocess isolation is slower (startup overhead) and uses more memory than thread-level isolation. For a system running at most 3 concurrent experiments on a single machine, this is acceptable. The safety guarantee is worth the overhead.

**Example:**

```python
# agent/experiment.py

import subprocess
import json

class ExperimentRunner:
    def run_experiment(self, proposal: MutationProposal,
                       champion_config: dict) -> ExperimentResult:
        # 1. Fork champion config (deep copy)
        candidate_config = deep_copy(champion_config)
        apply_mutation(candidate_config, proposal)

        # 2. Serialize config to isolated experiment directory
        exp_dir = create_experiment_dir(proposal.experiment_id)
        save_config(candidate_config, exp_dir / "config.yaml")

        # 3. Run in isolated subprocess (no shared memory)
        result = subprocess.run(
            ["python", "-m", "hydra.sandbox.train_candidate",
             "--config", str(exp_dir / "config.yaml"),
             "--output", str(exp_dir / "results.json")],
            capture_output=True,
            timeout=3600  # 1 hour max
        )

        # 4. Collect results from file (no shared state)
        return ExperimentResult.from_file(exp_dir / "results.json")
```

### Pattern 6: Circuit Breaker State Machine

**What:** The system has explicit operational states with defined transitions. Only humans can exit lockdown.

**When to use:** Always active once the system is in paper trading or live.

**Trade-offs:** Conservative design -- may halt trading during temporary issues that would self-resolve. But in thin markets where errors are expensive, conservative is correct.

```python
class SystemState(Enum):
    RUNNING = "running"       # Normal operation
    DEGRADED = "degraded"     # Reduced position sizing, heightened monitoring
    HALTED = "halted"         # No new positions, existing positions maintained
    LOCKDOWN = "lockdown"     # Flatten all positions, no trading

TRANSITIONS = {
    (SystemState.RUNNING, "warning_alert"): SystemState.DEGRADED,
    (SystemState.RUNNING, "critical_alert"): SystemState.HALTED,
    (SystemState.DEGRADED, "critical_alert"): SystemState.HALTED,
    (SystemState.DEGRADED, "resolved"): SystemState.RUNNING,
    (SystemState.HALTED, "manual_override"): SystemState.RUNNING,
    (SystemState.HALTED, "cascade_failure"): SystemState.LOCKDOWN,
    (SystemState.LOCKDOWN, "manual_override"): SystemState.RUNNING,  # Human only
}
```

### Pattern 7: Configuration as Code with Override Hierarchy

**What:** All tunable parameters live in YAML config with typed Python dataclass loading. Override hierarchy: defaults -> market-specific -> environment variables -> CLI flags.

**When to use:** Anywhere a magic number or threshold appears.

**Why this matters:** The agent's mutation playbook includes "adjust retraining frequency," "adjust lookback window." These mutations are config changes, not code changes. Typed config makes self-modification safe and auditable.

```yaml
# config/default.yaml
observer:
  drift_detection:
    method: "adwin"
    delta: 0.002
  alert_thresholds:
    sharpe_decline_warning: 0.5
    sharpe_critical: 0.0
    psi_warning: 0.25
  check_interval_minutes: 60

agent:
  max_concurrent_experiments: 3
  max_mutations_per_experiment: 2
  autonomy_level: "semi-auto"

evaluator:
  sharpe_improvement_min: 0.1
  drawdown_tolerance: 0.10
  min_evaluation_windows: 3
  paper_trading_hours: 48
```

---

## Data Flow

### Flow 1: Ingestion to Signal (runs on schedule)

```
External Sources (CME/Databento, CFTC, News APIs)
    │
    ↓ [scheduled fetch via APScheduler, per-source interval]
Ingestion Pipelines (one per source)
    │
    ├──→ Raw Data Lake (Parquet, immutable, partitioned by market/date)
    │
    ↓ [transform + compute, triggered after ingestion completes]
Signal Layer (stateless compute)
    │
    ├── Options Math Engine:
    │     raw chain → filter/clean → spline interpolation → vol surface
    │     → B-L density → implied moments → Greeks flows
    │     ──→ Feature Store: [implied_mean, implied_var, implied_skew,
    │          implied_kurt, gex, vanna_flow, charm_flow, skew_slope,
    │          vol_term_spread, put_call_oi_ratio]
    │
    ├── Sentiment Engine:
    │     COT CSV → positioning ratios
    │     Headlines → FinBERT → sentiment scores
    │     ──→ Feature Store: [composite_sentiment, cot_score,
    │          nlp_score, social_score, flow_imbalance,
    │          cross_market_risk_appetite, sentiment_confidence]
    │
    └── Divergence Detector:
          options features + sentiment features → divergence classification
          ──→ Feature Store: [divergence_direction, divergence_magnitude,
               divergence_type, confidence, suggested_bias]
```

### Flow 2: Inference (champion prediction)

```
Feature Store (point-in-time feature vector)
    │
    ↓ [assemble feature row with all signal components]
Champion Model (loaded from MLflow Registry, cached in memory)
    │
    ↓ [prediction: direction + confidence + suggested size]
Performance DB (log prediction + full input context)
    │
    ↓ [Phase 5+ only]
Order Manager → Risk Manager → IB Gateway → Market
```

### Flow 3: Agent Loop (observe -> diagnose -> hypothesize -> experiment -> evaluate)

```
Performance DB ──→ Observer ←── Feature Store
                      │
                      │ [alert raised: INFO / WARNING / CRITICAL]
                      ↓
                 Diagnostician
                      │ (reads: feature health, SHAP attribution, regime state)
                      │ (calls: LLM API for structured diagnosis)
                      │ (fallback: rule-based diagnosis if LLM fails)
                      ↓
                 Hypothesis Engine
                      │ (reads: diagnosis + Experiment Journal history)
                      │ (calls: LLM API for mutation proposal)
                      │ (validates: playbook constraints + dedup check)
                      │ (fallback: weighted random from playbook if LLM fails)
                      ↓
                 Experiment Runner [isolated subprocess]
                      │ (forks champion config, applies mutation)
                      │
                      ├──→ Market Replay Engine (walk-forward backtest)
                      ├──→ Synthetic Data Generator (stress tests)
                      │
                      ↓
                 Evaluator/Judge [deterministic]
                      │ (multi-objective fitness comparison vs. champion)
                      │
                      ├── PROMOTE → MLflow Registry: candidate becomes champion
                      │              → EventBus: "model.promoted"
                      │              → [optional: 48h paper trading validation first]
                      │
                      ├── REJECT  → Graveyard archive + Experiment Journal log
                      │
                      ↓
                 Experiment Journal
                      (immutable record: hypothesis, config diff, results, decision)
```

### Flow 4: Memory / Knowledge Retrieval

```
Agent Component (any) ──write──→ Experiment Journal ←──read── Agent Component (any)
                                         ↑
                                         │
                                   CLI Query (human)
                                   "What worked for high-vol regimes?"
                                         │
                                         ↓
                                   LLM translates NL → SQL
                                         │
                                         ↓
                                   Structured response with experiment IDs
```

---

## Where the LLM Agent Sits and How It Interfaces

### Placement Principle

The LLM is NOT in the prediction pipeline. It is a **meta-controller** that sits above the ML models and makes strategic decisions about how to improve them.

```
                    ┌──────────────────────────┐
                    │  LLM (Qwen/DeepSeek/GLM) │
                    │  via API provider         │
                    └────────────┬─────────────┘
                                 │
                    Structured output (JSON)
                                 │
                    ┌────────────┴─────────────┐
                    │  Deterministic Envelope   │
                    │  - Pydantic validation    │
                    │  - Playbook constraints   │
                    │  - Journal dedup check    │
                    │  - Budget enforcement     │
                    └────────────┬─────────────┘
                                 │
                    Valid typed object or None
                                 │
                    ┌────────────┴─────────────┐
                    │  Deterministic Execution  │
                    │  (pure Python, no LLM)    │
                    └──────────────────────────┘
```

### LLM Touchpoints (exactly 3)

| Touchpoint | Input | Output | Fallback |
|------------|-------|--------|----------|
| **Diagnostician.diagnose()** | Structured health data (Sharpe trend, PSI scores, feature importance, regime indicators) | Typed `Diagnosis` object: `{root_cause, affected_components, severity, recommended_action_type}` | Rule-based: if Sharpe < threshold -> "performance_decay"; if PSI > threshold -> "feature_drift" |
| **HypothesisEngine.propose()** | Diagnosis + recent experiment summaries from Journal + Mutation Playbook description | Typed `MutationProposal` object: `{mutation_type, target_component, parameters, hypothesis_text}` | Weighted random selection from Mutation Playbook based on diagnosis type |
| **Journal.query_natural_language()** | Human's natural language question (e.g., "what worked in high-vol regimes?") | SQL query against Journal + formatted results | Expose structured query parameters directly via CLI (`--tag`, `--mutation-type`, `--date-range`) |

### LLM Failure Handling

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| API timeout | HTTP timeout (30s) | Retry once with backoff, then skip cycle. Agent continues on next interval. |
| Malformed JSON output | Pydantic validation failure | Discard response. Log. Use rule-based fallback for this cycle. |
| Hallucinated mutation type | Not found in MutationPlaybook enum | Reject proposal. Log the hallucination for prompt improvement. |
| Repetitive proposals | Journal similarity check (cosine similarity > 0.85) | Reject as duplicate. LLM is not informed -- deterministic gate catches it. |
| API provider down | Multiple consecutive failures (3+) | Switch to rule-based mode entirely. Alert operator via CLI. |
| Cost spike | Token usage counter per cycle exceeds threshold | Rate-limit agent cycles (increase interval). Alert operator. |

**Key design principle:** The system MUST be able to run (in degraded mode) without the LLM. The LLM adds intelligence to the agent loop, but the loop's skeleton can run with rule-based fallbacks at every LLM touchpoint. An LLM API outage degrades agent quality but does not halt the system.

---

## Anti-Patterns

### Anti-Pattern 1: LLM in the Hot Path

**What people do:** Put the LLM in the inference/prediction pipeline so every trading decision goes through it.

**Why it is wrong:** Latency (100ms-2s per API call), cost (per-token on every prediction), reliability (API outages halt trading), determinism (same inputs may give different outputs).

**Do this instead:** LLM operates on slow cycles (hourly/daily). Predictions come from deterministic ML models (LightGBM/PyTorch) running locally in microseconds.

### Anti-Pattern 2: Shared Mutable State Between Champion and Candidate

**What people do:** Candidates read from the same feature cache, model registry, or state as champion "for efficiency."

**Why it is wrong:** A training bug in a candidate can corrupt shared state. Any shared mutable resource is a vector for candidates to affect production.

**Do this instead:** Candidates read from immutable snapshots (Parquet files). They write to experiment-specific directories. The only shared resource is the read-only Feature Store, which uses database-level isolation (transactions).

### Anti-Pattern 3: Look-Ahead Bias in Features

**What people do:** Using information from the future when computing features or training models.

**Why it is wrong:** Model appears to work in backtesting but fails completely live. This is the single most common reason quant strategies fail at deployment.

**Do this instead:** Strict point-in-time feature computation. Features at time T use only data available at time T. Walk-forward validation with embargo periods. Never shuffle time-series data. The Feature Store enforces this by storing features keyed by (market, timestamp).

### Anti-Pattern 4: Single Evaluation Metric

**What people do:** Judge model quality by Sharpe ratio alone.

**Why it is wrong:** High Sharpe with 90% drawdown. Or high Sharpe via overfitting to one regime.

**Do this instead:** PRD's multi-objective fitness: Sharpe (0.25) + drawdown (0.20) + calibration (0.15) + robustness (0.15) + slippage-adjusted return (0.15) + simplicity (0.10). Must pass on 3 of 5 independent evaluation windows.

### Anti-Pattern 5: Monolithic Agent Loop

**What people do:** Build observe/diagnose/hypothesize/experiment/evaluate as one giant function.

**Why it is wrong:** Untestable, undebuggable, unextendable. Cannot test the Evaluator without running the entire pipeline.

**Do this instead:** Each stage is a separate module with defined inputs/outputs. `loop.py` connects them. Each stage can be tested independently with synthetic inputs.

### Anti-Pattern 6: Building Execution Before Signal Validation

**What people do:** Wire up broker connections early because "we need to test with real money."

**Why it is wrong:** Weeks debugging IB API instead of validating alpha. Bugs in early execution lose real money.

**Do this instead:** Follow the PRD phase order. Execution is Phase 5. The sandbox (Phase 3) provides all feedback needed to validate the system.

---

## Integration Points

### External Services

| Service | Integration Pattern | Gotchas |
|---------|---------------------|---------|
| **CME DataMine / Databento** | REST API or FTP download, scheduled via APScheduler | Rate limits, auth, variable data formats. Abstract behind ingestion pipeline interface. Options chain data may have gaps in thin markets. |
| **CFTC COT Data** | Weekly FTP download (free, `cftc.gov`) | Release lag: Tuesday data, Friday publication. Must handle holidays gracefully. Historical data format changes occasionally. |
| **News APIs (NewsAPI, Benzinga)** | REST polling, hourly | Rate limits vary by tier. Need graceful degradation when quota exhausted. FinBERT inference adds latency -- batch process. |
| **Chinese LLM API provider** | REST API, async HTTP client (httpx) | Rate limits, outages, response format changes between model versions. Retry with exponential backoff. **Never block agent loop on failed LLM call.** Structured output reliability varies by model -- DeepSeek-R1 reported better than base Qwen for JSON (LOW confidence, needs validation). |
| **Interactive Brokers (Phase 5+)** | ib_insync Python library, persistent TCP connection to TWS/Gateway | TWS must be running locally. Connection drops are common -- implement automatic reconnection. Market hours enforcement required. Paper trading mode available for validation. |
| **MLflow** | Python SDK, local file backend | No network dependency if local. Model Registry for versioning and promotion tracking. |
| **TimescaleDB** | psycopg2 or asyncpg, SQL interface | Requires PostgreSQL instance with TimescaleDB extension. For Phase 1 development, SQLite is acceptable as stand-in for features (migrate to TimescaleDB before Phase 3). |

### Internal Boundaries

| Boundary | Communication Pattern | Notes |
|----------|----------------------|-------|
| Data Layer <-> Signal Layer | Feature Store SQL (read raw, write computed) | Same DB, different tables. Signal Layer never sees raw data outside the Feature Store. |
| Signal Layer <-> ML Models | Feature Store read-only | Models read assembled feature vectors. Never write back. Never compute features inline. |
| Agent Layer <-> Sandbox | Function calls (simple ops) or subprocess (training) | Agent proposes, sandbox executes. Results returned as serialized JSON/dataclass. |
| Agent Layer <-> LLM | HTTP API via `agent/llm/client.py` | Isolated. All other agent code unaware of which LLM is used. Client handles retries, parsing, fallback. |
| Agent <-> CLI | Shared service layer | CLI calls same methods the agent loop calls. No separate internal API. |
| Sandbox <-> Execution (Phase 5+) | EventBus: `model.promoted` event | One-way: sandbox tells execution which model is champion. Execution never tells sandbox anything. |

---

## Scaling Considerations

HYDRA is a single-operator, single-machine system targeting low-volume markets. "Scaling" means handling more markets or more concurrent experiments, not millions of users.

| Concern | 1 Market (MVP) | 3-5 Markets | 10+ Markets |
|---------|----------------|-------------|-------------|
| Data ingestion | Single APScheduler, sequential pipelines | Parallel pipelines per market, still single process | Dedicated ingestion worker per market group |
| Feature Store | SQLite for dev, TimescaleDB for production | TimescaleDB with market_id partitioning | TimescaleDB with compression + retention policies |
| Model training | Sequential experiments in subprocesses | Subprocess pool (3 concurrent across all markets) | Consider multiprocessing.Pool or Celery task queue |
| Agent reasoning | Sequential LLM API calls, one market context | Market-specific agent instances, each with own journal | Shared LLM client, market-specific prompts |
| Memory | Single SQLite journal | Single PostgreSQL DB, market_id column | Same with indexes on market_id |
| LLM costs | ~$5-10/month | ~$20-30/month | ~$60-100/month |
| IB API | Single TWS connection | Same connection handles multiple instruments | Same -- IB gateway supports dozens on one connection |

### Scaling Priorities

1. **First bottleneck: Data ingestion latency.** Adding markets makes ingestion I/O bound. Fix: parallelize with asyncio or threading.
2. **Second bottleneck: Experiment training time.** More markets = more models. Fix: subprocess isolation already enables parallelism; add simple job queue if needed.
3. **Not a bottleneck: LLM API calls.** Agent reasoning runs on slow cycles. Even 10 markets at one call/hour = 240 calls/day -- trivial.

---

## Build Order (Dependency-Driven)

The build order follows data flow dependencies. Each phase produces the inputs the next phase needs.

```
Phase 1: DATA INFRASTRUCTURE + OPTIONS MATH ENGINE
         ┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
         │ Ingestion   │ ──→ │ Feature Store     │ ──→ │ Options Math │
         │ (futures +  │     │ (SQLite/Timescale) │     │ Engine       │
         │  options)   │     └──────────────────┘     └──────────────┘
         └─────────────┘
         VALIDATES: Can compute and store implied distributions.
                    Plot implied vs. realized to confirm theoretical correctness.

Phase 2: SIGNAL LAYER + BASELINE MODEL
         ┌──────────────┐  ┌──────────────┐  ┌───────────────┐
         │ Sentiment    │  │ Divergence   │  │ Baseline      │
         │ Engine       │  │ Detector     │  │ LightGBM      │
         └──────────────┘  └──────────────┘  └───────────────┘
         REQUIRES: Phase 1 (features to compute sentiment against, data to train on)
         VALIDATES: Divergence signal has predictive power (Sharpe > 0 OOS).
                    If Sharpe <= 0, stop and investigate before continuing.

Phase 3: SANDBOX + EXPERIMENT INFRASTRUCTURE
         ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
         │ Market Replay  │  │ Synthetic    │  │ MLflow       │
         │ Engine         │  │ Data Gen     │  │ Registry     │
         └────────────────┘  └──────────────┘  └──────────────┘
         ┌──────────────┐  ┌──────────────────┐
         │ Evaluator    │  │ Model Versioning │
         └──────────────┘  └──────────────────┘
         REQUIRES: Phase 2 (a model to test, features to replay)
         VALIDATES: Can run full experiment cycle end-to-end.
                    Train candidate, evaluate vs. champion, log results.

Phase 4: AGENT CORE + MEMORY + CLI
         ┌──────────┐  ┌──────────────┐  ┌──────────────┐
         │ Observer │  │ Diagnostician│  │ Hypothesis   │
         │          │  │ (LLM)       │  │ Engine (LLM) │
         └──────────┘  └──────────────┘  └──────────────┘
         ┌──────────────┐  ┌──────────────┐  ┌─────┐
         │ Experiment   │  │ Experiment   │  │ CLI │
         │ Runner       │  │ Journal      │  │     │
         └──────────────┘  └──────────────┘  └─────┘
         REQUIRES: Phase 3 (sandbox for experiments to run in)
         VALIDATES: Agent detects injected drift, diagnoses it,
                    proposes fix, tests it, promotes if better.

Phase 5: EXECUTION + HARDENING
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │ Order Manager│  │ Risk Manager │  │ IB Gateway   │
         └──────────────┘  └──────────────┘  └──────────────┘
         ┌────────────────┐  ┌──────────────────┐
         │ Circuit Breakers│ │ Paper Trading     │
         └────────────────┘  └──────────────────┘
         REQUIRES: Phase 4 (autonomous agent managing the champion)
         VALIDATES: 4+ weeks paper trading with stable/improving performance
                    and at least 1 successful self-healing cycle.
```

**Why this exact order:**

1. **Data first** -- every downstream component needs features. You cannot validate signals, train models, or run experiments without data flowing through the system.
2. **Signals before sandbox** -- the sandbox needs a model, and a model needs features from the signal layer. More importantly: validating signal quality early prevents building infrastructure around a worthless signal. If divergence signals have no alpha, the architecture is the same but the signal layer needs rethinking.
3. **Sandbox before agent** -- the agent's experiment loop depends on the sandbox for isolated training and evaluation. Building the agent without a sandbox means experiments have nowhere to run safely.
4. **Agent before execution** -- the agent is the core value proposition. A working agent with paper trading validation is infinitely more valuable than a live execution layer with a static model. The agent proves the self-healing thesis; execution is "just" connecting to a broker.
5. **Execution last** -- highest risk (real money) and depends on everything else working. Paper trading validation within this phase provides the final safety gate.

---

## Sources

- **HYDRA PRD** (`/Users/tristanfarmer/Documents/HYDRA/prd-get-shit-done.md`): Architecture diagram (Section 4), module specifications (Section 5), phase structure (Section 9), risk register (Section 10). HIGH confidence -- this is the authoritative design document.
- **HYDRA PROJECT.md** (`/Users/tristanfarmer/Documents/HYDRA/.planning/PROJECT.md`): Constraints (cheap Chinese LLM, IB for execution, Python 3.11+), decisions (single market first, PRD architecture followed to the letter). HIGH confidence.
- **Architectural patterns** (deterministic envelope, event bus, champion-challenger, abstract model interface, circuit breaker state machine, config-as-code): Established software engineering and MLOps patterns. MEDIUM confidence -- patterns are well-known but specific library APIs should be verified during implementation phases.
- **LLM integration patterns** (structured output validation, fallback to rule-based, token budget management): Based on general agent architecture patterns from training data. MEDIUM confidence -- specific cheap Chinese LLM API structured output capabilities need validation during Phase 4 research.
- **Library-specific claims** (TimescaleDB hypertables, MLflow Model Registry, ib_insync, ADWIN/CUSUM drift detection): Based on training data knowledge. MEDIUM confidence -- version-specific APIs should be verified via official docs during each implementation phase.

**What could not be verified (web tools unavailable during research):**
- Current LangGraph architecture and suitability for agent loop (needs Phase 4 research)
- Current MLflow Model Registry API for champion/challenger workflows
- TimescaleDB current Python client library status and feature set
- Specific cheap Chinese LLM API providers structured output reliability comparison
- ib_insync current maintenance status and IB API compatibility
- ADWIN/CUSUM implementations in river or alibi-detect libraries

---

*Architecture research for: HYDRA -- Autonomous Self-Healing ML Trading Agent*
*Researched: 2026-02-18*
