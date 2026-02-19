# Phase 4: Agent Core + LLM Integration (Multi-Headed Architecture) - Research

**Researched:** 2026-02-19
**Domain:** Autonomous agent loop, LLM structured output, multi-head hypothesis generation and tournament evaluation
**Confidence:** MEDIUM (structured output reliability with reasoning models is actively evolving; financial domain prompt engineering is emergent)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Research Head Scope**: Full autonomy -- Research Head can discover, test, and integrate new signals without human approval as long as they pass fitness thresholds in the sandbox. User gets notified after integration, not before.
- **Broad data sources**: Financial APIs (USDA, CFTC, Fed), political signals (congressional STOCK Act trades, lobbying, committee hearings), AND macro cycles (commodity supercycles, El Nino/seasonal, historical regime data). Start with all three categories.
- **Cycle detection**: Use BOTH historical decomposition (extracting seasonal/cyclical components from price data) AND external event calendars (planting dates, USDA schedule, El Nino forecasts) combined for richer cyclical features.
- **Cost-optimized multi-model routing**: DO NOT use a reasoning model for tasks that don't require reasoning. Route each task to the cheapest capable model. Key principle: minimize loss to LLM API costs.
- **Models to research**: DeepSeek-R1 (reasoning), Kimi K2.5, GLM-4, Qwen 2.5/3, Claude Sonnet 4.6 (high-end option alongside DeepSeek). Research the optimal combination for different task types.
- **Dynamic budget**: Cap at $20/day but adjust dynamically based on results. If heads are producing winners, allocate more budget. If heads are producing garbage, reduce frequency and budget.
- **Research-driven approach to head communication**: Determine through research what balance between competition and collaboration is most likely to succeed. Avoid echo chambers AND hallucination amplification.

### Claude's Discretion
- Web search implementation for Research Head (pre-configured APIs vs web search vs hybrid)
- Fallback chain model sequence and retry logic
- Head communication pattern (the research should determine the optimal balance)
- Structured output format (Pydantic models, JSON schema, etc.)
- Tournament bracket vs ranked list for arena competition
- Head reputation scoring algorithm

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

## Summary

Phase 4 builds the autonomous self-healing agent system on top of the Phase 3 sandbox infrastructure. The core architecture is a state machine loop (observe -> diagnose -> hypothesize -> experiment -> evaluate) powered by LLM calls for diagnosis and hypothesis generation, with structured Pydantic-validated output at every step. The multi-headed design dispatches drift diagnoses to three specialized heads (Technical, Research, Structural) that compete through a sandbox tournament to produce winning improvements.

The critical technical risk is structured output reliability from reasoning models. Together AI now supports JSON schema-constrained output from DeepSeek-R1, but the `<think>` token reasoning phase adds complexity -- the JSON schema applies only to the output portion after `</think>`. Fallback chains (DeepSeek-R1 -> Qwen3/DeepSeek-V3.1 -> rule-based) with the Instructor library provide retry-on-validation-failure resilience. Cost optimization through task-aware model routing is essential: reasoning tasks (diagnosis, hypothesis) use DeepSeek-R1 ($3.00/$7.00 per 1M tokens on Together AI), while classification and formatting tasks use Qwen3-235B ($0.20/$0.60) or DeepSeek-V3.1 ($0.60/$1.70), achieving 80%+ cost reduction on non-reasoning tasks.

For the Research Head, pre-configured API integrations (FMP for congressional trades, USDA ERS for WASDE/crop data, NOAA for ENSO/seasonal) are safer and more reliable than live web search. Web search should be reserved as a discovery fallback for novel signal types. Semantic deduplication via `all-MiniLM-L6-v2` embeddings with cosine similarity > 0.85 rejection threshold prevents degenerate experiment loops. The tournament system should use a ranked-list approach (simpler than bracket) since head counts are small (3-5), with top-K advancing to full sandbox evaluation.

**Primary recommendation:** Build the single-head agent loop first as a complete, tested subsystem (AGNT-01 through AGNT-10), then layer multi-head coordination on top. Use `instructor` library with Together AI's OpenAI-compatible API for structured output with automatic retry. Route tasks to cheapest capable model via a simple task-type classifier.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| openai | >=1.60 | OpenAI-compatible API client for Together AI, DeepSeek, Anthropic | Standard interface; Together AI and DeepSeek both expose OpenAI-compatible endpoints |
| instructor | >=1.8 | Structured output extraction with Pydantic validation and automatic retry | 3M+ monthly downloads; handles retry-on-parse-failure natively; supports DeepSeek, Together AI, Anthropic |
| pydantic | >=2.7 | Schema definition for all LLM inputs/outputs | Already transitive dependency; defines type-safe contracts between agent components |
| sentence-transformers | >=3.0 | Semantic embedding for hypothesis deduplication | Industry standard for text similarity; `all-MiniLM-L6-v2` is 22M params, runs on CPU in <10ms |
| anthropic | >=0.40 | Claude Sonnet 4.6 API client (high-end fallback) | Official SDK; needed only if Claude is in the fallback chain |
| httpx | >=0.27 | HTTP client for Research Head data source APIs | Async-capable, already a transitive dependency of openai |
| statsmodels | >=0.14 | MSTL/STL seasonal decomposition for cycle detection | Standard for time series decomposition; MSTL handles multi-seasonal data |
| apscheduler | >=3.10 | Head scheduling (Technical every cycle, Research weekly, Structural monthly) | Already in dependencies; cron-like scheduling |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tenacity | >=8.2 | Retry logic for API calls beyond Instructor's built-in retry | Complex fallback chains with exponential backoff, circuit-breaking on provider outages |
| cot-reports | (existing) | CFTC COT data download | Already in dependencies; Research Head extends existing COT pipeline |
| wasdeparser | >=1.0 | USDA WASDE report parsing | Research Head: agricultural supply/demand signal extraction |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| instructor | pydantic-ai | pydantic-ai is newer (Pydantic team), but instructor has wider adoption and explicit DeepSeek support; instructor is safer for now |
| instructor | raw openai + manual JSON parsing | Loses automatic retry-on-validation, Pydantic integration; not worth the manual work |
| sentence-transformers | openai embeddings API | Would add API cost and latency for something that runs locally in <10ms |
| statsmodels MSTL | Prophet | Prophet is heavier dependency, overkill for decomposition-only use case |
| httpx | requests | httpx supports async natively; better for non-blocking API calls |

**Installation:**
```bash
uv add openai instructor sentence-transformers anthropic tenacity statsmodels httpx wasdeparser
```

## Architecture Patterns

### Recommended Project Structure
```
src/hydra/
├── agent/                     # Phase 4: Agent core
│   ├── __init__.py
│   ├── loop.py                # Main agent loop state machine (AGNT-01)
│   ├── diagnostician.py       # Structured triage: data audit, SHAP, regime, overfitting (AGNT-02)
│   ├── hypothesis.py          # Mutation playbook + hypothesis engine (AGNT-03)
│   ├── experiment_runner.py   # Isolated subprocess candidate training (AGNT-04)
│   ├── autonomy.py            # Autonomy levels + action gating (AGNT-06)
│   ├── rollback.py            # Hysteresis-based rollback triggers (AGNT-07)
│   ├── promotion.py           # 3-of-5 window promotion logic (AGNT-08)
│   ├── dedup.py               # Semantic deduplication + cooldowns (AGNT-09)
│   ├── budget.py              # Mutation budgets + dynamic allocation (AGNT-09, AGNT-17)
│   ├── coordinator.py         # Head Coordinator: dispatch + collection (AGNT-11)
│   ├── arena.py               # Tournament ranking of competing hypotheses (AGNT-15)
│   └── scheduler.py           # Head scheduling: frequency + budget (AGNT-18)
├── agent/heads/               # Specialized heads
│   ├── __init__.py
│   ├── base.py                # Abstract head interface
│   ├── technical.py           # Hyperparameter/feature/architecture mutations (AGNT-12)
│   ├── research.py            # New signal discovery via APIs + web (AGNT-13)
│   └── structural.py          # Ensemble/target/interaction proposals (AGNT-14)
├── agent/llm/                 # LLM integration layer
│   ├── __init__.py
│   ├── client.py              # Multi-provider LLM client with fallback (AGNT-05)
│   ├── router.py              # Task-type -> model routing for cost optimization
│   ├── schemas.py             # All Pydantic schemas for LLM I/O
│   └── prompts.py             # Prompt templates for each task type
├── agent/research/            # Research Head data sources
│   ├── __init__.py
│   ├── congressional.py       # Senate STOCK Act trading data (FMP API)
│   ├── usda.py                # WASDE + crop reports (USDA ERS API)
│   ├── seasonal.py            # ENSO/El Nino + planting calendars (NOAA)
│   ├── cycles.py              # Supercycle detection via MSTL decomposition
│   └── proposals.py           # Proposal system: new signal -> sandbox validation (AGNT-16)
```

### Pattern 1: Agent Loop State Machine
**What:** Fixed-state agent loop with explicit state transitions and journal logging at each step
**When to use:** The core observe-diagnose-hypothesize-experiment-evaluate cycle
**Example:**
```python
# src/hydra/agent/loop.py
from enum import Enum
from dataclasses import dataclass
from hydra.sandbox.journal import ExperimentJournal, ExperimentRecord
from hydra.sandbox.observer import DriftObserver, DriftReport
from hydra.agent.autonomy import AutonomyLevel, check_permission

class AgentPhase(Enum):
    OBSERVE = "observe"
    DIAGNOSE = "diagnose"
    HYPOTHESIZE = "hypothesize"
    EXPERIMENT = "experiment"
    EVALUATE = "evaluate"
    IDLE = "idle"

@dataclass
class AgentState:
    phase: AgentPhase
    drift_report: DriftReport | None = None
    diagnosis: "DiagnosisResult | None" = None
    hypothesis: "Hypothesis | None" = None
    experiment_result: "ExperimentResult | None" = None

class AgentLoop:
    """Single-head agent loop -- the foundation for multi-head."""

    def __init__(
        self,
        observer: DriftObserver,
        diagnostician: "Diagnostician",
        hypothesis_engine: "HypothesisEngine",
        experiment_runner: "ExperimentRunner",
        evaluator: "CompositeEvaluator",
        journal: ExperimentJournal,
        autonomy: AutonomyLevel = AutonomyLevel.SUPERVISED,
    ):
        self.observer = observer
        self.diagnostician = diagnostician
        self.hypothesis_engine = hypothesis_engine
        self.experiment_runner = experiment_runner
        self.evaluator = evaluator
        self.journal = journal
        self.autonomy = autonomy
        self.state = AgentState(phase=AgentPhase.IDLE)

    def step(self) -> AgentPhase:
        """Execute one step of the agent loop."""
        if self.state.phase == AgentPhase.IDLE:
            # OBSERVE: Check for drift
            report = self.observer.get_full_report(...)
            if report.needs_diagnosis:
                self.state = AgentState(
                    phase=AgentPhase.DIAGNOSE,
                    drift_report=report,
                )
            return self.state.phase

        elif self.state.phase == AgentPhase.DIAGNOSE:
            diagnosis = self.diagnostician.diagnose(self.state.drift_report)
            self.state.diagnosis = diagnosis
            self.state.phase = AgentPhase.HYPOTHESIZE
            return self.state.phase

        # ... continue through phases
```

### Pattern 2: LLM Client with Fallback Chain
**What:** Multi-provider LLM client using instructor for structured output, with automatic fallback on parse failure
**When to use:** Every LLM call in the system
**Example:**
```python
# src/hydra/agent/llm/client.py
import instructor
from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

class LLMClient:
    """Multi-provider LLM client with fallback chain and cost tracking."""

    def __init__(self, config: dict):
        # Together AI client (hosts DeepSeek-R1, Qwen, GLM)
        self.together = instructor.from_openai(
            OpenAI(
                api_key=config["together_api_key"],
                base_url="https://api.together.ai/v1",
            ),
        )
        # DeepSeek direct (cheaper for non-reasoning)
        self.deepseek = instructor.from_openai(
            OpenAI(
                api_key=config["deepseek_api_key"],
                base_url="https://api.deepseek.com",
            ),
        )
        self._daily_cost = 0.0
        self._daily_budget = config.get("daily_budget", 20.0)

    def call(
        self,
        response_model: type[BaseModel],
        messages: list[dict],
        task_type: str = "reasoning",
        max_retries: int = 3,
    ) -> BaseModel:
        """Call LLM with structured output, fallback chain, cost tracking."""
        chain = self._get_fallback_chain(task_type)
        last_error = None

        for provider, model in chain:
            try:
                client = getattr(self, provider)
                result = client.chat.completions.create(
                    model=model,
                    response_model=response_model,
                    messages=messages,
                    max_retries=max_retries,  # instructor handles retry
                )
                self._track_cost(provider, model, messages, result)
                return result
            except Exception as e:
                last_error = e
                continue

        # All LLM providers failed -- fall back to rule-based
        raise LLMUnavailableError(f"All providers failed: {last_error}")

    def _get_fallback_chain(self, task_type: str) -> list[tuple[str, str]]:
        """Route task to cheapest capable model sequence."""
        if task_type == "reasoning":
            return [
                ("together", "deepseek-ai/DeepSeek-R1-0528"),
                ("together", "Qwen/Qwen3-235B-A22B-Instruct"),
                # rule-based fallback handled by caller
            ]
        elif task_type == "classification":
            return [
                ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
                ("deepseek", "deepseek-chat"),  # DeepSeek-V3
            ]
        elif task_type == "formatting":
            return [
                ("together", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
            ]
        else:  # general
            return [
                ("together", "deepseek-ai/DeepSeek-V3.1"),
                ("together", "Qwen/Qwen3-235B-A22B-Instruct"),
            ]
```

### Pattern 3: Task-Type Model Router
**What:** Route each LLM task to the cheapest model capable of handling it
**When to use:** Every LLM invocation -- the router sits between agent code and LLM client
**Task type classification:**

| Task Type | Requires Reasoning | Model Tier | Together AI Cost (in/out per 1M) |
|-----------|-------------------|------------|----------------------------------|
| Diagnosis (root cause analysis from drift report) | YES | DeepSeek-R1 | $3.00/$7.00 |
| Hypothesis generation (propose mutations from diagnosis) | YES | DeepSeek-R1 | $3.00/$7.00 |
| Research signal discovery (novel data source proposals) | YES | DeepSeek-R1 or Claude Sonnet 4.6 | $3.00/$7.00 or $3.00/$15.00 |
| Drift classification (categorize drift type) | NO | Qwen2.5-7B-Turbo | $0.30/$0.30 |
| Hypothesis formatting (structure raw idea into playbook format) | NO | DeepSeek-V3.1 | $0.60/$1.70 |
| Deduplication check (is this hypothesis novel?) | NO | Embedding only (no LLM) | $0 (local) |
| Journal summary (summarize experiment results) | NO | Qwen2.5-7B-Turbo | $0.30/$0.30 |
| Reputation update (compute head scores) | NO | Pure computation | $0 |

**Estimated daily cost at full utilization:**
- Technical Head (daily): ~2 reasoning calls + 3 classification = ~$0.30/day
- Research Head (weekly): ~3 reasoning calls + 2 formatting = ~$0.15/day averaged
- Structural Head (monthly): ~2 reasoning calls + 1 formatting = ~$0.03/day averaged
- Coordinator overhead: ~1 classification per cycle = ~$0.05/day
- **Total estimated: $1-5/day typical, well under $20 cap**

### Pattern 4: Head Communication -- Competitive Independence with Shared Memory
**What:** Heads operate independently per round but share the experiment journal as collective memory
**Recommendation:** Use **competitive independence** -- each head sees the same drift diagnosis but generates hypotheses without seeing other heads' current-round proposals. The experiment journal serves as shared memory of what has been tried (and what worked/failed), preventing repetition without creating echo chambers.

**Why this prevents both failure modes:**
- **No echo chambers:** Heads never see each other's current proposals, so they cannot copy or converge on the same idea. Each head's approach is architecturally different (hyperparams vs. new data vs. ensemble structure).
- **No hallucination amplification:** Since heads do not build on each other's outputs within a round, a hallucinated proposal from one head cannot propagate to others. The sandbox evaluation catches hallucination-driven bad proposals before they affect the system.
- **Shared journal prevents repetition:** Before generating a proposal, each head queries the journal for recent experiments of its type and the deduplication engine rejects semantically similar hypotheses (cosine similarity > 0.85).

### Pattern 5: Autonomy Level Gating
**What:** Four-level autonomy system that gates agent actions
**Example:**
```python
# src/hydra/agent/autonomy.py
from enum import IntEnum

class AutonomyLevel(IntEnum):
    LOCKDOWN = 0    # No agent actions; manual only
    SUPERVISED = 1  # Agent proposes, human approves
    SEMI_AUTO = 2   # Agent executes, human approves promotion
    AUTONOMOUS = 3  # Full autonomous operation

# Action permission matrix
PERMISSIONS = {
    "observe":    {AutonomyLevel.LOCKDOWN: False, AutonomyLevel.SUPERVISED: True,  AutonomyLevel.SEMI_AUTO: True,  AutonomyLevel.AUTONOMOUS: True},
    "diagnose":   {AutonomyLevel.LOCKDOWN: False, AutonomyLevel.SUPERVISED: True,  AutonomyLevel.SEMI_AUTO: True,  AutonomyLevel.AUTONOMOUS: True},
    "hypothesize":{AutonomyLevel.LOCKDOWN: False, AutonomyLevel.SUPERVISED: True,  AutonomyLevel.SEMI_AUTO: True,  AutonomyLevel.AUTONOMOUS: True},
    "experiment": {AutonomyLevel.LOCKDOWN: False, AutonomyLevel.SUPERVISED: False, AutonomyLevel.SEMI_AUTO: True,  AutonomyLevel.AUTONOMOUS: True},
    "promote":    {AutonomyLevel.LOCKDOWN: False, AutonomyLevel.SUPERVISED: False, AutonomyLevel.SEMI_AUTO: False, AutonomyLevel.AUTONOMOUS: True},
    "rollback":   {AutonomyLevel.LOCKDOWN: False, AutonomyLevel.SUPERVISED: True,  AutonomyLevel.SEMI_AUTO: True,  AutonomyLevel.AUTONOMOUS: True},
}

def check_permission(action: str, level: AutonomyLevel) -> bool:
    return PERMISSIONS.get(action, {}).get(level, False)
```

### Pattern 6: Hysteresis-Based Rollback
**What:** Rollback triggers with hysteresis to prevent flapping between champion and candidate
**When to use:** Automatic rollback on sustained performance degradation
**Example:**
```python
# src/hydra/agent/rollback.py
from dataclasses import dataclass

@dataclass
class RollbackConfig:
    degradation_threshold: float = 0.15  # 15% composite fitness drop
    sustained_periods: int = 3           # Must be degraded for 3 consecutive checks
    recovery_periods: int = 5            # Must be healthy for 5 periods before re-arming
    cooldown_after_rollback: int = 10    # Wait 10 periods after rollback before new experiments

class HysteresisRollbackTrigger:
    """Prevents flapping by requiring sustained degradation before rollback
    and sustained recovery before re-arming."""

    def __init__(self, config: RollbackConfig | None = None):
        self.config = config or RollbackConfig()
        self._degraded_count = 0
        self._healthy_count = 0
        self._armed = True  # Whether trigger is active
        self._cooldown_remaining = 0

    def update(self, current_fitness: float, champion_fitness: float) -> bool:
        """Returns True if rollback should execute."""
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        drop = (champion_fitness - current_fitness) / champion_fitness
        if drop > self.config.degradation_threshold:
            self._degraded_count += 1
            self._healthy_count = 0
        else:
            self._healthy_count += 1
            self._degraded_count = 0

        # Trigger rollback after sustained degradation
        if self._armed and self._degraded_count >= self.config.sustained_periods:
            self._armed = False
            self._cooldown_remaining = self.config.cooldown_after_rollback
            self._degraded_count = 0
            return True

        # Re-arm after sustained recovery
        if not self._armed and self._healthy_count >= self.config.recovery_periods:
            self._armed = True

        return False
```

### Anti-Patterns to Avoid
- **LLM in the hot path:** The LLM never touches the prediction pipeline. It proposes mutations (config changes, new features, structural changes) that deterministic code executes. The trained LightGBM model runs predictions; the LLM just decides what to try next.
- **Unbounded experiment loops:** Without mutation budgets and deduplication, the agent can enter degenerate loops trying the same or similar mutations endlessly. Always enforce per-head budgets, cooldowns, and semantic dedup.
- **Synchronous LLM calls in the agent loop:** API calls can timeout or fail. Always use the fallback chain pattern with timeouts. Never let a single provider outage halt the entire agent loop.
- **Coupling heads to each other:** Heads should be independently deployable units. The coordinator dispatches to them; they do not call each other. This ensures one head's failure doesn't cascade.
- **Stateful LLM interactions:** Each LLM call should be stateless (full context in the prompt). Do not rely on conversation history or threading -- it creates fragile dependencies on provider state.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Structured LLM output | Custom JSON parser with regex | `instructor` library | Handles retry-on-validation, Pydantic integration, multi-provider support; 3M+ downloads |
| Text embedding similarity | TF-IDF + cosine similarity | `sentence-transformers` with `all-MiniLM-L6-v2` | Pre-trained semantic embeddings outperform bag-of-words; 22M params runs on CPU |
| API retry with backoff | Custom retry loops | `tenacity` library | Battle-tested exponential backoff, jitter, circuit-breaking |
| Seasonal decomposition | Custom FFT-based cycle finder | `statsmodels.tsa.seasonal.MSTL` | Multi-seasonal support, robust to outliers, lowest RMSE of decomposition methods |
| WASDE report parsing | Custom HTML/PDF scraper | `wasdeparser` library | Purpose-built for USDA data format changes |
| Cron-like scheduling | Custom timer threads | `apscheduler` (already in deps) | Handles missed executions, timezone awareness, persistent job store |

**Key insight:** The agent core's value is in the orchestration logic (which mutations to try, when to rollback, how to rank hypotheses). Every commodity subtask (parsing, embedding, scheduling, retrying) has a battle-tested library that handles edge cases you would miss in a hand-rolled version.

## Common Pitfalls

### Pitfall 1: DeepSeek-R1 Structured Output with `<think>` Tokens
**What goes wrong:** DeepSeek-R1 emits `<think>...</think>` reasoning tokens before the JSON output. Naive JSON parsers fail because they try to parse the reasoning prefix.
**Why it happens:** Reasoning models have a distinct thinking phase; JSON schema constraining applies only to the output after `</think>`.
**How to avoid:** Use Together AI's native structured output support (response_format with json_schema), which handles the think/output separation automatically. The `instructor` library with Together AI already handles this. When using DeepSeek's direct API, use `json_object` mode and explicitly instruct the model to emit JSON after thinking.
**Warning signs:** Parse failures that contain `<think>` in the error message; empty `content` responses from DeepSeek API.

### Pitfall 2: Cost Explosion from Reasoning Models on Simple Tasks
**What goes wrong:** Using DeepSeek-R1 ($3.00/$7.00 per 1M tokens) for tasks that Qwen2.5-7B ($0.30/$0.30) handles perfectly.
**Why it happens:** Developers default to the most capable model without task classification.
**How to avoid:** Implement the task-type router (Pattern 3). Classification, formatting, and summarization tasks do not need reasoning traces. Only diagnosis and hypothesis generation benefit from R1's chain-of-thought.
**Warning signs:** Daily costs consistently near the $20 cap without proportional improvement in head success rates.

### Pitfall 3: Degenerate Experiment Loops
**What goes wrong:** The agent repeatedly proposes the same or very similar mutations, wasting compute and API budget.
**Why it happens:** Without deduplication, the LLM may re-propose previously tried ideas, especially when prompted with the same drift diagnosis.
**How to avoid:** Three-layer defense: (1) Semantic dedup via `all-MiniLM-L6-v2` embeddings rejects hypotheses with cosine similarity > 0.85 to any hypothesis tried in the last 30 days. (2) Per-head mutation budgets limit experiments per cycle. (3) Cooldown timers prevent the same mutation type from being retried within a configurable window (e.g., 7 days for hyperparameter mutations).
**Warning signs:** Journal shows many rejected experiments with high similarity scores; same mutation type appearing in consecutive cycles.

### Pitfall 4: Flapping Between Champion and Candidate
**What goes wrong:** A candidate beats the champion, gets promoted, then performs worse in the next evaluation, triggering rollback. The old champion gets re-promoted, and the cycle repeats.
**Why it happens:** Noisy evaluation on a single window. Small differences between champion and candidate are within measurement noise.
**How to avoid:** Require the candidate to beat the champion on composite fitness across 3 of 5 independent evaluation windows (AGNT-08). Add hysteresis to rollback triggers (sustained degradation over N periods, not a single bad period). Post-rollback cooldown prevents immediate re-experimentation.
**Warning signs:** Rapidly alternating promotions/rollbacks in the experiment journal; rollback count exceeding promotion count.

### Pitfall 5: LLM Hallucination in Financial Domain
**What goes wrong:** The LLM proposes mutations or data sources that sound plausible but are nonsensical (e.g., suggesting a feature that requires future data, proposing a data source that doesn't exist).
**Why it happens:** LLMs are trained on general text; financial domain reasoning requires specific constraints.
**How to avoid:** Every LLM output must be validated against a Pydantic schema that encodes domain constraints. The hypothesis schema should require mutation_type from a known enum, config_diff that references only existing config keys, and a testable prediction. The sandbox evaluation catches proposals that fail to improve fitness. The Research Head's proposals are validated through sandbox testing before integration (AGNT-16).
**Warning signs:** Hypotheses that reference nonexistent features, parameters outside valid ranges, or data sources that return errors.

### Pitfall 6: Research Head Data Source Reliability
**What goes wrong:** APIs for congressional trades, USDA data, or ENSO forecasts change format, rate-limit, or go offline, causing the Research Head to fail.
**Why it happens:** External APIs are outside HYDRA's control; government data portals have downtime.
**How to avoid:** Pre-configured API integrations with local caching (24-hour cache for frequently accessed data). Graceful degradation: if a data source is unavailable, the Research Head skips that signal category and logs a warning rather than failing the entire cycle. Rate-limit headers respected; exponential backoff on 429 responses.
**Warning signs:** Increasing error rates from specific APIs; data staleness alerts; Research Head proposals declining in quality.

## Code Examples

### Pydantic Schemas for LLM I/O
```python
# src/hydra/agent/llm/schemas.py
from enum import Enum
from pydantic import BaseModel, Field

class DriftCategory(str, Enum):
    PERFORMANCE = "performance_degradation"
    FEATURE_DRIFT = "feature_distribution_drift"
    REGIME_CHANGE = "regime_change"
    OVERFITTING = "overfitting"
    DATA_QUALITY = "data_quality_issue"

class DiagnosisResult(BaseModel):
    """Structured output from LLM diagnosis of drift report."""
    primary_cause: DriftCategory = Field(
        description="The most likely root cause of the detected drift"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the diagnosis (0-1)"
    )
    evidence: list[str] = Field(
        description="Specific evidence from the drift report supporting this diagnosis"
    )
    recommended_mutation_types: list[str] = Field(
        description="Mutation types from the playbook that address this root cause"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning chain explaining the diagnosis"
    )

class MutationType(str, Enum):
    HYPERPARAMETER = "hyperparameter"
    FEATURE_ADD = "feature_add"
    FEATURE_REMOVE = "feature_remove"
    FEATURE_ENGINEERING = "feature_engineering"
    ENSEMBLE_METHOD = "ensemble_method"
    PREDICTION_TARGET = "prediction_target"
    NEW_DATA_SIGNAL = "new_data_signal"

class Hypothesis(BaseModel):
    """Structured hypothesis proposed by a head."""
    mutation_type: MutationType
    description: str = Field(
        description="Human-readable description of the proposed change"
    )
    config_diff: dict = Field(
        description="Specific config changes to apply (key: new_value)"
    )
    expected_impact: str = Field(
        description="Predicted impact on composite fitness"
    )
    testable_prediction: str = Field(
        description="A specific, falsifiable prediction about the experiment outcome"
    )
    head_name: str = Field(
        description="Which head generated this hypothesis"
    )

class TournamentEntry(BaseModel):
    """A hypothesis with its sandbox evaluation score."""
    hypothesis: Hypothesis
    fitness_score: float
    evaluation_windows_won: int = Field(ge=0, le=5)
    promoted: bool = False
```

### Mutation Playbook Design
```python
# src/hydra/agent/hypothesis.py
# The playbook maps diagnosed root causes to candidate mutations

MUTATION_PLAYBOOK: dict[str, list[dict]] = {
    "performance_degradation": [
        {
            "type": "hyperparameter",
            "name": "reduce_learning_rate",
            "config_diff": {"learning_rate": "current * 0.5"},
            "rationale": "Slower learning may reduce overshoot in changed regime",
        },
        {
            "type": "hyperparameter",
            "name": "increase_regularization",
            "config_diff": {"reg_alpha": "current * 2", "reg_lambda": "current * 2"},
            "rationale": "Stronger regularization combats overfitting to recent data",
        },
        {
            "type": "feature_remove",
            "name": "drop_low_importance_features",
            "config_diff": {"remove_features": "bottom_3_by_shap"},
            "rationale": "Noisy features may be driving spurious predictions",
        },
    ],
    "feature_distribution_drift": [
        {
            "type": "hyperparameter",
            "name": "shorten_training_window",
            "config_diff": {"training_window_days": "current * 0.7"},
            "rationale": "Focus on recent regime where feature distributions match",
        },
        {
            "type": "feature_engineering",
            "name": "add_rolling_z_scores",
            "config_diff": {"add_features": ["z_score_30d", "z_score_90d"]},
            "rationale": "Z-scored features are distribution-invariant",
        },
    ],
    "regime_change": [
        {
            "type": "ensemble_method",
            "name": "add_regime_conditioning",
            "config_diff": {"ensemble_type": "regime_conditional"},
            "rationale": "Separate models for different regimes",
        },
    ],
    "overfitting": [
        {
            "type": "hyperparameter",
            "name": "reduce_num_leaves",
            "config_diff": {"num_leaves": "max(8, current // 2)"},
            "rationale": "Simpler trees generalize better",
        },
        {
            "type": "hyperparameter",
            "name": "increase_min_child_samples",
            "config_diff": {"min_child_samples": "current * 2"},
            "rationale": "Require more samples per leaf to reduce memorization",
        },
    ],
    "data_quality_issue": [
        {
            "type": "feature_remove",
            "name": "remove_degraded_features",
            "config_diff": {"remove_features": "features_with_high_psi"},
            "rationale": "Remove features whose data quality has degraded",
        },
    ],
}
```

### Semantic Deduplication
```python
# src/hydra/agent/dedup.py
from sentence_transformers import SentenceTransformer
import numpy as np

class HypothesisDeduplicator:
    """Reject hypotheses semantically similar to recently tried ones."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        self._recent_embeddings: list[np.ndarray] = []
        self._recent_descriptions: list[str] = []

    def is_duplicate(self, hypothesis_description: str) -> bool:
        """Check if hypothesis is too similar to a recent one."""
        if not self._recent_embeddings:
            return False

        new_embedding = self.model.encode([hypothesis_description])[0]
        recent_matrix = np.stack(self._recent_embeddings)

        # Cosine similarity
        similarities = np.dot(recent_matrix, new_embedding) / (
            np.linalg.norm(recent_matrix, axis=1) * np.linalg.norm(new_embedding)
        )

        return float(np.max(similarities)) > self.threshold

    def register(self, hypothesis_description: str) -> None:
        """Add a hypothesis to the dedup memory."""
        embedding = self.model.encode([hypothesis_description])[0]
        self._recent_embeddings.append(embedding)
        self._recent_descriptions.append(hypothesis_description)

    def load_from_journal(
        self, journal: "ExperimentJournal", days: int = 30
    ) -> None:
        """Load recent hypotheses from experiment journal."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        recent = journal.query(date_from=cutoff)
        for record in recent:
            self.register(record.hypothesis)
```

### Head Reputation Scoring
```python
# src/hydra/agent/budget.py
from dataclasses import dataclass, field

@dataclass
class HeadReputation:
    """Track head success rate and adjust budget accordingly."""
    name: str
    total_proposals: int = 0
    promoted_count: int = 0
    rejected_count: int = 0
    # Exponential moving average of success rate
    ema_success_rate: float = 0.5  # Start at 50% (neutral prior)
    ema_alpha: float = 0.3  # Smoothing factor

    @property
    def success_rate(self) -> float:
        if self.total_proposals == 0:
            return 0.5  # Neutral prior
        return self.promoted_count / self.total_proposals

    def record_outcome(self, promoted: bool) -> None:
        self.total_proposals += 1
        if promoted:
            self.promoted_count += 1
        else:
            self.rejected_count += 1
        # Update EMA
        outcome = 1.0 if promoted else 0.0
        self.ema_success_rate = (
            self.ema_alpha * outcome
            + (1 - self.ema_alpha) * self.ema_success_rate
        )

    def budget_multiplier(self) -> float:
        """Budget allocation multiplier based on reputation.

        Range: [0.25, 2.0] -- struggling heads get reduced budget,
        successful heads get amplified budget.
        """
        return max(0.25, min(2.0, self.ema_success_rate * 2.0))
```

### Research Head Data Source Integration
```python
# src/hydra/agent/research/congressional.py
import httpx
from pydantic import BaseModel
from datetime import date

class CongressionalTrade(BaseModel):
    senator: str
    transaction_date: date
    ticker: str
    asset_description: str
    transaction_type: str  # "Purchase" | "Sale" | "Exchange"
    amount_range: str  # e.g. "$1,001 - $15,000"
    committees: list[str]  # Senator's committee memberships

class CongressionalTradesFetcher:
    """Fetch Senate STOCK Act trading data from Financial Modeling Prep API."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.Client(timeout=30.0)

    def get_recent_senate_trades(
        self, days: int = 30
    ) -> list[CongressionalTrade]:
        """Fetch recent Senate trading disclosures."""
        resp = self.client.get(
            f"{self.BASE_URL}/senate-trading",
            params={"apikey": self.api_key},
        )
        resp.raise_for_status()
        # Filter and parse into domain objects
        # ... (implementation depends on exact API response format)

    def get_agriculture_committee_trades(self) -> list[CongressionalTrade]:
        """Filter trades by Agriculture Committee members.

        This is the key signal: senators on ag committees trading
        commodity-related assets before COT reports.
        """
        trades = self.get_recent_senate_trades(days=90)
        ag_tickers = {"ADM", "BG", "CTVA", "DE", "FMC", "MOS", "NTR", "CF"}
        return [
            t for t in trades
            if t.ticker in ag_tickers
            or "Agriculture" in (t.committees or [])
        ]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual JSON parsing from LLM | Constrained decoding via JSON schema in API | 2024-2025 | Parse failure rate drops from ~15% to <2% with schema constraints |
| Single-model LLM deployment | Multi-model routing by task complexity | 2025 | 60-85% cost reduction; reasoning models only where needed |
| Fixed fallback chains | Dynamic routing with cost/quality tradeoff | 2025-2026 | Models like Qwen3-235B ($0.20/$0.60) approach GPT-4-level quality for structured output |
| Single-agent improvement loop | Multi-head competitive hypothesis generation | 2025 | Research shows diversity in proposals prevents premature convergence |
| Human-curated mutation playbook only | LLM-augmented playbook with domain reasoning | 2025 | LLMs can propose novel mutations not in the playbook, but require guardrails |
| DeepSeek-R1 (Jan 2025) | DeepSeek-R1-0528 (May 2025) | May 2025 | Significantly improved reasoning depth and reduced hallucination in structured output |

**Deprecated/outdated:**
- MLflow stages (Stage.PRODUCTION, etc.): Deprecated in favor of aliases (champion/archived) -- already handled in Phase 3
- DeepSeek `json_object` mode (basic): Superceded by `json_schema` mode with full schema validation on Together AI
- LangChain for agent orchestration: Explicitly out of scope per REQUIREMENTS.md; direct API calls with instructor are simpler and more debuggable

## Research Head: Data Source Recommendations

### Recommendation: Pre-Configured APIs as Primary, Web Search as Fallback

**Rationale:** Pre-configured APIs provide structured, reliable, versioned data that can be cached locally and tested deterministically. Web search is useful for discovering new signal ideas but introduces HTML parsing fragility, rate limiting, and content changes.

### Data Sources by Category

**Political Signals:**
| Source | API | Free Tier | Data Quality | Update Frequency |
|--------|-----|-----------|-------------|-----------------|
| Senate STOCK Act trades | Financial Modeling Prep `/senate-trading` | Yes (limited) | HIGH -- structured JSON | Daily |
| House trades | Financial Modeling Prep `/house-trading` | Yes (limited) | HIGH | Daily |
| Alternative: QuiverQuant | `quiverquant.com/congresstrading` | Free web data | MEDIUM | Daily |
| Alternative: Capitol Trades | `capitoltrades.com` | Free web data | MEDIUM | Daily |

**Agricultural/USDA:**
| Source | API | Free Tier | Data Quality | Update Frequency |
|--------|-----|-----------|-------------|-----------------|
| WASDE reports | USDA ERS Data API (REST, requires key) | Yes | HIGH -- official USDA | Monthly |
| WASDE parsing | `wasdeparser` Python library | N/A (local) | HIGH | Per-release |
| Crop progress | USDA NASS QuickStats API | Yes | HIGH | Weekly during season |
| CFTC COT | Already implemented (`cot-reports` library) | N/A | HIGH | Weekly (Friday 3:30 ET) |

**Seasonal/Macro Cycles:**
| Source | API | Free Tier | Data Quality | Update Frequency |
|--------|-----|-----------|-------------|-----------------|
| ENSO/ONI data | NOAA CPC (direct download, no API key) | Yes | HIGH -- official NOAA | Monthly |
| Historical SST | IRI ENSO forecast (Columbia) | Yes | HIGH | Monthly |
| Planting/harvest calendar | Static reference data (embedded) | N/A | HIGH | Annual update |

**Cycle Detection (Local Computation, No API):**
- `statsmodels.tsa.seasonal.MSTL` for multi-seasonal decomposition (handles 3-7 year, 10-year, seasonal cycles simultaneously)
- Applied to existing price data in the feature store -- no external API needed

## Fallback Chain: Recommended Design

```
Task: Reasoning (diagnosis, hypothesis generation)
  1. Together AI -> DeepSeek-R1-0528 (structured output via json_schema)
  2. Together AI -> Qwen3-235B-A22B-Instruct (structured output via json_schema)
  3. Rule-based fallback (curated playbook lookup by drift category)

Task: Classification (drift categorization, simple decisions)
  1. Together AI -> Qwen2.5-7B-Instruct-Turbo
  2. Together AI -> DeepSeek-V3.1
  3. Rule-based (threshold-based classification from drift report values)

Task: High-stakes research (novel signal proposals, complex analysis)
  1. Together AI -> DeepSeek-R1-0528
  2. Anthropic -> Claude Sonnet 4.6 (if budget allows and R1 fails)
  3. Rule-based (skip novel proposals, retry next cycle)
```

**Parse failure handling:** The `instructor` library retries with the validation error appended to the prompt (showing the model what went wrong). After `max_retries` exhausted on a provider, move to next in chain. After all LLM providers fail, fall back to rule-based. The rule-based fallback uses the mutation playbook directly: map drift category -> top-ranked mutation in playbook.

**Target: <5% parse failure rate** on the primary provider (DeepSeek-R1 via Together AI with json_schema). Based on research, Together AI's constrained decoding on reasoning models achieves <2% parse failure with properly defined schemas.

## Tournament Design: Ranked List (Not Bracket)

**Recommendation: Simple ranked list, not tournament bracket.**

**Rationale:** With only 3-5 heads and 1-3 proposals per head per cycle, a full bracket tournament is overengineered. A ranked list is simpler, more transparent, and sufficient.

**Process:**
1. All heads submit proposals to the Coordinator
2. Semantic dedup removes duplicates (cosine similarity > 0.85)
3. Remaining proposals enter the sandbox: each is trained and evaluated via MarketReplayEngine
4. Proposals are ranked by composite fitness score (using existing `CompositeEvaluator`)
5. Top-K (default K=3) advance to full 5-window evaluation
6. Any proposal beating the champion on 3 of 5 windows gets promoted (AGNT-08)
7. Results logged to experiment journal; head reputations updated

## Open Questions

1. **Claude Sonnet 4.6 in fallback chain: worth the cost?**
   - What we know: Claude Sonnet 4.6 costs $3/$15 per 1M tokens (output is 2x DeepSeek-R1). It has excellent structured output support.
   - What's unclear: Whether the quality improvement over DeepSeek-R1 justifies the 2x output cost for financial domain reasoning.
   - Recommendation: Include as optional high-end fallback for Research Head proposals only. Make it configurable; default to disabled. The dynamic budget can enable it when the Research Head is producing winners.

2. **Kimi K2.5 viability**
   - What we know: Kimi K2.5 (reasoning) costs $0.60/$3.00 per 1M tokens; released January 2026; multimodal.
   - What's unclear: Structured output reliability with Kimi; availability on Together AI; whether it adds value beyond Qwen3 for this use case.
   - Recommendation: Defer Kimi integration. DeepSeek-R1 + Qwen3 + DeepSeek-V3.1 cover the reasoning/classification spectrum. Add Kimi later if those providers prove unreliable.

3. **GLM-4.7 role**
   - What we know: GLM-4.7 costs $0.45/$2.00 on Together AI; supports structured output; 203K context.
   - What's unclear: Quality of financial domain reasoning vs Qwen3 and DeepSeek.
   - Recommendation: Consider as a middle-tier alternative to Qwen3 if needed. Not a priority for initial implementation.

4. **Optimal embedding similarity threshold for dedup**
   - What we know: Literature suggests 0.85-0.97 range. Higher thresholds miss paraphrased duplicates; lower thresholds reject valid novel ideas.
   - What's unclear: The right threshold for financial mutation hypotheses specifically.
   - Recommendation: Start at 0.85 (per AGNT-09 spec), log all similarity scores to the journal, and tune empirically based on false-positive/false-negative rates in the first month.

5. **Subprocess isolation for experiment runner**
   - What we know: AGNT-04 requires experiments run in isolation via subprocess with configurable timeout.
   - What's unclear: Whether `subprocess.Popen` with timeout is sufficient or if containerization (Docker) is needed for true isolation.
   - Recommendation: Start with `subprocess.run(timeout=N)` wrapping the training script. This provides process isolation and timeout. Docker adds unnecessary complexity for single-machine operation. If memory leaks become an issue, re-evaluate.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AGNT-01 | Agent runs full observe -> diagnose -> hypothesize -> experiment -> evaluate loop | Pattern 1 (Agent Loop State Machine); builds directly on Phase 3 DriftObserver, CompositeEvaluator, ExperimentJournal, ModelRegistry |
| AGNT-02 | Diagnostician performs structured triage (data audit, SHAP, regime check, overfitting test) | DiagnosisResult Pydantic schema; LLM-powered analysis with DriftReport as input; SHAP via existing LightGBM feature_importance |
| AGNT-03 | Hypothesis engine proposes mutations from curated playbook matched to diagnosed root causes | Mutation Playbook code example; MUTATION_PLAYBOOK dict maps DriftCategory -> candidate mutations; LLM augments with domain reasoning |
| AGNT-04 | Experiment runner executes candidate training in isolation (subprocess) with configurable timeout | Use `subprocess.run(timeout=N)` wrapping training script; Open Question #5 addresses isolation approach |
| AGNT-05 | LLM client with DeepSeek-R1 fallback chain and Pydantic validation | Pattern 2 (LLM Client); `instructor` library + Together AI OpenAI-compatible endpoint; fallback chain design in dedicated section |
| AGNT-06 | Autonomy levels (supervised, semi-auto, autonomous, lockdown) gate agent actions | Pattern 5 (Autonomy Level Gating); IntEnum with permission matrix; extends existing `cli/state.py` AgentState |
| AGNT-07 | Automatic rollback with hysteresis to prevent flapping | Pattern 6 (Hysteresis-Based Rollback); sustained_periods + recovery_periods + cooldown; builds on existing ModelRegistry.rollback() |
| AGNT-08 | Candidate must beat champion on 3 of 5 independent evaluation windows | Tournament Design section; uses existing CompositeEvaluator + MarketReplayEngine on 5 disjoint windows |
| AGNT-09 | Mutation budgets, semantic deduplication, and cooldowns | Semantic Dedup code example; `all-MiniLM-L6-v2` with 0.85 threshold; HeadReputation.budget_multiplier() for dynamic budgets |
| AGNT-10 | System operates in degraded mode when LLM unavailable | Fallback chain terminates at rule-based (playbook lookup); `LLMUnavailableError` triggers deterministic path |
| AGNT-11 | Head Coordinator dispatches diagnosis to multiple heads and collects hypotheses | `coordinator.py` dispatches DriftReport + DiagnosisResult to each head; collects Hypothesis objects; feeds to arena |
| AGNT-12 | Technical Head mutates hyperparameters, features, architecture from playbook | `heads/technical.py` wraps AGNT-03 hypothesis engine; uses MUTATION_PLAYBOOK with LLM augmentation |
| AGNT-13 | Research Head discovers new data signals via web search | `heads/research.py` + `agent/research/` module; FMP API for congressional trades, USDA ERS for WASDE, NOAA for ENSO; data source table in research |
| AGNT-14 | Structural Head proposes ensemble methods, prediction targets, feature interactions | `heads/structural.py`; distinct playbook section for ensemble_method, prediction_target, feature_engineering mutations |
| AGNT-15 | Arena tournament ranks competing hypotheses from all heads | Tournament Design section: ranked list, top-K -> full 5-window evaluation via MarketReplayEngine |
| AGNT-16 | Proposal system: heads propose toolkit expansions validated through sandbox | `agent/research/proposals.py`; new signal -> build feature pipeline -> sandbox test -> integrate if fitness improves |
| AGNT-17 | Head reputation scoring tracks success rates, adjusts budgets/frequency | HeadReputation code example; EMA success rate with alpha=0.3; budget_multiplier in [0.25, 2.0] |
| AGNT-18 | Head scheduling: Technical every cycle, Research weekly, Structural monthly | `agent/scheduler.py` using APScheduler; frequency adjusted by reputation multiplier |
</phase_requirements>

## Sources

### Primary (HIGH confidence)
- Together AI official docs: Structured Outputs -- https://docs.together.ai/docs/json-mode (verified models, pricing, JSON schema support for DeepSeek-R1)
- Together AI pricing page -- https://www.together.ai/pricing (verified Feb 2026 pricing for all models)
- DeepSeek API docs: JSON Output -- https://api-docs.deepseek.com/guides/json_mode (verified json_object mode, caveats about empty content)
- Instructor library docs: DeepSeek integration -- https://python.useinstructor.com/integrations/deepseek/ (verified setup, Pydantic models, MD_JSON mode for reasoning)
- sentence-transformers `all-MiniLM-L6-v2` -- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (verified 384-dim embeddings, production-ready)
- CFTC COT Release Schedule -- https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm (verified Friday 3:30 ET releases)
- USDA ERS Data APIs -- https://www.ers.usda.gov/developer/data-apis (verified REST API, requires API key)
- Financial Modeling Prep Senate Trading API -- https://site.financialmodelingprep.com/developer/docs/senate-trading-api (verified endpoint, free tier)

### Secondary (MEDIUM confidence)
- Claude Sonnet 4.6 pricing -- https://platform.claude.com/docs/en/about-claude/pricing ($3/$15 per 1M tokens; verified from official Anthropic docs)
- DeepSeek-R1 financial applications paper -- https://link.springer.com/article/10.1631/FITEE.2500227 (academic benchmarking of R1 in finance; published 2025)
- Fireworks blog on constrained generation with R1 -- https://fireworks.ai/blog/constrained-generation-with-reasoning (JSON schema constraining on reasoning output)
- CATArena tournament evaluation -- https://arxiv.org/html/2510.26852v1 (iterative tournament design for LLM agent evaluation)
- RouteLLM cost-effective routing -- https://lmsys.org/blog/2024-07-01-routellm/ (LMSYS framework for model routing, 85% cost reduction)
- wasdeparser GitHub -- https://github.com/fdfoneill/wasdeparser (Python package for USDA WASDE data)
- NOAA ENSO data -- https://www.climate.gov/enso (official ENSO monitoring and forecast)

### Tertiary (LOW confidence)
- Kimi K2.5 pricing -- https://artificialanalysis.ai/models/kimi-k2-5 (third-party aggregator; not verified with official docs)
- GLM-4.7 pricing -- https://llm-stats.com/models/glm-4.7 (third-party; pricing varies by provider)
- Multi-agent failure rates (41-86.7%) -- https://arxiv.org/html/2503.13657v1 (academic paper; may not apply to constrained-domain systems like HYDRA)
- AgentAuditor reasoning tree dedup -- https://arxiv.org/pdf/2602.09341 (preprint; structural semantic dedup concept applicable to hypothesis dedup)

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM-HIGH -- openai/instructor/sentence-transformers are well-established; Together AI pricing verified; DeepSeek-R1 structured output support confirmed but edge cases exist (empty content issue noted in official docs)
- Architecture: MEDIUM -- Agent loop pattern is well-understood; multi-head competitive design is novel but grounded in published multi-agent research; head communication recommendation (competitive independence + shared journal) is a design choice, not a proven pattern
- Pitfalls: HIGH -- Structured output parsing failures, cost explosion, degenerate loops, and flapping are well-documented in LLM agent literature and production experience
- Data sources: MEDIUM -- API endpoints verified; but government data portals have variable reliability; FMP free tier limits unknown

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (30 days -- LLM pricing and model availability change frequently; re-verify pricing before implementation)
