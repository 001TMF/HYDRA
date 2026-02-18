# HYDRA

## What This Is

An autonomous, self-healing machine learning system that exploits pricing inefficiencies in low-volume futures markets (oats, lean hogs, ethanol, dairy, niche crypto derivatives) by combining options-derived mathematical signals with sentiment analysis. The system runs as a Claude Code extension, continuously improving itself through an agent-driven experimentation loop with minimal human intervention after deployment.

## Core Value

The agent loop must reliably detect model degradation, diagnose root causes, generate and test improvement hypotheses, and promote better models — all without human intervention. Everything else (signal quality, data pipeline, execution) serves this self-healing capability.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Options Math Engine computes implied distributions, Greeks flows, and volatility surfaces from raw options chains
- [ ] Sentiment Engine produces normalized composite sentiment scores from COT, NLP, social, and order flow data
- [ ] Divergence Detector identifies mispricings between options-implied expectations and sentiment signals
- [ ] Agent Core runs the full observe → diagnose → hypothesize → experiment → evaluate loop autonomously
- [ ] Sandbox provides isolated market replay, synthetic data generation, and A/B testing for candidate models
- [ ] Experiment Journal logs every experiment, diagnosis, and decision with queryable long-term memory
- [ ] Champion/Candidate model promotion protocol with multi-objective fitness evaluation
- [ ] CLI interface exposes agent commands (status, diagnose, experiment, journal, sandbox, rollback)
- [ ] Agent autonomy levels (supervised, semi-auto, autonomous, lockdown) are configurable
- [ ] Circuit breakers and automatic rollback on critical performance degradation
- [ ] Paper trading validation pipeline before any live capital deployment

### Out of Scope

- Multi-market expansion — single market first, scale architecture later
- Custom execution layer — using Interactive Brokers API for live execution
- Mobile/web dashboard — CLI-first, power user tool
- Real-time tick-level order flow — start with EOD/intraday bars, add tick data later if needed
- Consumer-facing features — this is a personal quant tool

## Context

**Market thesis:** Low-volume futures markets are structurally inefficient. Price discovery is slow, spreads are wide, and information takes hours or days to get priced in. Most quant firms ignore these markets due to insufficient liquidity for institutional-scale capital. A smaller, smarter system can exploit this gap.

**Core signal:** Options chains embed the market's probabilistic view of future prices (via Breeden-Litzenberger risk-neutral density extraction). When this implied view diverges from crowd sentiment (COT positioning, news NLP, social media), the divergence is the primary trading signal.

**Agent architecture:** The system treats the ML model as a living organism. The agent tends to it via a continuous loop: Observer monitors health and drift → Diagnostician triages issues → Hypothesis Engine proposes mutations from a playbook → Experiment Runner tests in sandbox → Evaluator/Judge decides promotion. The Experiment Journal provides long-term memory so the agent doesn't repeat failed experiments.

**Agent LLM decision:** The agent reasoning layer (hypothesis generation, diagnosis, natural language journal queries) will use a cheap Chinese open-source LLM via API provider — Qwen2.5, GLM-4, or DeepSeek-R1. Specific model to be determined by research based on reasoning quality, structured output reliability, and cost. This replaces the PRD's assumption of Claude API for agent reasoning.

**Data situation:** Starting from zero — no existing market data subscriptions. All data pipelines (futures prices, options chains, COT, news, social) need to be built from scratch during Phase 1-2.

**Target market:** To be determined by research — oat futures and lean hogs are leading candidates based on options availability, low volume, and data history.

## Constraints

- **Agent LLM**: Must use a cheap Chinese open-source model (Qwen, GLM, or DeepSeek) via API provider — not Claude API for agent reasoning
- **Broker**: Interactive Brokers for live execution (Phase 5+)
- **Language**: Python 3.11+ (ML ecosystem, rapid prototyping)
- **Budget**: Minimize API costs — cheap LLM provider, free data sources where possible (CFTC COT is free), paid data only where necessary
- **Risk**: No live capital until 4+ weeks of stable paper trading with at least one successful self-healing cycle
- **Isolation**: All experiments must be fully isolated from champion model and live systems

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Cheap Chinese LLM for agent reasoning | Cost control — Claude API too expensive for continuous agent loop | — Pending (research will select specific model) |
| Interactive Brokers for execution | Most common algo trading broker, solid API, supports futures + options | — Pending |
| Single market first | Prove the system works before scaling to multiple markets | — Pending |
| Follow PRD architecture to the letter | Comprehensive design already exists — no need to reinvent | — Pending |
| Scope-driven, not calendar-driven | Quality over speed — build what the PRD says, don't rush to hit arbitrary dates | — Pending |

---
*Last updated: 2026-02-18 after initialization*
