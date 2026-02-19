# Phase 4: Agent Core + LLM Integration (Multi-Headed Architecture) - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the multi-headed autonomous agent system: a coordinator dispatches drift diagnoses to specialized heads (Technical, Research, Structural), which compete through the sandbox arena to generate improvement hypotheses. The arena ranks candidates via tournament, and winners get promoted. Includes LLM client with structured output, fallback chains, autonomy levels, rollback triggers, mutation budgets, and head reputation scoring. "Cut off one head, two grow back."

</domain>

<decisions>
## Implementation Decisions

### Research Head Scope
- **Full autonomy**: Research Head can discover, test, and integrate new signals without human approval -- as long as they pass fitness thresholds in the sandbox. User gets notified after integration, not before.
- **Broad data sources**: Financial APIs (USDA, CFTC, Fed), political signals (congressional STOCK Act trades, lobbying, committee hearings), AND macro cycles (commodity supercycles, El Nino/seasonal, historical regime data). Start with all three categories.
- **Cycle detection**: Use BOTH historical decomposition (extracting seasonal/cyclical components from price data) AND external event calendars (planting dates, USDA schedule, El Nino forecasts) combined for richer cyclical features.
- **Web search**: Claude's discretion on whether to use web search vs pre-configured APIs. Research should determine the safest effective approach.

### LLM Provider & Reasoning
- **Cost-optimized multi-model routing**: DO NOT use a reasoning model for tasks that don't require reasoning. Route each task to the cheapest capable model. Key principle: minimize loss to LLM API costs.
- **Models to research**: DeepSeek-R1 (reasoning), Kimi K2.5, GLM-4, Qwen 2.5/3, Claude Sonnet 4.6 (high-end option alongside DeepSeek). Research the optimal combination for different task types (diagnosis, hypothesis generation, structured output, simple classification).
- **Dynamic budget**: Cap at $20/day but adjust dynamically based on results. If heads are producing winners, allocate more budget. If heads are producing garbage, reduce frequency and budget.
- **Fallback chain**: Claude's discretion on design. Research the reliability of each model's structured output and design the optimal cascade.

### Head Communication Strategy
- **Research-driven approach**: Determine through research what balance between competition and collaboration is most likely to succeed. Key constraints: avoid echo chambers (heads just copying each other) AND avoid hallucination amplification (heads reinforcing bad ideas). The experiment journal should serve as shared memory of what's been tried, but current-round proposals should have some independence.

### Claude's Discretion
- Web search implementation for Research Head (pre-configured APIs vs web search vs hybrid)
- Fallback chain model sequence and retry logic
- Head communication pattern (the research should determine the optimal balance)
- Structured output format (Pydantic models, JSON schema, etc.)
- Tournament bracket vs ranked list for arena competition
- Head reputation scoring algorithm

</decisions>

<specifics>
## Specific Ideas

- "The key is to minimise loss to LLM API costs" -- cost efficiency is a first-class concern, not an afterthought
- "Tasks that don't require reasoning don't use a reasoning model" -- smart routing is essential
- "We don't want echo chambers but we also don't want hallucinations" -- the communication design should prevent both failure modes
- "Max $20/day but keep it dynamic based on results" -- budget follows performance, not a fixed allocation
- Multi-headed Hydra metaphor: each head is an independent approach to self-healing. When one model/approach fails, the others continue. True resilience through diversity.
- Senator trading on Agricultural Committee before COT reports is a specific signal of interest
- Decade cycles (3-7-10-50-100 year) impacting commodities -- long-horizon signals the Research Head should discover

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 04-agent-core-llm-integration*
*Context gathered: 2026-02-19*
