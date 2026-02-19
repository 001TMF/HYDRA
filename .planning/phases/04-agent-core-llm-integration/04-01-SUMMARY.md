---
phase: 04-agent-core-llm-integration
plan: 01
subsystem: llm
tags: [instructor, openai, pydantic, deepseek, together-ai, structured-output, fallback-chain]

# Dependency graph
requires:
  - phase: 03-sandbox-experiment-infrastructure
    provides: "Experiment infrastructure (journal, sandbox, drift observer)"
provides:
  - "LLMClient with multi-provider fallback chain and cost tracking"
  - "LLMUnavailableError contract for AGNT-10 rule-based fallback"
  - "DiagnosisResultLLM, HypothesisLLM, ExperimentConfig Pydantic schemas"
  - "TaskRouter with cost-optimized model routing per task type"
  - "Canonical types.py with DriftCategory, MutationType enums and dataclasses"
affects: [04-02, 04-03, 04-04, 04-05]

# Tech tracking
tech-stack:
  added: [openai, instructor, tenacity, sentence-transformers]
  patterns: [instructor-structured-output, fallback-chain, task-type-routing, cost-tracking]

key-files:
  created:
    - src/hydra/agent/__init__.py
    - src/hydra/agent/types.py
    - src/hydra/agent/llm/__init__.py
    - src/hydra/agent/llm/client.py
    - src/hydra/agent/llm/schemas.py
    - src/hydra/agent/llm/router.py
    - tests/agent/__init__.py
    - tests/agent/llm/__init__.py
    - tests/agent/test_llm_client.py
  modified: [pyproject.toml, uv.lock]

key-decisions:
  - "Canonical types.py created as stub with DriftCategory/MutationType enums and DiagnosisResult/Hypothesis dataclasses -- 04-03 will extend"
  - "instructor.from_openai wraps OpenAI clients for Together AI and DeepSeek providers with Pydantic structured output"
  - "Token estimation uses 4 chars/token heuristic for cost tracking without API usage data"
  - "LLM schemas (Pydantic BaseModel) kept separate from core types (dataclasses) to avoid coupling agent to Pydantic"

patterns-established:
  - "Fallback chain pattern: iterate ModelSpec list, skip unconfigured providers, raise LLMUnavailableError on exhaustion"
  - "Task-type routing: TaskType enum maps to ordered ModelSpec chains for cost optimization"
  - "Budget cap: daily cost accumulator checked before each call, reset by scheduler"

requirements-completed: [AGNT-05, AGNT-10]

# Metrics
duration: 4min
completed: 2026-02-19
---

# Phase 4 Plan 01: LLM Client Summary

**Multi-provider LLM client with instructor-based structured output, task-type router for cost optimization, and Pydantic I/O schemas**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-19T13:12:10Z
- **Completed:** 2026-02-19T13:16:43Z
- **Tasks:** 2
- **Files modified:** 11

## Accomplishments
- Built LLMClient with fallback chain iterating through Together AI and DeepSeek providers via instructor library
- LLMUnavailableError contract satisfies AGNT-10: system operates entirely rule-based when no API keys configured
- TaskRouter maps reasoning/classification/formatting to cheapest capable model (DeepSeek-R1 for reasoning, Qwen2.5-7B-Turbo for simple tasks)
- Daily cost tracking with configurable budget cap prevents runaway spending
- 32 tests covering disabled mode, budget enforcement, fallback behavior, schema validation, and router logic

## Task Commits

Each task was committed atomically:

1. **Task 1: Pydantic schemas and task-type router** - `ae42e9c` (feat)
2. **Task 2: LLM client with fallback chain, cost tracking, and tests** - `66ec641` (feat)

## Files Created/Modified
- `src/hydra/agent/__init__.py` - Agent package init
- `src/hydra/agent/types.py` - Canonical DriftCategory, MutationType enums and DiagnosisResult, Hypothesis dataclasses
- `src/hydra/agent/llm/__init__.py` - LLM subpackage init
- `src/hydra/agent/llm/client.py` - LLMClient with fallback chain, cost tracking, LLMUnavailableError
- `src/hydra/agent/llm/schemas.py` - DiagnosisResultLLM, HypothesisLLM, ExperimentConfig Pydantic schemas
- `src/hydra/agent/llm/router.py` - TaskRouter with ModelSpec and default fallback chains
- `tests/agent/__init__.py` - Test package init
- `tests/agent/llm/__init__.py` - Test subpackage init
- `tests/agent/test_llm_client.py` - 32 tests for client, schemas, and router
- `pyproject.toml` - Added openai, instructor, tenacity, sentence-transformers dependencies
- `uv.lock` - Updated lockfile

## Decisions Made
- **Canonical types.py created as stub**: Since 04-03 hasn't run yet, types.py was created with DriftCategory, MutationType enums and DiagnosisResult, Hypothesis dataclasses. Plan 04-03 will extend this file.
- **instructor.from_openai for provider wrapping**: Uses instructor's OpenAI-compatible wrapper for both Together AI and DeepSeek, enabling automatic Pydantic validation and retry-on-parse-failure.
- **Token estimation heuristic (4 chars/token)**: Cost tracking uses approximate token counts from message/response character lengths rather than requiring API usage metadata. Sufficient for budget cap enforcement.
- **Separate LLM schemas from core types**: Pydantic BaseModels with Field(description=...) for LLM use are distinct from plain dataclasses in types.py, avoiding Pydantic dependency in non-LLM agent code.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Created types.py stub since 04-03 has not run**
- **Found during:** Task 1
- **Issue:** schemas.py imports from hydra.agent.types which doesn't exist yet (04-03 creates it)
- **Fix:** Created types.py with DriftCategory, MutationType enums per plan instructions. Linter extended with DiagnosisResult and Hypothesis dataclasses.
- **Files modified:** src/hydra/agent/types.py
- **Verification:** All imports resolve correctly
- **Committed in:** ae42e9c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Expected deviation -- plan explicitly anticipated this and provided instructions. No scope creep.

## Issues Encountered
None

## User Setup Required

External services are optional. The system works without any API keys (AGNT-10 rule-based mode).

For LLM-enhanced operation, set:
- `TOGETHER_API_KEY` - https://api.together.ai/settings/api-keys
- `DEEPSEEK_API_KEY` (optional) - https://platform.deepseek.com/api_keys

## Next Phase Readiness
- LLM client ready for use by diagnostician (04-03) and hypothesis engine (04-03)
- TaskRouter ready for cost optimization across all agent LLM calls
- LLMUnavailableError contract established for AGNT-10 fallback pattern
- types.py stub ready for 04-03 to extend with additional types

## Self-Check: PASSED

- All 9 created files verified present on disk
- Commit ae42e9c (Task 1) verified in git log
- Commit 66ec641 (Task 2) verified in git log
- 32/32 tests passing
- All plan verification commands succeed

---
*Phase: 04-agent-core-llm-integration*
*Completed: 2026-02-19*
