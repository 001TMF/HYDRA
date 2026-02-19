---
phase: 06-dashboard-monitoring-for-paper-trading-and-full-lightweight-containerisation
plan: 03
subsystem: dashboard
tags: [fastapi, htmx, jinja2, sse, lifespan, paper-trading]

requires:
  - phase: 06-dashboard-monitoring-for-paper-trading-and-full-lightweight-containerisation
    provides: FastAPI app factory, route structure, base template, dark theme CSS
provides:
  - 5 complete dashboard pages (overview, fills, agent, drift, system)
  - htmx auto-refresh on 60-second intervals
  - SSE live cycle updates on overview page
  - JSON API endpoints for htmx fragment updates
  - PaperTradingRunner lifespan integration
affects: [06-04]

tech-stack:
  added: []
  patterns: [lifespan-context-manager, htmx-fragment-swap, graceful-db-fallback]

key-files:
  created:
    - src/hydra/dashboard/templates/index.html
    - src/hydra/dashboard/templates/fills.html
    - src/hydra/dashboard/templates/agent.html
    - src/hydra/dashboard/templates/drift.html
    - src/hydra/dashboard/templates/system.html
  modified:
    - src/hydra/dashboard/routes/pages.py
    - src/hydra/dashboard/routes/api.py
    - src/hydra/dashboard/routes/sse.py
    - src/hydra/dashboard/app.py
    - src/hydra/dashboard/templates/base.html
    - src/hydra/dashboard/static/style.css

key-decisions:
  - "Graceful DB fallback: _open_fill_journal/_open_experiment_journal helpers return None on failure, pages show 'No data' instead of crashing"
  - "HYDRA_START_RUNNER env var activates runner without changing create_app call signature"
  - "Runner construction failures caught in lifespan -- dashboard starts even if IB Gateway unavailable"
  - "Inline HTML fragment generation for htmx /fills/recent and /agent/state endpoints (no extra partial templates)"

patterns-established:
  - "Lifespan context manager: _lifespan starts/stops PaperTradingRunner alongside dashboard"
  - "Active nav state: pass active_page from each route, base.html conditionally adds 'active' class"

requirements-completed: []

duration: 5min
completed: 2026-02-20
---

# Phase 6 Plan 3: Dashboard Pages and Lifespan Runner Summary

**5 dashboard pages with htmx auto-refresh, SSE live updates, and PaperTradingRunner lifespan integration for single-process deployment**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-19T20:22:33Z
- **Completed:** 2026-02-19T20:27:06Z
- **Tasks:** 3
- **Files modified:** 11

## Accomplishments
- 5 dashboard pages rendering data from FillJournal, ExperimentJournal, and agent state with graceful DB fallback
- htmx auto-refresh every 60s on overview, fills, and system pages plus SSE live updates
- PaperTradingRunner starts as FastAPI lifespan event when HYDRA_START_RUNNER=true
- 553 existing tests still passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create page routes and API endpoints for all 5 dashboard views** - `98b7680` (feat)
2. **Task 2: Create all 5 Jinja2 page templates with htmx auto-refresh** - `43c4062` (feat)
3. **Task 3: Integrate PaperTradingRunner as FastAPI lifespan event in app.py** - `61861b9` (feat)

## Files Created/Modified
- `src/hydra/dashboard/routes/pages.py` - 5 page routes with FillJournal/ExperimentJournal/agent state data
- `src/hydra/dashboard/routes/api.py` - 3 htmx API endpoints (/fills/recent, /agent/state, /system/status)
- `src/hydra/dashboard/routes/sse.py` - Enriched SSE with fill_count + agent_state + timestamp
- `src/hydra/dashboard/app.py` - Lifespan context manager, _build_runner helper, HYDRA_START_RUNNER env var
- `src/hydra/dashboard/templates/index.html` - Overview with fill count, agent state, SSE, recent fills
- `src/hydra/dashboard/templates/fills.html` - Fill table with slippage columns and reconciliation summary
- `src/hydra/dashboard/templates/agent.html` - Experiment history and agent state
- `src/hydra/dashboard/templates/drift.html` - Monitored metrics with thresholds (PSI, KS, ADWIN, CUSUM)
- `src/hydra/dashboard/templates/system.html` - DB status, reconciliation metrics, configuration
- `src/hydra/dashboard/templates/base.html` - Active nav state support
- `src/hydra/dashboard/static/style.css` - Card-row, metric, table, and layout styles

## Decisions Made
- Graceful DB fallback with helper functions returning None -- dashboard never crashes on missing data
- HYDRA_START_RUNNER env var for Docker activation without changing API signature
- Runner failures caught in lifespan -- dashboard starts independently of IB Gateway connectivity
- Inline HTML for htmx fragment endpoints avoids extra partial template files

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All dashboard pages and API endpoints ready for 06-04 testing
- Lifespan runner integration ready for Docker Compose HYDRA_START_RUNNER=true

## Self-Check: PASSED
