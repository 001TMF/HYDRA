---
phase: 06-dashboard-monitoring-for-paper-trading-and-full-lightweight-containerisation
plan: 04
subsystem: dashboard
tags: [fastapi, pytest, httpx, typer, uvicorn, cli]

requires:
  - phase: 06-dashboard-monitoring-for-paper-trading-and-full-lightweight-containerisation
    provides: FastAPI app factory, 5 page routes, API endpoints, lifespan runner
provides:
  - 16 new tests covering all 5 page routes and all API endpoints
  - CLI 'hydra serve' command for starting the dashboard
  - HYDRA_DATA_DIR env var support for Docker container configuration
affects: []

tech-stack:
  added: []
  patterns: [async-test-fixtures-with-populated-data, cli-serve-with-factory]

key-files:
  created:
    - tests/test_dashboard_pages.py
    - tests/test_dashboard_api.py
  modified:
    - src/hydra/cli/app.py
    - src/hydra/dashboard/app.py

key-decisions:
  - "HYDRA_DATA_DIR env var checked before ~/.hydra default in create_app -- allows Docker override without CLI flag"
  - "CLI serve validates app creation before starting uvicorn -- fail-fast on config errors"

patterns-established:
  - "Populated test fixtures: app_with_data creates FillJournal + ExperimentJournal with realistic test data for integration tests"

requirements-completed: []

duration: 3min
completed: 2026-02-20
---

# Phase 6 Plan 4: Dashboard Tests and CLI Serve Command Summary

**16 new tests for all dashboard page routes and API endpoints, plus CLI 'hydra serve' command with HYDRA_DATA_DIR env var for Docker configuration**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T20:29:31Z
- **Completed:** 2026-02-19T20:32:22Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- 10 page tests covering all 5 routes (/, /fills, /agent, /drift, /system) with empty and populated data
- 6 API tests covering fills/recent, agent/state, system/status, health, and fills/summary endpoints
- CLI 'hydra serve' command starts dashboard on localhost:8080 with configurable port, host, and data directory
- HYDRA_DATA_DIR env var allows Docker container to set data directory without CLI flags
- 569 total tests passing (16 new + 553 existing)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests for all dashboard page routes and API endpoints** - `863a272` (test)
2. **Task 2: Add CLI 'serve' command and verify full integration** - `b314a04` (feat)

## Files Created/Modified
- `tests/test_dashboard_pages.py` - 10 tests for all 5 page routes with empty and populated data states
- `tests/test_dashboard_api.py` - 6 tests for API endpoints with populated data verification
- `src/hydra/cli/app.py` - Added 'serve' command after 'fill-report' block
- `src/hydra/dashboard/app.py` - HYDRA_DATA_DIR env var fallback in create_app

## Decisions Made
- HYDRA_DATA_DIR env var checked before ~/.hydra default -- allows Docker override without CLI flag
- CLI serve validates app creation before starting uvicorn -- fail-fast on config errors

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 6 complete: all 4 plans executed
- Dashboard fully tested with 22 total tests (6 existing + 16 new)
- CLI has 9 commands including serve for dashboard startup
- Docker, dashboard, and tests all integrated and verified

## Self-Check: PASSED

All 4 modified/created files verified on disk. Both task commits (863a272, b314a04) found in git history.
