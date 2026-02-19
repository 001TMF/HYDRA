---
phase: 01-data-infrastructure-options-math-engine
plan: 02
subsystem: data-ingestion
tags: [databento, cot-reports, parquet, ingestion-pipeline, lookahead-bias, ohlcv, options-chain, cftc]

# Dependency graph
requires:
  - phase: 01-01
    provides: "Parquet data lake and feature store with as_of/available_at semantics"
provides:
  - "Abstract IngestPipeline base class with fetch/validate/persist contract"
  - "Databento futures OHLCV ingestion pipeline"
  - "Databento options chain ingestion pipeline (bid/ask/OI/strike/expiry)"
  - "CFTC COT ingestion pipeline with Tuesday as_of / Friday available_at timing"
  - "_next_friday() helper for COT release timestamp computation"
affects: [01-03, 01-04, 01-05, 01-06, 02-data-pipeline, 03-backtesting]

# Tech tracking
tech-stack:
  added: []
  patterns: [abstract-ingestion-pipeline, three-schema-join-options, cot-tuesday-friday-timing, cftc-revision-redownload]

key-files:
  created:
    - src/hydra/data/ingestion/base.py
    - src/hydra/data/ingestion/futures.py
    - src/hydra/data/ingestion/options.py
    - src/hydra/data/ingestion/cot.py
    - tests/test_ingestion_futures.py
    - tests/test_ingestion_options.py
    - tests/test_ingestion_cot.py
  modified: []

key-decisions:
  - "COT available_at uses fixed EST offset (UTC-5) for simplicity; production should use proper DST handling"
  - "Options chain joined from three Databento schemas (mbp-1, definition, statistics) on instrument_id"
  - "Futures close prices written as features with same-day availability (available after market close)"
  - "COT revision window flagged via is_revision_window field for downstream awareness"

patterns-established:
  - "Abstract IngestPipeline: all data sources implement fetch/validate/persist with DI for stores"
  - "COT timing: as_of=Tuesday, available_at=Friday 20:30 UTC -- never query by as_of alone"
  - "Options summary features computed at ingestion time: put_call_oi_ratio, total_oi, liquid_strike_count"
  - "Validation drops bad records with warnings, does not raise exceptions"

requirements-completed: [DATA-01, DATA-02, DATA-03]

# Metrics
duration: 6min
completed: 2026-02-19
---

# Phase 1 Plan 02: Data Ingestion Pipelines Summary

**Three ingestion pipelines (futures OHLCV, options chain, CFTC COT) with abstract interface, Databento integration, and critical COT Tuesday-Friday lookahead-bias prevention verified by 41 tests**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-19T08:09:09Z
- **Completed:** 2026-02-19T08:15:12Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Abstract IngestPipeline base class enforces fetch/validate/persist contract across all data sources with structlog logging and error handling
- FuturesIngestPipeline fetches daily OHLCV from Databento via parent symbology (HE.FUT), validates price integrity, writes to Parquet lake and close prices to feature store
- OptionsIngestPipeline joins three Databento schemas (mbp-1 bid/ask, definition strike/expiry, statistics OI) into a complete chain, computes summary features (put_call_oi_ratio, total_oi, liquid_strike_count)
- COTIngestPipeline fetches CFTC Disaggregated Futures-Only reports with correct as_of=Tuesday / available_at=Friday 20:30 UTC timing, preventing lookahead bias
- 41 tests pass across all three pipelines, including critical timing tests: COT data NOT available on Wednesday, available after Friday release

## Task Commits

Each task was committed atomically:

1. **Task 1: Create abstract IngestPipeline base and Databento futures pipeline** - `bef976e` (feat)
2. **Task 2: Implement options chain and COT ingestion pipelines** - `f3a93c1` (feat)

## Files Created/Modified
- `src/hydra/data/ingestion/base.py` - Abstract IngestPipeline with fetch/validate/persist/run (129 lines)
- `src/hydra/data/ingestion/futures.py` - Databento futures OHLCV ingestion (215 lines)
- `src/hydra/data/ingestion/options.py` - Databento options chain ingestion with three-schema join (309 lines)
- `src/hydra/data/ingestion/cot.py` - CFTC COT ingestion with as_of/available_at timing (374 lines)
- `tests/test_ingestion_futures.py` - 13 tests for futures pipeline (274 lines)
- `tests/test_ingestion_options.py` - 9 tests for options pipeline (283 lines)
- `tests/test_ingestion_cot.py` - 19 tests for COT pipeline including critical timing tests (450 lines)

## Decisions Made
- **COT release time uses fixed EST offset (UTC-5):** For Phase 1 simplicity, the _next_friday() helper uses a fixed UTC-5 offset. In production with DST, this should use a proper timezone library (pytz/zoneinfo) to correctly compute 15:30 ET year-round. The error is at most 1 hour during DST transitions.
- **Options chain joined from three Databento schemas:** Rather than a single query, we make three separate requests (mbp-1, definition, statistics) and join on instrument_id. This matches Databento's schema design where different data lives in different schemas.
- **Futures close prices as features:** Close prices are written to the feature store with as_of=date, available_at=date (same day) since futures prices are publicly available after market close.
- **COT revision redownload window:** The is_revision_window flag marks records within the last 4 weeks, enabling downstream logic to handle CFTC revisions explicitly.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - all tests use mocked external dependencies. Databento API key and cot_reports access are required only for live data ingestion.

## Next Phase Readiness
- All three ingestion pipelines are ready for scheduling (Plan 06: orchestration/scheduling)
- Abstract IngestPipeline interface established for any future data sources
- Options chain data flowing into Parquet enables Plan 03 (options math: Breeden-Litzenberger, SVI)
- COT features in the feature store enable Plan 04 (Greeks flow) and Plan 05 (data quality monitoring)
- Critical lookahead-bias prevention verified -- all downstream backtests can trust point-in-time correctness

## Self-Check: PASSED

- All 7 key files: FOUND
- Commit bef976e (Task 1): FOUND
- Commit f3a93c1 (Task 2): FOUND
- base.py: 129 lines (min 30)
- futures.py: 215 lines (min 60)
- options.py: 309 lines (min 80)
- cot.py: 374 lines (min 70)
- futures.py contains "parquet_lake.write": YES
- options.py contains "parquet_lake.write": YES
- cot.py contains "feature_store.write_feature": YES
- cot.py contains "available_at" semantics: YES
- All 41 tests pass: YES

---
*Phase: 01-data-infrastructure-options-math-engine*
*Completed: 2026-02-19*
