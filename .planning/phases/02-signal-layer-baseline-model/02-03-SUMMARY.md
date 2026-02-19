---
phase: 02-signal-layer-baseline-model
plan: 03
subsystem: signals
tags: [divergence, classification, rule-based, z-score, dataclass, taxonomy]

# Dependency graph
requires:
  - phase: 02-signal-layer-baseline-model
    provides: "SentimentScore with score [-1,+1] and confidence [0,1] from COT data"
  - phase: 01-data-infrastructure
    provides: "ImpliedMoments (mean, skew, kurtosis) and DataQuality enum from options math"
provides:
  - "DivergenceSignal dataclass with direction, magnitude, divergence_type, confidence, suggested_bias"
  - "classify_divergence function: rule-based 6-type taxonomy classification"
  - "Configurable threshold constants for all classification rules"
affects: [02-04-feature-engineering, 02-05-baseline-model]

# Tech tracking
tech-stack:
  added: []
  patterns: [priority-ordered rule classification, z-scored magnitude normalization, quality-degraded confidence penalty]

key-files:
  created:
    - src/hydra/signals/divergence/__init__.py
    - src/hydra/signals/divergence/detector.py
    - tests/test_divergence_detector.py
  modified: []

key-decisions:
  - "Priority-ordered classification: vol_play > bullish/bearish divergence > overreaction > early_signal > trend_follow > neutral"
  - "Configurable module-level threshold constants (not hardcoded magic numbers)"
  - "Z-scoring requires minimum 10 historical divergence values; falls back to raw magnitude"
  - "Degraded quality applies 0.5 confidence penalty factor; None implied_mean returns neutral"

patterns-established:
  - "DivergenceSignal dataclass pattern: direction + magnitude + type + confidence + bias for signal decomposition"
  - "Rule-based classification with configurable thresholds at module level"
  - "Quality-aware confidence: degraded options data automatically reduces signal confidence"

requirements-completed: [SGNL-02, SGNL-03]

# Metrics
duration: 7min
completed: 2026-02-19
---

# Phase 2 Plan 03: Divergence Detector Summary

**Rule-based divergence detector classifying options-vs-sentiment mismatch into 6-type PRD taxonomy with z-scored magnitude and quality-aware confidence**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-19T09:39:04Z
- **Completed:** 2026-02-19T09:46:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- DivergenceSignal dataclass with all 5 required fields (direction, magnitude, divergence_type, confidence, suggested_bias)
- classify_divergence function implementing all 6 PRD Section 5.3 taxonomy types plus neutral fallback
- Full TDD cycle: 10 failing tests -> implementation -> all 10 passing
- Z-scored magnitude against expanding-window historical divergences when >= 10 values available
- Quality-aware confidence: DEGRADED options data applies 0.5 penalty factor

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Write failing tests for all 6 divergence types** - `23feed9` (test)
2. **Task 2: GREEN + REFACTOR -- Implement divergence detector** - `f73f5ec` (feat)

## Files Created/Modified
- `src/hydra/signals/divergence/__init__.py` - Public re-exports for DivergenceSignal and classify_divergence
- `src/hydra/signals/divergence/detector.py` - Core detector: DivergenceSignal dataclass + classify_divergence function with 6-type taxonomy
- `tests/test_divergence_detector.py` - 10 TDD tests covering all 6 divergence types plus edge cases

## Decisions Made
- Priority-ordered classification rules: volatility_play evaluated first (high kurtosis is distinctive), then directional divergences, overreaction, early signal, trend follow, and neutral as default
- Module-level threshold constants (e.g., OPTIONS_DIRECTIONAL_THRESHOLD = 0.01, SENTIMENT_BEARISH_THRESHOLD = -0.3) for easy tuning without code changes
- Z-scoring requires minimum 10 historical values to produce meaningful statistics; falls back to raw magnitude below that
- Confidence formula: sentiment_confidence * quality_factor * max(strength_factor, 0.3) -- floor of 0.3 prevents zero confidence when direction is clear
- SENTIMENT_SCALE_FACTOR = 10.0 normalizes sentiment [-1,+1] to same order of magnitude as options_bias for raw divergence computation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Divergence detector ready for feature engineering (Plan 02-04)
- classify_divergence provides numerical features (direction, magnitude, confidence) for LightGBM model
- DivergenceSignal.divergence_type provides categorical feature for the model
- Configurable thresholds enable hyperparameter tuning in later phases

## Self-Check: PASSED

- [x] src/hydra/signals/divergence/__init__.py -- FOUND
- [x] src/hydra/signals/divergence/detector.py -- FOUND
- [x] tests/test_divergence_detector.py -- FOUND
- [x] Commit 23feed9 (test) -- FOUND
- [x] Commit f73f5ec (feat) -- FOUND

---
*Phase: 02-signal-layer-baseline-model*
*Completed: 2026-02-19*
