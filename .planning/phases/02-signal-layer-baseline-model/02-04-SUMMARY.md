---
phase: 02-signal-layer-baseline-model
plan: 04
subsystem: model
tags: [lightgbm, feature-matrix, binary-classification, point-in-time, nan-handling]

# Dependency graph
requires:
  - phase: 01-data-infrastructure
    provides: "FeatureStore with get_features_at for point-in-time correct feature retrieval"
  - phase: 02-signal-layer-baseline-model
    provides: "classify_divergence for live divergence computation, compute_cot_sentiment for live sentiment"
provides:
  - "FeatureAssembler: builds 17-feature matrix from feature store + live signal computation"
  - "BaselineModel: LightGBM binary classifier with conservative defaults (num_leaves=31, lr=0.1)"
  - "compute_binary_target: configurable horizon up/down target computation"
affects: [02-05-walk-forward-backtesting, phase-03-hyperparameter-tuning]

# Tech tracking
tech-stack:
  added: [lightgbm>=4.0, scikit-learn>=1.4]
  patterns: [feature-matrix-assembly, point-in-time-feature-retrieval, live-signal-computation, conservative-model-defaults]

key-files:
  created:
    - src/hydra/model/__init__.py
    - src/hydra/model/features.py
    - src/hydra/model/baseline.py
    - tests/test_model_features.py
    - tests/test_baseline_model.py
  modified:
    - pyproject.toml

key-decisions:
  - "Divergence and sentiment features computed live in assemble_at() rather than stored in feature store -- avoids separate pipeline"
  - "LightGBM conservative defaults: num_leaves=31, lr=0.1, n_estimators=100, min_child_samples=20 -- no tuning in Phase 2"
  - "NaN preservation for LightGBM's native NaN handling -- missing features passed as NaN, not imputed"

patterns-established:
  - "FeatureAssembler pattern: raw store features + live-computed derived signals into unified feature vector"
  - "Model wrapper pattern: BaselineModel encapsulates LGBMClassifier with train/predict_proba/feature_importance API"

requirements-completed: [MODL-01]

# Metrics
duration: 6min
completed: 2026-02-19
---

# Phase 2 Plan 04: Feature Engineering + Baseline Model Summary

**17-feature matrix assembler with point-in-time correct queries and LightGBM binary classifier using conservative defaults for directional prediction**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-19T09:52:42Z
- **Completed:** 2026-02-19T09:58:44Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- FeatureAssembler builds 17-feature vectors from feature store (COT, moments, Greeks) plus live-computed sentiment and divergence signals
- BaselineModel wraps LGBMClassifier with conservative Phase 2 defaults (no hyperparameter tuning)
- Binary target computation with configurable horizon (default 5 days)
- NaN values preserved throughout for LightGBM's native missing value handling
- 18 tests covering feature assembly, matrix construction, binary targets, model training, NaN handling, and feature importance

## Task Commits

Each task was committed atomically:

1. **Task 1: Feature matrix assembler + tests** - `f5a2840` (feat)
2. **Task 2: LightGBM baseline model wrapper + tests** - `63bdd51` (feat)

## Files Created/Modified
- `src/hydra/model/__init__.py` - Public re-exports for FeatureAssembler and BaselineModel
- `src/hydra/model/features.py` - FeatureAssembler: 17-feature matrix from store + live signals, binary target computation
- `src/hydra/model/baseline.py` - BaselineModel: LGBMClassifier wrapper with conservative defaults, predict_proba, feature_importance
- `tests/test_model_features.py` - 9 tests: assemble_at returns all features, missing = None, matrix shape, binary targets
- `tests/test_baseline_model.py` - 9 tests: train/predict, NaN handling, feature importance, NotFittedError, custom params
- `pyproject.toml` - Added lightgbm>=4.0 and scikit-learn>=1.4 dependencies

## Decisions Made
- Divergence and sentiment features computed live in `assemble_at()` via `classify_divergence()` and `compute_cot_sentiment()` rather than pre-stored in feature store -- avoids a separate write pipeline while maintaining point-in-time correctness for store features
- Conservative LightGBM defaults: `num_leaves=31, learning_rate=0.1, n_estimators=100, min_child_samples=20, subsample=0.8, colsample_bytree=0.8` -- deliberately not tuned; tuning belongs in Phase 3
- NaN values preserved throughout the pipeline: missing store features become NaN in the matrix, and LightGBM handles them natively without imputation

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added lightgbm and scikit-learn dependencies**
- **Found during:** Task 1 + Task 2
- **Issue:** lightgbm and scikit-learn not in pyproject.toml; LightGBM's sklearn API requires scikit-learn, and LightGBM requires libomp on macOS
- **Fix:** Added `lightgbm>=4.0` and `scikit-learn>=1.4` to pyproject.toml dependencies; installed libomp via Homebrew
- **Files modified:** pyproject.toml, uv.lock
- **Verification:** `uv run python -c "import lightgbm; print(lightgbm.__version__)"` succeeds
- **Committed in:** f5a2840 (Task 1) and 63bdd51 (Task 2)

---

**Total deviations:** 1 auto-fixed (1 blocking dependency)
**Impact on plan:** Necessary dependency addition. No scope creep.

## Issues Encountered
None beyond the dependency installation handled as a deviation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- FeatureAssembler and BaselineModel ready for walk-forward backtester (Plan 02-05)
- `FeatureAssembler.assemble_matrix()` provides training data matrices
- `FeatureAssembler.compute_binary_target()` provides binary labels
- `BaselineModel.train()` / `predict_proba()` / `predict()` provide full train-predict loop
- `BaselineModel.feature_importance()` enables diagnostic analysis

## Self-Check: PASSED

- [x] src/hydra/model/__init__.py -- FOUND
- [x] src/hydra/model/features.py -- FOUND
- [x] src/hydra/model/baseline.py -- FOUND
- [x] tests/test_model_features.py -- FOUND
- [x] tests/test_baseline_model.py -- FOUND
- [x] Commit f5a2840 (feat - features) -- FOUND
- [x] Commit 63bdd51 (feat - baseline) -- FOUND

---
*Phase: 02-signal-layer-baseline-model*
*Completed: 2026-02-19*
