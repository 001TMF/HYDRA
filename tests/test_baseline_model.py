"""Tests for LightGBM baseline model wrapper.

Tests that BaselineModel correctly wraps LightGBM with conservative defaults,
handles NaN features natively, and exposes feature importance for diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from hydra.model.baseline import BaselineModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_data(
    n_samples: int = 200, n_features: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    # Simple rule: label 1 if sum of features > 0
    y = (X.sum(axis=1) > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTrainAndPredict:
    """Tests for model training and prediction."""

    def test_train_and_predict(self) -> None:
        """Train on synthetic data, predict_proba returns values in [0, 1]."""
        X, y = _synthetic_data()
        model = BaselineModel()
        model.train(X, y)

        proba = model.predict_proba(X)

        assert proba.shape == (len(X),)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_binary(self) -> None:
        """predict() returns 0s and 1s."""
        X, y = _synthetic_data()
        model = BaselineModel()
        model.train(X, y)

        preds = model.predict(X)

        assert preds.shape == (len(X),)
        unique = set(np.unique(preds))
        assert unique.issubset({0, 1})

    def test_is_fitted_after_train(self) -> None:
        """is_fitted is True after training."""
        X, y = _synthetic_data()
        model = BaselineModel()

        assert model.is_fitted is False
        model.train(X, y)
        assert model.is_fitted is True


class TestFeatureImportance:
    """Tests for feature importance extraction."""

    def test_feature_importance(self) -> None:
        """After training, feature_importance() returns dict with correct keys."""
        X, y = _synthetic_data(n_features=4)
        names = ["alpha", "beta", "gamma", "delta"]
        model = BaselineModel()
        model.train(X, y, feature_names=names)

        importance = model.feature_importance()

        assert isinstance(importance, dict)
        assert set(importance.keys()) == set(names)
        # All importances should be non-negative
        for v in importance.values():
            assert v >= 0

    def test_feature_importance_default_names(self) -> None:
        """Without feature_names, uses f0, f1, ..."""
        X, y = _synthetic_data(n_features=3)
        model = BaselineModel()
        model.train(X, y)

        importance = model.feature_importance()

        assert set(importance.keys()) == {"f0", "f1", "f2"}


class TestNotFitted:
    """Tests for error handling when model is not fitted."""

    def test_not_fitted_raises(self) -> None:
        """predict_proba before train raises NotFittedError."""
        model = BaselineModel()
        X = np.random.randn(10, 5)

        with pytest.raises(NotFittedError):
            model.predict_proba(X)


class TestNaNHandling:
    """Tests for LightGBM's native NaN handling."""

    def test_handles_nan_features(self) -> None:
        """Training data with NaN values does not crash (LightGBM handles natively)."""
        X, y = _synthetic_data(n_samples=100, n_features=5)
        # Inject NaN values into ~20% of the data
        rng = np.random.RandomState(123)
        mask = rng.random(X.shape) < 0.2
        X[mask] = np.nan

        model = BaselineModel()
        # Should not raise
        model.train(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (100,)
        assert np.all(np.isfinite(proba))


class TestCustomParams:
    """Tests for parameter override."""

    def test_custom_params_override(self) -> None:
        """Passing custom params overrides defaults."""
        custom = {"n_estimators": 50, "learning_rate": 0.05}
        model = BaselineModel(params=custom)

        assert model.model.n_estimators == 50
        assert model.model.learning_rate == 0.05
        # Non-overridden defaults should remain
        assert model.model.num_leaves == 31

    def test_default_params_values(self) -> None:
        """Default model uses conservative params from DEFAULT_PARAMS."""
        model = BaselineModel()

        assert model.model.n_estimators == 100
        assert model.model.learning_rate == 0.1
        assert model.model.num_leaves == 31
        assert model.model.min_child_samples == 20
        assert model.model.random_state == 42
