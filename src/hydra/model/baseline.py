"""LightGBM baseline model wrapper for directional prediction.

Wraps LightGBM binary classifier with conservative defaults for Phase 2.
NO hyperparameter tuning -- conservative defaults only. Tuning belongs
in Phase 3.

The model predicts P(price up) for a configurable N-day horizon using
divergence signal components, implied moments, Greeks flows, and COT
features assembled by FeatureAssembler.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np


class BaselineModel:
    """LightGBM binary classifier for directional prediction.

    Conservative defaults -- NO hyperparameter tuning in Phase 2.

    Parameters
    ----------
    params : dict | None
        Optional parameter overrides. Merged with DEFAULT_PARAMS.
    """

    DEFAULT_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
    }

    def __init__(self, params: dict | None = None) -> None:
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model = lgb.LGBMClassifier(**merged)
        self._is_fitted = False
        self._feature_names: list[str] = []

    def train(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None
    ) -> None:
        """Train the model on feature matrix X and binary target y.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, F).
        y : np.ndarray
            Binary target array of length N.
        feature_names : list[str] | None
            Optional feature names for importance reporting.
        """
        self.model.fit(X, y)
        self._is_fitted = True
        self._feature_names = feature_names or [
            f"f{i}" for i in range(X.shape[1])
        ]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(price up) for each row.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, F).

        Returns
        -------
        np.ndarray
            Array of probabilities in [0, 1] of length N.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (N, F).
        threshold : float
            Probability threshold for positive class.

        Returns
        -------
        np.ndarray
            Binary array of predictions (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance as a dict.

        Returns
        -------
        dict[str, float]
            Mapping of feature_name -> importance (split-based).
        """
        return dict(
            zip(self._feature_names, self.model.feature_importances_)
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted
