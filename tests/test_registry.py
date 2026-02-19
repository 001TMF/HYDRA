"""Tests for MLflow model registry wrapper.

Tests use real MLflow operations against temporary directories.
Each test gets a fresh registry pointed at its own temp dir.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydra.model.baseline import BaselineModel
from hydra.sandbox.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    """Fresh ModelRegistry backed by a temp directory."""
    return ModelRegistry(tracking_uri=f"file://{tmp_path}/mlruns")


@pytest.fixture
def trained_model():
    """A trained BaselineModel on small synthetic data."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 5)
    y = (X[:, 0] > 0).astype(int)
    model = BaselineModel()
    model.train(X, y, [f"f{i}" for i in range(5)])
    return model


def _log_and_get(registry, trained_model, run_name="test-run", extra_tags=None):
    """Helper to log a candidate and return (run_id, version)."""
    metrics = {"accuracy": 0.75, "sharpe": 1.2}
    config = {"num_leaves": 31, "learning_rate": 0.1}
    tags = extra_tags or {"experiment": "test"}
    return registry.log_candidate(
        trained_model, metrics=metrics, config=config, tags=tags, run_name=run_name
    )


class TestLogCandidate:
    def test_log_candidate(self, registry, trained_model):
        """Log a candidate and verify return types."""
        run_id, version = _log_and_get(registry, trained_model)
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert isinstance(version, int)
        assert version >= 1


class TestPromoteAndLoad:
    def test_promote_and_load_champion(self, registry, trained_model):
        """Promote a candidate to champion, then load and predict."""
        _run_id, version = _log_and_get(registry, trained_model)
        registry.promote_to_champion(version)

        loaded = registry.load_champion()
        # MLflow loads LightGBM as a Booster -- use .predict()
        X_test = np.random.RandomState(99).randn(5, 5)
        preds = loaded.predict(X_test)
        assert len(preds) == 5


class TestPromoteArchivesPrevious:
    def test_promote_archives_previous(self, registry, trained_model):
        """Promoting B should archive A."""
        _run_id_a, ver_a = _log_and_get(registry, trained_model, run_name="model-A")
        registry.promote_to_champion(ver_a)

        _run_id_b, ver_b = _log_and_get(registry, trained_model, run_name="model-B")
        registry.promote_to_champion(ver_b)

        # Champion should be B
        info = registry.get_champion_info()
        assert info["version"] == ver_b

        # A should have "archived" alias
        versions = registry.list_versions()
        a_version = [v for v in versions if v["version"] == ver_a][0]
        assert "archived" in a_version["aliases"]


class TestRollback:
    def test_rollback(self, registry, trained_model):
        """Rollback restores previously archived version."""
        _run_id_a, ver_a = _log_and_get(registry, trained_model, run_name="model-A")
        registry.promote_to_champion(ver_a)

        _run_id_b, ver_b = _log_and_get(registry, trained_model, run_name="model-B")
        registry.promote_to_champion(ver_b)

        restored = registry.rollback()
        assert restored == ver_a

        info = registry.get_champion_info()
        assert info["version"] == ver_a

    def test_rollback_no_archived_raises(self, registry):
        """Rollback with no archived model raises ValueError."""
        with pytest.raises(ValueError, match="No archived model"):
            registry.rollback()


class TestListVersions:
    def test_list_versions(self, registry, trained_model):
        """list_versions returns correct structure for multiple candidates."""
        _log_and_get(registry, trained_model, run_name="v1")
        _log_and_get(registry, trained_model, run_name="v2")

        versions = registry.list_versions()
        assert len(versions) == 2

        for v in versions:
            assert "version" in v
            assert "run_id" in v
            assert "aliases" in v
            assert "created_at" in v
