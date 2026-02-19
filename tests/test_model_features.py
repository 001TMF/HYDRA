"""Tests for feature matrix assembler.

Tests that FeatureAssembler correctly pulls features from the feature store
with point-in-time correctness, computes derived signals, and builds
training-ready matrices.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from hydra.model.features import FeatureAssembler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_store(features: dict[str, float]) -> MagicMock:
    """Create a mock FeatureStore that returns the given features."""
    store = MagicMock()
    store.get_features_at.return_value = features
    return store


def _full_features() -> dict[str, float]:
    """Return a complete set of 12 raw store features."""
    return {
        "cot_managed_money_net": 5000.0,
        "cot_producer_net": -3000.0,
        "cot_swap_net": 1000.0,
        "cot_total_oi": 100000.0,
        "implied_mean": 100.5,
        "implied_variance": 0.04,
        "implied_skew": -0.3,
        "implied_kurtosis": 3.2,
        "atm_iv": 0.25,
        "gex": 1500.0,
        "vanna_flow": -200.0,
        "charm_flow": 50.0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAssembleAt:
    """Tests for FeatureAssembler.assemble_at."""

    def test_assemble_at_returns_all_features(self) -> None:
        """Mock feature store returns all raw features -> all 17 FEATURE_NAMES present."""
        store = _make_store(_full_features())
        assembler = FeatureAssembler(store)
        query_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        result = assembler.assemble_at("HE", query_time)

        assert set(result.keys()) == set(FeatureAssembler.FEATURE_NAMES)
        assert len(result) == 17

    def test_missing_features_are_none(self) -> None:
        """Feature store missing some features -> None in output."""
        # Only provide COT features, omit implied moments and Greeks
        partial = {
            "cot_managed_money_net": 5000.0,
            "cot_total_oi": 100000.0,
        }
        store = _make_store(partial)
        assembler = FeatureAssembler(store)
        query_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        result = assembler.assemble_at("HE", query_time)

        # All 17 keys should still be present
        assert len(result) == 17
        # Missing store features should be None
        assert result["implied_mean"] is None
        assert result["gex"] is None
        assert result["atm_iv"] is None
        # Divergence should be None because implied_mean is missing
        assert result["divergence_direction"] is None
        assert result["divergence_magnitude"] is None
        assert result["divergence_confidence"] is None

    def test_raw_store_features_passed_through(self) -> None:
        """Raw feature store values should be passed through unchanged."""
        features = _full_features()
        store = _make_store(features)
        assembler = FeatureAssembler(store)
        query_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        result = assembler.assemble_at("HE", query_time)

        for name in [
            "cot_managed_money_net", "cot_producer_net", "cot_swap_net", "cot_total_oi",
            "implied_mean", "implied_variance", "implied_skew", "implied_kurtosis", "atm_iv",
            "gex", "vanna_flow", "charm_flow",
        ]:
            assert result[name] == features[name], f"{name} mismatch"


class TestAssembleMatrix:
    """Tests for FeatureAssembler.assemble_matrix."""

    def test_assemble_matrix_shape(self) -> None:
        """N timestamps -> (N, len(FEATURE_NAMES)) matrix."""
        store = _make_store(_full_features())
        assembler = FeatureAssembler(store)
        timestamps = [
            datetime(2024, 6, i, tzinfo=timezone.utc) for i in range(1, 6)
        ]

        matrix, names = assembler.assemble_matrix("HE", timestamps)

        assert matrix.shape == (5, 17)
        assert names == FeatureAssembler.FEATURE_NAMES

    def test_matrix_missing_values_are_nan(self) -> None:
        """Missing features should be NaN in the matrix."""
        partial = {"cot_managed_money_net": 1000.0, "cot_total_oi": 50000.0}
        store = _make_store(partial)
        assembler = FeatureAssembler(store)
        timestamps = [datetime(2024, 6, 1, tzinfo=timezone.utc)]

        matrix, _ = assembler.assemble_matrix("HE", timestamps)

        # implied_mean index is 4 (0-indexed)
        idx_implied_mean = FeatureAssembler.FEATURE_NAMES.index("implied_mean")
        assert np.isnan(matrix[0, idx_implied_mean])


class TestBinaryTarget:
    """Tests for FeatureAssembler.compute_binary_target."""

    def test_binary_target_up(self) -> None:
        """Price increases -> target = 1."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        target = FeatureAssembler.compute_binary_target(prices, horizon=1)

        # First 5 should be 1 (each next price is higher)
        for i in range(5):
            assert target[i] == 1.0, f"index {i} should be 1.0"

    def test_binary_target_down(self) -> None:
        """Price decreases -> target = 0."""
        prices = np.array([105.0, 104.0, 103.0, 102.0, 101.0, 100.0])
        target = FeatureAssembler.compute_binary_target(prices, horizon=1)

        # First 5 should be 0 (each next price is lower)
        for i in range(5):
            assert target[i] == 0.0, f"index {i} should be 0.0"

    def test_binary_target_horizon(self) -> None:
        """Last ``horizon`` entries are NaN."""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        horizon = 3
        target = FeatureAssembler.compute_binary_target(prices, horizon=horizon)

        assert len(target) == len(prices)
        # First 2 should be numeric
        assert not np.isnan(target[0])
        assert not np.isnan(target[1])
        # Last 3 should be NaN
        assert np.isnan(target[2])
        assert np.isnan(target[3])
        assert np.isnan(target[4])

    def test_binary_target_default_horizon(self) -> None:
        """Default horizon is 5."""
        prices = np.arange(20, dtype=float)
        target = FeatureAssembler.compute_binary_target(prices)

        # Last 5 should be NaN
        for i in range(15, 20):
            assert np.isnan(target[i])
        # First 15 should all be 1.0 (monotonically increasing)
        for i in range(15):
            assert target[i] == 1.0
