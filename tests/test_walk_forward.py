"""Tests for PurgedWalkForwardSplit.

Tests verify:
- Correct number of splits produced
- No temporal overlap (train/test isolation)
- Expanding window property
- Embargo gap enforcement
- Minimum train size respected
- Graceful degradation with insufficient data
"""

import numpy as np
import pytest

from hydra.model.walk_forward import PurgedWalkForwardSplit


class TestPurgedWalkForwardSplit:
    """Tests for the purged walk-forward cross-validator."""

    def test_split_count(self):
        """n_splits=5 produces 5 (train, test) pairs with sufficient data."""
        splitter = PurgedWalkForwardSplit(
            n_splits=5, embargo_days=5, min_train_size=100, test_size=50
        )
        # Need: 100 + 5*50 + 5*5 = 100 + 250 + 25 = 375 (last fold needs
        # train_end=100+4*50=300, test_start=305, test_end=355)
        folds = splitter.split(n_samples=400)
        assert len(folds) == 5

    def test_no_overlap(self):
        """For every fold, max(train_idx) + embargo_days < min(test_idx)."""
        splitter = PurgedWalkForwardSplit(
            n_splits=5, embargo_days=5, min_train_size=100, test_size=50
        )
        folds = splitter.split(n_samples=500)

        for train_idx, test_idx in folds:
            # No index appears in both train and test
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            # Embargo gap is respected
            assert train_idx[-1] + splitter.embargo_days < test_idx[0]

    def test_expanding_window(self):
        """Train set grows across folds (len(train_k+1) > len(train_k))."""
        splitter = PurgedWalkForwardSplit(
            n_splits=4, embargo_days=5, min_train_size=100, test_size=50
        )
        folds = splitter.split(n_samples=500)
        assert len(folds) >= 2

        for k in range(len(folds) - 1):
            assert len(folds[k + 1][0]) > len(folds[k][0])

    def test_embargo_gap(self):
        """Gap between train end and test start >= embargo_days."""
        embargo = 10
        splitter = PurgedWalkForwardSplit(
            n_splits=3, embargo_days=embargo, min_train_size=100, test_size=50
        )
        folds = splitter.split(n_samples=500)

        for train_idx, test_idx in folds:
            gap = test_idx[0] - train_idx[-1]
            # gap should be embargo_days + 1 (since train_end is exclusive but
            # the last train index is train_end - 1)
            assert gap >= embargo

    def test_min_train_size_respected(self):
        """First fold has at least min_train_size training samples."""
        min_train = 200
        splitter = PurgedWalkForwardSplit(
            n_splits=3, embargo_days=5, min_train_size=min_train, test_size=50
        )
        folds = splitter.split(n_samples=500)
        assert len(folds) > 0
        assert len(folds[0][0]) >= min_train

    def test_insufficient_data_fewer_splits(self):
        """If n_samples too small for n_splits, return fewer folds."""
        splitter = PurgedWalkForwardSplit(
            n_splits=10, embargo_days=5, min_train_size=100, test_size=50
        )
        # Only enough data for a few folds
        folds = splitter.split(n_samples=300)
        assert len(folds) < 10
        assert len(folds) > 0

    def test_zero_samples_empty(self):
        """Zero samples produces no folds."""
        splitter = PurgedWalkForwardSplit(
            n_splits=5, embargo_days=5, min_train_size=100, test_size=50
        )
        folds = splitter.split(n_samples=0)
        assert len(folds) == 0

    def test_test_size_consistent(self):
        """Each fold has exactly test_size test samples."""
        test_size = 63
        splitter = PurgedWalkForwardSplit(
            n_splits=3, embargo_days=5, min_train_size=100, test_size=test_size
        )
        folds = splitter.split(n_samples=500)

        for _, test_idx in folds:
            assert len(test_idx) == test_size

    def test_get_n_splits(self):
        """get_n_splits returns configured number."""
        splitter = PurgedWalkForwardSplit(n_splits=7)
        assert splitter.get_n_splits() == 7
