"""Tests for SlippageReconciler -- predicted vs actual slippage comparison."""

from __future__ import annotations

import pytest

from hydra.execution.fill_journal import FillJournal, FillRecord
from hydra.execution.reconciler import SlippageReconciler


def _make_fill(
    predicted_slippage: float = 0.20,
    actual_slippage: float = 0.25,
    timestamp: str = "2026-01-15T14:30:00Z",
    symbol: str = "ZO",
) -> FillRecord:
    """Helper to build a FillRecord with configurable slippage."""
    return FillRecord(
        timestamp=timestamp,
        symbol=symbol,
        direction=1,
        n_contracts=2,
        order_price=350.0,
        fill_price=350.0 + actual_slippage,
        predicted_slippage=predicted_slippage,
        actual_slippage=actual_slippage,
        volume_at_fill=500.0,
        spread_at_fill=0.50,
        fill_latency_ms=120.0,
    )


def _insert_n_fills(
    journal: FillJournal,
    n: int,
    predicted: float,
    actual: float,
    symbol: str = "ZO",
) -> None:
    """Insert *n* fills with identical predicted/actual slippage."""
    for i in range(n):
        journal.log_fill(
            _make_fill(
                predicted_slippage=predicted,
                actual_slippage=actual,
                timestamp=f"2026-01-{15 + i:02d}T14:30:00Z",
                symbol=symbol,
            )
        )


@pytest.fixture()
def journal(tmp_path):
    """Create a FillJournal backed by a temporary SQLite database."""
    db_path = tmp_path / "fills.db"
    j = FillJournal(db_path)
    yield j
    j.close()


class TestReconcileInsufficientData:
    """Tests for the minimum fill threshold."""

    def test_fewer_than_10_fills_returns_none(self, journal):
        _insert_n_fills(journal, 5, predicted=0.5, actual=0.5)
        reconciler = SlippageReconciler(journal)
        assert reconciler.reconcile() is None

    def test_exactly_10_fills_returns_report(self, journal):
        _insert_n_fills(journal, 10, predicted=0.5, actual=0.5)
        reconciler = SlippageReconciler(journal)
        report = reconciler.reconcile()
        assert report is not None
        assert report.n_fills == 10


class TestReconcilePerfectCalibration:
    """Tests with predicted == actual (perfectly calibrated model)."""

    def test_bias_near_zero(self, journal):
        _insert_n_fills(journal, 15, predicted=0.5, actual=0.5)
        reconciler = SlippageReconciler(journal)
        report = reconciler.reconcile()

        assert report is not None
        assert abs(report.bias) < 1e-10
        assert abs(report.rmse) < 1e-10
        assert report.pessimism_multiplier == pytest.approx(1.0, abs=1e-6)

    def test_correlation_near_one_with_varying_data(self, journal):
        """With diverse values where predicted == actual, correlation ~ 1."""
        for i in range(15):
            val = 0.1 * (i + 1)
            journal.log_fill(
                _make_fill(
                    predicted_slippage=val,
                    actual_slippage=val,
                    timestamp=f"2026-01-{10 + i:02d}T14:30:00Z",
                )
            )
        reconciler = SlippageReconciler(journal)
        report = reconciler.reconcile()

        assert report is not None
        assert report.correlation == pytest.approx(1.0, abs=1e-6)


class TestReconcileSystematicBias:
    """Tests with systematic underestimation of slippage."""

    def test_positive_bias_when_actual_exceeds_predicted(self, journal):
        _insert_n_fills(journal, 15, predicted=0.5, actual=1.0)
        reconciler = SlippageReconciler(journal)
        report = reconciler.reconcile()

        assert report is not None
        assert report.bias == pytest.approx(0.5, abs=1e-10)
        assert report.mean_predicted == pytest.approx(0.5, abs=1e-10)
        assert report.mean_actual == pytest.approx(1.0, abs=1e-10)

    def test_pessimism_multiplier_two(self, journal):
        _insert_n_fills(journal, 15, predicted=0.5, actual=1.0)
        reconciler = SlippageReconciler(journal)
        report = reconciler.reconcile()

        assert report is not None
        assert report.pessimism_multiplier == pytest.approx(2.0, abs=1e-6)

    def test_rmse_equals_bias_for_constant_offset(self, journal):
        """When every pair has the same offset, RMSE == |bias|."""
        _insert_n_fills(journal, 15, predicted=0.5, actual=1.0)
        reconciler = SlippageReconciler(journal)
        report = reconciler.reconcile()

        assert report is not None
        assert report.rmse == pytest.approx(0.5, abs=1e-10)


class TestIsModelCalibrated:
    """Tests for the is_model_calibrated boolean check."""

    def test_calibrated_returns_true(self, journal):
        """Well-calibrated data should pass the check."""
        for i in range(15):
            val = 0.1 * (i + 1)
            journal.log_fill(
                _make_fill(
                    predicted_slippage=val,
                    actual_slippage=val + 0.01,  # tiny bias
                    timestamp=f"2026-01-{10 + i:02d}T14:30:00Z",
                )
            )
        reconciler = SlippageReconciler(journal)
        calibrated, reason = reconciler.is_model_calibrated()
        assert calibrated is True
        assert reason == "Model calibrated"

    def test_excessive_bias_returns_false(self, journal):
        _insert_n_fills(journal, 15, predicted=0.1, actual=1.0)
        reconciler = SlippageReconciler(journal)
        calibrated, reason = reconciler.is_model_calibrated(max_bias=0.5)
        assert calibrated is False
        assert "Bias" in reason

    def test_insufficient_data_returns_false(self, journal):
        _insert_n_fills(journal, 3, predicted=0.5, actual=0.5)
        reconciler = SlippageReconciler(journal)
        calibrated, reason = reconciler.is_model_calibrated()
        assert calibrated is False
        assert reason == "Insufficient data"
