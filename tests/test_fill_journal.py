"""Tests for FillJournal -- SQLite fill logging with slippage tracking."""

from __future__ import annotations

import pytest

from hydra.execution.fill_journal import FillJournal, FillRecord


def _make_fill(
    symbol: str = "ZO",
    timestamp: str = "2026-01-15T14:30:00Z",
    direction: int = 1,
    n_contracts: int = 2,
    order_price: float = 350.0,
    fill_price: float = 350.25,
    predicted_slippage: float = 0.20,
    actual_slippage: float = 0.25,
    volume_at_fill: float = 500.0,
    spread_at_fill: float = 0.50,
    fill_latency_ms: float = 120.0,
    order_id: int | None = None,
) -> FillRecord:
    """Helper to build a FillRecord with sensible defaults."""
    return FillRecord(
        timestamp=timestamp,
        symbol=symbol,
        direction=direction,
        n_contracts=n_contracts,
        order_price=order_price,
        fill_price=fill_price,
        predicted_slippage=predicted_slippage,
        actual_slippage=actual_slippage,
        volume_at_fill=volume_at_fill,
        spread_at_fill=spread_at_fill,
        fill_latency_ms=fill_latency_ms,
        order_id=order_id,
    )


@pytest.fixture()
def journal(tmp_path):
    """Create a FillJournal backed by a temporary SQLite database."""
    db_path = tmp_path / "fills.db"
    j = FillJournal(db_path)
    yield j
    j.close()


class TestLogFill:
    """Tests for log_fill insertion."""

    def test_log_fill_returns_auto_incremented_id(self, journal):
        fill1 = _make_fill()
        id1 = journal.log_fill(fill1)
        assert id1 == 1

        fill2 = _make_fill(timestamp="2026-01-15T14:31:00Z")
        id2 = journal.log_fill(fill2)
        assert id2 == 2

    def test_log_fill_stores_all_fields(self, journal):
        fill = _make_fill(
            symbol="HE",
            timestamp="2026-02-01T10:00:00Z",
            direction=-1,
            n_contracts=5,
            order_price=100.0,
            fill_price=100.50,
            predicted_slippage=0.30,
            actual_slippage=0.50,
            volume_at_fill=1200.0,
            spread_at_fill=0.25,
            fill_latency_ms=85.0,
            order_id=12345,
        )
        journal.log_fill(fill)

        records = journal.get_fills(symbol="HE")
        assert len(records) == 1
        r = records[0]
        assert r.symbol == "HE"
        assert r.direction == -1
        assert r.n_contracts == 5
        assert r.order_price == 100.0
        assert r.fill_price == 100.50
        assert r.predicted_slippage == 0.30
        assert r.actual_slippage == 0.50
        assert r.volume_at_fill == 1200.0
        assert r.spread_at_fill == 0.25
        assert r.fill_latency_ms == 85.0
        assert r.order_id == 12345


class TestGetFills:
    """Tests for get_fills querying."""

    def test_get_fills_filtered_by_symbol(self, journal):
        journal.log_fill(_make_fill(symbol="ZO"))
        journal.log_fill(_make_fill(symbol="HE"))
        journal.log_fill(_make_fill(symbol="ZO", timestamp="2026-01-15T14:31:00Z"))

        zo_fills = journal.get_fills(symbol="ZO")
        assert len(zo_fills) == 2
        assert all(f.symbol == "ZO" for f in zo_fills)

        he_fills = journal.get_fills(symbol="HE")
        assert len(he_fills) == 1
        assert he_fills[0].symbol == "HE"

    def test_get_fills_filtered_by_date_range(self, journal):
        journal.log_fill(_make_fill(timestamp="2026-01-10T10:00:00Z"))
        journal.log_fill(_make_fill(timestamp="2026-01-15T10:00:00Z"))
        journal.log_fill(_make_fill(timestamp="2026-01-20T10:00:00Z"))

        fills = journal.get_fills(since="2026-01-12T00:00:00Z", until="2026-01-18T00:00:00Z")
        assert len(fills) == 1
        assert fills[0].timestamp == "2026-01-15T10:00:00Z"

    def test_get_fills_respects_limit(self, journal):
        for i in range(5):
            journal.log_fill(_make_fill(timestamp=f"2026-01-{10+i:02d}T10:00:00Z"))

        fills = journal.get_fills(limit=3)
        assert len(fills) == 3

    def test_get_fills_ordered_by_timestamp_desc(self, journal):
        journal.log_fill(_make_fill(timestamp="2026-01-10T10:00:00Z"))
        journal.log_fill(_make_fill(timestamp="2026-01-20T10:00:00Z"))
        journal.log_fill(_make_fill(timestamp="2026-01-15T10:00:00Z"))

        fills = journal.get_fills()
        assert fills[0].timestamp == "2026-01-20T10:00:00Z"
        assert fills[1].timestamp == "2026-01-15T10:00:00Z"
        assert fills[2].timestamp == "2026-01-10T10:00:00Z"


class TestGetSlippagePairs:
    """Tests for get_slippage_pairs extraction."""

    def test_returns_predicted_actual_tuples(self, journal):
        journal.log_fill(_make_fill(predicted_slippage=0.20, actual_slippage=0.25))
        journal.log_fill(
            _make_fill(
                predicted_slippage=0.30,
                actual_slippage=0.50,
                timestamp="2026-01-15T14:31:00Z",
            )
        )

        pairs = journal.get_slippage_pairs()
        assert len(pairs) == 2
        # Each pair is (predicted, actual)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        # Check values (ordered by timestamp DESC)
        assert pairs[0] == (0.30, 0.50)
        assert pairs[1] == (0.20, 0.25)

    def test_slippage_pairs_filtered_by_symbol(self, journal):
        journal.log_fill(_make_fill(symbol="ZO", predicted_slippage=0.20, actual_slippage=0.25))
        journal.log_fill(
            _make_fill(
                symbol="HE",
                predicted_slippage=0.30,
                actual_slippage=0.50,
                timestamp="2026-01-15T14:31:00Z",
            )
        )

        zo_pairs = journal.get_slippage_pairs(symbol="ZO")
        assert len(zo_pairs) == 1
        assert zo_pairs[0] == (0.20, 0.25)


class TestCount:
    """Tests for count method."""

    def test_count_returns_correct_count(self, journal):
        assert journal.count() == 0
        journal.log_fill(_make_fill())
        assert journal.count() == 1
        journal.log_fill(_make_fill(timestamp="2026-01-15T14:31:00Z"))
        journal.log_fill(_make_fill(timestamp="2026-01-15T14:32:00Z"))
        assert journal.count() == 3


class TestEmptyJournal:
    """Tests for empty journal edge cases."""

    def test_empty_get_fills(self, journal):
        assert journal.get_fills() == []

    def test_empty_get_slippage_pairs(self, journal):
        assert journal.get_slippage_pairs() == []

    def test_empty_count(self, journal):
        assert journal.count() == 0
