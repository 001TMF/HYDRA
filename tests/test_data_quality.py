"""Tests for data quality monitoring: staleness, validators, and reporting.

Tests use configurable thresholds from a config dict (not hardcoded values)
and exercise weekend-aware staleness, options chain quality checks, and
the overall report status determination.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from hydra.data.quality import (
    COTFreshnessReport,
    DataQualityMonitor,
    OptionsQualityReport,
    QualityReport,
    StalenessAlert,
    is_trading_day,
)
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "staleness_thresholds": {
        "futures_days": 1,
        "options_days": 1,
        "cot_days": 7,
    },
    "quality": {
        "min_liquid_strikes": 8,
        "max_spread_pct": 0.20,
        "min_oi": 50,
    },
}


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def lake_and_store(tmp_dir):
    lake = ParquetLake(tmp_dir / "lake")
    store = FeatureStore(tmp_dir / "features.db")
    return lake, store


@pytest.fixture
def monitor(lake_and_store):
    lake, store = lake_and_store
    return DataQualityMonitor(lake, store, DEFAULT_CONFIG)


def _write_timestamp_feature(store: FeatureStore, market: str, source: str, ts: datetime):
    """Helper to write a last_timestamp feature to the store."""
    store.write_feature(
        market=market,
        feature_name=f"{source}_last_timestamp",
        as_of=ts,
        available_at=ts,
        value=ts.timestamp(),
    )


# ---------------------------------------------------------------------------
# Staleness detection tests
# ---------------------------------------------------------------------------


class TestStalenessDetection:
    def test_staleness_detects_stale_futures(self, lake_and_store, monitor):
        """Futures data 3 days ago checked on a weekday -> stale."""
        _, store = lake_and_store
        # Wednesday check, data from previous Friday (3 calendar days, 2 trading days)
        now = datetime(2026, 2, 18, 12, 0)  # Wednesday
        old = datetime(2026, 2, 13, 12, 0)  # Friday (data age: Mon + Tue + Wed = 3 trading days)
        _write_timestamp_feature(store, "HE", "futures", old)

        alerts = monitor.check_staleness("HE", now)
        futures_alert = next(a for a in alerts if a.source == "futures")
        assert futures_alert.is_stale is True
        assert futures_alert.days_since_update is not None
        assert futures_alert.days_since_update > 1

    def test_staleness_ignores_weekends(self, lake_and_store, monitor):
        """Futures data from Friday checked on Monday -> NOT stale (0 trading days gap)."""
        _, store = lake_and_store
        friday = datetime(2026, 2, 13, 16, 0)  # Friday
        monday = datetime(2026, 2, 16, 9, 0)  # Monday (next trading day)
        _write_timestamp_feature(store, "HE", "futures", friday)

        alerts = monitor.check_staleness("HE", monday)
        futures_alert = next(a for a in alerts if a.source == "futures")
        # Only 1 trading day gap (Monday), threshold is 1 -> not stale
        assert futures_alert.is_stale is False

    def test_weekend_aware_staleness_saturday_sunday(self, lake_and_store, monitor):
        """Friday data checked on Saturday/Sunday -> not stale."""
        _, store = lake_and_store
        friday = datetime(2026, 2, 13, 16, 0)
        saturday = datetime(2026, 2, 14, 12, 0)
        sunday = datetime(2026, 2, 15, 12, 0)

        _write_timestamp_feature(store, "HE", "futures", friday)

        # Saturday: 0 trading days since Friday
        alerts_sat = monitor.check_staleness("HE", saturday)
        futures_sat = next(a for a in alerts_sat if a.source == "futures")
        assert futures_sat.is_stale is False

        # Sunday: still 0 trading days since Friday
        alerts_sun = monitor.check_staleness("HE", sunday)
        futures_sun = next(a for a in alerts_sun if a.source == "futures")
        assert futures_sun.is_stale is False

    def test_staleness_no_data_is_stale(self, monitor):
        """No data at all -> stale for all sources."""
        now = datetime(2026, 2, 18, 12, 0)
        alerts = monitor.check_staleness("HE", now)
        assert len(alerts) == 3
        for alert in alerts:
            assert alert.is_stale is True
            assert alert.last_update is None

    def test_cot_freshness_within_threshold(self, lake_and_store, monitor):
        """COT data from 5 days ago -> fresh."""
        _, store = lake_and_store
        now = datetime(2026, 2, 18, 12, 0)

        # 5 days ago -> fresh (threshold is 7)
        recent = now - timedelta(days=5)
        _write_timestamp_feature(store, "HE", "cot", recent)
        report_fresh = monitor.check_cot_freshness("HE", now)
        assert report_fresh.is_stale is False
        assert report_fresh.days_since_update is not None
        assert report_fresh.days_since_update < 7

    def test_cot_freshness_stale_beyond_threshold(self, tmp_dir):
        """COT data from 10 days ago -> stale."""
        lake = ParquetLake(tmp_dir / "lake_cot_stale")
        store = FeatureStore(tmp_dir / "features_cot_stale.db")
        monitor = DataQualityMonitor(lake, store, DEFAULT_CONFIG)
        now = datetime(2026, 2, 18, 12, 0)

        # 10 days ago -> stale
        old = now - timedelta(days=10)
        _write_timestamp_feature(store, "HE", "cot", old)
        report_stale = monitor.check_cot_freshness("HE", now)
        assert report_stale.is_stale is True
        assert report_stale.days_since_update is not None
        assert report_stale.days_since_update > 7


# ---------------------------------------------------------------------------
# Options quality tests
# ---------------------------------------------------------------------------


class TestOptionsQuality:
    def test_options_quality_counts_liquid_strikes(self, lake_and_store, monitor):
        """Create chain with mixed quality, verify liquid count."""
        lake, _ = lake_and_store

        # Create 12 strikes: 9 liquid, 3 illiquid
        strikes = list(range(80, 92))
        oi = [100] * 9 + [10, 5, 2]  # 9 above min_oi, 3 below
        spreads = [0.05] * 9 + [0.05, 0.05, 0.05]  # all low spread
        call_prices = [12.0, 10.0, 8.0, 6.0, 4.5, 3.0, 2.0, 1.2, 0.6, 0.3, 0.1, 0.05]

        table = pa.table({
            "strike": strikes,
            "oi": oi,
            "bid_ask_spread_pct": spreads,
            "call_price": call_prices,
        })

        lake.write(table, "options", "HE", datetime(2026, 2, 18))

        report = monitor.check_options_quality("HE", datetime(2026, 2, 18))
        assert report.liquid_strike_count == 9
        assert report.total_strikes == 12

    def test_call_price_monotonicity_pass(self, lake_and_store, monitor):
        """Monotonically decreasing call prices -> no arbitrage."""
        lake, _ = lake_and_store

        strikes = list(range(80, 90))
        call_prices = [10.0, 8.5, 7.0, 5.5, 4.2, 3.0, 2.0, 1.2, 0.6, 0.2]
        oi = [100] * 10
        spreads = [0.05] * 10

        table = pa.table({
            "strike": strikes,
            "oi": oi,
            "bid_ask_spread_pct": spreads,
            "call_price": call_prices,
        })

        lake.write(table, "options", "HE", datetime(2026, 2, 18))

        report = monitor.check_options_quality("HE", datetime(2026, 2, 18))
        assert report.has_arbitrage_violation is False

    def test_call_price_monotonicity_fail(self, lake_and_store, monitor):
        """Non-monotonic call prices -> arbitrage detected."""
        lake, _ = lake_and_store

        strikes = list(range(80, 90))
        # Non-monotonic: price goes UP at strike 85
        call_prices = [10.0, 8.5, 7.0, 5.5, 4.2, 5.0, 2.0, 1.2, 0.6, 0.2]
        oi = [100] * 10
        spreads = [0.05] * 10

        table = pa.table({
            "strike": strikes,
            "oi": oi,
            "bid_ask_spread_pct": spreads,
            "call_price": call_prices,
        })

        lake.write(table, "options", "HE", datetime(2026, 2, 18))

        report = monitor.check_options_quality("HE", datetime(2026, 2, 18))
        assert report.has_arbitrage_violation is True

    def test_empty_options_returns_zero(self, monitor):
        """No options data -> zero strikes, no errors."""
        report = monitor.check_options_quality("HE", datetime(2026, 2, 18))
        assert report.liquid_strike_count == 0
        assert report.total_strikes == 0
        assert report.has_arbitrage_violation is False


# ---------------------------------------------------------------------------
# Report and overall status tests
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_generate_report_all_healthy(self, lake_and_store, monitor):
        """All healthy -> overall status 'healthy'."""
        lake, store = lake_and_store
        now = datetime(2026, 2, 18, 12, 0)

        # Write fresh data for all sources
        _write_timestamp_feature(store, "HE", "futures", now - timedelta(hours=2))
        _write_timestamp_feature(store, "HE", "options", now - timedelta(hours=2))
        _write_timestamp_feature(store, "HE", "cot", now - timedelta(days=3))

        # Write a good options chain with enough liquid strikes
        strikes = list(range(80, 92))
        call_prices = [12.0, 10.0, 8.0, 6.0, 4.5, 3.0, 2.0, 1.2, 0.6, 0.3, 0.1, 0.05]
        oi = [100] * 12
        spreads = [0.05] * 12

        table = pa.table({
            "strike": strikes,
            "oi": oi,
            "bid_ask_spread_pct": spreads,
            "call_price": call_prices,
        })
        lake.write(table, "options", "HE", now)

        report = monitor.generate_report("HE", now)
        assert report.overall_status == "healthy"
        assert report.market == "HE"

    def test_generate_report_stale_overrides_all(self, lake_and_store, monitor):
        """Any stale source -> overall status 'stale'."""
        lake, store = lake_and_store
        now = datetime(2026, 2, 18, 12, 0)

        # Futures fresh, but COT very old
        _write_timestamp_feature(store, "HE", "futures", now - timedelta(hours=2))
        _write_timestamp_feature(store, "HE", "options", now - timedelta(hours=2))
        _write_timestamp_feature(store, "HE", "cot", now - timedelta(days=15))

        # Good options chain
        strikes = list(range(80, 92))
        call_prices = [12.0, 10.0, 8.0, 6.0, 4.5, 3.0, 2.0, 1.2, 0.6, 0.3, 0.1, 0.05]
        oi = [100] * 12
        spreads = [0.05] * 12
        table = pa.table({
            "strike": strikes,
            "oi": oi,
            "bid_ask_spread_pct": spreads,
            "call_price": call_prices,
        })
        lake.write(table, "options", "HE", now)

        report = monitor.generate_report("HE", now)
        assert report.overall_status == "stale"

    def test_generate_report_degraded_when_low_liquidity(self, lake_and_store, monitor):
        """Low liquidity (no stale) -> overall status 'degraded'."""
        lake, store = lake_and_store
        now = datetime(2026, 2, 18, 12, 0)

        # All data fresh
        _write_timestamp_feature(store, "HE", "futures", now - timedelta(hours=2))
        _write_timestamp_feature(store, "HE", "options", now - timedelta(hours=2))
        _write_timestamp_feature(store, "HE", "cot", now - timedelta(days=3))

        # Poor options chain: only 4 liquid strikes
        strikes = list(range(80, 88))
        call_prices = [8.0, 6.0, 4.0, 2.5, 1.5, 0.8, 0.3, 0.1]
        oi = [100, 100, 100, 100, 10, 10, 10, 10]  # only 4 liquid
        spreads = [0.05] * 8
        table = pa.table({
            "strike": strikes,
            "oi": oi,
            "bid_ask_spread_pct": spreads,
            "call_price": call_prices,
        })
        lake.write(table, "options", "HE", now)

        report = monitor.generate_report("HE", now)
        assert report.overall_status == "degraded"


# ---------------------------------------------------------------------------
# is_trading_day helper tests
# ---------------------------------------------------------------------------


class TestIsTradingDay:
    def test_weekday_is_trading_day(self):
        assert is_trading_day(datetime(2026, 2, 16)) is True  # Monday
        assert is_trading_day(datetime(2026, 2, 17)) is True  # Tuesday
        assert is_trading_day(datetime(2026, 2, 18)) is True  # Wednesday
        assert is_trading_day(datetime(2026, 2, 19)) is True  # Thursday
        assert is_trading_day(datetime(2026, 2, 20)) is True  # Friday

    def test_weekend_is_not_trading_day(self):
        assert is_trading_day(datetime(2026, 2, 14)) is False  # Saturday
        assert is_trading_day(datetime(2026, 2, 15)) is False  # Sunday
