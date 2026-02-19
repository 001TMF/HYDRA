"""Tests for the COTIngestPipeline.

The cot_reports library is mocked -- no real CFTC downloads are made.

The COT timing test is the MOST CRITICAL test in Phase 1. If this fails,
every backtest built on top will have lookahead bias (COT data collected
Tuesday used as if available Tuesday, but it's not released until Friday).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hydra.data.ingestion.cot import COTIngestPipeline, _next_friday
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def parquet_lake(tmp_parquet_dir):
    """ParquetLake backed by a temp directory."""
    return ParquetLake(tmp_parquet_dir)


@pytest.fixture
def feature_store(tmp_feature_db):
    """FeatureStore backed by a temp SQLite database."""
    fs = FeatureStore(tmp_feature_db)
    yield fs
    fs.close()


def _make_mock_cot_df(cftc_code: str = "054642", num_weeks: int = 4):
    """Create a realistic mock COT DataFrame.

    Simulates weekly COT reports from Tuesdays, starting from 2026-01-06.
    """
    records = []
    base_date = datetime(2026, 1, 6)  # A Tuesday

    for i in range(num_weeks):
        report_date = base_date + timedelta(weeks=i)
        records.append({
            "CFTC_Contract_Market_Code": cftc_code,
            "As_of_Date_In_Form_YYMMDD": report_date.strftime("%Y-%m-%d"),
            "M_Money_Positions_Long_All": 15000 + i * 100,
            "M_Money_Positions_Short_All": 10000 + i * 50,
            "Prod_Merc_Positions_Long_All": 8000 + i * 80,
            "Prod_Merc_Positions_Short_All": 12000 + i * 60,
            "Swap_Positions_Long_All": 5000 + i * 30,
            "Swap_Positions_Short_All": 4000 + i * 20,
            "Open_Interest_All": 50000 + i * 500,
        })

    return pd.DataFrame(records)


@pytest.fixture
def mock_cot_data():
    """Create a mock COT DataFrame for lean hogs."""
    return _make_mock_cot_df()


@pytest.fixture
def pipeline(parquet_lake, feature_store):
    """COTIngestPipeline with mocked cot_reports library."""
    p = COTIngestPipeline(
        parquet_lake=parquet_lake,
        feature_store=feature_store,
        cftc_code="054642",
        redownload_weeks=4,
    )
    return p


# ---------------------------------------------------------------------------
# Tests: _next_friday helper
# ---------------------------------------------------------------------------

class TestNextFriday:
    """Tests for the _next_friday() helper function."""

    def test_tuesday_to_friday(self):
        """Tuesday Feb 17, 2026 -> Friday Feb 20, 2026 at 20:30 UTC."""
        tuesday = datetime(2026, 2, 17, tzinfo=timezone.utc)
        result = _next_friday(tuesday)

        assert result.weekday() == 4  # Friday
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 20
        assert result.hour == 20
        assert result.minute == 30
        assert result.tzinfo == timezone.utc

    def test_wednesday_to_friday(self):
        """Wednesday Feb 18 -> Friday Feb 20 same week."""
        wednesday = datetime(2026, 2, 18, tzinfo=timezone.utc)
        result = _next_friday(wednesday)
        assert result.day == 20
        assert result.weekday() == 4

    def test_monday_to_friday(self):
        """Monday Feb 16 -> Friday Feb 20 same week."""
        monday = datetime(2026, 2, 16, tzinfo=timezone.utc)
        result = _next_friday(monday)
        assert result.day == 20
        assert result.weekday() == 4

    def test_friday_goes_to_next_friday(self):
        """Friday Feb 20 -> Friday Feb 27 (next week, not same day)."""
        friday = datetime(2026, 2, 20, tzinfo=timezone.utc)
        result = _next_friday(friday)
        assert result.day == 27
        assert result.weekday() == 4

    def test_saturday_to_next_friday(self):
        """Saturday Feb 21 -> Friday Feb 27."""
        saturday = datetime(2026, 2, 21, tzinfo=timezone.utc)
        result = _next_friday(saturday)
        assert result.day == 27
        assert result.weekday() == 4

    def test_result_is_utc(self):
        """Result should always be timezone-aware UTC."""
        tuesday = datetime(2026, 2, 17, tzinfo=timezone.utc)
        result = _next_friday(tuesday)
        assert result.tzinfo is not None
        assert result.tzinfo == timezone.utc

    def test_release_time_is_2030_utc(self):
        """Release time should be 20:30 UTC (15:30 ET)."""
        tuesday = datetime(2026, 2, 17, tzinfo=timezone.utc)
        result = _next_friday(tuesday)
        assert result.hour == 20
        assert result.minute == 30
        assert result.second == 0


# ---------------------------------------------------------------------------
# Tests: COT fetch
# ---------------------------------------------------------------------------

class TestCOTFetch:
    """Tests for COTIngestPipeline.fetch()."""

    def test_fetch_filters_by_cftc_code(self, pipeline, mock_cot_data):
        """fetch() should only return rows matching the CFTC code."""
        # Add some rows for a different contract
        other_df = _make_mock_cot_df(cftc_code="099999", num_weeks=2)
        combined = pd.concat([mock_cot_data, other_df], ignore_index=True)

        with patch("hydra.data.ingestion.cot.cot") as mock_cot:
            mock_cot.cot_year.return_value = combined
            date = datetime(2026, 2, 18, tzinfo=timezone.utc)
            result = pipeline.fetch("HE", date)

        # Should only get lean hogs records (054642), not the other ones
        assert len(result["records"]) == 4

    def test_fetch_returns_position_data(self, pipeline, mock_cot_data):
        """fetch() should return complete position data."""
        with patch("hydra.data.ingestion.cot.cot") as mock_cot:
            mock_cot.cot_year.return_value = mock_cot_data
            date = datetime(2026, 2, 18, tzinfo=timezone.utc)
            result = pipeline.fetch("HE", date)

        rec = result["records"][0]
        assert "managed_money_net" in rec
        assert "producer_net" in rec
        assert "swap_net" in rec
        assert "total_oi" in rec
        assert "report_date" in rec
        assert "available_at" in rec

    def test_revision_redownload(self, pipeline, mock_cot_data):
        """Records in the revision window should be flagged."""
        with patch("hydra.data.ingestion.cot.cot") as mock_cot:
            mock_cot.cot_year.return_value = mock_cot_data
            # Date is far enough forward that earlier records are outside window
            date = datetime(2026, 2, 18, tzinfo=timezone.utc)
            result = pipeline.fetch("HE", date)

        # At least some records should have is_revision_window flag
        revision_records = [r for r in result["records"] if r["is_revision_window"]]
        non_revision = [r for r in result["records"] if not r["is_revision_window"]]
        # With 4 weeks of data and a 4-week redownload window relative to Feb 18,
        # records from Jan 6, 13 are outside the window, Jan 20, 27 are inside
        assert len(revision_records) >= 1


# ---------------------------------------------------------------------------
# Tests: COT validation
# ---------------------------------------------------------------------------

class TestCOTValidation:
    """Tests for COTIngestPipeline.validate()."""

    def test_validate_passes_good_data(self, pipeline):
        """Good COT data should pass without warnings."""
        raw = {
            "records": [
                {
                    "report_date": datetime(2026, 2, 17, tzinfo=timezone.utc),
                    "available_at": datetime(2026, 2, 20, 20, 30, tzinfo=timezone.utc),
                    "managed_money_long": 15000, "managed_money_short": 10000,
                    "managed_money_net": 5000,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }
        cleaned, warnings = pipeline.validate(raw)
        assert len(cleaned["records"]) == 1
        assert len(warnings) == 0

    def test_validate_catches_negative_positions(self, pipeline):
        """Negative position values should be flagged."""
        raw = {
            "records": [
                {
                    "report_date": datetime(2026, 2, 17, tzinfo=timezone.utc),
                    "available_at": datetime(2026, 2, 20, 20, 30, tzinfo=timezone.utc),
                    "managed_money_long": -100, "managed_money_short": 10000,
                    "managed_money_net": -10100,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }
        cleaned, warnings = pipeline.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("managed_money_long" in w for w in warnings)

    def test_validate_warns_on_empty_records(self, pipeline):
        """Empty records should produce a warning."""
        raw = {"records": []}
        cleaned, warnings = pipeline.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("No COT" in w for w in warnings)


# ---------------------------------------------------------------------------
# Tests: COT timing (CRITICAL -- prevents lookahead bias)
# ---------------------------------------------------------------------------

class TestCOTTiming:
    """CRITICAL tests for COT as_of/available_at timing semantics.

    These tests verify that COT data collected on Tuesday is NOT available
    to queries until Friday. Getting this wrong introduces lookahead bias
    that invalidates all backtest results.
    """

    def test_as_of_available_at_timing(self, pipeline, feature_store):
        """COT data should have as_of=Tuesday, available_at=next Friday 20:30 UTC."""
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        # Tuesday Feb 17 report
        report_date = datetime(2026, 2, 17, tzinfo=timezone.utc)
        expected_available = datetime(2026, 2, 20, 20, 30, 0, tzinfo=timezone.utc)

        data = {
            "records": [
                {
                    "report_date": report_date,
                    "available_at": expected_available,
                    "managed_money_long": 15000, "managed_money_short": 10000,
                    "managed_money_net": 5000,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }

        pipeline.persist(data, "HE", date)

        # Query on Thursday (before Friday release) -- should NOT see COT data
        thursday = datetime(2026, 2, 19, 12, 0, 0, tzinfo=timezone.utc)
        features_thursday = feature_store.get_features_at("HE", thursday)
        assert "cot_managed_money_net" not in features_thursday

        # Query on Saturday (after Friday release) -- SHOULD see COT data
        saturday = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)
        features_saturday = feature_store.get_features_at("HE", saturday)
        assert "cot_managed_money_net" in features_saturday
        assert features_saturday["cot_managed_money_net"] == 5000.0

    def test_features_not_available_on_wednesday(self, pipeline, feature_store):
        """COT data from Tuesday should NOT be available on Wednesday.

        This is the canonical lookahead-bias prevention test.
        """
        report_tuesday = datetime(2026, 2, 17, tzinfo=timezone.utc)
        available_friday = _next_friday(report_tuesday)

        data = {
            "records": [
                {
                    "report_date": report_tuesday,
                    "available_at": available_friday,
                    "managed_money_long": 15000, "managed_money_short": 10000,
                    "managed_money_net": 5000,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        pipeline.persist(data, "HE", date)

        # Wednesday query: data NOT available
        wednesday = datetime(2026, 2, 18, 12, 0, 0, tzinfo=timezone.utc)
        features = feature_store.get_features_at("HE", wednesday)
        assert "cot_managed_money_net" not in features
        assert "cot_producer_net" not in features
        assert "cot_swap_net" not in features
        assert "cot_total_oi" not in features

    def test_features_available_after_friday_release(self, pipeline, feature_store):
        """COT data should become available after Friday 20:30 UTC."""
        report_tuesday = datetime(2026, 2, 17, tzinfo=timezone.utc)
        available_friday = _next_friday(report_tuesday)

        data = {
            "records": [
                {
                    "report_date": report_tuesday,
                    "available_at": available_friday,
                    "managed_money_long": 15000, "managed_money_short": 10000,
                    "managed_money_net": 5000,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        pipeline.persist(data, "HE", date)

        # Just before release: NOT available
        just_before = datetime(2026, 2, 20, 20, 29, 0, tzinfo=timezone.utc)
        features_before = feature_store.get_features_at("HE", just_before)
        assert "cot_managed_money_net" not in features_before

        # At release time: AVAILABLE
        at_release = datetime(2026, 2, 20, 20, 30, 0, tzinfo=timezone.utc)
        features_at = feature_store.get_features_at("HE", at_release)
        assert "cot_managed_money_net" in features_at

        # After release: AVAILABLE
        after_release = datetime(2026, 2, 20, 21, 0, 0, tzinfo=timezone.utc)
        features_after = feature_store.get_features_at("HE", after_release)
        assert "cot_managed_money_net" in features_after
        assert features_after["cot_managed_money_net"] == 5000.0

    def test_all_four_features_written_with_timing(self, pipeline, feature_store):
        """All four COT features should be written with correct timing."""
        report_tuesday = datetime(2026, 2, 17, tzinfo=timezone.utc)
        available_friday = _next_friday(report_tuesday)

        data = {
            "records": [
                {
                    "report_date": report_tuesday,
                    "available_at": available_friday,
                    "managed_money_long": 15000, "managed_money_short": 10000,
                    "managed_money_net": 5000,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        pipeline.persist(data, "HE", date)

        # Query after release
        saturday = datetime(2026, 2, 21, 12, 0, 0, tzinfo=timezone.utc)
        features = feature_store.get_features_at("HE", saturday)

        assert features["cot_managed_money_net"] == 5000.0
        assert features["cot_producer_net"] == -4000.0
        assert features["cot_swap_net"] == 1000.0
        assert features["cot_total_oi"] == 50000.0


# ---------------------------------------------------------------------------
# Tests: COT persist
# ---------------------------------------------------------------------------

class TestCOTPersist:
    """Tests for COTIngestPipeline.persist()."""

    def test_persist_writes_to_parquet(self, pipeline):
        """persist() should write COT data to ParquetLake with data_type='cot'."""
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        report_date = datetime(2026, 2, 17, tzinfo=timezone.utc)

        data = {
            "records": [
                {
                    "report_date": report_date,
                    "available_at": _next_friday(report_date),
                    "managed_money_long": 15000, "managed_money_short": 10000,
                    "managed_money_net": 5000,
                    "producer_long": 8000, "producer_short": 12000,
                    "producer_net": -4000,
                    "swap_long": 5000, "swap_short": 4000, "swap_net": 1000,
                    "total_oi": 50000,
                    "is_revision_window": False,
                },
            ]
        }

        pipeline.persist(data, "HE", date)

        result = pipeline.parquet_lake.read(data_type="cot", market="HE")
        assert len(result) == 1

    def test_persist_skips_empty_records(self, pipeline):
        """persist() should not write anything when there are no records."""
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        pipeline.persist({"records": []}, "HE", date)
