"""Tests for the OptionsIngestPipeline.

All Databento API calls are mocked -- no real network requests are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hydra.data.ingestion.options import OptionsIngestPipeline
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


def _make_mock_mbp_df():
    """Create a mock mbp-1 DataFrame with bid/ask data."""
    return pd.DataFrame({
        "instrument_id": [1001, 1002, 1003, 1004],
        "bid_px_00": [2.50, 1.80, 3.10, 0.90],
        "ask_px_00": [2.70, 2.00, 3.30, 1.10],
        "bid_sz_00": [50, 30, 40, 20],
        "ask_sz_00": [55, 35, 45, 25],
        "volume": [100, 80, 120, 60],
    })


def _make_mock_def_df():
    """Create a mock definition DataFrame with strike/expiry metadata."""
    return pd.DataFrame({
        "instrument_id": [1001, 1002, 1003, 1004],
        "strike_price": [60.0, 65.0, 60.0, 65.0],
        "expiration": [
            "2026-04-15", "2026-04-15", "2026-04-15", "2026-04-15"
        ],
        "instrument_class": ["C", "C", "P", "P"],
    })


def _make_mock_stat_df():
    """Create a mock statistics DataFrame with OI data."""
    return pd.DataFrame({
        "instrument_id": [1001, 1002, 1003, 1004],
        "open_interest": [200, 150, 180, 90],
    })


@pytest.fixture
def mock_databento_responses():
    """Create mock Databento responses for all three schema requests."""
    mbp_mock = MagicMock()
    mbp_mock.to_df.return_value = _make_mock_mbp_df()

    def_mock = MagicMock()
    def_mock.to_df.return_value = _make_mock_def_df()

    stat_mock = MagicMock()
    stat_mock.to_df.return_value = _make_mock_stat_df()

    return mbp_mock, def_mock, stat_mock


@pytest.fixture
def pipeline(parquet_lake, feature_store):
    """OptionsIngestPipeline with a mocked Databento client."""
    with patch("hydra.data.ingestion.options.db") as mock_db:
        p = OptionsIngestPipeline(
            api_key="TEST_KEY",
            parquet_lake=parquet_lake,
            feature_store=feature_store,
        )
        yield p, mock_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptionsFetch:
    """Tests for OptionsIngestPipeline.fetch()."""

    def test_fetch_joins_mbp_definition_statistics(
        self, pipeline, mock_databento_responses
    ):
        """fetch() should join all three schemas into a complete chain."""
        p, _ = pipeline
        mbp_mock, def_mock, stat_mock = mock_databento_responses
        p.client.timeseries.get_range.side_effect = [mbp_mock, def_mock, stat_mock]

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        result = p.fetch("HE", date)

        assert "records" in result
        assert len(result["records"]) == 4

        # Check that join produced complete records
        rec = result["records"][0]
        assert "strike" in rec
        assert "expiry" in rec
        assert "bid" in rec
        assert "ask" in rec
        assert "oi" in rec
        assert "is_call" in rec
        assert rec["strike"] == 60.0
        assert rec["bid"] == 2.50
        assert rec["ask"] == 2.70
        assert rec["oi"] == 200.0
        assert rec["is_call"] is True

    def test_fetch_uses_parent_symbology(
        self, pipeline, mock_databento_responses
    ):
        """fetch() should use parent symbology (e.g., HE.OPT)."""
        p, _ = pipeline
        mbp_mock, def_mock, stat_mock = mock_databento_responses
        p.client.timeseries.get_range.side_effect = [mbp_mock, def_mock, stat_mock]

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        p.fetch("HE", date)

        # All three calls should use HE.OPT
        for call in p.client.timeseries.get_range.call_args_list:
            assert call.kwargs["symbols"] == ["HE.OPT"]
            assert call.kwargs["stype_in"] == "parent"


class TestOptionsValidation:
    """Tests for OptionsIngestPipeline.validate()."""

    def test_validate_passes_good_data(self, pipeline):
        """Good options data should pass validation."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "instrument_id": 1001, "strike": 60.0, "expiry": "2026-04-15",
                    "is_call": True, "bid": 2.50, "ask": 2.70,
                    "bid_size": 50, "ask_size": 55, "oi": 200, "volume": 100,
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 1
        assert len(warnings) == 0

    def test_validate_warns_on_bad_spreads(self, pipeline):
        """Inverted market (ask < bid) should produce a warning."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "instrument_id": 1001, "strike": 60.0, "expiry": "2026-04-15",
                    "is_call": True, "bid": 3.00, "ask": 2.50,
                    "bid_size": 50, "ask_size": 55, "oi": 200, "volume": 100,
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("inverted market" in w for w in warnings)

    def test_validate_catches_zero_strike(self, pipeline):
        """Strike <= 0 should fail validation."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "instrument_id": 1001, "strike": 0.0, "expiry": "2026-04-15",
                    "is_call": True, "bid": 2.50, "ask": 2.70,
                    "bid_size": 50, "ask_size": 55, "oi": 200, "volume": 100,
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("strike" in w for w in warnings)

    def test_validate_warns_on_empty_chain(self, pipeline):
        """Empty chain should produce a warning."""
        p, _ = pipeline
        raw = {"records": []}
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("No options" in w for w in warnings)


class TestOptionsPersist:
    """Tests for OptionsIngestPipeline.persist()."""

    def test_persist_writes_to_parquet(self, pipeline):
        """persist() should write chain to ParquetLake with data_type='options'."""
        p, _ = pipeline
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        data = {
            "records": [
                {
                    "strike": 60.0, "expiry": "2026-04-15",
                    "is_call": True, "bid": 2.50, "ask": 2.70,
                    "bid_size": 50, "ask_size": 55, "oi": 200, "volume": 100,
                },
                {
                    "strike": 60.0, "expiry": "2026-04-15",
                    "is_call": False, "bid": 3.10, "ask": 3.30,
                    "bid_size": 40, "ask_size": 45, "oi": 180, "volume": 120,
                },
            ]
        }

        p.persist(data, "HE", date)

        result = p.parquet_lake.read(data_type="options", market="HE")
        assert len(result) == 2

    def test_summary_features_written_to_feature_store(self, pipeline):
        """persist() should write put_call_oi_ratio and liquid_strike_count."""
        p, _ = pipeline
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        data = {
            "records": [
                {
                    "strike": 60.0, "expiry": "2026-04-15",
                    "is_call": True, "bid": 2.50, "ask": 2.70,
                    "bid_size": 50, "ask_size": 55, "oi": 200, "volume": 100,
                },
                {
                    "strike": 65.0, "expiry": "2026-04-15",
                    "is_call": True, "bid": 1.80, "ask": 2.00,
                    "bid_size": 30, "ask_size": 35, "oi": 150, "volume": 80,
                },
                {
                    "strike": 60.0, "expiry": "2026-04-15",
                    "is_call": False, "bid": 3.10, "ask": 3.30,
                    "bid_size": 40, "ask_size": 45, "oi": 180, "volume": 120,
                },
                {
                    "strike": 65.0, "expiry": "2026-04-15",
                    "is_call": False, "bid": 0.90, "ask": 1.10,
                    "bid_size": 20, "ask_size": 25, "oi": 90, "volume": 60,
                },
            ]
        }

        p.persist(data, "HE", date)

        features = p.feature_store.get_features_at("HE", date)
        assert "put_call_oi_ratio" in features
        assert "total_oi" in features
        assert "liquid_strike_count" in features

        # Call OI = 200 + 150 = 350, Put OI = 180 + 90 = 270
        expected_ratio = 270.0 / 350.0
        assert abs(features["put_call_oi_ratio"] - expected_ratio) < 0.001
        assert features["total_oi"] == 620.0

        # All 4 records have OI >= 50, so liquid count = 4
        assert features["liquid_strike_count"] == 4.0

    def test_persist_skips_empty_records(self, pipeline):
        """persist() should not write anything for empty records."""
        p, _ = pipeline
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        p.persist({"records": []}, "HE", date)
