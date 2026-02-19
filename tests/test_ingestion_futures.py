"""Tests for the FuturesIngestPipeline.

All Databento API calls are mocked -- no real network requests are made.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from hydra.data.ingestion.futures import FuturesIngestPipeline
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


@pytest.fixture
def mock_databento_response():
    """Create a realistic mock Databento response with OHLCV data."""
    df = pd.DataFrame(
        {
            "open": [65.50, 66.00],
            "high": [66.75, 67.10],
            "low": [65.00, 65.80],
            "close": [66.25, 66.90],
            "volume": [1234, 5678],
            "symbol": ["HEG6", "HEJ6"],
        },
        index=pd.to_datetime(
            ["2026-02-18T00:00:00Z", "2026-02-18T00:00:00Z"]
        ),
    )
    mock_data = MagicMock()
    mock_data.to_df.return_value = df
    return mock_data


@pytest.fixture
def pipeline(parquet_lake, feature_store):
    """FuturesIngestPipeline with a mocked Databento client."""
    with patch("hydra.data.ingestion.futures.db") as mock_db:
        p = FuturesIngestPipeline(
            api_key="TEST_KEY",
            parquet_lake=parquet_lake,
            feature_store=feature_store,
        )
        yield p, mock_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFuturesFetch:
    """Tests for FuturesIngestPipeline.fetch()."""

    def test_fetch_returns_raw_data(self, pipeline, mock_databento_response):
        """fetch() should call Databento and return records."""
        p, mock_db = pipeline
        p.client.timeseries.get_range.return_value = mock_databento_response

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        result = p.fetch("HE", date)

        assert "records" in result
        assert len(result["records"]) == 2
        assert result["records"][0]["open"] == 65.50
        assert result["records"][0]["close"] == 66.25
        assert result["records"][1]["volume"] == 5678

    def test_fetch_uses_correct_symbology(self, pipeline, mock_databento_response):
        """fetch() should use parent symbology (e.g., HE.FUT)."""
        p, _ = pipeline
        p.client.timeseries.get_range.return_value = mock_databento_response

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        p.fetch("HE", date)

        call_kwargs = p.client.timeseries.get_range.call_args
        assert call_kwargs.kwargs["symbols"] == ["HE.FUT"]
        assert call_kwargs.kwargs["stype_in"] == "parent"
        assert call_kwargs.kwargs["schema"] == "ohlcv-1d"
        assert call_kwargs.kwargs["dataset"] == "GLBX.MDP3"


class TestFuturesValidation:
    """Tests for FuturesIngestPipeline.validate()."""

    def test_validate_passes_good_data(self, pipeline):
        """Good OHLCV data should pass validation without warnings."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "open": 65.50, "high": 66.75, "low": 65.00,
                    "close": 66.25, "volume": 1234,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 1
        assert len(warnings) == 0

    def test_validate_catches_negative_prices(self, pipeline):
        """Negative prices should trigger validation warnings."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "open": -5.0, "high": 66.75, "low": 65.00,
                    "close": 66.25, "volume": 1234,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("open=-5.0" in w for w in warnings)

    def test_validate_catches_high_less_than_low(self, pipeline):
        """high < low should trigger a validation warning."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "open": 65.50, "high": 64.00, "low": 65.00,
                    "close": 64.50, "volume": 100,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("high" in w and "low" in w for w in warnings)

    def test_validate_catches_close_out_of_range(self, pipeline):
        """close outside [low, high] should trigger a warning."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "open": 65.50, "high": 66.75, "low": 65.00,
                    "close": 67.00, "volume": 100,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("close" in w and "not in" in w for w in warnings)

    def test_validate_warns_on_empty_records(self, pipeline):
        """Empty records list should produce a warning."""
        p, _ = pipeline
        raw = {"records": []}
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("No records" in w for w in warnings)

    def test_validate_negative_volume(self, pipeline):
        """Negative volume should trigger a validation warning."""
        p, _ = pipeline
        raw = {
            "records": [
                {
                    "open": 65.50, "high": 66.75, "low": 65.00,
                    "close": 66.25, "volume": -10,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }
        cleaned, warnings = p.validate(raw)
        assert len(cleaned["records"]) == 0
        assert any("volume" in w for w in warnings)


class TestFuturesPersist:
    """Tests for FuturesIngestPipeline.persist()."""

    def test_persist_writes_to_parquet_lake(self, pipeline, tmp_parquet_dir):
        """persist() should write OHLCV records to ParquetLake with data_type='futures'."""
        p, _ = pipeline
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        data = {
            "records": [
                {
                    "open": 65.50, "high": 66.75, "low": 65.00,
                    "close": 66.25, "volume": 1234,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }

        p.persist(data, "HE", date)

        # Read back from lake
        result = p.parquet_lake.read(data_type="futures", market="HE")
        assert len(result) == 1
        assert result.column("close")[0].as_py() == 66.25

    def test_persist_writes_close_to_feature_store(self, pipeline, tmp_feature_db):
        """persist() should write close prices to the feature store."""
        p, _ = pipeline
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        data = {
            "records": [
                {
                    "open": 65.50, "high": 66.75, "low": 65.00,
                    "close": 66.25, "volume": 1234,
                    "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
                },
            ]
        }

        p.persist(data, "HE", date)

        # Query feature store -- same-day availability for futures
        features = p.feature_store.get_features_at("HE", date)
        assert "futures_close_HEG6" in features
        assert features["futures_close_HEG6"] == 66.25

    def test_persist_skips_empty_records(self, pipeline):
        """persist() should not write anything when there are no records."""
        p, _ = pipeline
        date = datetime(2026, 2, 18, tzinfo=timezone.utc)

        # Should not raise
        p.persist({"records": []}, "HE", date)


class TestFuturesRunOrchestration:
    """Tests for the full run() pipeline orchestration."""

    def test_run_orchestrates_full_pipeline(self, pipeline, mock_databento_response):
        """run() should call fetch -> validate -> persist in order."""
        p, _ = pipeline
        p.client.timeseries.get_range.return_value = mock_databento_response

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        result = p.run("HE", date)

        assert result is True

    def test_run_returns_false_on_fetch_error(self, pipeline):
        """run() should return False when fetch raises an exception."""
        p, _ = pipeline
        p.client.timeseries.get_range.side_effect = RuntimeError("API error")

        date = datetime(2026, 2, 18, tzinfo=timezone.utc)
        result = p.run("HE", date)

        assert result is False
