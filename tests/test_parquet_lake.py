"""Tests for the Parquet data lake with hive partitioning."""

from datetime import datetime, timezone

import pyarrow as pa
import pytest

from hydra.data.store.parquet_lake import ParquetLake


@pytest.fixture
def sample_table() -> pa.Table:
    """Create a small sample Arrow table for testing."""
    return pa.table(
        {
            "timestamp": [
                "2026-01-15T00:00:00Z",
                "2026-01-15T01:00:00Z",
            ],
            "close": [65.50, 65.75],
            "volume": [1200, 800],
        }
    )


@pytest.fixture
def lake(tmp_parquet_dir) -> ParquetLake:
    """Create a ParquetLake instance backed by a temporary directory."""
    return ParquetLake(tmp_parquet_dir)


class TestParquetLakeWriteRead:
    """Verify basic write/read roundtrip with data integrity."""

    def test_write_and_read_roundtrip(
        self, lake: ParquetLake, sample_table: pa.Table
    ) -> None:
        """Data written to the lake can be read back with matching values."""
        date = datetime(2026, 1, 15, tzinfo=timezone.utc)
        lake.write(sample_table, data_type="futures", market="HE", date=date)

        result = lake.read(data_type="futures", market="HE")

        assert len(result) == 2
        assert result.column("close").to_pylist() == [65.50, 65.75]
        assert result.column("volume").to_pylist() == [1200, 800]
        assert result.column("timestamp").to_pylist() == [
            "2026-01-15T00:00:00Z",
            "2026-01-15T01:00:00Z",
        ]

    def test_read_filters_by_data_type_and_market(
        self, lake: ParquetLake, sample_table: pa.Table
    ) -> None:
        """Reading with different data_type or market returns empty."""
        date = datetime(2026, 1, 15, tzinfo=timezone.utc)
        lake.write(sample_table, data_type="futures", market="HE", date=date)

        # Different market
        result = lake.read(data_type="futures", market="ZO")
        assert len(result) == 0

        # Different data_type
        result = lake.read(data_type="options", market="HE")
        assert len(result) == 0


class TestParquetLakeAppendOnly:
    """Verify append-only semantics: no overwrites, unique files."""

    def test_append_only_creates_unique_files(
        self, lake: ParquetLake, sample_table: pa.Table
    ) -> None:
        """Writing the same date twice creates two separate Parquet files."""
        date = datetime(2026, 1, 15, tzinfo=timezone.utc)
        lake.write(sample_table, data_type="futures", market="HE", date=date)
        lake.write(sample_table, data_type="futures", market="HE", date=date)

        # Read should return all 4 rows (2 from each write)
        result = lake.read(data_type="futures", market="HE")
        assert len(result) == 4

        # Count actual parquet files in the partition directory
        partition_dir = (
            lake.base_path
            / "data_type=futures"
            / "market=HE"
            / "year=2026"
            / "month=01"
        )
        parquet_files = list(partition_dir.glob("*.parquet"))
        assert len(parquet_files) == 2


class TestParquetLakeHivePartitioning:
    """Verify directory structure matches hive partitioning scheme."""

    def test_hive_partitioning_structure(
        self, lake: ParquetLake, sample_table: pa.Table
    ) -> None:
        """Directory tree follows data_type=X/market=Y/year=Z/month=W."""
        date = datetime(2026, 2, 10, tzinfo=timezone.utc)
        lake.write(sample_table, data_type="cot", market="HE", date=date)

        expected_dir = (
            lake.base_path
            / "data_type=cot"
            / "market=HE"
            / "year=2026"
            / "month=02"
        )
        assert expected_dir.exists()

        parquet_files = list(expected_dir.glob("*.parquet"))
        assert len(parquet_files) == 1
        assert parquet_files[0].name.startswith("batch_20260210_")
