"""Shared test fixtures for HYDRA test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_parquet_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for Parquet lake tests."""
    parquet_dir = tmp_path / "parquet_lake"
    parquet_dir.mkdir()
    return parquet_dir


@pytest.fixture
def tmp_feature_db(tmp_path: Path) -> Path:
    """Provide a temporary SQLite database path for feature store tests."""
    return tmp_path / "features.db"
