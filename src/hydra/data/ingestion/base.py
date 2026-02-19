"""Abstract base class for all data ingestion pipelines.

Every data source (futures, options, COT) implements this interface:
    fetch(market, date) -> raw data dict
    validate(raw_data) -> (cleaned_data, warnings)
    persist(data, market, date) -> None

The concrete ``run()`` method orchestrates the full pipeline with structured
logging and error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import structlog

from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

logger = structlog.get_logger()


class IngestPipeline(ABC):
    """Abstract base for all data ingestion pipelines.

    Subclasses must implement ``fetch``, ``validate``, and ``persist``.
    The ``run`` method orchestrates them in order with logging and error
    handling.

    Parameters
    ----------
    parquet_lake : ParquetLake
        Data lake for raw data persistence.
    feature_store : FeatureStore
        Point-in-time feature store for derived features.
    """

    def __init__(
        self,
        parquet_lake: ParquetLake,
        feature_store: FeatureStore,
    ) -> None:
        self.parquet_lake = parquet_lake
        self.feature_store = feature_store

    @abstractmethod
    def fetch(self, market: str, date: datetime) -> dict:
        """Fetch raw data from external source.

        Parameters
        ----------
        market : str
            Market identifier (e.g., "HE" for lean hogs).
        date : datetime
            Reference date for the data (timezone-aware UTC).

        Returns
        -------
        dict
            Raw payload from the data source.
        """
        ...

    @abstractmethod
    def validate(self, raw_data: dict) -> tuple[dict, list[str]]:
        """Validate and clean raw data.

        Parameters
        ----------
        raw_data : dict
            Raw payload from ``fetch()``.

        Returns
        -------
        tuple[dict, list[str]]
            A tuple of (cleaned_data, list_of_warning_messages).
        """
        ...

    @abstractmethod
    def persist(self, data: dict, market: str, date: datetime) -> None:
        """Write validated data to storage.

        Parameters
        ----------
        data : dict
            Cleaned data from ``validate()``.
        market : str
            Market identifier.
        date : datetime
            Reference date (timezone-aware UTC).
        """
        ...

    def run(self, market: str, date: datetime) -> bool:
        """Execute the full pipeline: fetch -> validate -> persist.

        Parameters
        ----------
        market : str
            Market identifier.
        date : datetime
            Reference date (timezone-aware UTC).

        Returns
        -------
        bool
            True if the pipeline completed successfully, False otherwise.
        """
        log = logger.bind(
            pipeline=self.__class__.__name__,
            market=market,
            date=str(date),
        )
        try:
            log.info("ingestion_started")
            raw = self.fetch(market, date)
            cleaned, warnings = self.validate(raw)
            for w in warnings:
                log.warning("validation_warning", detail=w)
            self.persist(cleaned, market, date)
            record_count = len(cleaned.get("records", []))
            log.info("ingestion_complete", record_count=record_count)
            return True
        except Exception as e:
            log.error("ingestion_failed", error=str(e), exc_info=True)
            return False
