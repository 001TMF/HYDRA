"""Append-only Parquet data lake with hive partitioning.

All raw data (futures OHLCV, options chains, COT reports) is persisted as
Parquet files organized in hive-partitioned directories. Each ingestion run
creates uniquely named files to guarantee append-only semantics -- old data
is never overwritten or deleted.

Partitioning scheme:
    base_path / data_type=X / market=Y / year=Z / month=W / batch_YYYYMMDD_uuid.parquet
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import structlog

logger = structlog.get_logger()

PARTITION_COLUMNS = ["data_type", "market", "year", "month"]


class ParquetLake:
    """Append-only Parquet data lake with hive partitioning.

    Parameters
    ----------
    base_path : str | Path
        Root directory for the data lake. Hive-partitioned subdirectories
        are created automatically on first write.
    """

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        table: pa.Table,
        data_type: str,
        market: str,
        date: datetime,
    ) -> Path:
        """Write a batch of records to the lake.

        Each call creates a uniquely named Parquet file within the hive
        partition directory, ensuring append-only semantics.

        Parameters
        ----------
        table : pa.Table
            The data to write. Must NOT already contain partition columns.
        data_type : str
            Category of data (e.g., "futures", "options", "cot").
        market : str
            Market identifier (e.g., "HE" for lean hogs).
        date : datetime
            Reference date for partitioning (must be timezone-aware UTC).

        Returns
        -------
        Path
            The base path of the data lake.
        """
        log = logger.bind(data_type=data_type, market=market, date=str(date))
        n_rows = len(table)

        # Add partition columns to the table
        table = table.append_column(
            "data_type", pa.array([data_type] * n_rows)
        )
        table = table.append_column("market", pa.array([market] * n_rows))
        table = table.append_column(
            "year", pa.array([str(date.year)] * n_rows)
        )
        table = table.append_column(
            "month", pa.array([f"{date.month:02d}"] * n_rows)
        )

        # Unique basename prevents overwriting on repeat ingestion.
        # pyarrow requires {i} placeholder in basename_template for file indexing.
        unique_id = uuid.uuid4().hex[:8]
        basename = f"batch_{date.strftime('%Y%m%d')}_{unique_id}_{{i}}.parquet"

        ds.write_dataset(
            table,
            base_dir=str(self.base_path),
            format="parquet",
            partitioning=PARTITION_COLUMNS,
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
            basename_template=basename,
        )

        log.info("parquet_lake_write", rows=n_rows, basename=basename)
        return self.base_path

    def read(
        self,
        data_type: str,
        market: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pa.Table:
        """Read records from the lake with optional date filtering.

        Parameters
        ----------
        data_type : str
            Category of data to read.
        market : str
            Market identifier.
        start : datetime, optional
            Inclusive start date filter (by year/month partitions).
        end : datetime, optional
            Inclusive end date filter (by year/month partitions).

        Returns
        -------
        pa.Table
            Combined table of all matching records.
        """
        dataset = ds.dataset(
            str(self.base_path),
            format="parquet",
            partitioning="hive",
        )

        # Build filter expression
        filter_expr = (ds.field("data_type") == data_type) & (
            ds.field("market") == market
        )

        if start is not None:
            start_ym = f"{start.year}{start.month:02d}"
            filter_expr = filter_expr & (
                (ds.field("year") + ds.field("month")) >= start_ym
            )

        if end is not None:
            end_ym = f"{end.year}{end.month:02d}"
            filter_expr = filter_expr & (
                (ds.field("year") + ds.field("month")) <= end_ym
            )

        table = dataset.to_table(filter=filter_expr)

        # Drop partition columns from the result for clean consumption
        columns_to_drop = [
            col for col in PARTITION_COLUMNS if col in table.column_names
        ]
        table = table.drop(columns_to_drop)

        logger.info(
            "parquet_lake_read",
            data_type=data_type,
            market=market,
            rows=len(table),
        )
        return table
