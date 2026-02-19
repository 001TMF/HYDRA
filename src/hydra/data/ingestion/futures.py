"""Databento futures OHLCV ingestion pipeline.

Fetches daily OHLCV bars for CME futures from Databento's Historical API,
validates price integrity, and persists to the Parquet data lake. Close
prices are also written to the feature store with same-day availability
(futures prices are available after market close).

Uses parent symbology (e.g., "HE.FUT") to fetch all expirations for a
given root symbol in a single request.
"""

from __future__ import annotations

from datetime import datetime, timezone

import databento as db
import pyarrow as pa
import structlog

from hydra.data.ingestion.base import IngestPipeline
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

logger = structlog.get_logger()


class FuturesIngestPipeline(IngestPipeline):
    """Ingest daily futures OHLCV bars from Databento.

    Parameters
    ----------
    api_key : str
        Databento API key for the Historical client.
    parquet_lake : ParquetLake
        Data lake for raw bar persistence.
    feature_store : FeatureStore
        Feature store for close-price features.
    """

    def __init__(
        self,
        api_key: str,
        parquet_lake: ParquetLake,
        feature_store: FeatureStore,
    ) -> None:
        super().__init__(parquet_lake=parquet_lake, feature_store=feature_store)
        self.client = db.Historical(api_key)

    def fetch(self, market: str, date: datetime) -> dict:
        """Fetch daily OHLCV bars from Databento.

        Parameters
        ----------
        market : str
            Root symbol (e.g., "HE" for lean hogs).
        date : datetime
            The trading day to fetch (timezone-aware UTC).

        Returns
        -------
        dict
            Keys: "records" (list of dicts with open, high, low, close,
            volume, symbol, ts_event).
        """
        log = logger.bind(pipeline="FuturesIngestPipeline", market=market)
        symbol = f"{market}.FUT"

        # Format dates as strings for Databento API
        start_str = date.strftime("%Y-%m-%d")
        # End is exclusive in Databento, so add one day
        end_date = datetime(
            date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc
        )
        end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")

        log.info("fetching_futures", symbol=symbol, start=start_str, end=end_str)

        data = self.client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema="ohlcv-1d",
            start=start_str,
            end=end_str,
        )

        # Convert Databento response to list of record dicts
        df = data.to_df()
        records = []
        for _, row in df.iterrows():
            records.append({
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]),
                "symbol": str(row.get("symbol", symbol)),
                "ts_event": str(row.name if hasattr(row, "name") else date.isoformat()),
            })

        log.info("fetched_futures", record_count=len(records))
        return {"records": records}

    def validate(self, raw_data: dict) -> tuple[dict, list[str]]:
        """Validate OHLCV data integrity.

        Checks:
        - Non-empty records
        - open, high, low, close > 0
        - volume >= 0
        - high >= low
        - close between low and high (inclusive)

        Parameters
        ----------
        raw_data : dict
            Raw payload from ``fetch()``.

        Returns
        -------
        tuple[dict, list[str]]
            (cleaned_data, warnings)
        """
        warnings: list[str] = []
        records = raw_data.get("records", [])

        if not records:
            warnings.append("No records returned from Databento")
            return {"records": []}, warnings

        valid_records = []
        for i, rec in enumerate(records):
            rec_warnings: list[str] = []
            o, h, l, c = rec["open"], rec["high"], rec["low"], rec["close"]
            v = rec["volume"]

            if o <= 0:
                rec_warnings.append(f"Record {i}: open={o} <= 0")
            if h <= 0:
                rec_warnings.append(f"Record {i}: high={h} <= 0")
            if l <= 0:
                rec_warnings.append(f"Record {i}: low={l} <= 0")
            if c <= 0:
                rec_warnings.append(f"Record {i}: close={c} <= 0")
            if v < 0:
                rec_warnings.append(f"Record {i}: volume={v} < 0")
            if h < l:
                rec_warnings.append(f"Record {i}: high={h} < low={l}")
            if c < l or c > h:
                rec_warnings.append(
                    f"Record {i}: close={c} not in [{l}, {h}]"
                )

            if rec_warnings:
                warnings.extend(rec_warnings)
            else:
                valid_records.append(rec)

        if len(valid_records) < len(records):
            warnings.append(
                f"Dropped {len(records) - len(valid_records)} of "
                f"{len(records)} records due to validation failures"
            )

        return {"records": valid_records}, warnings

    def persist(self, data: dict, market: str, date: datetime) -> None:
        """Write validated OHLCV data to Parquet lake and close prices to feature store.

        Parameters
        ----------
        data : dict
            Cleaned data from ``validate()``.
        market : str
            Market identifier (e.g., "HE").
        date : datetime
            Reference date (timezone-aware UTC).
        """
        records = data.get("records", [])
        if not records:
            logger.warning("no_records_to_persist", market=market, date=str(date))
            return

        # Build pyarrow table for Parquet lake
        table = pa.table({
            "open": pa.array([r["open"] for r in records], type=pa.float64()),
            "high": pa.array([r["high"] for r in records], type=pa.float64()),
            "low": pa.array([r["low"] for r in records], type=pa.float64()),
            "close": pa.array([r["close"] for r in records], type=pa.float64()),
            "volume": pa.array([r["volume"] for r in records], type=pa.int64()),
            "symbol": pa.array([r["symbol"] for r in records], type=pa.string()),
            "ts_event": pa.array(
                [r["ts_event"] for r in records], type=pa.string()
            ),
        })

        self.parquet_lake.write(table, data_type="futures", market=market, date=date)

        # Write close prices as features.
        # Futures prices are available same-day after market close.
        for rec in records:
            self.feature_store.write_feature(
                market=market,
                feature_name=f"futures_close_{rec['symbol']}",
                as_of=date,
                available_at=date,
                value=rec["close"],
            )

        logger.info(
            "futures_persisted",
            market=market,
            records=len(records),
            date=str(date),
        )
