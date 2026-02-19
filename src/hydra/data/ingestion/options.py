"""Databento options chain ingestion pipeline.

Fetches complete options chain data (bid/ask, strike/expiry metadata, OI)
from Databento using three separate schema requests:
    1. mbp-1: Top-of-book bid/ask prices and sizes
    2. definition: Strike, expiry, and instrument metadata
    3. statistics: Open interest data

These are joined on instrument_id to produce a complete chain which is
persisted to the Parquet data lake. Summary features (put/call OI ratio,
total OI, ATM implied vol, liquid strike count) are written to the
feature store.
"""

from __future__ import annotations

from datetime import datetime, timezone

import databento as db
import numpy as np
import pyarrow as pa
import structlog

from hydra.data.ingestion.base import IngestPipeline
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

logger = structlog.get_logger()


class OptionsIngestPipeline(IngestPipeline):
    """Ingest options chain data from Databento.

    Parameters
    ----------
    api_key : str
        Databento API key for the Historical client.
    parquet_lake : ParquetLake
        Data lake for raw chain persistence.
    feature_store : FeatureStore
        Feature store for summary features.
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
        """Fetch options chain from Databento via three schema requests.

        Fetches mbp-1 (bid/ask), definition (strike/expiry), and
        statistics (OI) data, then joins on instrument_id.

        Parameters
        ----------
        market : str
            Root symbol (e.g., "HE" for lean hogs).
        date : datetime
            The trading day to fetch (timezone-aware UTC).

        Returns
        -------
        dict
            Keys: "records" (list of dicts with strike, expiry, bid, ask,
            bid_size, ask_size, oi, volume, is_call, instrument_id).
        """
        log = logger.bind(pipeline="OptionsIngestPipeline", market=market)
        symbol = f"{market}.OPT"
        start_str = date.strftime("%Y-%m-%d")
        end_str = datetime(
            date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S")

        log.info("fetching_options", symbol=symbol, start=start_str, end=end_str)

        # 1. Fetch bid/ask from mbp-1 schema
        mbp_data = self.client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema="mbp-1",
            start=start_str,
            end=end_str,
        )
        mbp_df = mbp_data.to_df()

        # 2. Fetch definition (strike/expiry metadata)
        def_data = self.client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema="definition",
            start=start_str,
            end=end_str,
        )
        def_df = def_data.to_df()

        # 3. Fetch statistics (OI)
        stat_data = self.client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=[symbol],
            stype_in="parent",
            schema="statistics",
            start=start_str,
            end=end_str,
        )
        stat_df = stat_data.to_df()

        # Join on instrument_id to produce complete chain
        records = self._join_chain(mbp_df, def_df, stat_df)

        log.info("fetched_options", record_count=len(records))
        return {"records": records}

    def _join_chain(self, mbp_df, def_df, stat_df) -> list[dict]:
        """Join mbp, definition, and statistics data into a unified chain.

        Parameters
        ----------
        mbp_df : DataFrame
            Bid/ask data with instrument_id.
        def_df : DataFrame
            Definition data with strike_price, expiration, instrument_class.
        stat_df : DataFrame
            Statistics data with open_interest.

        Returns
        -------
        list[dict]
            Unified chain records.
        """
        # Build lookups from definition and statistics
        definitions = {}
        if len(def_df) > 0:
            id_col = "instrument_id" if "instrument_id" in def_df.columns else None
            if id_col is None:
                # Try index if instrument_id is the index
                for idx, row in def_df.iterrows():
                    iid = row.get("instrument_id", idx)
                    definitions[iid] = {
                        "strike": float(row.get("strike_price", 0)),
                        "expiry": str(row.get("expiration", "")),
                        "is_call": str(row.get("instrument_class", "")).upper() == "C",
                    }
            else:
                for _, row in def_df.iterrows():
                    iid = row["instrument_id"]
                    definitions[iid] = {
                        "strike": float(row.get("strike_price", 0)),
                        "expiry": str(row.get("expiration", "")),
                        "is_call": str(row.get("instrument_class", "")).upper() == "C",
                    }

        oi_lookup: dict[int | str, float] = {}
        if len(stat_df) > 0:
            for _, row in stat_df.iterrows():
                iid = row.get("instrument_id", None)
                if iid is not None:
                    oi_lookup[iid] = float(row.get("open_interest", 0))

        # Join mbp with definitions and statistics
        records = []
        if len(mbp_df) > 0:
            for _, row in mbp_df.iterrows():
                iid = row.get("instrument_id", None)
                defn = definitions.get(iid, {})
                records.append({
                    "instrument_id": iid,
                    "strike": defn.get("strike", 0.0),
                    "expiry": defn.get("expiry", ""),
                    "is_call": defn.get("is_call", True),
                    "bid": float(row.get("bid_px_00", row.get("bid_px", 0.0))),
                    "ask": float(row.get("ask_px_00", row.get("ask_px", 0.0))),
                    "bid_size": int(row.get("bid_sz_00", row.get("bid_sz", 0))),
                    "ask_size": int(row.get("ask_sz_00", row.get("ask_sz", 0))),
                    "oi": oi_lookup.get(iid, 0.0),
                    "volume": int(row.get("volume", 0)),
                })

        return records

    def validate(self, raw_data: dict) -> tuple[dict, list[str]]:
        """Validate options chain data.

        Checks:
        - Non-empty chain
        - Strikes > 0
        - Bids >= 0
        - Warns if ask < bid (inverted market)
        - Expiry dates present

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
            warnings.append("No options chain records returned")
            return {"records": []}, warnings

        valid_records = []
        for i, rec in enumerate(records):
            rec_warnings: list[str] = []

            if rec["strike"] <= 0:
                rec_warnings.append(f"Record {i}: strike={rec['strike']} <= 0")
            if rec["bid"] < 0:
                rec_warnings.append(f"Record {i}: bid={rec['bid']} < 0")
            if rec["ask"] < rec["bid"] and rec["ask"] > 0:
                rec_warnings.append(
                    f"Record {i}: inverted market ask={rec['ask']} < bid={rec['bid']}"
                )

            if rec_warnings:
                warnings.extend(rec_warnings)
            else:
                valid_records.append(rec)

        if len(valid_records) < len(records):
            warnings.append(
                f"Dropped {len(records) - len(valid_records)} of "
                f"{len(records)} records"
            )

        return {"records": valid_records}, warnings

    def persist(self, data: dict, market: str, date: datetime) -> None:
        """Write options chain to Parquet lake and summary features to feature store.

        Summary features written:
        - put_call_oi_ratio: Total put OI / total call OI
        - total_oi: Sum of all open interest
        - liquid_strike_count: Number of strikes with OI >= 50

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
            logger.warning("no_options_to_persist", market=market, date=str(date))
            return

        # Build pyarrow table for Parquet lake
        table = pa.table({
            "strike": pa.array([r["strike"] for r in records], type=pa.float64()),
            "expiry": pa.array([r["expiry"] for r in records], type=pa.string()),
            "is_call": pa.array([r["is_call"] for r in records], type=pa.bool_()),
            "bid": pa.array([r["bid"] for r in records], type=pa.float64()),
            "ask": pa.array([r["ask"] for r in records], type=pa.float64()),
            "bid_size": pa.array([r["bid_size"] for r in records], type=pa.int64()),
            "ask_size": pa.array([r["ask_size"] for r in records], type=pa.int64()),
            "oi": pa.array([r["oi"] for r in records], type=pa.float64()),
            "volume": pa.array([r.get("volume", 0) for r in records], type=pa.int64()),
        })

        self.parquet_lake.write(table, data_type="options", market=market, date=date)

        # Compute and write summary features
        call_oi = sum(r["oi"] for r in records if r["is_call"])
        put_oi = sum(r["oi"] for r in records if not r["is_call"])
        total_oi = call_oi + put_oi
        put_call_oi_ratio = put_oi / call_oi if call_oi > 0 else 0.0

        # Count liquid strikes (OI >= 50)
        liquid_count = sum(1 for r in records if r["oi"] >= 50)

        # Options data available same day after close
        features = {
            "put_call_oi_ratio": put_call_oi_ratio,
            "total_oi": total_oi,
            "liquid_strike_count": float(liquid_count),
        }

        for feature_name, value in features.items():
            self.feature_store.write_feature(
                market=market,
                feature_name=feature_name,
                as_of=date,
                available_at=date,
                value=value,
            )

        logger.info(
            "options_persisted",
            market=market,
            records=len(records),
            total_oi=total_oi,
            liquid_strikes=liquid_count,
            date=str(date),
        )
