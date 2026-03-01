"""IB Gateway futures OHLCV ingestion pipeline.

Fetches daily OHLCV bars (or snapshot fallback) for CME futures via
Interactive Brokers Gateway, validates price integrity, and persists
to the Parquet data lake. Close prices are written to the feature store.

Uses the shared BrokerGateway connection — no extra client ID needed
since ingestion runs sequentially before trading in the daily cycle.
Delayed data (type 3, 15-min) is sufficient for end-of-day signals.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime
from typing import TYPE_CHECKING

import pyarrow as pa
import structlog

from hydra.data.ingestion.base import IngestPipeline
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake

if TYPE_CHECKING:
    from hydra.execution.broker import BrokerGateway

logger = structlog.get_logger()


class IBFuturesIngestPipeline(IngestPipeline):
    """Ingest daily futures OHLCV bars via IB Gateway.

    Parameters
    ----------
    broker : BrokerGateway
        Shared broker connection wrapping ib_async.IB.
    parquet_lake : ParquetLake
        Data lake for raw bar persistence.
    feature_store : FeatureStore
        Feature store for close-price features.
    exchange : str
        IB exchange identifier (default: "GLOBEX").
    """

    def __init__(
        self,
        broker: BrokerGateway,
        parquet_lake: ParquetLake,
        feature_store: FeatureStore,
        exchange: str = "GLOBEX",
    ) -> None:
        super().__init__(parquet_lake=parquet_lake, feature_store=feature_store)
        self._broker = broker
        self._exchange = exchange

    def fetch(self, market: str, date: datetime) -> dict:
        """Not supported — use run_async() for IB pipelines."""
        raise NotImplementedError("Use run_async() for IB pipelines")

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
            Raw payload from fetch (or run_async).

        Returns
        -------
        tuple[dict, list[str]]
            (cleaned_data, warnings)
        """
        warnings: list[str] = []
        records = raw_data.get("records", [])

        if not records:
            warnings.append("No records returned from IB Gateway")
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

        Each bar's ``ts_event`` is used as the ``as_of`` date for the feature
        store so that backfill (multi-bar) writes have correct temporal keys.

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

        for rec in records:
            bar_date = self._parse_bar_date(rec["ts_event"], date)
            self.feature_store.write_feature(
                market=market,
                feature_name=f"futures_close_{rec['symbol']}",
                as_of=bar_date,
                available_at=bar_date,
                value=rec["close"],
            )

        logger.info(
            "futures_persisted",
            market=market,
            records=len(records),
            date=str(date),
        )

    @staticmethod
    def _parse_bar_date(ts_event: str, fallback: datetime) -> datetime:
        """Parse a bar's ts_event string to a timezone-aware datetime."""
        from datetime import timezone as tz

        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                dt = datetime.strptime(ts_event.strip()[:10], fmt)
                return dt.replace(tzinfo=tz.utc)
            except (ValueError, AttributeError):
                continue
        # ts_event might be an ISO string from a snapshot
        try:
            dt = datetime.fromisoformat(ts_event)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz.utc)
            return dt
        except (ValueError, AttributeError):
            return fallback

    async def run_async(self, market: str, date: datetime) -> bool:
        """Fetch, validate, and persist futures OHLCV via IB Gateway.

        Uses delayed data (type 3). Tries historical bars first; falls
        back to a market-data snapshot if no bars are returned.

        Parameters
        ----------
        market : str
            Root symbol (e.g., "HE" for lean hogs).
        date : datetime
            The trading day to ingest (timezone-aware UTC).

        Returns
        -------
        bool
            True if the pipeline completed successfully, False otherwise.
        """
        log = logger.bind(
            pipeline="IBFuturesIngestPipeline",
            market=market,
            date=str(date),
        )
        try:
            log.info("ingestion_started")
            self._broker.ib.reqMarketDataType(3)

            contract = await self._qualify_futures_contract(market, log)
            if contract is None:
                log.error("qualify_contract_failed", market=market)
                return False

            bars = await self._fetch_historical(contract, log)
            if not bars:
                log.warning("no_historical_bars_falling_back_to_snapshot", market=market)
                records = await self._fetch_snapshot(contract, market, date, log)
            else:
                records = [
                    {
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                        "symbol": market,
                        "ts_event": str(bar.date),
                    }
                    for bar in bars
                ]

            cleaned, warnings = self.validate({"records": records})
            for w in warnings:
                log.warning("validation_warning", detail=w)

            self.persist(cleaned, market, date)
            log.info("ingestion_complete", record_count=len(cleaned.get("records", [])))
            return True

        except Exception as e:
            log.error("ingestion_failed", error=str(e), exc_info=True)
            return False

    async def _qualify_futures_contract(self, market: str, log):
        """Qualify a futures contract, handling ib_async 2.x None returns.

        Tries ContFuture first (ideal for historical bars). If that returns
        None (unknown/ambiguous), falls back to Future with returnAll=True
        and picks the nearest expiry.
        """
        from ib_async import ContFuture, Future

        # Try ContFuture first — ideal for continuous historical data
        contract = ContFuture(symbol=market, exchange=self._exchange)
        qualified = await self._broker.ib.qualifyContractsAsync(contract)
        result = qualified[0] if qualified else None

        if result is not None:
            log.info(
                "contract_qualified",
                contract_type="ContFuture",
                con_id=getattr(result, "conId", None),
            )
            return result

        log.warning("contfuture_qualify_failed_trying_future", market=market)

        # Fallback: use Future with returnAll to handle ambiguity
        contract = Future(symbol=market, exchange=self._exchange)
        qualified = await self._broker.ib.qualifyContractsAsync(
            contract, returnAll=True
        )
        result = qualified[0] if qualified else None

        if result is None:
            log.error("future_qualify_failed", market=market)
            return None

        # If ambiguous, we get a list of contracts — pick nearest expiry
        if isinstance(result, list):
            log.info("ambiguous_future_resolving", count=len(result))
            # Sort by lastTradeDateOrContractMonth, pick nearest future date
            today = datetime.now().strftime("%Y%m%d")
            future_contracts = [
                c for c in result
                if c is not None
                and getattr(c, "lastTradeDateOrContractMonth", "") >= today
            ]
            if not future_contracts:
                future_contracts = [c for c in result if c is not None]
            if not future_contracts:
                log.error("no_valid_futures_in_ambiguous_result")
                return None
            result = sorted(
                future_contracts,
                key=lambda c: getattr(c, "lastTradeDateOrContractMonth", ""),
            )[0]
            log.info(
                "front_month_selected",
                con_id=getattr(result, "conId", None),
                expiry=getattr(result, "lastTradeDateOrContractMonth", None),
            )
            return result

        log.info(
            "contract_qualified",
            contract_type="Future",
            con_id=getattr(result, "conId", None),
        )
        return result

    async def run_backfill_async(
        self, market: str, date: datetime, duration: str = "1 Y"
    ) -> bool:
        """Fetch historical daily bars and backfill the feature store.

        Same as ``run_async`` but requests a longer duration (default 1 year)
        so that ~250 trading days of ``futures_close_{market}`` are written.

        Parameters
        ----------
        market : str
            Root symbol (e.g., "HE").
        date : datetime
            Reference date (timezone-aware UTC).
        duration : str
            IB duration string (e.g., "1 Y", "6 M", "30 D").
        """
        log = logger.bind(
            pipeline="IBFuturesIngestPipeline",
            market=market,
            date=str(date),
            mode="backfill",
        )
        try:
            log.info("backfill_started", duration=duration)
            self._broker.ib.reqMarketDataType(3)

            contract = await self._qualify_futures_contract(market, log)
            if contract is None:
                log.error("qualify_contract_failed", market=market)
                return False

            bars = await self._fetch_historical(contract, log, duration_str=duration)
            if not bars:
                log.warning("no_historical_bars_for_backfill", market=market)
                return False

            records = [
                {
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "symbol": market,
                    "ts_event": str(bar.date),
                }
                for bar in bars
            ]

            cleaned, warnings = self.validate({"records": records})
            for w in warnings:
                log.warning("validation_warning", detail=w)

            self.persist(cleaned, market, date)
            log.info("backfill_complete", record_count=len(cleaned.get("records", [])))
            return True

        except Exception as e:
            log.error("backfill_failed", error=str(e), exc_info=True)
            return False

    async def _fetch_historical(
        self, contract, log, duration_str: str = "1 D"
    ) -> list:
        """Fetch OHLCV bars via reqHistoricalDataAsync."""
        bars = await self._broker.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration_str,
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=False,
        )
        log.info("historical_bars_fetched", count=len(bars) if bars else 0, duration=duration_str)
        return bars or []

    async def _fetch_snapshot(
        self, contract, market: str, date: datetime, log
    ) -> list[dict]:
        """Fetch a market-data snapshot as OHLCV fallback."""
        ticker = self._broker.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(3)
        self._broker.ib.cancelMktData(contract)

        def _valid(val) -> bool:
            return val is not None and not math.isnan(val) and val > 0

        close = ticker.last if _valid(ticker.last) else ticker.close
        if not _valid(close):
            log.warning("snapshot_no_valid_price", market=market)
            return []

        high = ticker.high if _valid(ticker.high) else close
        low = ticker.low if _valid(ticker.low) else close
        open_ = ticker.open if _valid(ticker.open) else close
        volume = int(ticker.volume) if ticker.volume is not None and not math.isnan(ticker.volume) else 0

        log.info("snapshot_fetched", market=market, close=close)
        return [
            {
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": volume,
                "symbol": market,
                "ts_event": date.isoformat(),
            }
        ]
