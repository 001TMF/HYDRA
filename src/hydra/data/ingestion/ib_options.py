"""IB Gateway options chain ingestion pipeline.

Fetches options chain data (bid/ask, strike/expiry, OI) via Interactive
Brokers Gateway using a multi-step workflow:
    1. Qualify the underlying future
    2. Get underlying price via snapshot
    3. Fetch available strikes/expiries via reqSecDefOptParams
    4. Filter to nearest expiries and ATM strikes
    5. Request market data snapshots in batches of 50
    6. Persist to Parquet lake and write summary features

Uses the shared BrokerGateway connection with delayed data (type 3).
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

BATCH_SIZE = 50
BATCH_SLEEP_SECONDS = 1.0
SNAPSHOT_WAIT_SECONDS = 3.0
N_EXPIRIES = 3
STRIKE_RANGE = 15


class IBOptionsIngestPipeline(IngestPipeline):
    """Ingest options chain data via IB Gateway.

    Parameters
    ----------
    broker : BrokerGateway
        Shared broker connection wrapping ib_async.IB.
    parquet_lake : ParquetLake
        Data lake for raw chain persistence.
    feature_store : FeatureStore
        Feature store for summary features.
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
        """Validate options chain data.

        Checks:
        - Non-empty chain
        - Strikes > 0
        - Bids >= 0
        - Warns if ask < bid (inverted market)

        Parameters
        ----------
        raw_data : dict
            Raw payload from run_async.

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

        table = pa.table({
            "instrument_id": pa.array(
                [r["instrument_id"] for r in records], type=pa.int64()
            ),
            "strike": pa.array([r["strike"] for r in records], type=pa.float64()),
            "expiry": pa.array([r["expiry"] for r in records], type=pa.string()),
            "is_call": pa.array([r["is_call"] for r in records], type=pa.bool_()),
            "bid": pa.array([r["bid"] for r in records], type=pa.float64()),
            "ask": pa.array([r["ask"] for r in records], type=pa.float64()),
            "bid_size": pa.array([r["bid_size"] for r in records], type=pa.int64()),
            "ask_size": pa.array([r["ask_size"] for r in records], type=pa.int64()),
            "oi": pa.array([r["oi"] for r in records], type=pa.float64()),
            "volume": pa.array([r["volume"] for r in records], type=pa.int64()),
        })

        self.parquet_lake.write(table, data_type="options", market=market, date=date)

        call_oi = sum(r["oi"] for r in records if r["is_call"])
        put_oi = sum(r["oi"] for r in records if not r["is_call"])
        total_oi = call_oi + put_oi
        put_call_oi_ratio = put_oi / call_oi if call_oi > 0 else 0.0
        liquid_count = sum(1 for r in records if r["oi"] >= 50)

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

    async def run_async(self, market: str, date: datetime) -> bool:
        """Fetch, validate, and persist options chain via IB Gateway.

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
        from ib_async import Future, FuturesOption

        log = logger.bind(
            pipeline="IBOptionsIngestPipeline",
            market=market,
            date=str(date),
        )
        try:
            log.info("ingestion_started")
            self._broker.ib.reqMarketDataType(3)

            underlying = await self._qualify_underlying(market, log)
            if underlying is None:
                log.error("qualify_underlying_failed", market=market)
                return False

            underlying_price = await self._get_underlying_price(underlying, log)
            if underlying_price is None:
                log.error("no_underlying_price", market=market)
                return False

            log.info("underlying_price", market=market, price=underlying_price)

            chains = await self._broker.ib.reqSecDefOptParamsAsync(
                underlying.symbol,
                self._exchange,
                "FUT",
                underlying.conId,
            )

            if not chains:
                log.error("no_option_chains_returned", market=market)
                return False

            # Prefer chain matching trading class; fallback to first
            chain = next(
                (c for c in chains if c.tradingClass == market), chains[0]
            )

            expiries = sorted(chain.expirations)[:N_EXPIRIES]
            strikes = sorted(
                s for s in chain.strikes
                if abs(s - underlying_price) <= STRIKE_RANGE
            )

            if not expiries or not strikes:
                log.warning(
                    "no_expiries_or_strikes_after_filter",
                    market=market,
                    expiries=len(expiries),
                    strikes=len(strikes),
                )
                return False

            log.info(
                "chain_filtered",
                expiries=len(expiries),
                strikes=len(strikes),
            )

            contracts = [
                FuturesOption(
                    symbol=market,
                    lastTradeDateOrContractMonth=exp,
                    strike=strike,
                    right=right,
                    exchange=self._exchange,
                )
                for exp in expiries
                for strike in strikes
                for right in ("C", "P")
            ]

            qualified_opts = await self._broker.ib.qualifyContractsAsync(*contracts)
            qualified_opts = [
                c for c in qualified_opts
                if c is not None and getattr(c, "conId", None) and c.conId > 0
            ]

            log.info("contracts_qualified", count=len(qualified_opts))

            if not qualified_opts:
                log.warning("no_qualified_option_contracts", market=market)
                return False

            records = await self._fetch_chain_batched(qualified_opts, log)

            cleaned, warnings = self.validate({"records": records})
            for w in warnings:
                log.warning("validation_warning", detail=w)

            self.persist(cleaned, market, date)
            log.info("ingestion_complete", record_count=len(cleaned.get("records", [])))
            return True

        except Exception as e:
            log.error("ingestion_failed", error=str(e), exc_info=True)
            return False

    async def _qualify_underlying(self, market: str, log):
        """Qualify the underlying futures contract for options chain lookup.

        Handles ib_async 2.x returning None for ambiguous contracts by
        using returnAll=True and selecting the front-month contract.
        """
        from ib_async import Future

        contract = Future(symbol=market, exchange=self._exchange)
        qualified = await self._broker.ib.qualifyContractsAsync(
            contract, returnAll=True
        )
        result = qualified[0] if qualified else None

        if result is None:
            log.error("future_qualify_returned_none", market=market)
            return None

        # If ambiguous, we get a list of contracts — pick nearest expiry
        if isinstance(result, list):
            log.info("ambiguous_underlying_resolving", count=len(result))
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
                "front_month_underlying_selected",
                con_id=getattr(result, "conId", None),
                expiry=getattr(result, "lastTradeDateOrContractMonth", None),
            )
            return result

        log.info(
            "underlying_qualified",
            con_id=getattr(result, "conId", None),
            expiry=getattr(result, "lastTradeDateOrContractMonth", None),
        )
        return result

    async def _get_underlying_price(self, contract, log) -> float | None:
        """Get the underlying futures price via market-data snapshot."""
        ticker = self._broker.ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(SNAPSHOT_WAIT_SECONDS)
        self._broker.ib.cancelMktData(contract)

        def _valid(val) -> bool:
            return val is not None and not math.isnan(val) and val > 0

        if _valid(ticker.last):
            return float(ticker.last)
        if _valid(ticker.close):
            return float(ticker.close)
        mid = None
        if _valid(ticker.bid) and _valid(ticker.ask):
            mid = (ticker.bid + ticker.ask) / 2
        if mid and mid > 0:
            return float(mid)

        log.warning("underlying_snapshot_no_valid_price")
        return None

    async def _fetch_chain_batched(self, contracts: list, log) -> list[dict]:
        """Fetch market-data snapshots for options in batches of BATCH_SIZE."""
        records: list[dict] = []

        def _safe(val, default=0.0):
            if val is None:
                return default
            try:
                if math.isnan(val):
                    return default
            except TypeError:
                return default
            return val

        for batch_start in range(0, len(contracts), BATCH_SIZE):
            batch = contracts[batch_start: batch_start + BATCH_SIZE]

            tickers = [
                self._broker.ib.reqMktData(c, "101", False, False)
                for c in batch
            ]
            await asyncio.sleep(SNAPSHOT_WAIT_SECONDS)

            for contract, ticker in zip(batch, tickers):
                self._broker.ib.cancelMktData(contract)

                is_call = getattr(contract, "right", "C").upper() == "C"
                oi = (
                    _safe(getattr(ticker, "callOpenInterest", None))
                    if is_call
                    else _safe(getattr(ticker, "putOpenInterest", None))
                )

                records.append({
                    "instrument_id": int(contract.conId),
                    "strike": float(contract.strike),
                    "expiry": str(contract.lastTradeDateOrContractMonth),
                    "is_call": is_call,
                    "bid": float(_safe(ticker.bid)),
                    "ask": float(_safe(ticker.ask)),
                    "bid_size": int(_safe(ticker.bidSize, 0)),
                    "ask_size": int(_safe(ticker.askSize, 0)),
                    "oi": float(oi),
                    "volume": int(_safe(ticker.volume, 0)),
                })

            log.info(
                "batch_fetched",
                batch_start=batch_start,
                batch_size=len(batch),
                total_so_far=len(records),
            )

            if batch_start + BATCH_SIZE < len(contracts):
                await asyncio.sleep(BATCH_SLEEP_SECONDS)

        return records
