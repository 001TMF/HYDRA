"""Tests for IB Gateway data ingestion pipelines.

All IB calls are mocked — no real network requests are made.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hydra.data.ingestion.ib_futures import IBFuturesIngestPipeline
from hydra.data.ingestion.ib_options import IBOptionsIngestPipeline
from hydra.data.store.feature_store import FeatureStore
from hydra.data.store.parquet_lake import ParquetLake


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DATE = datetime(2026, 2, 18, tzinfo=timezone.utc)


def _make_broker(
    historical_bars=None,
    snapshot_ticker=None,
    chain_params=None,
    option_tickers=None,
):
    """Build a mock BrokerGateway with IB sub-mock."""
    broker = MagicMock()
    broker.ib = MagicMock()
    broker.ib.reqMarketDataType = MagicMock()
    broker.ib.cancelMktData = MagicMock()

    # Default qualified contract
    mock_contract = MagicMock()
    mock_contract.conId = 123
    broker.ib.qualifyContractsAsync = AsyncMock(return_value=[mock_contract])

    # Historical bars
    bars = historical_bars if historical_bars is not None else []
    broker.ib.reqHistoricalDataAsync = AsyncMock(return_value=bars)

    # Snapshot ticker
    if snapshot_ticker is not None:
        broker.ib.reqMktData = MagicMock(return_value=snapshot_ticker)
    else:
        broker.ib.reqMktData = MagicMock(return_value=MagicMock())

    # Options chain params
    chains = chain_params if chain_params is not None else []
    broker.ib.reqSecDefOptParamsAsync = AsyncMock(return_value=chains)

    return broker


def _make_bar(open=65.5, high=67.0, low=65.0, close=66.25, volume=1234, date="20260218"):
    bar = MagicMock()
    bar.open = open
    bar.high = high
    bar.low = low
    bar.close = close
    bar.volume = volume
    bar.date = date
    return bar


def _make_snapshot_ticker(
    last=66.25, high=67.0, low=65.0, open=65.5, volume=100.0, close=float("nan")
):
    ticker = MagicMock()
    ticker.last = last
    ticker.bid = 66.0
    ticker.ask = 66.5
    ticker.bidSize = 10
    ticker.askSize = 15
    ticker.volume = volume
    ticker.high = high
    ticker.low = low
    ticker.open = open
    ticker.close = close
    ticker.putOpenInterest = 500.0
    ticker.callOpenInterest = 800.0
    return ticker


def _make_chain(
    trading_class="HE",
    expirations=None,
    strikes=None,
    multiplier="40000",
):
    chain = MagicMock()
    chain.tradingClass = trading_class
    chain.expirations = expirations or ["20260315", "20260415", "20260515"]
    chain.strikes = strikes or [60.0, 62.0, 64.0, 66.0, 68.0, 70.0]
    chain.multiplier = multiplier
    return chain


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parquet_lake(tmp_parquet_dir):
    return ParquetLake(tmp_parquet_dir)


@pytest.fixture
def feature_store(tmp_feature_db):
    fs = FeatureStore(tmp_feature_db)
    yield fs
    fs.close()


@pytest.fixture
def futures_pipeline(parquet_lake, feature_store):
    broker = _make_broker()
    return IBFuturesIngestPipeline(
        broker=broker,
        parquet_lake=parquet_lake,
        feature_store=feature_store,
    )


@pytest.fixture
def options_pipeline(parquet_lake, feature_store):
    broker = _make_broker()
    return IBOptionsIngestPipeline(
        broker=broker,
        parquet_lake=parquet_lake,
        feature_store=feature_store,
    )


# ---------------------------------------------------------------------------
# Test 1: IBFuturesIngestPipeline — historical bars happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("hydra.data.ingestion.ib_futures.asyncio.sleep", new_callable=AsyncMock)
@patch("ib_async.ContFuture")
async def test_ib_futures_fetch_historical_bars(mock_cont_future, mock_sleep, parquet_lake, feature_store):
    """run_async returns True and persists when historical bars are returned."""
    bar = _make_bar()
    broker = _make_broker(historical_bars=[bar])
    pipeline = IBFuturesIngestPipeline(
        broker=broker, parquet_lake=parquet_lake, feature_store=feature_store
    )

    result = await pipeline.run_async("HE", DATE)

    assert result is True
    broker.ib.reqMarketDataType.assert_called_with(3)

    # Parquet lake should have the record
    table = parquet_lake.read(data_type="futures", market="HE")
    assert len(table) == 1
    assert table.column("close")[0].as_py() == 66.25


# ---------------------------------------------------------------------------
# Test 2: IBFuturesIngestPipeline — snapshot fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("hydra.data.ingestion.ib_futures.asyncio.sleep", new_callable=AsyncMock)
@patch("ib_async.ContFuture")
async def test_ib_futures_fetch_snapshot_fallback(mock_cont_future, mock_sleep, parquet_lake, feature_store):
    """run_async falls back to snapshot when no historical bars returned."""
    ticker = _make_snapshot_ticker()
    broker = _make_broker(historical_bars=[], snapshot_ticker=ticker)
    pipeline = IBFuturesIngestPipeline(
        broker=broker, parquet_lake=parquet_lake, feature_store=feature_store
    )

    result = await pipeline.run_async("HE", DATE)

    assert result is True

    # Snapshot data should have been persisted
    table = parquet_lake.read(data_type="futures", market="HE")
    assert len(table) == 1
    assert table.column("close")[0].as_py() == 66.25


# ---------------------------------------------------------------------------
# Test 3: IBFuturesIngestPipeline — validate() reuses same rules
# ---------------------------------------------------------------------------


def test_ib_futures_validate_reuses_rules(futures_pipeline):
    """validate() enforces the same OHLCV rules as Databento futures."""
    p = futures_pipeline

    # Good record
    good = {
        "records": [
            {
                "open": 65.5, "high": 67.0, "low": 65.0,
                "close": 66.25, "volume": 1234,
                "symbol": "HE", "ts_event": "2026-02-18T00:00:00Z",
            }
        ]
    }
    cleaned, warnings = p.validate(good)
    assert len(cleaned["records"]) == 1
    assert len(warnings) == 0

    # Negative price
    bad_open = {
        "records": [
            {
                "open": -5.0, "high": 67.0, "low": 65.0,
                "close": 66.25, "volume": 100,
                "symbol": "HE", "ts_event": "2026-02-18T00:00:00Z",
            }
        ]
    }
    cleaned, warnings = p.validate(bad_open)
    assert len(cleaned["records"]) == 0
    assert any("open=-5.0" in w for w in warnings)

    # high < low
    bad_hl = {
        "records": [
            {
                "open": 65.5, "high": 64.0, "low": 65.0,
                "close": 64.5, "volume": 100,
                "symbol": "HE", "ts_event": "2026-02-18T00:00:00Z",
            }
        ]
    }
    cleaned, warnings = p.validate(bad_hl)
    assert len(cleaned["records"]) == 0
    assert any("high" in w and "low" in w for w in warnings)

    # close out of [low, high]
    bad_close = {
        "records": [
            {
                "open": 65.5, "high": 66.75, "low": 65.0,
                "close": 67.0, "volume": 100,
                "symbol": "HE", "ts_event": "2026-02-18T00:00:00Z",
            }
        ]
    }
    cleaned, warnings = p.validate(bad_close)
    assert len(cleaned["records"]) == 0
    assert any("close" in w and "not in" in w for w in warnings)

    # Negative volume
    bad_volume = {
        "records": [
            {
                "open": 65.5, "high": 67.0, "low": 65.0,
                "close": 66.25, "volume": -10,
                "symbol": "HE", "ts_event": "2026-02-18T00:00:00Z",
            }
        ]
    }
    cleaned, warnings = p.validate(bad_volume)
    assert len(cleaned["records"]) == 0
    assert any("volume" in w for w in warnings)

    # Empty records
    empty = {"records": []}
    cleaned, warnings = p.validate(empty)
    assert len(cleaned["records"]) == 0
    assert any("No records" in w for w in warnings)


# ---------------------------------------------------------------------------
# Test 4: IBFuturesIngestPipeline — persist() writes parquet + features
# ---------------------------------------------------------------------------


def test_ib_futures_persist_writes_parquet_and_features(futures_pipeline, tmp_parquet_dir, tmp_feature_db):
    """persist() writes records to parquet lake and close feature to feature store."""
    p = futures_pipeline
    data = {
        "records": [
            {
                "open": 65.5, "high": 67.0, "low": 65.0,
                "close": 66.25, "volume": 1234,
                "symbol": "HEG6", "ts_event": "2026-02-18T00:00:00Z",
            }
        ]
    }

    p.persist(data, "HE", DATE)

    table = p.parquet_lake.read(data_type="futures", market="HE")
    assert len(table) == 1
    assert table.column("close")[0].as_py() == 66.25

    features = p.feature_store.get_features_at("HE", DATE)
    assert "futures_close_HEG6" in features
    assert features["futures_close_HEG6"] == 66.25


# ---------------------------------------------------------------------------
# Test 5: IBOptionsIngestPipeline — full chain fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("hydra.data.ingestion.ib_options.asyncio.sleep", new_callable=AsyncMock)
@patch("ib_async.FuturesOption")
@patch("ib_async.Future")
async def test_ib_options_fetch_chain(
    mock_future, mock_futures_option, mock_sleep, parquet_lake, feature_store
):
    """run_async returns True after fetching and processing the full options chain."""
    chain = _make_chain()

    underlying_ticker = MagicMock()
    underlying_ticker.last = 66.25
    underlying_ticker.bid = 66.0
    underlying_ticker.ask = 66.5
    underlying_ticker.close = float("nan")

    opt_ticker = MagicMock()
    opt_ticker.bid = 1.5
    opt_ticker.ask = 1.6
    opt_ticker.bidSize = 10
    opt_ticker.askSize = 15
    opt_ticker.putOpenInterest = 500.0
    opt_ticker.callOpenInterest = 800.0
    opt_ticker.volume = 100.0

    mock_underlying = MagicMock()
    mock_underlying.conId = 123
    mock_underlying.symbol = "HE"

    mock_opt_contract = MagicMock()
    mock_opt_contract.conId = 456
    mock_opt_contract.strike = 66.0
    mock_opt_contract.right = "C"
    mock_opt_contract.lastTradeDateOrContractMonth = "20260315"

    broker = MagicMock()
    broker.ib = MagicMock()
    broker.ib.reqMarketDataType = MagicMock()
    broker.ib.cancelMktData = MagicMock()
    broker.ib.qualifyContractsAsync = AsyncMock(
        side_effect=[[mock_underlying], [mock_opt_contract]]
    )
    broker.ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[chain])

    call_count = [0]

    def _mkt_data_side_effect(contract, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return underlying_ticker
        return opt_ticker

    broker.ib.reqMktData = MagicMock(side_effect=_mkt_data_side_effect)

    pipeline = IBOptionsIngestPipeline(
        broker=broker, parquet_lake=parquet_lake, feature_store=feature_store
    )

    result = await pipeline.run_async("HE", DATE)

    assert result is True


# ---------------------------------------------------------------------------
# Test 6: IBOptionsIngestPipeline — batching (>50 contracts)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("hydra.data.ingestion.ib_options.asyncio.sleep", new_callable=AsyncMock)
@patch("ib_async.FuturesOption")
@patch("ib_async.Future")
async def test_ib_options_batching(
    mock_future, mock_futures_option, mock_sleep, parquet_lake, feature_store
):
    """run_async processes >50 contracts in batches of BATCH_SIZE=50."""
    # 9 strikes * 2 rights * 3 expiries = 54 contracts (> BATCH_SIZE=50)
    # underlying=66, STRIKE_RANGE=15 -> keeps strikes in [51,81]; all 9 below pass
    many_strikes = [float(60 + i * 2) for i in range(9)]  # 60,62,...,76
    many_expiries = ["20260315", "20260415", "20260515"]
    chain = _make_chain(strikes=many_strikes, expirations=many_expiries)

    underlying_ticker = MagicMock()
    underlying_ticker.last = 66.0  # within STRIKE_RANGE=15 of all strikes 60-76
    underlying_ticker.bid = None
    underlying_ticker.ask = None
    underlying_ticker.close = float("nan")

    opt_ticker = MagicMock()
    opt_ticker.bid = 1.5
    opt_ticker.ask = 1.6
    opt_ticker.bidSize = 5
    opt_ticker.askSize = 5
    opt_ticker.putOpenInterest = 100.0
    opt_ticker.callOpenInterest = 200.0
    opt_ticker.volume = 50.0

    mock_underlying = MagicMock()
    mock_underlying.conId = 123
    mock_underlying.symbol = "HE"

    def _make_opt_contract(i):
        c = MagicMock()
        c.conId = 1000 + i
        c.strike = 66.0
        c.right = "C" if i % 2 == 0 else "P"
        c.lastTradeDateOrContractMonth = "20260315"
        return c

    broker = MagicMock()
    broker.ib = MagicMock()
    broker.ib.reqMarketDataType = MagicMock()
    broker.ib.cancelMktData = MagicMock()
    broker.ib.reqSecDefOptParamsAsync = AsyncMock(return_value=[chain])

    call_count = [0]

    def _qualify_side_effect(*contracts):
        if call_count[0] == 0:
            call_count[0] += 1
            return [mock_underlying]
        # Return one mock contract per input
        n = len(contracts) if contracts else 1
        return [_make_opt_contract(i) for i in range(n)]

    broker.ib.qualifyContractsAsync = AsyncMock(side_effect=_qualify_side_effect)

    mkt_call_count = [0]

    def _mkt_data(contract, *args, **kwargs):
        mkt_call_count[0] += 1
        if mkt_call_count[0] == 1:
            return underlying_ticker
        return opt_ticker

    broker.ib.reqMktData = MagicMock(side_effect=_mkt_data)

    pipeline = IBOptionsIngestPipeline(
        broker=broker, parquet_lake=parquet_lake, feature_store=feature_store
    )

    result = await pipeline.run_async("HE", DATE)

    assert result is True
    # 54 option contracts (9 strikes * 2 rights * 3 expiries) + 1 underlying = 55 total
    # reqMktData calls; with BATCH_SIZE=50, at least 2 batches must have been processed
    assert mkt_call_count[0] > 50


# ---------------------------------------------------------------------------
# Test 7: IBOptionsIngestPipeline — persist() writes summary features
# ---------------------------------------------------------------------------


def test_ib_options_persist_summary_features(options_pipeline):
    """persist() writes put_call_oi_ratio, total_oi, liquid_strike_count to feature store."""
    p = options_pipeline
    # Mix of puts and calls with OI — uses the actual schema from persist()
    data = {
        "records": [
            {
                "instrument_id": 1001,
                "strike": 64.0, "expiry": "20260315", "is_call": False,
                "bid": 1.2, "ask": 1.3, "bid_size": 10, "ask_size": 12,
                "oi": 300.0, "volume": 50,
            },
            {
                "instrument_id": 1002,
                "strike": 66.0, "expiry": "20260315", "is_call": True,
                "bid": 1.5, "ask": 1.6, "bid_size": 20, "ask_size": 25,
                "oi": 500.0, "volume": 80,
            },
            {
                "instrument_id": 1003,
                "strike": 68.0, "expiry": "20260315", "is_call": True,
                "bid": 0.8, "ask": 0.9, "bid_size": 8, "ask_size": 10,
                "oi": 200.0, "volume": 30,
            },
        ]
    }

    p.persist(data, "HE", DATE)

    features = p.feature_store.get_features_at("HE", DATE)
    assert "put_call_oi_ratio" in features
    assert "total_oi" in features
    assert "liquid_strike_count" in features

    # put_call_oi_ratio = put_OI / call_OI = 300 / (500+200) = 300/700 ≈ 0.4286
    assert abs(features["put_call_oi_ratio"] - 300.0 / 700.0) < 1e-4
    assert features["total_oi"] == 1000.0
    # liquid_strike_count = records with oi >= 50: all 3 (300, 500, 200 all >= 50)
    assert features["liquid_strike_count"] == 3.0


# ---------------------------------------------------------------------------
# Helpers for runner tests (mirror test_runner.py _make_runner style)
# ---------------------------------------------------------------------------


def _make_runner(ingestion_pipelines=None, model_fitted=True):
    """Create a PaperTradingRunner with all dependencies mocked."""
    from hydra.execution.runner import PaperTradingRunner

    broker = MagicMock()
    broker.is_connected = MagicMock(return_value=True)
    broker._port = 4002
    broker.connect = AsyncMock()
    broker.disconnect = AsyncMock()
    broker.reconnect = AsyncMock()
    broker.get_account_summary = AsyncMock(return_value=[])
    broker.get_positions = AsyncMock(return_value=[])

    risk_gate = AsyncMock()

    order_manager = AsyncMock()
    mock_fill = MagicMock()
    mock_fill.execution.price = 350.5
    mock_fill.execution.shares = 1
    mock_fill.time = 50.0
    mock_trade = MagicMock()
    mock_trade.fills = [mock_fill]
    mock_trade.order.orderId = 42
    order_manager.route_order = AsyncMock(return_value=[mock_trade])

    fill_journal = MagicMock()
    fill_journal.log_fill = MagicMock(return_value=1)
    fill_journal.close = MagicMock()
    fill_journal.get_fills = MagicMock(return_value=[])

    agent_loop = MagicMock()
    agent_result = MagicMock()
    agent_result.phase_reached.value = "observe"
    agent_result.promoted = False
    agent_result.rolled_back = False
    agent_result.skipped_reason = "No drift detected"
    agent_loop.run_cycle = MagicMock(return_value=agent_result)

    import numpy as np

    model = MagicMock()
    model.is_fitted = model_fitted
    model.predict = MagicMock(return_value=np.array([1]))

    reconciler = MagicMock()
    reconciler.reconcile = MagicMock(return_value=None)

    feature_assembler = MagicMock()
    feature_assembler.FEATURE_NAMES = [f"f{i}" for i in range(17)]
    feature_assembler.assemble_at = MagicMock(
        return_value={f"f{i}": float(i) for i in range(17)}
    )

    config = {"trading_mode": "paper"}

    runner = PaperTradingRunner(
        broker=broker,
        risk_gate=risk_gate,
        order_manager=order_manager,
        fill_journal=fill_journal,
        agent_loop=agent_loop,
        model=model,
        reconciler=reconciler,
        feature_assembler=feature_assembler,
        config=config,
        ingestion_pipelines=ingestion_pipelines,
    )

    return runner, {
        "broker": broker,
        "agent_loop": agent_loop,
        "model": model,
        "order_manager": order_manager,
        "fill_journal": fill_journal,
    }


# ---------------------------------------------------------------------------
# Test 8: PaperTradingRunner calls ingestion pipelines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_calls_ingestion():
    """run_daily_cycle calls run_async on each ingestion pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.run_async = AsyncMock(return_value=True)

    runner, mocks = _make_runner(ingestion_pipelines=[mock_pipeline], model_fitted=True)

    market_snapshot = {
        "mid_price": 350.0,
        "spread": 0.25,
        "adv": 500.0,
        "volatility": 0.02,
        "contract": MagicMock(),
    }
    with patch.object(
        runner, "_fetch_market_snapshot", new_callable=AsyncMock, return_value=market_snapshot
    ):
        await runner.run_daily_cycle()

    mock_pipeline.run_async.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 9: PaperTradingRunner continues on ingestion failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_continues_on_ingestion_failure():
    """run_daily_cycle continues the agent loop even when an ingestion pipeline raises."""
    mock_pipeline = MagicMock()
    mock_pipeline.run_async = AsyncMock(side_effect=Exception("IB timeout"))

    runner, mocks = _make_runner(ingestion_pipelines=[mock_pipeline], model_fitted=False)

    result = await runner.run_daily_cycle()

    # Agent loop must still have been called despite the ingestion failure
    mocks["agent_loop"].run_cycle.assert_called_once()

    # Cycle wasn't aborted — we get a valid summary dict
    assert "cycle_time" in result
