"""PaperTradingRunner: daily cycle orchestrator with APScheduler (EXEC-01, EXEC-05).

Orchestrates the complete daily trading cycle:
    1. Check broker connection (reconnect if needed)
    2. Get account state for risk parameters
    3. Run agent loop (self-healing: observe/diagnose/hypothesize/experiment/evaluate)
    4. Generate trading signal via BaselineModel.predict()
    5. Execute signal through OrderManager -> RiskGate -> BrokerGateway
    6. Log fills to FillJournal with predicted and actual slippage

The agent loop and model serve independent roles:
    - AgentLoop: maintains model quality over time (detect drift -> retrain -> evaluate)
    - BaselineModel.predict(): produces today's trading decision

APScheduler runs the daily cycle at a configurable time (default 2 PM CT,
after 1:15 PM CT pit close for agricultural futures).

Live mode requires explicit ``HYDRA_LIVE_CONFIRMED=true`` env var.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from hydra.agent.loop import AgentLoop
    from hydra.execution.broker import BrokerGateway
    from hydra.execution.fill_journal import FillJournal
    from hydra.execution.order_manager import OrderManager
    from hydra.execution.reconciler import ReconciliationReport, SlippageReconciler
    from hydra.execution.risk_gate import RiskGate
    from hydra.model.baseline import BaselineModel
    from hydra.model.features import FeatureAssembler

logger = structlog.get_logger(__name__)


class PaperTradingRunner:
    """Orchestrates the daily paper trading cycle with APScheduler.

    All dependencies are injected for testability. The runner ties together
    the agent loop (self-healing), the trading model (signal generation),
    and the execution pipeline (OrderManager -> RiskGate -> BrokerGateway).

    Parameters
    ----------
    broker : BrokerGateway
        IB connection wrapper.
    risk_gate : RiskGate
        Mandatory pre-trade risk middleware.
    order_manager : OrderManager
        Smart order routing (limit-patience + TWAP).
    fill_journal : FillJournal
        SQLite fill logging with slippage tracking.
    agent_loop : AgentLoop
        Self-healing cycle (observe/diagnose/hypothesize/experiment/evaluate).
    model : BaselineModel
        Trading signal source (directional prediction).
    reconciler : SlippageReconciler
        Predicted vs actual slippage comparison.
    feature_assembler : FeatureAssembler | None
        Feature matrix assembler for model prediction. If None, model
        prediction is skipped (useful for tests or before feature store
        is populated).
    config : dict | None
        Runtime configuration with keys:
        - schedule_hour (int, default 14): hour for daily cycle
        - schedule_minute (int, default 0): minute for daily cycle
        - schedule_timezone (str, default "US/Central"): timezone
        - trading_mode (str, default "paper"): "paper" or "live"
        - market (str, default "ZO"): market identifier for feature store
        - contract_symbol (str, default "ZO"): IB contract symbol
        - contract_exchange (str, default "CBOT"): IB exchange
        - contract_sec_type (str, default "FUT"): IB security type
        - n_contracts (int, default 1): order size per signal
    """

    def __init__(
        self,
        broker: BrokerGateway,
        risk_gate: RiskGate,
        order_manager: OrderManager,
        fill_journal: FillJournal,
        agent_loop: AgentLoop,
        model: BaselineModel,
        reconciler: SlippageReconciler,
        feature_assembler: FeatureAssembler | None = None,
        config: dict | None = None,
    ) -> None:
        self._broker = broker
        self._risk_gate = risk_gate
        self._order_manager = order_manager
        self._fill_journal = fill_journal
        self._agent_loop = agent_loop
        self._model = model
        self._reconciler = reconciler
        self._feature_assembler = feature_assembler

        cfg = config or {}
        self._schedule_hour: int = cfg.get("schedule_hour", 14)
        self._schedule_minute: int = cfg.get("schedule_minute", 0)
        self._schedule_timezone: str = cfg.get("schedule_timezone", "US/Central")
        self._trading_mode: str = cfg.get("trading_mode", "paper")
        self._market: str = cfg.get("market", "ZO")
        self._contract_symbol: str = cfg.get("contract_symbol", "ZO")
        self._contract_exchange: str = cfg.get("contract_exchange", "CBOT")
        self._contract_sec_type: str = cfg.get("contract_sec_type", "FUT")
        self._n_contracts: int = cfg.get("n_contracts", 1)

        self._scheduler = None
        self._qualified_contract = None  # cached after first qualification

    async def start(self) -> None:
        """Start the paper trading runner.

        1. Validate trading mode (live requires HYDRA_LIVE_CONFIRMED=true).
        2. Connect to broker.
        3. Set up APScheduler with CronTrigger for daily cycle.
        4. Start the scheduler.

        Raises
        ------
        ValueError
            If trading_mode is "live" and HYDRA_LIVE_CONFIRMED is not set.
        """
        if self._trading_mode == "live":
            confirmed = os.environ.get("HYDRA_LIVE_CONFIRMED", "").lower()
            if confirmed != "true":
                raise ValueError(
                    "Live trading requires HYDRA_LIVE_CONFIRMED=true env var. "
                    "Set it explicitly to confirm live trading intent."
                )

        await self._broker.connect()

        logger.info(
            "runner_starting",
            trading_mode=self._trading_mode,
            schedule=f"{self._schedule_hour:02d}:{self._schedule_minute:02d}",
            timezone=self._schedule_timezone,
            port=self._broker._port,
        )

        self._scheduler = AsyncIOScheduler()
        trigger = CronTrigger(
            hour=self._schedule_hour,
            minute=self._schedule_minute,
            timezone=self._schedule_timezone,
        )
        self._scheduler.add_job(
            self.run_daily_cycle,
            trigger=trigger,
            id="daily_cycle",
            name="HYDRA Daily Trading Cycle",
        )
        self._scheduler.start()

        logger.info("runner_started", trading_mode=self._trading_mode)

    async def stop(self) -> None:
        """Stop the runner: shut down scheduler, disconnect broker, close journal."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

        await self._broker.disconnect()
        self._fill_journal.close()

        logger.info("runner_stopped")

    async def run_daily_cycle(self) -> dict:
        """Execute one complete daily trading cycle.

        Steps:
            1. Check/restore broker connection.
            2. Get account state (positions, summary) for risk parameters.
            3. Run agent loop (self-healing cycle).
            4. Generate trading signal via model.predict().
            5. Execute signal through risk-checked pipeline.
            6. Log fills with predicted/actual slippage.

        Returns
        -------
        dict
            Cycle summary with keys: cycle_time, agent_result, signal,
            n_fills, n_orders_blocked.
        """
        import numpy as np

        from hydra.execution.fill_journal import FillRecord
        from hydra.risk.slippage import estimate_slippage

        cycle_time = datetime.now(timezone.utc).isoformat()
        summary: dict = {
            "cycle_time": cycle_time,
            "agent_result": None,
            "signal": None,
            "n_fills": 0,
            "n_orders_blocked": 0,
        }

        # ------------------------------------------------------------------
        # 1. Check connection
        # ------------------------------------------------------------------
        if not self._broker.is_connected():
            logger.warning("broker_not_connected_attempting_reconnect")
            try:
                await self._broker.reconnect()
            except ConnectionError:
                logger.error("broker_reconnect_failed_skipping_cycle")
                return summary

        # ------------------------------------------------------------------
        # 2. Get account state
        # ------------------------------------------------------------------
        try:
            account_summary = await self._broker.get_account_summary()
            positions = await self._broker.get_positions()
        except Exception as exc:
            logger.error("account_state_failed", error=str(exc))
            return summary

        # Extract risk parameters from account state
        daily_pnl = 0.0
        current_equity = 100_000.0  # Default for paper
        peak_equity = 100_000.0
        position_value = 0.0

        for item in account_summary:
            tag = getattr(item, "tag", "")
            val = getattr(item, "value", "0")
            if tag == "DailyPnL":
                daily_pnl = float(val) if val else 0.0
            elif tag == "NetLiquidation":
                current_equity = float(val) if val else current_equity
            elif tag == "GrossPositionValue":
                position_value = float(val) if val else 0.0

        if current_equity > 0:
            daily_pnl = daily_pnl / current_equity  # Normalize to fraction
            position_value = position_value / current_equity
            peak_equity = max(peak_equity, current_equity)

        # ------------------------------------------------------------------
        # 3. Run agent loop (self-healing)
        # ------------------------------------------------------------------
        perf = self._get_recent_performance()
        try:
            agent_result = self._agent_loop.run_cycle(
                recent_returns=perf["recent_returns"],
                predictions=perf["predictions"],
                actuals=perf["actuals"],
                probabilities=perf["probabilities"],
                baseline_sharpe=perf["baseline_sharpe"],
            )
            summary["agent_result"] = {
                "phase_reached": agent_result.phase_reached.value,
                "promoted": agent_result.promoted,
                "rolled_back": agent_result.rolled_back,
                "skipped_reason": agent_result.skipped_reason,
            }
            logger.info(
                "agent_cycle_complete",
                phase_reached=agent_result.phase_reached.value,
                promoted=agent_result.promoted,
                skipped_reason=agent_result.skipped_reason,
            )
        except Exception as exc:
            logger.warning("agent_cycle_failed", error=str(exc))
            summary["agent_result"] = {"error": str(exc)}

        # ------------------------------------------------------------------
        # 4. Generate trading signal
        # ------------------------------------------------------------------
        if not self._model.is_fitted:
            logger.info("model_not_fitted_skipping_signal")
            summary["signal"] = None
            return summary

        try:
            features = self._assemble_features()
            if features is None:
                logger.warning("feature_assembly_failed_skipping_signal")
                summary["signal"] = None
                return summary

            prediction = self._model.predict(features)
            direction = "BUY" if prediction[0] == 1 else "SELL"
            n_contracts = self._n_contracts

            summary["signal"] = {
                "direction": direction,
                "n_contracts": n_contracts,
            }
            logger.info(
                "signal_generated",
                direction=direction,
                n_contracts=n_contracts,
            )
        except Exception as exc:
            logger.warning("signal_generation_failed", error=str(exc))
            summary["signal"] = None
            return summary

        # ------------------------------------------------------------------
        # 5. Get live market data + execute signal
        # ------------------------------------------------------------------
        try:
            market_data = await self._fetch_market_snapshot()
            mid_price = market_data["mid_price"]
            adv = market_data["adv"]
            spread = market_data["spread"]
            volatility = market_data["volatility"]
            contract = market_data["contract"]
            trade_loss = 0.0

            trades = await self._order_manager.route_order(
                contract=contract,
                direction=direction,
                n_contracts=n_contracts,
                adv=adv,
                mid_price=mid_price,
                daily_pnl=daily_pnl,
                peak_equity=peak_equity,
                current_equity=current_equity,
                position_value=position_value,
                trade_loss=trade_loss,
            )

            # ------------------------------------------------------------------
            # 6. Log fills
            # ------------------------------------------------------------------
            n_fills = 0
            n_blocked = 0

            for trade in trades:
                if trade is None:
                    n_blocked += 1
                    continue

                fills = getattr(trade, "fills", [])
                for fill in fills:
                    fill_price = fill.execution.price
                    actual_slippage = abs(fill_price - mid_price)
                    predicted_slippage = estimate_slippage(
                        order_size=n_contracts,
                        daily_volume=adv,
                        spread=spread,
                        daily_volatility=volatility,
                    )

                    record = FillRecord(
                        timestamp=cycle_time,
                        symbol=self._contract_symbol,
                        direction=1 if direction == "BUY" else -1,
                        n_contracts=fill.execution.shares,
                        order_price=mid_price,
                        fill_price=fill_price,
                        predicted_slippage=predicted_slippage,
                        actual_slippage=actual_slippage,
                        volume_at_fill=adv,
                        spread_at_fill=spread,
                        fill_latency_ms=getattr(fill, "time", 0),
                        order_id=getattr(trade.order, "orderId", None),
                    )
                    self._fill_journal.log_fill(record)
                    n_fills += 1

            summary["n_fills"] = n_fills
            summary["n_orders_blocked"] = n_blocked

            logger.info(
                "cycle_execution_complete",
                n_fills=n_fills,
                n_blocked=n_blocked,
                direction=direction,
            )

        except Exception as exc:
            logger.error("order_execution_failed", error=str(exc))

        return summary

    # ------------------------------------------------------------------
    # Data helpers — replace placeholders with real data sources
    # ------------------------------------------------------------------

    def _assemble_features(self) -> np.ndarray | None:
        """Assemble feature vector for model prediction.

        Uses FeatureAssembler to pull real features from the feature store.
        Returns None if assembler is not configured or data is unavailable.
        """
        import numpy as np

        if self._feature_assembler is None:
            logger.info("no_feature_assembler_configured")
            return None

        try:
            now = datetime.now(timezone.utc)
            feat_dict = self._feature_assembler.assemble_at(self._market, now)

            # Convert dict to numpy array in FEATURE_NAMES order
            names = self._feature_assembler.FEATURE_NAMES
            row = [feat_dict.get(n) for n in names]
            # Replace None with NaN for LightGBM native NaN handling
            row = [v if v is not None else float("nan") for v in row]
            return np.array([row])
        except Exception as exc:
            logger.warning("feature_assembly_error", error=str(exc))
            return None

    async def _fetch_market_snapshot(self) -> dict:
        """Fetch live bid/ask/volume from IB for the configured contract.

        Returns dict with mid_price, spread, adv, volatility, contract.
        Qualifies the contract on first call and caches it.
        """
        import math

        import numpy as np
        from ib_async import Contract

        # Qualify contract once and cache
        if self._qualified_contract is None:
            raw = Contract(
                symbol=self._contract_symbol,
                exchange=self._contract_exchange,
                secType=self._contract_sec_type,
            )
            self._qualified_contract = await self._broker.qualify_contract(raw)
            logger.info(
                "contract_qualified",
                symbol=self._contract_symbol,
                con_id=getattr(self._qualified_contract, "conId", None),
            )

        contract = self._qualified_contract

        # Use delayed data (free, 15-min delay) when live isn't subscribed.
        # Type 3 = delayed, falls back to live if subscribed.
        self._broker.ib.reqMarketDataType(3)

        # reqMktData is async-safe (reqTickers has nested-loop issues).
        # snapshot=True requests a single update then auto-cancels.
        import asyncio

        ticker = self._broker.ib.reqMktData(contract, "", True, False)
        await asyncio.sleep(3)  # allow delayed snapshot to arrive
        self._broker.ib.cancelMktData(contract)

        def _valid(val: float) -> bool:
            """Check IB value is usable (not NaN and not -1 sentinel)."""
            return not math.isnan(val) and val > 0

        if ticker and _valid(ticker.bid) and _valid(ticker.ask):
            mid_price = (ticker.bid + ticker.ask) / 2.0
            spread = ticker.ask - ticker.bid
        elif ticker and _valid(ticker.last):
            mid_price = ticker.last
            spread = mid_price * 0.002  # estimate 0.2% spread
            logger.warning("no_bid_ask_using_last", last=mid_price)
        elif ticker and _valid(ticker.close):
            mid_price = ticker.close
            spread = mid_price * 0.002
            logger.warning("no_last_using_close", close=mid_price)
        else:
            raise ValueError("No market data available from IB")

        # Request 20-day historical bars for ADV and volatility
        bars = await self._broker.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="20 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
        )

        if bars:
            volumes = [b.volume for b in bars]
            closes = [b.close for b in bars]
            adv = float(np.mean(volumes))
            # Daily volatility from log returns
            if len(closes) >= 2:
                log_returns = np.diff(np.log(closes))
                volatility = float(np.std(log_returns))
            else:
                volatility = 0.02
        else:
            adv = 500.0
            volatility = 0.02
            logger.warning("no_historical_bars_using_defaults")

        logger.info(
            "market_snapshot",
            mid_price=round(mid_price, 4),
            spread=round(spread, 4),
            adv=round(adv, 1),
            volatility=round(volatility, 4),
        )

        return {
            "mid_price": mid_price,
            "spread": spread,
            "adv": adv,
            "volatility": volatility,
            "contract": contract,
        }

    def _get_recent_performance(self, lookback: int = 30) -> dict:
        """Pull recent performance data from fill journal for the agent loop.

        Returns arrays for recent_returns, predictions, actuals, probabilities,
        and baseline_sharpe. If insufficient data, returns minimal arrays that
        will cause the agent loop to report 'no drift detected' (correct
        behavior when we don't have enough history).
        """
        import numpy as np

        fills = self._fill_journal.get_fills(
            symbol=self._contract_symbol, limit=lookback
        )

        if len(fills) < 2:
            # Not enough fills for meaningful performance analysis.
            # Agent loop will see no drift, which is correct.
            return {
                "recent_returns": np.array([0.0]),
                "predictions": np.array([0]),
                "actuals": np.array([0]),
                "probabilities": np.array([0.5]),
                "baseline_sharpe": 0.0,
            }

        # Compute per-fill returns: (fill_price - order_price) / order_price * direction
        returns = []
        predictions = []
        actuals = []
        for f in fills:
            ret = (f.fill_price - f.order_price) / max(f.order_price, 1e-10)
            ret *= f.direction  # +1 for long, -1 for short
            returns.append(ret)
            # direction=+1 means we predicted up (1), direction=-1 means down (0)
            predictions.append(1 if f.direction > 0 else 0)
            # actual: was the fill profitable?
            actuals.append(1 if ret > 0 else 0)

        returns_arr = np.array(returns)
        mean_ret = float(np.mean(returns_arr))
        std_ret = float(np.std(returns_arr)) if len(returns_arr) > 1 else 1.0
        sharpe = mean_ret / max(std_ret, 1e-10)

        return {
            "recent_returns": returns_arr,
            "predictions": np.array(predictions),
            "actuals": np.array(actuals),
            "probabilities": np.full(len(predictions), 0.5),
            "baseline_sharpe": sharpe,
        }

    async def run_reconciliation(
        self, symbol: str | None = None
    ) -> ReconciliationReport | None:
        """Run slippage reconciliation.

        Delegates to the ``SlippageReconciler`` for predicted vs actual
        slippage comparison.

        Parameters
        ----------
        symbol : str | None
            Restrict analysis to a specific instrument.

        Returns
        -------
        ReconciliationReport | None
            The report, or None if insufficient data.
        """
        return self._reconciler.reconcile(symbol=symbol)
