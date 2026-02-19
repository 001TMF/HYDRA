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
    config : dict | None
        Runtime configuration with keys:
        - schedule_hour (int, default 14): hour for daily cycle
        - schedule_minute (int, default 0): minute for daily cycle
        - schedule_timezone (str, default "US/Central"): timezone
        - trading_mode (str, default "paper"): "paper" or "live"
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
        config: dict | None = None,
    ) -> None:
        self._broker = broker
        self._risk_gate = risk_gate
        self._order_manager = order_manager
        self._fill_journal = fill_journal
        self._agent_loop = agent_loop
        self._model = model
        self._reconciler = reconciler

        cfg = config or {}
        self._schedule_hour: int = cfg.get("schedule_hour", 14)
        self._schedule_minute: int = cfg.get("schedule_minute", 0)
        self._schedule_timezone: str = cfg.get("schedule_timezone", "US/Central")
        self._trading_mode: str = cfg.get("trading_mode", "paper")

        self._scheduler = None

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
        try:
            agent_result = self._agent_loop.run_cycle(
                recent_returns=np.array([0.0]),  # Minimal data for cycle
                predictions=np.array([0]),
                actuals=np.array([0]),
                probabilities=np.array([0.5]),
                baseline_sharpe=0.0,
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
            # Assemble minimal features from account state
            features = np.zeros((1, 10))  # Placeholder feature vector
            prediction = self._model.predict(features)
            direction = "BUY" if prediction[0] == 1 else "SELL"
            n_contracts = 1  # Conservative single-contract paper trading

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
        # 5. Execute signal
        # ------------------------------------------------------------------
        try:
            from ib_async import Contract

            contract = Contract(symbol="ZO", exchange="CBOT", secType="FUT")
            mid_price = 350.0  # Placeholder -- real data from broker in production
            adv = 500.0  # Average daily volume placeholder
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

                # Extract fills from trade
                fills = getattr(trade, "fills", [])
                for fill in fills:
                    fill_price = fill.execution.price
                    actual_slippage = abs(fill_price - mid_price)
                    predicted_slippage = estimate_slippage(
                        order_size=n_contracts,
                        daily_volume=adv,
                        spread=0.25,  # Typical thin-market spread
                        daily_volatility=0.02,
                    )

                    record = FillRecord(
                        timestamp=cycle_time,
                        symbol="ZO",
                        direction=1 if direction == "BUY" else -1,
                        n_contracts=fill.execution.shares,
                        order_price=mid_price,
                        fill_price=fill_price,
                        predicted_slippage=predicted_slippage,
                        actual_slippage=actual_slippage,
                        volume_at_fill=adv,
                        spread_at_fill=0.25,
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
