"""FastAPI dashboard application factory.

Creates the HYDRA monitoring dashboard with route registration,
Jinja2 template rendering, and static file serving. Optionally
starts the PaperTradingRunner as a FastAPI lifespan event.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

from hydra.dashboard.routes import api, api_data, api_model, api_agent, api_system, pages, sse

logger = structlog.get_logger(__name__)


def _build_runner(data_dir: Path):
    """Construct a PaperTradingRunner with all dependencies from data_dir.

    Mirrors the construction logic from CLI. Returns the runner instance.
    """
    from hydra.agent.budget import MutationBudget
    from hydra.agent.dedup import HypothesisDeduplicator
    from hydra.agent.diagnostician import Diagnostician
    from hydra.agent.experiment_runner import ExperimentRunner
    from hydra.agent.hypothesis import HypothesisEngine
    from hydra.agent.loop import AgentLoop
    from hydra.agent.promotion import PromotionEvaluator
    from hydra.agent.rollback import HysteresisRollbackTrigger
    from hydra.config.markets import MARKETS
    from hydra.data.ingestion.cot import COTIngestPipeline
    from hydra.data.ingestion.ib_futures import IBFuturesIngestPipeline
    from hydra.data.ingestion.ib_options import IBOptionsIngestPipeline
    from hydra.data.ingestion.options_features import OptionsFeaturePipeline
    from hydra.data.store.feature_store import FeatureStore
    from hydra.data.store.parquet_lake import ParquetLake
    from hydra.execution.broker import BrokerGateway
    from hydra.execution.fill_journal import FillJournal
    from hydra.execution.order_manager import OrderManager
    from hydra.execution.reconciler import SlippageReconciler
    from hydra.execution.risk_gate import RiskGate
    from hydra.execution.runner import PaperTradingRunner
    from hydra.model.baseline import BaselineModel
    from hydra.model.features import FeatureAssembler
    from hydra.risk.circuit_breakers import CircuitBreakerManager
    from hydra.sandbox.evaluator import CompositeEvaluator
    from hydra.sandbox.journal import ExperimentJournal
    from hydra.sandbox.observer import DriftObserver
    from hydra.sandbox.registry import ModelRegistry

    fill_journal = FillJournal(data_dir / "fill_journal.db")
    experiment_journal = ExperimentJournal(data_dir / "experiment_journal.db")

    host = os.environ.get("IB_GATEWAY_HOST", "127.0.0.1")
    port = int(os.environ.get("IB_GATEWAY_PORT", "4002"))
    broker = BrokerGateway(host=host, port=port, client_id=1)

    breakers = CircuitBreakerManager()
    risk_gate = RiskGate(broker=broker, breakers=breakers)
    order_manager = OrderManager(risk_gate=risk_gate)
    agent_loop = AgentLoop(
        observer=DriftObserver(),
        diagnostician=Diagnostician(),
        hypothesis_engine=HypothesisEngine(),
        experiment_runner=ExperimentRunner(),
        evaluator=CompositeEvaluator(),
        journal=experiment_journal,
        registry=ModelRegistry(),
        rollback_trigger=HysteresisRollbackTrigger(),
        promotion_evaluator=PromotionEvaluator(),
        deduplicator=HypothesisDeduplicator(),
        budget=MutationBudget(),
    )
    model = BaselineModel()
    reconciler = SlippageReconciler(fill_journal)

    # Data ingestion pipelines (multi-market)
    parquet_lake = ParquetLake(data_dir / "lake")
    feature_store = FeatureStore(data_dir / "feature_store.db")

    symbols_raw = os.environ.get("HYDRA_MARKETS", "HE,LE,GF")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    market_configs = []
    market_pipelines = {}
    for symbol in symbols:
        if symbol not in MARKETS:
            logger.warning("unknown_market_symbol_skipping", symbol=symbol)
            continue
        cfg = MARKETS[symbol]
        market_configs.append(cfg)
        market_pipelines[symbol] = [
            IBFuturesIngestPipeline(
                broker=broker, parquet_lake=parquet_lake, feature_store=feature_store,
                exchange=cfg.exchange,
            ),
            IBOptionsIngestPipeline(
                broker=broker, parquet_lake=parquet_lake, feature_store=feature_store,
                exchange=cfg.exchange,
            ),
            COTIngestPipeline(
                parquet_lake=parquet_lake, feature_store=feature_store,
                cftc_code=cfg.cftc_code,
            ),
            OptionsFeaturePipeline(
                parquet_lake=parquet_lake, feature_store=feature_store,
            ),
        ]

    primary_market = market_configs[0].symbol if market_configs else "HE"
    feature_assembler = FeatureAssembler(feature_store=feature_store)

    runner_config = {
        "market": primary_market,
        "contract_symbol": primary_market,
        "contract_exchange": market_configs[0].exchange if market_configs else "CME",
        "trading_mode": os.environ.get("TRADING_MODE", "paper"),
    }

    return PaperTradingRunner(
        broker=broker,
        risk_gate=risk_gate,
        order_manager=order_manager,
        fill_journal=fill_journal,
        agent_loop=agent_loop,
        model=model,
        reconciler=reconciler,
        feature_assembler=feature_assembler,
        market_configs=market_configs,
        market_pipelines=market_pipelines,
        config=runner_config,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """FastAPI lifespan: start/stop PaperTradingRunner if enabled."""
    if getattr(app.state, "start_runner", False):
        try:
            # Patch event loop globally so ib_async works inside
            # uvicorn's loop (APScheduler + broker async calls).
            import nest_asyncio
            nest_asyncio.apply()

            runner = _build_runner(app.state.data_dir)
            await runner.start()
            app.state.runner = runner
            logger.info("paper_trading_runner_started_via_lifespan")
        except Exception as exc:
            logger.warning(
                "paper_trading_runner_start_failed",
                error=str(exc),
            )
            app.state.runner = None
    yield
    if hasattr(app.state, "runner") and app.state.runner is not None:
        if app.state.runner._scheduler is not None:
            app.state.runner._scheduler.shutdown(wait=False)
        logger.info("paper_trading_runner_stopped")


def create_app(
    data_dir: str | None = None, start_runner: bool = False
) -> FastAPI:
    """Create and configure the HYDRA Dashboard FastAPI application.

    Parameters
    ----------
    data_dir : str | None
        Path to the HYDRA data directory containing SQLite databases.
        Defaults to ``~/.hydra``. Tilde is expanded.
    start_runner : bool
        If True, start PaperTradingRunner as a lifespan event.
        Also activated by ``HYDRA_START_RUNNER=true`` env var.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    if data_dir is None:
        data_dir = os.environ.get("HYDRA_DATA_DIR", "~/.hydra")

    if not start_runner:
        start_runner = os.environ.get("HYDRA_START_RUNNER", "").lower() == "true"

    app = FastAPI(
        title="HYDRA Dashboard", docs_url=None, redoc_url=None, lifespan=_lifespan
    )

    app.state.data_dir = Path(data_dir).expanduser()
    app.state.start_runner = start_runner
    app.state.templates = Jinja2Templates(
        directory=str(Path(__file__).parent / "templates")
    )

    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    app.include_router(pages.router)
    app.include_router(api.router, prefix="/api")
    app.include_router(sse.router, prefix="/api/sse")
    app.include_router(api_model.router, prefix="/api/model")
    app.include_router(api_agent.router, prefix="/api/agent")
    app.include_router(api_data.router, prefix="/api/data")
    app.include_router(api_system.router, prefix="/api/system")

    return app
