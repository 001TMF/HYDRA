"""Execution layer: broker abstraction, risk middleware, order management, and fill tracking.

Public API:
    - BrokerGateway: ib_async wrapper with connection management and reconnection
    - RiskGate: Mandatory pre-trade circuit breaker middleware
    - OrderManager: Smart order routing (limit-patience + TWAP)
    - FillJournal: SQLite fill logging with slippage tracking
    - FillRecord: Dataclass for a single fill entry
    - SlippageReconciler: Predicted vs actual slippage comparison
    - ReconciliationReport: Statistical comparison metrics
    - PaperTradingRunner: Daily cycle orchestrator with APScheduler
"""

from hydra.execution.broker import BrokerGateway
from hydra.execution.fill_journal import FillJournal, FillRecord
from hydra.execution.order_manager import OrderManager
from hydra.execution.reconciler import ReconciliationReport, SlippageReconciler
from hydra.execution.risk_gate import RiskGate
from hydra.execution.runner import PaperTradingRunner

__all__ = [
    "BrokerGateway",
    "FillJournal",
    "FillRecord",
    "OrderManager",
    "PaperTradingRunner",
    "ReconciliationReport",
    "RiskGate",
    "SlippageReconciler",
]
