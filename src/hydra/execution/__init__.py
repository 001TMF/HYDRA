"""Execution layer: broker abstraction, risk middleware, and order management.

Public API:
    - BrokerGateway: ib_async wrapper with connection management and reconnection
    - RiskGate: Mandatory pre-trade circuit breaker middleware
    - OrderManager: Smart order routing (limit-patience + TWAP)
"""

from hydra.execution.broker import BrokerGateway
from hydra.execution.order_manager import OrderManager
from hydra.execution.risk_gate import RiskGate

__all__ = [
    "BrokerGateway",
    "OrderManager",
    "RiskGate",
]
