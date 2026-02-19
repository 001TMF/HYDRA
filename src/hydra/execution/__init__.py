"""Execution layer: broker abstraction and risk middleware.

Public API:
    - BrokerGateway: ib_async wrapper with connection management and reconnection
    - RiskGate: Mandatory pre-trade circuit breaker middleware
"""

from hydra.execution.broker import BrokerGateway

__all__ = [
    "BrokerGateway",
]
