"""Risk management module: slippage, position sizing, and circuit breakers.

Public API:
    - estimate_slippage: Volume-adaptive slippage using square-root impact model
    - fractional_kelly: Fractional Kelly position sizing
    - volume_capped_position: Convert Kelly % to integer contracts with volume cap
    - CircuitBreaker: Single circuit breaker with state machine
    - CircuitBreakerState: Enum for breaker lifecycle states
    - CircuitBreakerManager: Manages 4 independent circuit breakers
"""

from hydra.risk.circuit_breakers import (
    CircuitBreaker,
    CircuitBreakerManager,
    CircuitBreakerState,
)
from hydra.risk.position_sizing import fractional_kelly, volume_capped_position
from hydra.risk.slippage import estimate_slippage

__all__ = [
    "estimate_slippage",
    "fractional_kelly",
    "volume_capped_position",
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerManager",
]
