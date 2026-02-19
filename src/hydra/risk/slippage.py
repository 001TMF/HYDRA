"""Volume-adaptive slippage model using square-root market impact.

Implements the Almgren & Chriss (2001) square-root impact model:
    slippage = spread/2 + k * sigma_daily * sqrt(V_order / V_daily)

Components:
    - spread/2: half the bid-ask spread (crossing cost)
    - k * sigma * sqrt(participation_rate): market impact

Used by the walk-forward backtester (Plan 02-05) for every simulated trade
and by the runtime paper-trading engine in Phase 5.
"""

import math


def estimate_slippage(
    order_size: float,
    daily_volume: float,
    spread: float,
    daily_volatility: float,
    impact_coefficient: float = 0.1,
) -> float:
    """Estimate per-contract slippage using volume-adaptive square-root model.

    Args:
        order_size: Number of contracts in the order.
        daily_volume: Average daily volume (contracts).
        spread: Bid-ask spread in price units.
        daily_volatility: Daily price volatility (e.g., annualized / sqrt(252)).
        impact_coefficient: Market impact scaling factor (default 0.1,
            conservative for thin commodity markets).

    Returns:
        Estimated slippage per contract in price units.
    """
    crossing_cost = spread / 2.0
    participation_rate = order_size / max(daily_volume, 1)
    market_impact = impact_coefficient * daily_volatility * math.sqrt(participation_rate)
    return crossing_cost + market_impact
