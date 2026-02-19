"""Black-76 Greeks computation and aggregated flow metrics (GEX, vanna, charm).

Greeks flows represent market maker hedging pressure and predict short-term
price dynamics. GEX (gamma exposure) indicates whether dealers will amplify
or dampen price moves. Vanna and charm flows predict how the hedging pressure
will change with vol moves and time decay.

Sign convention (dealer-short assumption):
- Dealer assumed net short options (customers buy, dealers sell)
- Calls: dealer short -> sign = +1 for GEX (positive gamma)
- Puts: dealer short -> sign = -1 for GEX (negative gamma)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Shared quality enum (will be consolidated with density.py when created)
# ---------------------------------------------------------------------------

class DataQuality(Enum):
    """Quality level of computed results."""
    FULL = "full"           # >= min_liquid_strikes, all flows computed
    DEGRADED = "degraded"   # < min_liquid_strikes, flows zeroed
    STALE = "stale"         # data older than staleness threshold
    MISSING = "missing"     # no data available


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GreeksFlowResult:
    """Result of aggregated Greeks flow computation across an options chain."""
    gex: float                      # gamma exposure (GEX)
    vanna_flow: float               # aggregated vanna exposure
    charm_flow: float               # aggregated charm exposure
    quality: DataQuality            # data quality assessment
    liquid_strike_count: int        # number of strikes passing liquidity filters
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual Black-76 Greeks
# ---------------------------------------------------------------------------

def black76_greeks(
    F: float,
    K: float,
    r: float,
    T: float,
    sigma: float,
    is_call: bool,
) -> dict[str, float]:
    """Compute Greeks for a single option under the Black-76 model.

    Parameters
    ----------
    F : float
        Futures/forward price.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    T : float
        Time to expiry in years.
    sigma : float
        Implied volatility.
    is_call : bool
        True for call, False for put.

    Returns
    -------
    dict with keys: gamma, vanna, charm, delta, vega
    """
    # Guard: degenerate inputs return zeros
    if T <= 1e-10 or sigma <= 1e-10:
        return {
            "gamma": 0.0,
            "vanna": 0.0,
            "charm": 0.0,
            "delta": 0.0,
            "vega": 0.0,
        }

    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1)

    gamma = discount * pdf_d1 / (F * sigma * sqrt_T)
    vanna = -discount * pdf_d1 * d2 / sigma
    charm = -discount * pdf_d1 * (
        2 * r * T - d2 * sigma * sqrt_T
    ) / (2 * T * sigma * sqrt_T)

    if is_call:
        delta = discount * norm.cdf(d1)
    else:
        delta = discount * (norm.cdf(d1) - 1.0)

    vega = F * discount * pdf_d1 * sqrt_T

    return {
        "gamma": gamma,
        "vanna": vanna,
        "charm": charm,
        "delta": delta,
        "vega": vega,
    }


# ---------------------------------------------------------------------------
# Aggregated flow computation
# ---------------------------------------------------------------------------

def compute_greeks_flow(
    chain: dict,
    spot: float,
    r: float,
    contract_multiplier: float,
    min_liquid_strikes: int = 8,
    max_spread_pct: float = 0.20,
    min_oi: int = 50,
) -> GreeksFlowResult:
    """Aggregate Greeks flows across the full options chain.

    Parameters
    ----------
    chain : dict
        Keys: strikes, call_ivs, put_ivs, call_oi, put_oi, expiries_T,
        bid_ask_spread_pct.
    spot : float
        Current spot / futures price.
    r : float
        Risk-free rate.
    contract_multiplier : float
        Contract multiplier (e.g. 400 for lean hogs).
    min_liquid_strikes : int
        Minimum number of liquid strikes required for FULL quality.
    max_spread_pct : float
        Maximum bid-ask spread as fraction of mid for a strike to be liquid.
    min_oi : int
        Minimum open interest for a strike to be liquid.

    Returns
    -------
    GreeksFlowResult with aggregated GEX, vanna flow, charm flow.
    """
    strikes = chain["strikes"]
    call_ivs = chain["call_ivs"]
    put_ivs = chain["put_ivs"]
    call_oi = chain["call_oi"]
    put_oi = chain["put_oi"]
    expiries_T = chain["expiries_T"]
    bid_ask_spread_pct = chain.get("bid_ask_spread_pct", [0.0] * len(strikes))

    n = len(strikes)
    warnings: list[str] = []

    # Step 1: Count liquid strikes
    # A strike is liquid if EITHER call or put has OI >= min_oi AND spread <= max_spread_pct
    liquid_count = 0
    for i in range(n):
        oi_ok = call_oi[i] >= min_oi or put_oi[i] >= min_oi
        spread_ok = bid_ask_spread_pct[i] <= max_spread_pct
        if oi_ok and spread_ok:
            liquid_count += 1

    # Step 2: Degradation check
    if liquid_count < min_liquid_strikes:
        warnings.append(
            f"Only {liquid_count} liquid strikes "
            f"(need {min_liquid_strikes}); returning degraded result with zero flows"
        )
        return GreeksFlowResult(
            gex=0.0,
            vanna_flow=0.0,
            charm_flow=0.0,
            quality=DataQuality.DEGRADED,
            liquid_strike_count=liquid_count,
            warnings=warnings,
        )

    # Step 3: Aggregate flows across all strikes
    gex_total = 0.0
    vanna_total = 0.0
    charm_total = 0.0

    for i in range(n):
        K = strikes[i]
        T = expiries_T[i]

        # Skip expired or too-far-out options
        if T <= 0 or T > 2.0:
            continue

        # Call contribution (dealer short calls -> sign = +1)
        if call_ivs[i] > 0 and call_oi[i] > 0:
            g = black76_greeks(spot, K, r, T, call_ivs[i], is_call=True)
            sign = 1
            gex_total += sign * g["gamma"] * call_oi[i] * contract_multiplier * spot**2 / 100
            vanna_total += sign * g["vanna"] * call_oi[i] * contract_multiplier * spot
            charm_total += sign * g["charm"] * call_oi[i] * contract_multiplier

        # Put contribution (dealer short puts -> sign = -1)
        if put_ivs[i] > 0 and put_oi[i] > 0:
            g = black76_greeks(spot, K, r, T, put_ivs[i], is_call=False)
            sign = -1
            gex_total += sign * g["gamma"] * put_oi[i] * contract_multiplier * spot**2 / 100
            vanna_total += sign * g["vanna"] * put_oi[i] * contract_multiplier * spot
            charm_total += sign * g["charm"] * put_oi[i] * contract_multiplier

    return GreeksFlowResult(
        gex=gex_total,
        vanna_flow=vanna_total,
        charm_flow=charm_total,
        quality=DataQuality.FULL,
        liquid_strike_count=liquid_count,
        warnings=warnings,
    )
