"""Breeden-Litzenberger risk-neutral density extraction with graceful degradation.

Extracts the risk-neutral probability density from SVI-smoothed call prices
using the Breeden-Litzenberger second derivative approach:

    f(K) = e^{rT} * d^2 C / dK^2

where C(K) are smooth call prices from the SVI volatility surface. The density
tells us what the options market implies about future price distribution.

Graceful degradation (OPTS-05): when fewer than min_liquid_strikes pass
liquidity filters, return DEGRADED result with ATM IV only.

References:
    Breeden & Litzenberger, "Prices of State-Contingent Claims" (1978)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm

from hydra.signals.options_math.surface import (
    calibrate_svi,
    svi_to_call_prices,
)


# ---------------------------------------------------------------------------
# Shared data quality enum (canonical location; greeks.py re-exports this)
# ---------------------------------------------------------------------------


class DataQuality(Enum):
    """Quality level of computed results."""

    FULL = "full"  # >= min_liquid_strikes, all features computed
    DEGRADED = "degraded"  # < min_liquid_strikes, ATM IV only
    STALE = "stale"  # data older than staleness threshold
    MISSING = "missing"  # no data available


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ImpliedDensityResult:
    """Result of Breeden-Litzenberger density extraction.

    Attributes:
        strikes: Strike price grid used for the density computation.
        density: Estimated risk-neutral probability density at each strike.
        quality: Data quality assessment (FULL or DEGRADED).
        liquid_strike_count: Number of strikes that passed liquidity filters.
        atm_iv: At-the-money implied volatility (always computed as fallback).
        warnings: List of warning messages from the extraction process.
    """

    strikes: np.ndarray
    density: np.ndarray
    quality: DataQuality
    liquid_strike_count: int
    atm_iv: Optional[float]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Black-76 helpers for ATM IV extraction
# ---------------------------------------------------------------------------


def _black76_call(F: float, K: float, r: float, T: float, sigma: float) -> float:
    """Black-76 call price for a single strike."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = np.exp(-r * T)
    return float(discount * (F * sp_norm.cdf(d1) - K * sp_norm.cdf(d2)))


def _implied_vol_from_price(
    price: float,
    F: float,
    K: float,
    r: float,
    T: float,
    vol_lo: float = 0.001,
    vol_hi: float = 5.0,
) -> Optional[float]:
    """Invert Black-76 to find implied volatility using Brent's method.

    Returns None if the solver fails to converge.
    """
    discount = np.exp(-r * T)
    intrinsic = max(discount * (F - K), 0.0)

    # If price is below intrinsic or near zero, IV is degenerate
    if price <= intrinsic + 1e-12:
        return None

    def objective(sigma: float) -> float:
        return _black76_call(F, K, r, T, sigma) - price

    try:
        iv = brentq(objective, vol_lo, vol_hi, xtol=1e-10, maxiter=200)
        return float(iv)
    except (ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Main density extraction
# ---------------------------------------------------------------------------


def extract_density(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    oi: np.ndarray,
    bid_ask_spread_pct: np.ndarray,
    spot: float,
    r: float,
    T: float,
    min_liquid_strikes: int = 8,
    max_spread_pct: float = 0.20,
    min_oi: int = 50,
) -> ImpliedDensityResult:
    """Extract risk-neutral density from options chain data via Breeden-Litzenberger.

    Pipeline:
      1. Filter to liquid strikes (OI >= min_oi AND spread <= max_spread_pct)
      2. Compute ATM IV as universal fallback
      3. If too few liquid strikes -> DEGRADED with ATM IV only
      4. Invert call prices to implied vols (brentq per strike)
      5. Calibrate SVI to implied vols (surface.calibrate_svi)
      6. Generate smooth call prices on fine grid (svi_to_call_prices)
      7. Second derivative d2C/dK2 -> density = exp(rT) * d2C/dK2
      8. Clip negatives, normalize, validate

    Parameters
    ----------
    strikes : np.ndarray
        Strike prices from the options chain.
    call_prices : np.ndarray
        Observed call prices at each strike.
    oi : np.ndarray
        Open interest at each strike.
    bid_ask_spread_pct : np.ndarray
        Bid-ask spread as fraction of mid price at each strike.
    spot : float
        Current spot / futures price (used as forward approximation).
    r : float
        Risk-free rate.
    T : float
        Time to expiry in years.
    min_liquid_strikes : int
        Minimum liquid strikes required for FULL quality.
    max_spread_pct : float
        Maximum spread for a strike to be considered liquid.
    min_oi : int
        Minimum open interest for a strike to be considered liquid.

    Returns
    -------
    ImpliedDensityResult with density, quality, ATM IV, and diagnostics.
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    call_prices = np.asarray(call_prices, dtype=np.float64)
    oi = np.asarray(oi, dtype=np.float64)
    bid_ask_spread_pct = np.asarray(bid_ask_spread_pct, dtype=np.float64)

    warn_list: list[str] = []
    forward = spot  # use spot as forward approximation

    # -----------------------------------------------------------------------
    # Step 1: Filter to liquid strikes
    # -----------------------------------------------------------------------
    liquid_mask = (oi >= min_oi) & (bid_ask_spread_pct <= max_spread_pct)
    liquid_count = int(np.sum(liquid_mask))

    # -----------------------------------------------------------------------
    # Step 2: Compute ATM IV as fallback (use the strike nearest to spot)
    # -----------------------------------------------------------------------
    atm_idx = int(np.argmin(np.abs(strikes - spot)))
    atm_iv = _implied_vol_from_price(
        call_prices[atm_idx], forward, strikes[atm_idx], r, T
    )

    # -----------------------------------------------------------------------
    # Step 3: Degradation check
    # -----------------------------------------------------------------------
    if liquid_count < min_liquid_strikes:
        warn_list.append(
            f"Only {liquid_count} liquid strikes "
            f"(need {min_liquid_strikes}); returning degraded result with ATM IV only"
        )
        return ImpliedDensityResult(
            strikes=strikes[liquid_mask] if liquid_count > 0 else np.array([spot]),
            density=np.zeros(max(liquid_count, 1)),
            quality=DataQuality.DEGRADED,
            liquid_strike_count=liquid_count,
            atm_iv=atm_iv,
            warnings=warn_list,
        )

    # -----------------------------------------------------------------------
    # Step 4: Invert call prices to implied vols on liquid strikes
    # -----------------------------------------------------------------------
    liq_strikes = strikes[liquid_mask]
    liq_prices = call_prices[liquid_mask]

    implied_vols = np.zeros(liquid_count)
    valid_mask = np.ones(liquid_count, dtype=bool)

    for i in range(liquid_count):
        iv = _implied_vol_from_price(liq_prices[i], forward, liq_strikes[i], r, T)
        if iv is not None and iv > 0:
            implied_vols[i] = iv
        else:
            valid_mask[i] = False

    # Need enough valid IV points for SVI
    valid_count = int(np.sum(valid_mask))
    if valid_count < min_liquid_strikes:
        warn_list.append(
            f"Only {valid_count} valid IV inversions "
            f"(need {min_liquid_strikes}); returning degraded"
        )
        return ImpliedDensityResult(
            strikes=liq_strikes,
            density=np.zeros(liquid_count),
            quality=DataQuality.DEGRADED,
            liquid_strike_count=valid_count,
            atm_iv=atm_iv,
            warnings=warn_list,
        )

    # Use only valid points for SVI calibration
    svi_strikes = liq_strikes[valid_mask]
    svi_ivs = implied_vols[valid_mask]

    # -----------------------------------------------------------------------
    # Step 5: Calibrate SVI to implied vols
    # -----------------------------------------------------------------------
    svi_result = calibrate_svi(svi_strikes, svi_ivs, forward, T)
    warn_list.extend(svi_result.warnings)

    # -----------------------------------------------------------------------
    # Step 6: Generate smooth call prices on a fine grid
    # -----------------------------------------------------------------------
    k_min = svi_strikes.min()
    k_max = svi_strikes.max()
    # Narrow the grid slightly to avoid edge effects
    margin = (k_max - k_min) * 0.05
    fine_strikes = np.linspace(k_min + margin, k_max - margin, 200)

    fine_call_prices = svi_to_call_prices(
        svi_result.params, fine_strikes, forward, r, T
    )

    # -----------------------------------------------------------------------
    # Step 7: Second derivative -> density
    # -----------------------------------------------------------------------
    d2C = np.gradient(np.gradient(fine_call_prices, fine_strikes), fine_strikes)
    density = np.exp(r * T) * d2C

    # -----------------------------------------------------------------------
    # Step 8: Clip negatives, normalize, validate
    # -----------------------------------------------------------------------
    has_negatives = bool(np.any(density < 0))
    if has_negatives:
        negative_frac = float(np.sum(density < 0)) / len(density)
        warn_list.append(
            f"Negative density values clipped to zero "
            f"({negative_frac:.1%} of grid points)"
        )
        warnings.warn(
            "Breeden-Litzenberger density has negative regions; clipping to zero.",
            stacklevel=2,
        )
        density = np.maximum(density, 0.0)

    # Normalize to integrate to 1.0
    integral = np.trapezoid(density, fine_strikes)
    if integral > 1e-12:
        density = density / integral

    # Validate integral
    final_integral = np.trapezoid(density, fine_strikes)
    if abs(final_integral - 1.0) > 0.05:
        warn_list.append(
            f"Density integral {final_integral:.4f} deviates from 1.0 by more than 0.05"
        )

    # Validate mean vs forward
    density_mean = float(np.trapezoid(fine_strikes * density, fine_strikes))
    mean_err = abs(density_mean - forward) / forward
    if mean_err > 0.02:
        warn_list.append(
            f"Density mean {density_mean:.2f} differs from forward {forward:.2f} "
            f"by {mean_err:.2%} (> 2% threshold)"
        )

    return ImpliedDensityResult(
        strikes=fine_strikes,
        density=density,
        quality=DataQuality.FULL,
        liquid_strike_count=liquid_count,
        atm_iv=atm_iv,
        warnings=warn_list,
    )
