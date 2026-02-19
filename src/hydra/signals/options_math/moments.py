"""Implied moments computed from Breeden-Litzenberger density.

Computes mean, variance, skew, and kurtosis of the risk-neutral distribution
via numerical integration over the B-L density. These moments are the features
that feed the divergence signal in Phase 2.

For degraded input (insufficient data for density), returns None for all
moments except ATM IV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hydra.signals.options_math.density import DataQuality, ImpliedDensityResult


@dataclass
class ImpliedMoments:
    """Implied moments of the risk-neutral distribution.

    Attributes:
        mean: First moment (expected value under risk-neutral measure).
        variance: Second central moment.
        skew: Third standardized moment.
        kurtosis: Fourth standardized moment (raw, not excess).
        atm_iv: At-the-money implied volatility (always available).
        quality: Data quality (FULL or DEGRADED).
        warnings: List of warnings from moment computation.
    """

    mean: Optional[float]
    variance: Optional[float]
    skew: Optional[float]
    kurtosis: Optional[float]
    atm_iv: Optional[float]
    quality: DataQuality
    warnings: list[str] = field(default_factory=list)


def compute_moments(density_result: ImpliedDensityResult) -> ImpliedMoments:
    """Compute implied moments from a B-L density result.

    For FULL quality density, computes all four moments via numerical
    integration (trapezoid rule). For DEGRADED density, returns None
    for all moments except ATM IV.

    Parameters
    ----------
    density_result : ImpliedDensityResult
        Output from extract_density(), containing strikes, density, and quality.

    Returns
    -------
    ImpliedMoments with computed moments and quality indicator.
    """
    warn_list: list[str] = []

    # Degraded path: no density to integrate
    if density_result.quality != DataQuality.FULL:
        warn_list.append(
            "Density quality is not FULL; moments cannot be computed. "
            "Returning ATM IV only."
        )
        return ImpliedMoments(
            mean=None,
            variance=None,
            skew=None,
            kurtosis=None,
            atm_iv=density_result.atm_iv,
            quality=density_result.quality,
            warnings=warn_list,
        )

    K = density_result.strikes
    f = density_result.density

    # Mean: E[K] = integral(K * f(K) dK)
    mean = float(np.trapezoid(K * f, K))

    # Variance: E[(K - mean)^2] = integral((K - mean)^2 * f(K) dK)
    variance = float(np.trapezoid((K - mean) ** 2 * f, K))

    # Guard against zero/near-zero variance (degenerate density)
    if variance < 1e-12:
        warn_list.append("Near-zero variance detected; skew and kurtosis unreliable")
        return ImpliedMoments(
            mean=mean,
            variance=variance,
            skew=0.0,
            kurtosis=0.0,
            atm_iv=density_result.atm_iv,
            quality=DataQuality.FULL,
            warnings=warn_list,
        )

    std = np.sqrt(variance)

    # Skew: E[((K - mean) / std)^3]
    skew = float(np.trapezoid(((K - mean) / std) ** 3 * f, K))

    # Kurtosis: E[((K - mean) / std)^4]  (raw, not excess)
    kurtosis = float(np.trapezoid(((K - mean) / std) ** 4 * f, K))

    return ImpliedMoments(
        mean=mean,
        variance=variance,
        skew=skew,
        kurtosis=kurtosis,
        atm_iv=density_result.atm_iv,
        quality=DataQuality.FULL,
        warnings=warn_list,
    )
