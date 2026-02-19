"""Kolmogorov-Smirnov test wrapper for distribution drift detection.

Uses the two-sample KS test from scipy.stats to determine whether
two samples come from the same distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp


def check_ks_drift(
    baseline: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """Check for distribution drift via two-sample KS test.

    Parameters
    ----------
    baseline : np.ndarray
        Reference distribution (1-D array).
    current : np.ndarray
        Current distribution to compare.
    alpha : float
        Significance level (default 0.05).

    Returns
    -------
    tuple[bool, float, float]
        (drifted, statistic, pvalue) where drifted is True if pvalue < alpha.
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    # Edge case: insufficient data
    if len(baseline) < 2 or len(current) < 2:
        return (False, 0.0, 1.0)

    stat, pvalue = ks_2samp(baseline, current)
    return (pvalue < alpha, float(stat), float(pvalue))
