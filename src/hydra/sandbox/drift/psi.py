"""Population Stability Index (PSI) computation.

Detects distribution shift between a baseline and current sample
using quantile-based binning with epsilon smoothing to handle zero bins.

Interpretation:
  PSI < 0.10  -- no significant shift
  0.10 <= PSI < 0.25 -- moderate shift
  PSI >= 0.25 -- significant shift
"""

from __future__ import annotations

import numpy as np


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Compute Population Stability Index between two distributions.

    Parameters
    ----------
    baseline : np.ndarray
        Reference distribution (1-D array).
    current : np.ndarray
        Current distribution to compare against baseline.
    n_bins : int
        Number of quantile-based bins (default 10).
    epsilon : float
        Smoothing constant to avoid log(0) when bins are empty.

    Returns
    -------
    float
        PSI value. Non-negative; higher means more distribution shift.
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()

    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    # Adapt n_bins to available unique values in baseline
    n_unique = len(np.unique(baseline))
    effective_bins = max(2, min(n_bins, n_unique - 1))

    # Quantile-based bin edges from baseline
    quantiles = np.linspace(0, 1, effective_bins + 1)
    bin_edges = np.quantile(baseline, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Histogram both distributions against same bin edges
    baseline_counts = np.histogram(baseline, bins=bin_edges)[0].astype(float)
    current_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    # Convert to proportions with epsilon smoothing, then re-normalize
    baseline_pct = baseline_counts + epsilon
    current_pct = current_counts + epsilon
    baseline_pct = baseline_pct / baseline_pct.sum()
    current_pct = current_pct / current_pct.sum()

    # PSI formula
    psi = float(np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct)))
    return psi
