"""COT sentiment scoring: normalized crowd positioning from CFTC managed money data.

Transforms raw CFTC managed money net positioning into a sentiment score
in [-1, +1] via 52-week percentile rank, with a confidence weight in [0, 1]
derived from open interest magnitude and positioning concentration.

This is the first half of the divergence signal. The divergence detector
(Plan 02-03) will compare this sentiment against options-implied expectations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import percentileofscore


@dataclass
class SentimentScore:
    """Normalized sentiment score from COT managed money positioning.

    Attributes
    ----------
    score : float
        Sentiment in [-1, +1]: -1 extreme bearish, +1 extreme bullish.
    confidence : float
        Confidence weight in [0, 1] based on OI magnitude and concentration.
    components : dict
        Breakdown: managed_money_pct_rank, oi_rank, concentration.
    """

    score: float
    confidence: float
    components: dict


def compute_cot_sentiment(
    managed_money_net: float,
    producer_net: float,
    total_oi: float,
    history_managed: np.ndarray,
    history_oi: np.ndarray,
) -> SentimentScore:
    """Compute a normalized sentiment score from COT managed money positioning.

    Parameters
    ----------
    managed_money_net : float
        Current managed money net position (long - short).
    producer_net : float
        Current producer/merchant net position (reserved for future use).
    total_oi : float
        Current total open interest.
    history_managed : np.ndarray
        Historical managed money net positions (e.g., 52-week lookback).
    history_oi : np.ndarray
        Historical total open interest values (same lookback period).

    Returns
    -------
    SentimentScore
        Normalized sentiment score with confidence weight and component breakdown.
    """
    # Insufficient history: return neutral with zero confidence
    if len(history_managed) < 4:
        return SentimentScore(score=0.0, confidence=0.0, components={})

    # Percentile rank of current positioning within history
    pct_rank = percentileofscore(history_managed, managed_money_net) / 100.0

    # Score: map [0, 1] percentile to [-1, +1] sentiment
    score = float(np.clip(2.0 * pct_rank - 1.0, -1.0, 1.0))

    # Confidence components
    oi_rank = percentileofscore(history_oi, total_oi) / 100.0
    concentration = abs(managed_money_net) / max(total_oi, 1.0)

    # Confidence: weighted blend of OI rank and concentration
    # Higher OI = more representative market; higher concentration = stronger signal
    confidence = float(np.clip(
        0.6 * oi_rank + 0.4 * min(concentration * 5.0, 1.0),
        0.0,
        1.0,
    ))

    return SentimentScore(
        score=score,
        confidence=confidence,
        components={
            "managed_money_pct_rank": float(pct_rank),
            "oi_rank": float(oi_rank),
            "concentration": float(concentration),
        },
    )
