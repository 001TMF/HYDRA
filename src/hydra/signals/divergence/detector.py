"""Divergence detector: classifies options vs. sentiment divergence into 6-type taxonomy.

Compares what the options market implies (risk-neutral distribution moments)
against crowd positioning (COT sentiment) to identify exploitable mispricings.

Classification is rule-based using configurable thresholds -- no ML involved.
The LightGBM model (Plan 02-04) consumes the numerical components as features.

Taxonomy (PRD Section 5.3):
  1. options_bullish_sentiment_bearish: Options bullish + sentiment bearish -> long
  2. options_bearish_sentiment_bullish: Options bearish + sentiment bullish -> short
  3. sentiment_overreaction: Options neutral + extreme sentiment -> fade
  4. early_signal: Options directional + significant skew + flat sentiment -> early entry
  5. trend_follow: Both aligned in same direction -> follow
  6. volatility_play: High kurtosis + flat mean -> vol trade
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydra.signals.options_math.density import DataQuality


# ---------------------------------------------------------------------------
# Configurable thresholds (module-level constants)
# ---------------------------------------------------------------------------

# Options bias thresholds (as fraction of spot)
OPTIONS_DIRECTIONAL_THRESHOLD = 0.01   # |options_bias| > this = directional
OPTIONS_NEUTRAL_THRESHOLD = 0.02       # |options_bias| < this = neutral

# Sentiment thresholds
SENTIMENT_BEARISH_THRESHOLD = -0.3     # sentiment < this = bearish
SENTIMENT_BULLISH_THRESHOLD = 0.3      # sentiment > this = bullish
SENTIMENT_EXTREME_THRESHOLD = 0.6      # |sentiment| > this = extreme
SENTIMENT_FLAT_THRESHOLD = 0.3         # |sentiment| < this = flat

# Implied moments thresholds
SKEW_SIGNIFICANT_THRESHOLD = 0.5       # |skew| > this = significant skew
KURTOSIS_HIGH_THRESHOLD = 4.0          # kurtosis > this = fat tails

# Trend-follow alignment threshold
TREND_MIN_STRENGTH = 0.01             # both sides must exceed this

# Minimum history length for z-scoring
MIN_HISTORY_FOR_ZSCORE = 10

# Quality penalty for degraded data
DEGRADED_CONFIDENCE_PENALTY = 0.5

# Sentiment scaling factor for raw divergence computation
SENTIMENT_SCALE_FACTOR = 10.0


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class DivergenceSignal:
    """Result of divergence classification between options and sentiment.

    Attributes
    ----------
    direction : int
        +1 for long bias, -1 for short bias, 0 for neutral/vol.
    magnitude : float
        Z-scored magnitude of the divergence (raw if insufficient history).
    divergence_type : str
        One of the 6 taxonomy types or "neutral".
    confidence : float
        Composite confidence in [0, 1].
    suggested_bias : str
        Action recommendation: long, short, fade_sentiment, early_entry,
        trend_follow, vol_play, or neutral.
    """

    direction: int
    magnitude: float
    divergence_type: str
    confidence: float
    suggested_bias: str


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def classify_divergence(
    implied_mean: float | None,
    spot: float,
    implied_skew: float | None,
    implied_kurtosis: float | None,
    sentiment_score: float,
    sentiment_confidence: float,
    options_quality: DataQuality,
    history: list[float] | None = None,
) -> DivergenceSignal:
    """Classify the divergence between options-implied signals and COT sentiment.

    Parameters
    ----------
    implied_mean : float | None
        Risk-neutral expected value from implied moments. None if degraded.
    spot : float
        Current spot / futures price.
    implied_skew : float | None
        Third standardized moment of risk-neutral distribution.
    implied_kurtosis : float | None
        Fourth standardized moment (raw, not excess).
    sentiment_score : float
        COT sentiment in [-1, +1] from compute_cot_sentiment.
    sentiment_confidence : float
        Confidence of sentiment score in [0, 1].
    options_quality : DataQuality
        Quality level of options data (FULL, DEGRADED, etc.).
    history : list[float] | None
        Historical raw divergence values for z-scoring. If len >= 10,
        magnitude is z-scored against this distribution.

    Returns
    -------
    DivergenceSignal
        Classified signal with direction, magnitude, type, confidence, bias.
    """
    # -----------------------------------------------------------------------
    # Step 1: Handle degraded / missing options data
    # -----------------------------------------------------------------------
    quality_factor = 1.0
    if options_quality != DataQuality.FULL:
        quality_factor = DEGRADED_CONFIDENCE_PENALTY

    if implied_mean is None:
        confidence = _compute_confidence(
            0.0, sentiment_confidence, quality_factor
        )
        return DivergenceSignal(
            direction=0,
            magnitude=0.0,
            divergence_type="neutral",
            confidence=confidence,
            suggested_bias="neutral",
        )

    # -----------------------------------------------------------------------
    # Step 2: Compute options bias (positive = bullish, negative = bearish)
    # -----------------------------------------------------------------------
    options_bias = (implied_mean - spot) / spot

    # -----------------------------------------------------------------------
    # Step 3: Compute raw divergence and z-score if history available
    # -----------------------------------------------------------------------
    raw_divergence = options_bias - (sentiment_score / SENTIMENT_SCALE_FACTOR)
    magnitude = _zscore_magnitude(raw_divergence, history)

    # -----------------------------------------------------------------------
    # Step 4: Classify using priority-ordered rules
    # -----------------------------------------------------------------------
    divergence_type, direction, suggested_bias = _classify(
        options_bias=options_bias,
        sentiment_score=sentiment_score,
        implied_skew=implied_skew,
        implied_kurtosis=implied_kurtosis,
    )

    # -----------------------------------------------------------------------
    # Step 5: Compute composite confidence
    # -----------------------------------------------------------------------
    confidence = _compute_confidence(
        abs(magnitude), sentiment_confidence, quality_factor
    )

    return DivergenceSignal(
        direction=direction,
        magnitude=magnitude,
        divergence_type=divergence_type,
        confidence=confidence,
        suggested_bias=suggested_bias,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _classify(
    options_bias: float,
    sentiment_score: float,
    implied_skew: float | None,
    implied_kurtosis: float | None,
) -> tuple[str, int, str]:
    """Apply priority-ordered classification rules.

    Returns (divergence_type, direction, suggested_bias).
    """
    abs_bias = abs(options_bias)

    # Rule 1: Volatility play -- high kurtosis + flat mean (highest priority)
    if (
        implied_kurtosis is not None
        and implied_kurtosis > KURTOSIS_HIGH_THRESHOLD
        and abs_bias < OPTIONS_NEUTRAL_THRESHOLD
    ):
        return "volatility_play", 0, "vol_play"

    # Rule 2: Options bullish + sentiment bearish
    if (
        options_bias > OPTIONS_DIRECTIONAL_THRESHOLD
        and sentiment_score < SENTIMENT_BEARISH_THRESHOLD
    ):
        return "options_bullish_sentiment_bearish", 1, "long"

    # Rule 3: Options bearish + sentiment bullish
    if (
        options_bias < -OPTIONS_DIRECTIONAL_THRESHOLD
        and sentiment_score > SENTIMENT_BULLISH_THRESHOLD
    ):
        return "options_bearish_sentiment_bullish", -1, "short"

    # Rule 4: Sentiment overreaction -- options neutral + extreme sentiment
    if (
        abs_bias < OPTIONS_NEUTRAL_THRESHOLD
        and abs(sentiment_score) > SENTIMENT_EXTREME_THRESHOLD
    ):
        # Fade direction = opposite of sentiment
        fade_direction = -1 if sentiment_score > 0 else 1
        return "sentiment_overreaction", fade_direction, "fade_sentiment"

    # Rule 5: Early signal -- directional options + significant skew + flat sentiment
    if (
        abs_bias > OPTIONS_DIRECTIONAL_THRESHOLD
        and implied_skew is not None
        and abs(implied_skew) > SKEW_SIGNIFICANT_THRESHOLD
        and abs(sentiment_score) < SENTIMENT_FLAT_THRESHOLD
    ):
        direction = 1 if options_bias > 0 else -1
        return "early_signal", direction, "early_entry"

    # Rule 6: Trend follow -- both aligned in same direction
    if (
        abs_bias > TREND_MIN_STRENGTH
        and abs(sentiment_score) > TREND_MIN_STRENGTH
        and _same_sign(options_bias, sentiment_score)
    ):
        direction = 1 if options_bias > 0 else -1
        return "trend_follow", direction, "trend_follow"

    # Default: Neutral
    return "neutral", 0, "neutral"


def _zscore_magnitude(
    raw_divergence: float,
    history: list[float] | None,
) -> float:
    """Z-score the raw divergence against historical values.

    If history is None or too short (< MIN_HISTORY_FOR_ZSCORE), returns
    the raw divergence value.
    """
    if history is None or len(history) < MIN_HISTORY_FOR_ZSCORE:
        return float(raw_divergence)

    arr = np.array(history, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))

    if std < 1e-12:
        return float(raw_divergence)

    return float((raw_divergence - mean) / std)


def _compute_confidence(
    signal_strength: float,
    sentiment_confidence: float,
    quality_factor: float,
) -> float:
    """Compute composite confidence in [0, 1].

    Blends sentiment confidence, options quality factor, and
    signal strength (magnitude-based, capped at 1.0).
    """
    # Signal strength contribution: diminishing returns via tanh-like sigmoid
    strength_factor = min(signal_strength / 2.0, 1.0)

    raw_confidence = sentiment_confidence * quality_factor * max(strength_factor, 0.3)
    return float(np.clip(raw_confidence, 0.0, 1.0))


def _same_sign(a: float, b: float) -> bool:
    """Check if two values have the same sign (both positive or both negative)."""
    return (a > 0 and b > 0) or (a < 0 and b < 0)
