"""TDD tests for divergence detector -- classifies options vs. sentiment divergence.

Tests all 6 divergence types from PRD Section 5.3 taxonomy plus edge cases:
1. options_bullish_sentiment_bearish
2. options_bearish_sentiment_bullish
3. sentiment_overreaction
4. early_signal
5. trend_follow
6. volatility_play
Plus: degraded quality, z-scored magnitude, dataclass fields, neutral fallback.
"""

import pytest

from hydra.signals.divergence.detector import DivergenceSignal, classify_divergence
from hydra.signals.options_math.density import DataQuality


# ---------------------------------------------------------------------------
# 1. Options bullish + sentiment bearish -> long bias
# ---------------------------------------------------------------------------
def test_options_bullish_sentiment_bearish():
    """Implied mean > spot (bullish options) + bearish sentiment -> long."""
    signal = classify_divergence(
        implied_mean=105.0,   # bullish: above spot
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=-0.5,  # bearish sentiment
        sentiment_confidence=0.8,
        options_quality=DataQuality.FULL,
    )
    assert signal.divergence_type == "options_bullish_sentiment_bearish"
    assert signal.direction == 1
    assert signal.suggested_bias == "long"
    assert signal.confidence > 0.0


# ---------------------------------------------------------------------------
# 2. Options bearish + sentiment bullish -> short bias
# ---------------------------------------------------------------------------
def test_options_bearish_sentiment_bullish():
    """Implied mean < spot (bearish options) + bullish sentiment -> short."""
    signal = classify_divergence(
        implied_mean=95.0,    # bearish: below spot
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=0.5,   # bullish sentiment
        sentiment_confidence=0.8,
        options_quality=DataQuality.FULL,
    )
    assert signal.divergence_type == "options_bearish_sentiment_bullish"
    assert signal.direction == -1
    assert signal.suggested_bias == "short"
    assert signal.confidence > 0.0


# ---------------------------------------------------------------------------
# 3. Sentiment overreaction: options neutral, extreme sentiment
# ---------------------------------------------------------------------------
def test_sentiment_overreaction():
    """Options near neutral + extreme sentiment -> fade sentiment."""
    signal = classify_divergence(
        implied_mean=100.5,   # ~neutral (within 2% of spot)
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=0.8,   # extreme bullish sentiment
        sentiment_confidence=0.7,
        options_quality=DataQuality.FULL,
    )
    assert signal.divergence_type == "sentiment_overreaction"
    assert signal.direction == -1  # opposite of bullish sentiment
    assert signal.suggested_bias == "fade_sentiment"


# ---------------------------------------------------------------------------
# 4. Early signal: options directional + significant skew + flat sentiment
# ---------------------------------------------------------------------------
def test_early_signal():
    """Options directional with significant skew, sentiment near zero."""
    signal = classify_divergence(
        implied_mean=103.0,   # directional (3% above spot)
        spot=100.0,
        implied_skew=-0.8,    # significant negative skew
        implied_kurtosis=3.0,
        sentiment_score=0.1,   # near zero sentiment
        sentiment_confidence=0.5,
        options_quality=DataQuality.FULL,
    )
    assert signal.divergence_type == "early_signal"
    assert signal.suggested_bias == "early_entry"


# ---------------------------------------------------------------------------
# 5. Trend follow: options and sentiment aligned
# ---------------------------------------------------------------------------
def test_trend_follow():
    """Both options and sentiment aligned in same bullish direction."""
    signal = classify_divergence(
        implied_mean=104.0,   # bullish
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=0.5,   # also bullish
        sentiment_confidence=0.8,
        options_quality=DataQuality.FULL,
    )
    assert signal.divergence_type == "trend_follow"
    assert signal.direction == 1
    assert signal.suggested_bias == "trend_follow"


# ---------------------------------------------------------------------------
# 6. Volatility play: high kurtosis + flat mean
# ---------------------------------------------------------------------------
def test_volatility_play():
    """High kurtosis with flat mean -> vol play."""
    signal = classify_divergence(
        implied_mean=100.5,   # near spot (flat)
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=5.5,  # high kurtosis (> 4.0)
        sentiment_score=0.1,
        sentiment_confidence=0.5,
        options_quality=DataQuality.FULL,
    )
    assert signal.divergence_type == "volatility_play"
    assert signal.direction == 0
    assert signal.suggested_bias == "vol_play"


# ---------------------------------------------------------------------------
# 7. Degraded quality reduces confidence
# ---------------------------------------------------------------------------
def test_degraded_quality_reduces_confidence():
    """DataQuality.DEGRADED should reduce confidence below 0.5."""
    signal = classify_divergence(
        implied_mean=105.0,
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=-0.5,
        sentiment_confidence=0.8,
        options_quality=DataQuality.DEGRADED,
    )
    assert signal.confidence < 0.5


# ---------------------------------------------------------------------------
# 8. Magnitude is z-scored when history provided
# ---------------------------------------------------------------------------
def test_magnitude_is_z_scored():
    """Magnitude should be z-scored against historical divergences."""
    # Provide enough history for z-scoring (>= 10 values)
    history = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.0, -0.01, 0.02]
    signal = classify_divergence(
        implied_mean=105.0,
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=-0.5,
        sentiment_confidence=0.8,
        options_quality=DataQuality.FULL,
        history=history,
    )
    # Z-scored magnitude should typically be > 1.0 for a significant divergence
    # (the raw divergence is much larger than historical values)
    assert abs(signal.magnitude) > 1.0


# ---------------------------------------------------------------------------
# 9. Output dataclass has all 5 required fields
# ---------------------------------------------------------------------------
def test_output_dataclass_fields():
    """DivergenceSignal must have direction, magnitude, divergence_type, confidence, suggested_bias."""
    signal = classify_divergence(
        implied_mean=100.0,
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=0.0,
        sentiment_confidence=0.5,
        options_quality=DataQuality.FULL,
    )
    assert hasattr(signal, "direction")
    assert hasattr(signal, "magnitude")
    assert hasattr(signal, "divergence_type")
    assert hasattr(signal, "confidence")
    assert hasattr(signal, "suggested_bias")

    assert isinstance(signal.direction, int)
    assert isinstance(signal.magnitude, float)
    assert isinstance(signal.divergence_type, str)
    assert isinstance(signal.confidence, float)
    assert isinstance(signal.suggested_bias, str)


# ---------------------------------------------------------------------------
# 10. Neutral when no clear signal
# ---------------------------------------------------------------------------
def test_neutral_when_no_clear_signal():
    """Neither side strong -> neutral direction and type."""
    signal = classify_divergence(
        implied_mean=100.1,   # basically at spot
        spot=100.0,
        implied_skew=0.0,
        implied_kurtosis=3.0,
        sentiment_score=0.05,  # near zero
        sentiment_confidence=0.3,
        options_quality=DataQuality.FULL,
    )
    assert signal.direction == 0
    assert signal.divergence_type == "neutral"
    assert signal.suggested_bias == "neutral"
