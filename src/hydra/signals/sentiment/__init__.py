"""COT sentiment scoring: crowd positioning signals from CFTC data."""

from hydra.signals.sentiment.cot_scoring import SentimentScore, compute_cot_sentiment

__all__ = ["SentimentScore", "compute_cot_sentiment"]
