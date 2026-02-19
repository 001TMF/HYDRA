"""Divergence detector: classifies options vs. sentiment divergence.

Public API:
  - DivergenceSignal: dataclass with direction, magnitude, divergence_type, confidence, suggested_bias
  - classify_divergence: rule-based classification into 6-type taxonomy
"""

from hydra.signals.divergence.detector import DivergenceSignal, classify_divergence

__all__ = ["DivergenceSignal", "classify_divergence"]
