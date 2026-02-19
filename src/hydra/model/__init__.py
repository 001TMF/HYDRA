"""Model package: feature assembly and baseline LightGBM classifier.

Public API:
  - FeatureAssembler: builds feature matrix from feature store + computed signals
  - BaselineModel: LightGBM binary classifier with conservative defaults
"""

from hydra.model.baseline import BaselineModel
from hydra.model.features import FeatureAssembler

__all__ = ["FeatureAssembler", "BaselineModel"]
