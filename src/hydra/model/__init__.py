"""Model package: feature assembly, baseline model, and walk-forward backtesting.

Public API:
  - FeatureAssembler: builds feature matrix from feature store + computed signals
  - BaselineModel: LightGBM binary classifier with conservative defaults
  - PurgedWalkForwardSplit: expanding-window cross-validator with embargo gap
  - WalkForwardEngine: full walk-forward backtest with risk stack integration
  - BacktestResult: dataclass holding all backtest metrics
  - compute_backtest_metrics: computes slippage-adjusted backtest metrics
"""

from hydra.model.baseline import BaselineModel
from hydra.model.evaluation import BacktestResult, compute_backtest_metrics
from hydra.model.features import FeatureAssembler
from hydra.model.walk_forward import PurgedWalkForwardSplit, WalkForwardEngine

__all__ = [
    "FeatureAssembler",
    "BaselineModel",
    "PurgedWalkForwardSplit",
    "WalkForwardEngine",
    "BacktestResult",
    "compute_backtest_metrics",
]
