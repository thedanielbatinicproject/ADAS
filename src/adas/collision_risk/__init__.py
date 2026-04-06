"""
Kinematic risk estimation package.

Modules:
  types.py     - RiskResult, SystemAction
  estimator.py - RiskEstimator class and helpers
  decision.py  - decide() -> (SystemAction, intensity)
  metrics.py   - evaluate_action(), aggregate_evaluation()
"""

from .types import RiskResult, SystemAction
from .estimator import RiskEstimator, EstimatorConfig, DEFAULT_ESTIMATOR_CONFIG
from .decision import decide, DecisionConfig, DEFAULT_DECISION_CONFIG
from .metrics import RiskEvalResult, evaluate_action, aggregate_evaluation

__all__ = [
    # main API
    "decide",
    "RiskEstimator",
    "evaluate_action",
    "aggregate_evaluation",
    # types
    "RiskResult",
    "SystemAction",
    "EstimatorConfig",
    "DEFAULT_ESTIMATOR_CONFIG",
    "DecisionConfig",
    "DEFAULT_DECISION_CONFIG",
    "RiskEvalResult",
    # legacy lazy-loader kept for backward compatibility
    "load_estimator",
]


def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)


def load_estimator():
    """Lazy-load estimator module (estimator.py)."""
    return _import("estimator")