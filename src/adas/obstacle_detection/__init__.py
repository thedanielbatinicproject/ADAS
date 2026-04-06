"""
Obstacle detection package.

Modules:
  types.py    - DetectedObject dataclass
  detector.py - Detector class and detect_obstacles() functional API
  tracking.py - SimpleTracker for ID assignment across frames
  metrics.py  - evaluate_detections(), aggregate_evaluation()
"""

from .types import DetectedObject
from .detector import Detector, DetectorConfig, DEFAULT_DETECTOR_CONFIG, detect_obstacles
from .tracking import SimpleTracker
from .metrics import DetectionEvalResult, evaluate_detections, aggregate_evaluation

__all__ = [
    # main API
    "detect_obstacles",
    "Detector",
    "SimpleTracker",
    "evaluate_detections",
    "aggregate_evaluation",
    # types
    "DetectedObject",
    "DetectorConfig",
    "DEFAULT_DETECTOR_CONFIG",
    "DetectionEvalResult",
    # legacy lazy-loader kept for backward compatibility
    "load_detector",
]


def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)


def load_detector():
    """Lazy-load detector module (detector.py)."""
    return _import("detector")