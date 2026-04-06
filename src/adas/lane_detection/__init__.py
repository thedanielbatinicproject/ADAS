"""
Lane detection package.

Modules:
  processing.py    - process_frame() -> LaneOutput (geometric lane detection)
  visualization.py - draw_lanes(), draw_edges() helpers
  metrics.py       - evaluate_batch() for benchmarking
  hough.py         - placeholder for future geometric Hough implementation
"""

from .processing import LaneOutput, LaneProcessingConfig, DEFAULT_PROCESSING_CONFIG, process_frame
from .visualization import draw_lanes, draw_edges
from .metrics import LaneEvalResult, evaluate_detection, evaluate_batch

__all__ = [
    # main API
    "process_frame",
    "draw_lanes",
    "draw_edges",
    "evaluate_detection",
    "evaluate_batch",
    # types
    "LaneOutput",
    "LaneProcessingConfig",
    "DEFAULT_PROCESSING_CONFIG",
    "LaneEvalResult",
    # legacy lazy-loaders kept for backward compatibility
    "load_processing",
    "load_visualization",
]


def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)


def load_processing():
    """Lazy-load lane detection processing module (processing.py)."""
    return _import("processing")


def load_visualization():
    """Lazy-load visualization helpers (visualization.py)."""
    return _import("visualization")