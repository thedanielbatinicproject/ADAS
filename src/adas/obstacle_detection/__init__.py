"""
Obstacle detection package.
Planned modules: detector.py (classical methods), detector_ml.py (NN wrappers), evaluation.py
"""
__all__ = ["load_detector"]

def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)

def load_detector():
    """Lazy-load detector implementation (detector.py or detector_ml.py)."""
    return _import("detector")