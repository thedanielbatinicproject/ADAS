"""
Kinematic risk estimation package (TTC, relative velocity, brake model).
Expose load_estimator() to get estimator module/class.
"""
__all__ = ["load_estimator"]

def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)

def load_estimator():
    """Lazy-load estimator implementation (estimator.py)."""
    return _import("estimator")