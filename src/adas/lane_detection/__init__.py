"""
Lane detection package.
Planned modules: processing.py, visualization.py, metrics.py.
Use load_processing() to get the implementation when ready.
"""
__all__ = ["load_processing", "load_visualization"]

def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)

def load_processing():
    """Lazy-load lane detection processing implementation (processing.py)."""
    return _import("processing")

def load_visualization():
    """Lazy-load visualization helpers (visualization.py)."""
    return _import("visualization")