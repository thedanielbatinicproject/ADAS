"""
adas package
"""
__version__ = "0.1"

_submodules = ("dataset", "lane_detection", "obstacle_detection", "collision_risk", "utils")
__all__ = list(_submodules)

def __getattr__(name):
    # lazy import subpackages: `import adas; adas.dataset` radi samo kada je potrebno
    if name in _submodules:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals().keys()) + list(_submodules))