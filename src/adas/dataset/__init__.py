"""
Dataset / IO utilities.
"""
__all__ = ["load_parser", "DATASET_ROOT_HINT"]

DATASET_ROOT_HINT = "data/raw"  # promijeniti po potrebi

def _import(name):
    import importlib
    return importlib.import_module(f".{name}", __name__)

def load_parser():
    """
    Lazy-load parser module (expected at src/adas/dataset/parser.py).
    Returns the parser module, for example:
      parser = adas.dataset.load_parser()
      parser.export_random_samples(...)
    """
    return _import("parser")