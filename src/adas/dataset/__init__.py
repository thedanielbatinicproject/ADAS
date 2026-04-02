"""Dataset package with parser, indexing, sampling, and loader utilities."""

__all__ = [
    "load_parser",
    "load_indexer",
    "load_lotvs_reader",
    "load_sampler",
    "load_loader_wrappers",
    "load_annotation",
    "load_utils_io",
    "DATASET_ROOT_HINT",
]

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


def load_indexer():
    """Lazy-load indexer module."""
    return _import("indexer")


def load_lotvs_reader():
    """Lazy-load DADA reader module."""
    return _import("lotvs_reader")


def load_sampler():
    """Lazy-load sampling utilities module."""
    return _import("sampler")


def load_loader_wrappers():
    """Lazy-load loader wrappers module."""
    return _import("loader_wrappers")


def load_annotation():
    """Lazy-load annotation helpers module."""
    return _import("annotation")


def load_utils_io():
    """Lazy-load I/O helpers module."""
    return _import("utils_io")