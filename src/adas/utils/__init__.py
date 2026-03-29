"""
Utility helpers (IO, image utils, coordinate transforms).
Only small utilities here (stateless functions). Heavy deps should be in separate modules and lazy-imported.
"""
__all__ = ["ensure_cv2", "split_path"]

def ensure_cv2():
    """Return cv2 module, raise clear error if not installed."""
    try:
        import cv2
        return cv2
    except Exception as e:
        raise ImportError("opencv-python is required for image operations: pip install opencv-python") from e

def split_path(path):
    """Small helper: (dir, base, ext)."""
    import os
    d = os.path.dirname(path)
    b = os.path.basename(path)
    name, ext = os.path.splitext(b)
    return d, name, ext