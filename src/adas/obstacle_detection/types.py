"""Obstacle detection types shared across the obstacle_detection package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DetectedObject:
    """A single obstacle detected in a frame.

    Attributes
    ----------
    bbox : tuple[int, int, int, int]
        Bounding box in full-frame pixel coordinates: (x, y, width, height).
    area : float
        Area of the bounding box in pixels.
    centroid : tuple[float, float]
        Center of the bounding box: (cx, cy) in full-frame pixel coordinates.
    track_id : int
        Unique ID assigned by the tracker. -1 means not yet tracked.
    distance_estimate : float or None
        Rough depth estimate in meters (from bounding-box size heuristic).
        None when not computable.
    confidence : float
        Detection confidence in [0, 1].
    frame_idx : int
        Frame index this detection belongs to.
    """

    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    area: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    track_id: int = -1
    distance_estimate: Optional[float] = None
    confidence: float = 1.0
    frame_idx: int = -1
