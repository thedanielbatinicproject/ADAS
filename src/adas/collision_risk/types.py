"""Collision risk estimation types."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class SystemAction(enum.Enum):
    """Action recommended by the collision risk decision module.

    NONE  - no intervention needed
    WARN  - audible/visual warning to driver
    BRAKE - emergency brake recommendation
    """

    NONE = "none"
    WARN = "warn"
    BRAKE = "brake"


@dataclass(frozen=True)
class RiskResult:
    """Collision risk estimate for a single detected obstacle.

    Attributes
    ----------
    object_id : int
        Track ID of the obstacle (from SimpleTracker). -1 if not tracked.
    ttc : float
        Time-to-collision in seconds. float('inf') when object is not
        approaching or distance/velocity is unavailable.
    distance_m : float
        Estimated longitudinal distance to the obstacle in meters.
    risk_score : float
        Combined risk score in [0, 1]. Higher means more dangerous.
    lateral_offset_m : float
        Lateral offset of the obstacle centroid from the ego lane center
        in meters. Positive = right of center, negative = left.
    in_ego_lane : bool
        True when the obstacle is estimated to be in the ego vehicle's lane.
    """

    object_id: int = -1
    ttc: float = float("inf")
    distance_m: float = float("inf")
    risk_score: float = 0.0
    lateral_offset_m: float = 0.0
    in_ego_lane: bool = False
