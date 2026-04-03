"""
Shared types, enums, and data classes for the context package.

All modules in ``adas.context`` import their data structures from here
so that downstream code can depend on a single, stable set of definitions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional, Tuple, Any

# ---------------------------------------------------------------------------
# Type alias – avoids importing numpy at module level.
# At runtime this is just ``Any``; static checkers can narrow it further.
# ---------------------------------------------------------------------------
Frame = Any  # numpy.ndarray, BGR uint8, shape (H, W, 3)


# ===================================================================== Enums


class LaneAvailability(enum.Enum):
    """Lane detection availability status."""

    HAS_LANES = "has_lanes"
    DEGRADED_LANES = "degraded_lanes"
    NO_LANES = "no_lanes"


class RoadSurfaceType(enum.Enum):
    """Estimated road surface category."""

    ASPHALT_DRY = "asphalt_dry"
    ASPHALT_WET = "asphalt_wet"
    GRAVEL = "gravel"
    UNKNOWN = "unknown"


class Mode(enum.Enum):
    """System operating mode determined by the context router.

    NORMAL_MARKED       - lanes reliable, visibility OK
    DEGRADED_MARKED     - lanes exist but visibility poor (rain/fog/night)
    UNMARKED_GOOD_VIS   - no reliable lanes, but image quality OK
    UNMARKED_DEGRADED   - no lanes **and** poor visibility
    EMERGENCY_OVERRIDE  - critical situation, bypass normal logic
    """

    NORMAL_MARKED = "normal_marked"
    DEGRADED_MARKED = "degraded_marked"
    UNMARKED_GOOD_VIS = "unmarked_good_vis"
    UNMARKED_DEGRADED = "unmarked_degraded"
    EMERGENCY_OVERRIDE = "emergency_override"


# ============================================================== Data classes


@dataclass(frozen=True)
class SceneMetrics:
    """Raw image-quality / visibility metrics computed from a single frame."""

    brightness_mean: float = 0.0
    brightness_p05: float = 0.0
    brightness_p95: float = 0.0
    contrast_std: float = 0.0
    blur_laplacian_var: float = 0.0
    edge_density: float = 0.0
    saturation_mean: float = 0.0
    glare_score: float = 0.0


@dataclass(frozen=True)
class VisibilityEstimate:
    """Derived visibility assessment from :class:`SceneMetrics`."""

    confidence: float = 0.0
    is_night: bool = False
    is_degraded: bool = False
    is_glare: bool = False


@dataclass(frozen=True)
class LaneDetectionInput:
    """Raw output consumed **from** the lane-detection module.

    The context package does not detect lanes – it evaluates their quality.
    """

    left_detected: bool = False
    right_detected: bool = False
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    lane_width_px: Optional[float] = None
    left_poly_coeffs: Optional[Tuple[float, ...]] = None
    right_poly_coeffs: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class LaneState:
    """Smoothed lane-availability state (output of :func:`compute_lane_state`)."""

    availability: LaneAvailability = LaneAvailability.NO_LANES
    confidence: float = 0.0
    has_lanes: bool = False
    lanes_degraded: bool = False
    lane_width_px: Optional[float] = None
    stability: float = 0.0


@dataclass(frozen=True)
class RoadSurfaceHint:
    """Heuristic road-surface estimate with confidence."""

    surface_type: RoadSurfaceType = RoadSurfaceType.UNKNOWN
    confidence: float = 0.0


@dataclass(frozen=True)
class EmergencySignal:
    """External or internal emergency-override signal."""

    active: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ContextState:
    """Complete context-evaluation result for one frame.

    Carries the determined :attr:`mode`, all intermediate analyses, and the
    bookkeeping fields needed by the hysteresis logic on the next call.
    """

    mode: Mode = Mode.UNMARKED_DEGRADED
    scene_metrics: Optional[SceneMetrics] = None
    visibility: Optional[VisibilityEstimate] = None
    lane_state: Optional[LaneState] = None
    road_surface: Optional[RoadSurfaceHint] = None
    braking_multiplier: float = 1.0
    timestamp_s: float = 0.0
    fps: Optional[float] = None

    # --- hysteresis bookkeeping (carried between frames) ---
    mode_hold_count: int = 0
    pending_mode: Optional[Mode] = None
    pending_count: int = 0
