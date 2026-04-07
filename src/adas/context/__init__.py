"""
Context package - scene analysis, lane-state evaluation, road-surface
estimation, and mode routing for the ADAS pipeline.

Public API
----------
- :func:`route`- main entry point (one call per frame)
- :func:`compute_scene_metrics`
- :func:`estimate_visibility`
- :func:`compute_lane_state`
- :func:`estimate_road_surface`
- :func:`braking_multiplier`
- All types from :mod:`.types`
- :class:`ContextConfig` / :data:`DEFAULT_CONFIG`
"""

from .types import (
    Frame,
    LaneAvailability,
    RoadSurfaceType,
    Mode,
    WeatherCondition,
    LightCondition,
    SceneMetrics,
    VisibilityEstimate,
    LaneDetectionInput,
    LaneState,
    RoadSurfaceHint,
    EmergencySignal,
    ContextState,
)
from .defaults import ContextConfig, DEFAULT_CONFIG
from .router import route
from .service import ContextService
from .scene_metrics import compute_scene_metrics, estimate_visibility
from .lane_state import compute_lane_state
from .road_surface import estimate_road_surface, braking_multiplier

__all__ = [
    # main entry point
    "route",
    "ContextService",
    # scene metrics
    "compute_scene_metrics",
    "estimate_visibility",
    # lane state
    "compute_lane_state",
    # road surface
    "estimate_road_surface",
    "braking_multiplier",
    # types
    "Frame",
    "LaneAvailability",
    "RoadSurfaceType",
    "Mode",
    "WeatherCondition",
    "LightCondition",
    "SceneMetrics",
    "VisibilityEstimate",
    "LaneDetectionInput",
    "LaneState",
    "RoadSurfaceHint",
    "EmergencySignal",
    "ContextState",
    # config
    "ContextConfig",
    "DEFAULT_CONFIG",
]
