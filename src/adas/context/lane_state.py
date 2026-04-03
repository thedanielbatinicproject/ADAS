"""
Lane-state evaluation with EMA smoothing.

This module does **not** detect lanes.  It consumes raw results from the
lane-detection module (:class:`LaneDetectionInput`) and produces a smoothed
:class:`LaneState` that the router uses for mode selection.
"""

from __future__ import annotations

from typing import Optional

from .types import LaneDetectionInput, LaneState, LaneAvailability
from .defaults import ContextConfig, DEFAULT_CONFIG


def compute_lane_state(
    lane_detection: Optional[LaneDetectionInput] = None,
    *,
    prev_lane_state: Optional[LaneState] = None,
    config: Optional[ContextConfig] = None,
) -> LaneState:
    """Evaluate lane availability and confidence.

    Parameters
    ----------
    lane_detection : LaneDetectionInput or None
        Raw output from the lane-detection module.
        *None* signals that no detection was performed / available.
    prev_lane_state : LaneState, optional
        Previous smoothed state - used for EMA continuity.
    config : ContextConfig, optional
    """
    cfg = config or DEFAULT_CONFIG

    # ---- extract raw values ----
    if lane_detection is None:
        raw_confidence = 0.0
        width_px = None
    else:
        raw_confidence = (
            lane_detection.left_confidence + lane_detection.right_confidence
        ) / 2.0
        width_px = lane_detection.lane_width_px

    # ---- EMA smoothing ----
    if prev_lane_state is not None:
        alpha = cfg.lane_ema_alpha
        smoothed = alpha * raw_confidence + (1.0 - alpha) * prev_lane_state.confidence
    else:
        smoothed = raw_confidence

    smoothed = max(0.0, min(1.0, smoothed))

    # ---- stability (low frame-to-frame change → high stability) ----
    if prev_lane_state is not None and prev_lane_state.confidence > 0.0:
        delta = abs(smoothed - prev_lane_state.confidence)
        stability = 1.0 - min(delta / prev_lane_state.confidence, 1.0)
    else:
        stability = 1.0 if smoothed > 0.0 else 0.0

    # ---- availability classification ----
    if smoothed >= cfg.t_lane:
        availability = LaneAvailability.HAS_LANES
    elif smoothed >= cfg.t_lane_low:
        availability = LaneAvailability.DEGRADED_LANES
    else:
        availability = LaneAvailability.NO_LANES

    return LaneState(
        availability=availability,
        confidence=smoothed,
        has_lanes=(availability == LaneAvailability.HAS_LANES),
        lanes_degraded=(availability == LaneAvailability.DEGRADED_LANES),
        lane_width_px=width_px,
        stability=stability,
    )
