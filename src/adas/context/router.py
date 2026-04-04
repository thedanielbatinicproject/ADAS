"""
Context router – main entry point for per-frame scene evaluation.

Combines scene metrics, lane-state, and road-surface heuristics into a
single :class:`ContextState` that tells the rest of the pipeline which
operating :class:`Mode` to use.

Hysteresis logic prevents rapid mode-switching by requiring
:pyattr:`ContextConfig.hysteresis_k` consecutive frames with the same
candidate mode before actually switching.
"""

from __future__ import annotations

from typing import Optional, Any

from .types import (
    Mode,
    ContextState,
    LaneDetectionInput,
    EmergencySignal,
    VisibilityEstimate,
    LaneState,
    RoadSurfaceType,
    WeatherCondition,
    LightCondition,
)
from .scene_metrics import compute_scene_metrics, estimate_visibility
from .lane_state import compute_lane_state
from .road_surface import estimate_road_surface, braking_multiplier
from .defaults import ContextConfig, DEFAULT_CONFIG

# Import Hough lane detector (optional – graceful fallback if cv2 not present)
try:
    from .lane_heuristic import detect_lanes_heuristic as _detect_lanes_hough
except Exception:  # pragma: no cover
    _detect_lanes_hough = None  # type: ignore[assignment]


# ------------------------------------------------------------------ internal


def _derive_conditions(
    visibility: VisibilityEstimate,
    surface: "RoadSurfaceType | None",
) -> tuple[WeatherCondition, LightCondition]:
    """Map low-level visibility/surface signals to human-readable conditions.

    Light condition (day/night) is determined independently of weather.
    Weather describes precipitation / visibility quality, not time of day:

    Glare:  is_glare → GLARE
    Rain:   is_degraded + wet surface → RAIN
    Fog:    is_degraded, not wet, not glare → FOG
    Clear:  not degraded, not glare → CLEAR  (holds at night too)
    """
    light = LightCondition.NIGHT if visibility.is_night else LightCondition.DAY

    if visibility.is_glare:
        weather = WeatherCondition.GLARE
    elif visibility.is_degraded:
        is_wet = (
            surface is not None
            and surface.surface_type == RoadSurfaceType.ASPHALT_WET
        )
        weather = WeatherCondition.RAIN if is_wet else WeatherCondition.FOG
    else:
        weather = WeatherCondition.CLEAR

    return weather, light


def _determine_candidate_mode(
    visibility: VisibilityEstimate,
    lane_state: LaneState,
    config: ContextConfig,
) -> Mode:
    """Apply mode-selection rules (without hysteresis).

    Decision matrix
    ---------------
    has_some_lanes AND good_vis  → NORMAL_MARKED
    has_some_lanes AND bad_vis   → DEGRADED_MARKED
    no_lanes       AND good_vis  → UNMARKED_GOOD_VIS
    no_lanes       AND bad_vis   → UNMARKED_DEGRADED
    """
    has_good_vis = visibility.confidence >= config.t_vis
    has_some_lanes = lane_state.confidence >= config.t_lane_low

    if has_some_lanes and has_good_vis:
        return Mode.NORMAL_MARKED
    if has_some_lanes and not has_good_vis:
        return Mode.DEGRADED_MARKED
    if not has_some_lanes and has_good_vis:
        return Mode.UNMARKED_GOOD_VIS
    return Mode.UNMARKED_DEGRADED


# -------------------------------------------------------------------- public


def route(
    frame: Any,
    *,
    lane_detection: Optional[LaneDetectionInput] = None,
    timestamp_s: float = 0.0,
    fps: Optional[float] = None,
    emergency: Optional[EmergencySignal] = None,
    prev_state: Optional[ContextState] = None,
    config: Optional[ContextConfig] = None,
) -> ContextState:
    """Analyse a single frame and determine the system operating mode.

    This is the main public entry point of the context package.  Downstream
    modules (lane detection, obstacle detection, collision-risk) use the
    returned :class:`ContextState` to adapt their thresholds and logic.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image ``(H, W, 3)``.
    lane_detection : LaneDetectionInput, optional
        Raw lane-detection result for this frame.
    timestamp_s : float
        Frame timestamp in seconds (monotonic or Unix).
    fps : float, optional
        Current processing frame rate.
    emergency : EmergencySignal, optional
        External (or internal) emergency-override signal.
    prev_state : ContextState, optional
        Previous frame's context state (carries hysteresis bookkeeping).
    config : ContextConfig, optional
    """
    cfg = config or DEFAULT_CONFIG

    # 1. scene metrics & visibility
    # Restrict analysis to the top portion of the frame to exclude the
    # car interior / dashboard, which inflates contrast in foggy / night scenes.
    h, w = frame.shape[:2] if hasattr(frame, "shape") else (0, 0)
    if h > 0 and w > 0 and 0.0 < cfg.scene_roi_top_fraction < 1.0:
        roi_y2 = int(h * cfg.scene_roi_top_fraction)
        scene_roi = (0, 0, w, roi_y2)
    else:
        scene_roi = None
    metrics = compute_scene_metrics(frame, roi=scene_roi, config=cfg)
    visibility = estimate_visibility(metrics, config=cfg)

    # 2. lane detection + state (EMA smoothing from previous frame)
    # If the caller does not supply a lane_detection result, run the built-in
    # Hough-transform detector on the full (unclipped) frame.
    if lane_detection is None and _detect_lanes_hough is not None:
        lane_detection = _detect_lanes_hough(frame, config=cfg)
    prev_lane = prev_state.lane_state if prev_state else None
    lane_state = compute_lane_state(
        lane_detection,
        prev_lane_state=prev_lane,
        config=cfg,
    )

    # 3. road surface & braking multiplier
    surface = estimate_road_surface(frame, config=cfg)
    brake_mult = braking_multiplier(surface, config=cfg)

    # 3b. Dark-channel disambiguation: solar glare ≠ atmospheric haze.
    #
    # When the raw visibility confidence is below t_vis and glare is
    # detected, two fundamentally different conditions produce the same
    # optical symptoms:
    #
    #   (a) Solar glare (sun in lens) – camera overloaded, road itself clear.
    #       Dark Channel Prior on road region stays LOW (asphalt is dark).
    #       Road surface is DRY.  → is_degraded should be False.
    #
    #   (b) Atmospheric haze / fog – road visibility genuinely reduced.
    #       Road DCP is ELEVATED (all channels raised by scattered airlight).
    #       Also applies to wet glare (rain puddle reflections): road is WET.
    #       → is_degraded must remain True.
    #
    # We correct is_degraded only when ALL signals agree on solar glare:
    #   • glare_score > t_glare         (sun visible in sensor)
    #   • glare_score > t_glare_strong  (≥ 0.25: must be clearly solar, not
    #       marginal wet-road or streetlight reflection in [t_glare, 0.25))
    #   • glare_score < t_max_glare_dcp (cap: extreme exposure ≥ 0.40 is
    #       more likely rain-with-sunshine than simple solar glare)
    #   • dark_channel_road < t_dcp_haze (road itself is dark/clear)
    #   • road surface NOT wet          (wet reflection gives same low DCP)
    is_wet = (
        surface is not None
        and surface.surface_type == RoadSurfaceType.ASPHALT_WET
    )
    if (
        visibility.is_glare
        and visibility.is_degraded
        and metrics.glare_score > cfg.t_glare_strong
        and metrics.dark_channel_road < cfg.t_dcp_haze
        and metrics.glare_score < cfg.t_max_glare_dcp
        and not is_wet
    ):
        from dataclasses import replace as _dc_replace
        visibility = _dc_replace(visibility, is_degraded=False)

    # 3c. Clear-road override: motion blur on high-speed dashcam footage
    #     reduces Laplacian variance (blur_laplacian_var) and therefore
    #     suppresses the blur contribution to visibility confidence even on
    #     perfectly clear sunny days.  This causes borderline-confidence
    #     clear-day frames to be mis-labelled as "degraded".
    #
    # When ALL of the following conditions hold simultaneously:
    #   • no direct glare (3b handles that case)
    #   • road DCP < t_dcp_clear (0.20): road is clearly NOT hazy / foggy
    #       (all observed fog videos have road-DCP ≥ 0.219 > 0.20)
    #   • scene saturation > t_sat_clear (30): colourful scene (sunny, not
    #       grey overcast or foggy/rainy grey)
    #   • visibility.confidence ≥ t_vis_dcp_min (0.38): some minimum
    #       quality – truly terrible visibility is not cleared
    #   • road surface NOT wet: protects rainy-with-wet-road cases
    # → the "degraded" flag is an artifact of blur miscalibration, not
    #   genuine visibility impairment.  Clear it.
    if (
        not visibility.is_glare
        and visibility.is_degraded
        and metrics.dark_channel_road < cfg.t_dcp_clear
        and metrics.saturation_mean > cfg.t_sat_clear
        and visibility.confidence >= cfg.t_vis_dcp_min
        and not is_wet
    ):
        from dataclasses import replace as _dc_replace
        visibility = _dc_replace(visibility, is_degraded=False)

    # 4. candidate mode (rule-based)
    candidate = _determine_candidate_mode(visibility, lane_state, cfg)

    # 5. emergency override (bypasses hysteresis)
    if emergency is not None and emergency.active:
        candidate = Mode.EMERGENCY_OVERRIDE

    # 6. hysteresis
    mode, hold_count, pending_mode, pending_count = _apply_hysteresis(
        candidate,
        prev_state,
        cfg,
    )

    # 7. derive human-readable conditions
    weather_condition, light_condition = _derive_conditions(visibility, surface)

    return ContextState(
        mode=mode,
        scene_metrics=metrics,
        visibility=visibility,
        lane_state=lane_state,
        road_surface=surface,
        braking_multiplier=brake_mult,
        timestamp_s=timestamp_s,
        fps=fps,
        weather_condition=weather_condition,
        light_condition=light_condition,
        mode_hold_count=hold_count,
        pending_mode=pending_mode,
        pending_count=pending_count,
    )


# --------------------------------------------------------- hysteresis helper


def _apply_hysteresis(
    candidate: Mode,
    prev_state: Optional[ContextState],
    config: ContextConfig,
) -> tuple[Mode, int, Optional[Mode], int]:
    """Return ``(mode, hold_count, pending_mode, pending_count)``.

    Emergency override always takes effect immediately.
    """
    if prev_state is None:
        return candidate, 1, None, 0

    if candidate == Mode.EMERGENCY_OVERRIDE:
        return Mode.EMERGENCY_OVERRIDE, 1, None, 0

    if candidate == prev_state.mode:
        return candidate, prev_state.mode_hold_count + 1, None, 0

    # candidate differs from current mode
    if prev_state.pending_mode == candidate:
        new_pending = prev_state.pending_count + 1
        if new_pending >= config.hysteresis_k:
            return candidate, 1, None, 0
        return (
            prev_state.mode,
            prev_state.mode_hold_count + 1,
            candidate,
            new_pending,
        )

    # brand-new candidate – start tracking
    return (
        prev_state.mode,
        prev_state.mode_hold_count + 1,
        candidate,
        1,
    )
