"""Frame overlay drawing functions.

Each function takes a frame (numpy.ndarray, BGR) and a result object, and
returns a new frame with the overlay drawn on it. The original frame is
never modified in-place.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def draw_lanes(
    frame: Any,
    lane_output: Any,
    *,
    left_color: Tuple[int, int, int] = (0, 255, 0),
    right_color: Tuple[int, int, int] = (0, 200, 255),
    fill_alpha: float = 0.20,
) -> Any:
    """Draw lane boundaries over the frame.

    Delegates to adas.lane_detection.visualization.draw_lanes().
    Re-exported here so the UI module does not depend on lane_detection
    at import time (lazy import).

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image.
    lane_output : LaneOutput
        Result from adas.lane_detection.process_frame().
    left_color, right_color : tuple
        BGR colors for the lane boundary lines.
    fill_alpha : float
        Opacity of the lane-region fill.

    Returns
    -------
    numpy.ndarray
        Copy of frame with overlay drawn.
    """
    from adas.lane_detection.visualization import draw_lanes as _draw
    return _draw(
        frame, lane_output,
        left_color=left_color,
        right_color=right_color,
        fill_alpha=fill_alpha,
    )


def draw_obstacles(
    frame: Any,
    obstacles: List[Any],
    *,
    box_color: Tuple[int, int, int] = (0, 80, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> Any:
    """Draw bounding boxes for detected obstacles.

    Parameters
    ----------
    frame : numpy.ndarray
    obstacles : list[DetectedObject]
    box_color : tuple
        Default bounding box BGR color.
    text_color : tuple
        Text label BGR color.
    thickness : int
        Border thickness in pixels.

    Returns
    -------
    numpy.ndarray
    """
    try:
        import cv2
    except ImportError:
        return frame.copy()

    if frame is None:
        return frame
    out = frame.copy()

    for obj in obstacles:
        x, y, w, h = obj.bbox
        tid = getattr(obj, "track_id", -1)
        dist = getattr(obj, "distance_estimate", None)

        cv2.rectangle(out, (x, y), (x + w, y + h), box_color, thickness)

        label_parts = []
        if tid >= 0:
            label_parts.append(f"#{tid}")
        if dist is not None:
            label_parts.append(f"{dist:.1f}m")
        label = " ".join(label_parts)

        if label:
            _put_label(out, label, (x, y - 4), text_color, cv2)

    return out


def draw_risk(
    frame: Any,
    risks: List[Any],
    context_state: Any = None,
    *,
    warn_color: Tuple[int, int, int] = (0, 180, 255),
    brake_color: Tuple[int, int, int] = (0, 0, 255),
    threshold_warn: float = 0.35,
    threshold_brake: float = 0.65,
) -> Any:
    """Overlay risk indicators onto detected object bounding boxes.

    This function requires obstacles drawn first (bbox positions). It re-draws
    colored boxes on top with the risk level encoded in color and label.

    Parameters
    ----------
    frame : numpy.ndarray
    risks : list[RiskResult]
        Output of RiskEstimator.estimate_risk().
    context_state : ContextState, optional
        Not used currently; reserved for future adaptive coloring.
    warn_color : tuple
        BGR color for warning-level risk boxes.
    brake_color : tuple
        BGR color for brake-level risk boxes.
    threshold_warn, threshold_brake : float
        Score thresholds for warn/brake coloring.

    Returns
    -------
    numpy.ndarray
    """
    # Risk visualization only annotates with text/color; actual bboxes are
    # already drawn by draw_obstacles(). Here we overlay a color-coded
    # risk score near each object's centroid.
    try:
        import cv2
    except ImportError:
        return frame.copy()

    if frame is None:
        return frame
    out = frame.copy()

    for risk in risks:
        score = risk.risk_score
        ttc = risk.ttc
        cx = int(risk.lateral_offset_m)  # placeholder - centroid not in RiskResult
        # Use a corner indicator instead since we don't have exact pixel pos
        if score >= threshold_brake:
            color = brake_color
            level = "BRAKE"
        elif score >= threshold_warn:
            color = warn_color
            level = "WARN"
        else:
            continue

        # Draw a risk badge at top-left area scaled by object id
        oid = risk.object_id if risk.object_id >= 0 else 0
        badge_y = 60 + oid * 24
        badge_x = 10
        label = f"#{oid} {level} score={score:.2f}"
        if ttc < float("inf"):
            label += f" TTC={ttc:.1f}s"
        _put_label(out, label, (badge_x, badge_y), color, cv2, scale=0.55)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _put_label(
    frame: Any,
    text: str,
    pos: Tuple[int, int],
    color: Tuple[int, int, int],
    cv2: Any,
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    """Draw a text label at pos with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = pos
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 3
    # Background rectangle
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        (20, 20, 20),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
