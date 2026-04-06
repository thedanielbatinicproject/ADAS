"""Dashboard panel rendering.

Draws a stats sidebar or overlay panel onto a frame showing contextual
information: mode, weather, light condition, lane state, TTC, risk score,
FPS, and braking multiplier.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PANEL_W = 260         # width of the sidebar in pixels
_PANEL_BG = (18, 18, 18)
_TEXT_COLOR = (210, 210, 210)
_HEADER_COLOR = (100, 200, 255)
_WARN_COLOR = (0, 180, 255)
_BRAKE_COLOR = (30, 30, 255)
_OK_COLOR = (60, 200, 60)
_FONT_SCALE = 0.44
_LINE_H = 20           # pixels per text line
_PAD_X = 8
_PAD_Y = 10


def draw_stats_panel(
    frame: Any,
    stats: Dict[str, Any],
    *,
    side: str = "right",
    panel_width: int = _PANEL_W,
) -> Any:
    """Attach a stats sidebar to the frame.

    The sidebar is appended to the left or right side of the frame
    (does not overlay the video content, just widens the canvas).
    For overlaying directly on the frame use draw_stats_overlay().

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image.
    stats : dict
        Keys (all optional):
          "mode"               - str (e.g. "NORMAL_MARKED")
          "weather"            - str
          "light"              - str
          "lane_state"         - str
          "road_surface"       - str
          "braking_mult"       - float
          "visibility_conf"    - float
          "ttc"                - float or None
          "risk_score"         - float
          "action"             - str (e.g. "WARN", "BRAKE", "NONE")
          "fps"                - float
          "frame_idx"          - int
          "total_frames"       - int
          "annotation_label"   - str
    side : str
        "right" or "left" - which side to attach the panel.
    panel_width : int
        Width of the stats panel in pixels.

    Returns
    -------
    numpy.ndarray
        Wider frame with the stats panel on the requested side.
    """
    try:
        import cv2
    except ImportError:
        return frame.copy() if frame is not None else frame

    if frame is None or not hasattr(frame, "shape"):
        return frame

    h = frame.shape[0]
    panel = _build_panel(h, panel_width, stats, cv2)

    if side == "left":
        return np.hstack([panel, frame])
    return np.hstack([frame, panel])


def draw_stats_overlay(
    frame: Any,
    stats: Dict[str, Any],
    *,
    alpha: float = 0.70,
    max_lines: int = 14,
) -> Any:
    """Draw a translucent stats box directly in the top-left corner.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image. Modified in-place.
    stats : dict
        Same keys as draw_stats_panel().
    alpha : float
        Opacity of the background rectangle [0, 1].
    max_lines : int
        Maximum number of stat lines to show.

    Returns
    -------
    numpy.ndarray
        Modified frame (same array).
    """
    try:
        import cv2
    except ImportError:
        return frame

    if frame is None or not hasattr(frame, "shape"):
        return frame

    lines = _build_stat_lines(stats)[:max_lines]
    box_w = _PANEL_W
    box_h = _PAD_Y * 2 + len(lines) * _LINE_H
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), _PANEL_BG, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    y = _PAD_Y + _LINE_H
    for text, color in lines:
        cv2.putText(
            frame, text, (_PAD_X, y),
            cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, 1, cv2.LINE_AA,
        )
        y += _LINE_H

    return frame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_panel(
    height: int,
    width: int,
    stats: Dict[str, Any],
    cv2: Any,
) -> Any:
    """Build the sidebar numpy array."""
    panel = np.full((height, width, 3), _PANEL_BG, dtype=np.uint8)
    lines = _build_stat_lines(stats)

    y = _PAD_Y + _LINE_H
    for text, color in lines:
        if y + _LINE_H > height:
            break
        cv2.putText(
            panel, text, (_PAD_X, y),
            cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, 1, cv2.LINE_AA,
        )
        y += _LINE_H

    return panel


def _build_stat_lines(stats: Dict[str, Any]):
    """Convert stats dict into list of (text, color) tuples."""
    lines = []

    def add(label: str, value: Any, color=_TEXT_COLOR):
        lines.append((f"{label}: {value}", color))

    def add_header(label: str):
        lines.append((label, _HEADER_COLOR))

    add_header("=== ADAS Status ===")

    frame_idx = stats.get("frame_idx")
    total = stats.get("total_frames")
    if frame_idx is not None:
        frame_str = f"{frame_idx}"
        if total:
            frame_str += f"/{total}"
        add("Frame", frame_str)

    fps = stats.get("fps")
    if fps is not None:
        add("FPS", f"{fps:.1f}")

    add_header("--- Scene ---")

    mode = stats.get("mode", "N/A")
    add("Mode", str(mode))

    weather = stats.get("weather", "N/A")
    add("Weather", str(weather))

    light = stats.get("light", "N/A")
    add("Light", str(light))

    add_header("--- Detection ---")

    lane = stats.get("lane_state", "N/A")
    add("Lane", str(lane))

    road = stats.get("road_surface", "N/A")
    add("Road", str(road))

    bm = stats.get("braking_mult")
    if bm is not None:
        bm_color = _WARN_COLOR if bm > 1.1 else _OK_COLOR
        add("BrakeMult", f"{bm:.2f}x", bm_color)

    vis = stats.get("visibility_conf")
    if vis is not None:
        add("VisConf", f"{vis:.2f}")

    add_header("--- Risk ---")

    ttc = stats.get("ttc")
    if ttc is not None and ttc < float("inf"):
        ttc_color = _BRAKE_COLOR if ttc < 1.5 else (
            _WARN_COLOR if ttc < 3.0 else _OK_COLOR
        )
        add("TTC", f"{ttc:.2f}s", ttc_color)
    else:
        add("TTC", "inf", _OK_COLOR)

    risk_score = stats.get("risk_score")
    if risk_score is not None:
        rs_color = _BRAKE_COLOR if risk_score > 0.65 else (
            _WARN_COLOR if risk_score > 0.35 else _OK_COLOR
        )
        add("Risk", f"{risk_score:.2f}", rs_color)

    action = stats.get("action", "NONE")
    if action == "BRAKE":
        action_color = _BRAKE_COLOR
    elif action == "WARN":
        action_color = _WARN_COLOR
    else:
        action_color = _OK_COLOR
    add("Action", str(action), action_color)

    annotation = stats.get("annotation_label")
    if annotation:
        add_header("--- GT Label ---")
        lines.append((str(annotation), (160, 160, 255)))

    return lines
