"""Lane detection visualization helpers.

Provides functions for drawing detected lane boundaries and masks onto a
BGR frame. Used by debug scripts and the UI overlay module.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from .processing import LaneOutput


def draw_lanes(
    frame: Any,
    lane_output: LaneOutput,
    *,
    left_color: Tuple[int, int, int] = (0, 255, 0),
    right_color: Tuple[int, int, int] = (0, 200, 255),
    fill_color: Tuple[int, int, int] = (0, 180, 0),
    fill_alpha: float = 0.25,
    line_thickness: int = 3,
) -> Any:
    """Draw lane boundaries on a copy of frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image.
    lane_output : LaneOutput
        Result from process_frame().
    left_color : tuple
        BGR color for the left boundary line.
    right_color : tuple
        BGR color for the right boundary line.
    fill_color : tuple
        BGR color for the filled lane region (when both sides detected).
    fill_alpha : float
        Opacity of the fill overlay in [0, 1].
    line_thickness : int
        Pixel thickness of the boundary lines.

    Returns
    -------
    numpy.ndarray
        A copy of frame with lane visualization drawn on it.
    """
    try:
        import cv2
    except ImportError:
        return frame.copy()

    if frame is None or not hasattr(frame, "shape"):
        return frame

    out = frame.copy()

    if not lane_output.has_lanes:
        return out

    h, w = frame.shape[:2]
    y1 = lane_output.roi_y1
    y2 = lane_output.roi_y2
    roi_h = y2 - y1

    ys = np.linspace(0, roi_h - 1, num=max(roi_h, 2))

    # Draw filled region when both boundaries are available
    if (
        lane_output.left_detected
        and lane_output.right_detected
        and lane_output.left_poly is not None
        and lane_output.right_poly is not None
        and fill_alpha > 0
    ):
        lxs = np.clip(
            (lane_output.left_poly[0] * ys + lane_output.left_poly[1]).astype(np.int32),
            0, w - 1,
        )
        rxs = np.clip(
            (lane_output.right_poly[0] * ys + lane_output.right_poly[1]).astype(np.int32),
            0, w - 1,
        )
        ys_int = (ys + y1).astype(np.int32)
        left_pts = list(zip(lxs.tolist(), ys_int.tolist()))
        right_pts = list(reversed(list(zip(rxs.tolist(), ys_int.tolist()))))
        polygon = np.array(left_pts + right_pts, dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [polygon], color=fill_color)
        cv2.addWeighted(overlay, fill_alpha, out, 1.0 - fill_alpha, 0, out)

    # Draw left boundary
    if lane_output.left_detected and lane_output.left_poly is not None:
        _draw_poly_line(out, lane_output.left_poly, ys, y1, w, left_color, line_thickness, cv2)

    # Draw right boundary
    if lane_output.right_detected and lane_output.right_poly is not None:
        _draw_poly_line(out, lane_output.right_poly, ys, y1, w, right_color, line_thickness, cv2)

    return out


def draw_edges(
    frame: Any,
    lane_output: LaneOutput,
    *,
    alpha: float = 0.4,
) -> Any:
    """Overlay the Canny edge map from lane_output onto frame.

    Useful for debugging the detector without a full UI.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image (original full frame).
    lane_output : LaneOutput
        Result from process_frame(). edges must be non-None.
    alpha : float
        Opacity of the edge overlay.

    Returns
    -------
    numpy.ndarray
        Copy of frame with edge ROI blended in.
    """
    try:
        import cv2
    except ImportError:
        return frame.copy()

    if frame is None or lane_output.edges is None:
        return frame.copy() if frame is not None else frame

    out = frame.copy()
    y1 = lane_output.roi_y1
    y2 = lane_output.roi_y2

    edges_bgr = cv2.cvtColor(lane_output.edges, cv2.COLOR_GRAY2BGR)
    overlay = out.copy()
    overlay[y1:y2, :] = edges_bgr
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_poly_line(
    frame: Any,
    poly: tuple,
    ys: Any,
    y_offset: int,
    frame_w: int,
    color: Tuple[int, int, int],
    thickness: int,
    cv2: Any,
) -> None:
    """Draw a polynomial curve x = a*y + b as a polyline on frame in-place."""
    xs = np.clip(
        (poly[0] * ys + poly[1]).astype(np.int32), 0, frame_w - 1
    )
    ys_abs = (ys + y_offset).astype(np.int32)
    pts = np.stack([xs, ys_abs], axis=1).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
