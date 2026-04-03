"""Heuristic lane-existence detector — part of the context package.

Answers the question: **do painted lane markings exist on this road?**

This is NOT geometric lane detection (finding exact line positions or
polynomials).  It produces a :class:`~adas.context.types.LaneDetectionInput`
confidence signal that :func:`~adas.context.lane_state.compute_lane_state`
uses to decide whether the road has visible lane markings at all.

The algorithm uses a Hough probabilistic line transform on the road-surface
region of the dashcam frame.  Lines are filtered by slope (to keep only
plausible lane-marking angles) and by position (left/right halves).
A one-sided detection penalty prevents single road-edges or guard-rails
from being misclassified as lane markings.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .types import LaneDetectionInput
from .defaults import ContextConfig, DEFAULT_CONFIG


def detect_lanes_heuristic(
    frame: Any,
    *,
    config: Optional[ContextConfig] = None,
) -> LaneDetectionInput:
    """Estimate whether lane markings are present in the dashcam frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image, shape ``(H, W, 3)``.
    config : ContextConfig, optional
        Tunable thresholds.  Falls back to :data:`DEFAULT_CONFIG`.

    Returns
    -------
    LaneDetectionInput
        Left/right detection flags and confidence scores in ``[0, 1]``.
        Low confidence on both sides → router infers NO_LANES.
    """
    import cv2

    cfg = config or DEFAULT_CONFIG

    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return LaneDetectionInput()

    h, w = frame.shape[:2]

    # ── ROI: road region only, excludes sky (top) and dashboard (bottom) ──
    y1 = int(h * cfg.lane_roi_top)
    y2 = int(h * cfg.lane_roi_bottom)
    if y2 <= y1:
        return LaneDetectionInput()
    roi = frame[y1:y2, :]

    # ── Pre-processing ─────────────────────────────────────────────────────
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # CLAHE: locally adaptive contrast — enhances faded markings without
    # amplifying random texture the way equalizeHist does.
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.lane_hough_threshold,
        minLineLength=cfg.lane_min_length,
        maxLineGap=cfg.lane_max_gap,
    )

    if lines is None:
        return LaneDetectionInput()

    mid_x = w / 2
    left_len = 0.0
    right_len = 0.0
    left_count = 0
    right_count = 0

    for line in lines:
        x1, y1_l, x2, y2_l = line[0]
        dx = x2 - x1
        dy = y2_l - y1_l
        if dx == 0:
            continue
        slope = dy / dx
        abs_slope = abs(slope)

        # Filter lines that are too flat or too steep to be lane markings.
        if abs_slope < cfg.lane_slope_min or abs_slope > cfg.lane_slope_max:
            continue

        cx = (x1 + x2) / 2.0
        length = (dx ** 2 + dy ** 2) ** 0.5

        # In perspective view:
        #   Left marking  → centre in left half, negative slope (rises toward right)
        #   Right marking → centre in right half, positive slope (falls toward right)
        if cx < mid_x and slope < 0:
            left_len += length
            left_count += 1
        elif cx >= mid_x and slope > 0:
            right_len += length
            right_count += 1

    # Normalise: expected detectable length ≈ 65 % of ROI height per side.
    roi_h = float(y2 - y1)
    norm = max(roi_h * 0.65, 1.0)

    left_conf = min(left_len / norm, 1.0)
    right_conf = min(right_len / norm, 1.0)

    # Require at least 2 segments for meaningful confidence.
    # A single streak is far more likely to be noise than a real marking.
    if left_count < 2:
        left_conf *= 0.15
    if right_count < 2:
        right_conf *= 0.15

    # One-sided detection penalty.
    # If only one side found, likely a road edge / barrier, NOT a lane marking.
    # Cap both sides at 20 % to stay below t_lane_low (0.3) and avoid
    # incorrectly claiming HAS_LANES or DEGRADED_LANES.
    if not (left_conf > 0.08 and right_conf > 0.08):
        left_conf *= 0.20
        right_conf *= 0.20

    return LaneDetectionInput(
        left_detected=left_conf > 0.08,
        right_detected=right_conf > 0.08,
        left_confidence=left_conf,
        right_confidence=right_conf,
    )
