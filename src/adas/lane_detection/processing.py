"""Lane detection processing module.

Provides process_frame() which produces a LaneOutput from a raw BGR frame.
This is geometric lane detection: it finds the actual pixel positions and
polynomial fits for left/right lane boundaries, not just a binary
has_lanes flag.

The pipeline:
  1. Extract road ROI (configurable fractions of frame height)
  2. Grayscale + CLAHE contrast enhancement
  3. Gaussian blur + Canny edge detection
  4. HoughLinesP line detection
  5. Classify lines as left/right by slope and position
  6. Fit degree-1 polynomials to left and right line clusters
  7. Produce LaneOutput with confidence scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List

import numpy as np


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaneOutput:
    """Result of geometric lane detection for one frame.

    Coordinates are in the original (full-frame) pixel space.
    Polynomial coefficients are for y = f(x) in the ROI-relative coordinate
    system (y=0 at top of ROI, increasing downward).

    Attributes
    ----------
    has_lanes : bool
        True when at least one lane boundary is detected with confidence
        above the minimum threshold.
    lane_confidence : float
        Overall confidence in [0, 1]. Average of left/right confidences when
        both are present; single-side value multiplied by a one-sided penalty
        otherwise.
    left_detected : bool
    right_detected : bool
    left_confidence : float
    right_confidence : float
    left_poly : tuple[float, ...] or None
        Coefficients [a, b] for left boundary, y-space: x = a*y + b.
        None if left was not detected.
    right_poly : tuple[float, ...] or None
        Coefficients [a, b] for right boundary.
    edges : numpy.ndarray or None
        Canny edge map of the ROI (uint8). Useful for debug overlays.
    mask : numpy.ndarray or None
        Binary mask of the detected lane region (uint8, same size as ROI).
    roi_y1 : int
        Top y-coordinate of the ROI in the original frame.
    roi_y2 : int
        Bottom y-coordinate of the ROI in the original frame.
    lane_width_px : float or None
        Estimated lane width in pixels at the bottom of the ROI.
    """

    has_lanes: bool = False
    lane_confidence: float = 0.0
    left_detected: bool = False
    right_detected: bool = False
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    left_poly: Optional[Tuple[float, ...]] = None
    right_poly: Optional[Tuple[float, ...]] = None
    edges: Optional[Any] = None  # numpy.ndarray
    mask: Optional[Any] = None   # numpy.ndarray
    roi_y1: int = 0
    roi_y2: int = 0
    lane_width_px: Optional[float] = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaneProcessingConfig:
    """Tunable parameters for process_frame().

    Default values are calibrated for DADA-2000 dashcam footage.
    """

    # ROI: fraction of frame height
    roi_top: float = 0.38         # top of lane-search ROI
    roi_bottom: float = 0.90      # bottom (excludes dashboard)

    # Canny
    canny_low: int = 40
    canny_high: int = 120

    # CLAHE
    clahe_clip: float = 1.8
    clahe_grid: int = 8

    # Gaussian blur kernel (must be odd)
    blur_ksize: int = 5

    # Hough probabilistic
    hough_threshold: int = 35
    hough_min_length: int = 30
    hough_max_gap: int = 130

    # Slope range for valid lane-marking lines
    slope_min: float = 0.20
    slope_max: float = 2.50

    # Confidence thresholds
    min_confidence: float = 0.10  # below this: no detection claimed
    one_side_penalty: float = 0.25  # multiplier when only one side detected

    # Normalisation cap for line length accumulation
    max_accum_length: float = 5000.0


DEFAULT_PROCESSING_CONFIG = LaneProcessingConfig()


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

def process_frame(
    frame: Any,
    context_state: Any = None,
    *,
    config: Optional[LaneProcessingConfig] = None,
) -> LaneOutput:
    """Detect lane boundaries in a dashcam frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image, shape (H, W, 3).
    context_state : ContextState, optional
        Current context state. Not used in the current CV implementation but
        kept in the signature for future adaptive tuning (e.g. relaxing
        thresholds at night).
    config : LaneProcessingConfig, optional
        Tunable parameters. Falls back to DEFAULT_PROCESSING_CONFIG.

    Returns
    -------
    LaneOutput
        Detected lane boundaries with confidence scores and polynomial fits.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for lane detection") from e

    cfg = config or DEFAULT_PROCESSING_CONFIG

    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return LaneOutput()

    h, w = frame.shape[:2]
    y1 = int(h * cfg.roi_top)
    y2 = int(h * cfg.roi_bottom)
    if y2 <= y1 or w == 0:
        return LaneOutput()

    roi = frame[y1:y2, :]

    # --- pre-processing ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    edges = cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_length,
        maxLineGap=cfg.hough_max_gap,
    )

    if lines is None:
        return LaneOutput(edges=edges, roi_y1=y1, roi_y2=y2)

    mid_x = w / 2.0
    roi_h = y2 - y1

    # Accumulate left/right line segments separately
    left_points: List[Tuple[float, float]] = []   # (x, y) pairs in ROI coords
    right_points: List[Tuple[float, float]] = []
    left_len_acc = 0.0
    right_len_acc = 0.0

    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        dx = x2_l - x1_l
        dy = y2_l - y1_l
        if dx == 0:
            continue
        slope = dy / dx
        abs_slope = abs(slope)
        if abs_slope < cfg.slope_min or abs_slope > cfg.slope_max:
            continue

        seg_len = float(np.hypot(dx, dy))
        cx = (x1_l + x2_l) / 2.0

        # Classify by position and slope direction
        if slope < 0 and cx < mid_x:
            # Left boundary: negative slope (going up-left to down-right),
            # located in the left half of the frame
            left_points.append((float(x1_l), float(y1_l)))
            left_points.append((float(x2_l), float(y2_l)))
            left_len_acc += seg_len
        elif slope > 0 and cx > mid_x:
            # Right boundary: positive slope, located in right half
            right_points.append((float(x1_l), float(y1_l)))
            right_points.append((float(x2_l), float(y2_l)))
            right_len_acc += seg_len

    # Compute confidence from accumulated line length (normalised)
    left_conf = min(1.0, left_len_acc / cfg.max_accum_length)
    right_conf = min(1.0, right_len_acc / cfg.max_accum_length)

    # Fit polynomials (x = a*y + b in ROI space)
    left_poly = _fit_line_poly(left_points) if len(left_points) >= 4 else None
    right_poly = _fit_line_poly(right_points) if len(right_points) >= 4 else None

    # One-sided penalty: a single visible boundary is more likely a barrier
    # or road edge than a genuine lane marking
    left_detected = left_conf >= cfg.min_confidence
    right_detected = right_conf >= cfg.min_confidence

    if left_detected and not right_detected:
        left_conf *= cfg.one_side_penalty
        left_detected = left_conf >= cfg.min_confidence
    elif right_detected and not left_detected:
        right_conf *= cfg.one_side_penalty
        right_detected = right_conf >= cfg.min_confidence

    has_lanes = left_detected or right_detected

    if left_detected and right_detected:
        lane_confidence = (left_conf + right_conf) / 2.0
    elif left_detected:
        lane_confidence = left_conf
    elif right_detected:
        lane_confidence = right_conf
    else:
        lane_confidence = 0.0

    # Estimate lane width from polynomial intersection at bottom of ROI
    lane_width_px: Optional[float] = None
    if left_poly is not None and right_poly is not None:
        y_bot = float(roi_h - 1)
        lx = left_poly[0] * y_bot + left_poly[1]
        rx = right_poly[0] * y_bot + right_poly[1]
        w_est = abs(rx - lx)
        if 50 < w_est < w * 0.95:
            lane_width_px = w_est

    # Build binary lane mask for overlay purposes
    mask = _build_lane_mask(
        roi_h, w, left_poly, right_poly, left_detected, right_detected
    )

    return LaneOutput(
        has_lanes=has_lanes,
        lane_confidence=lane_confidence,
        left_detected=left_detected,
        right_detected=right_detected,
        left_confidence=left_conf,
        right_confidence=right_conf,
        left_poly=left_poly,
        right_poly=right_poly,
        edges=edges,
        mask=mask,
        roi_y1=y1,
        roi_y2=y2,
        lane_width_px=lane_width_px,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_line_poly(
    points: List[Tuple[float, float]],
) -> Optional[Tuple[float, ...]]:
    """Fit a degree-1 polynomial x = a*y + b to a list of (x, y) points.

    Returns (a, b) as a tuple or None if fitting fails.
    """
    if len(points) < 4:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    try:
        coeffs = np.polyfit(ys, xs, deg=1)
        return tuple(float(c) for c in coeffs)
    except (np.linalg.LinAlgError, ValueError):
        return None


def _build_lane_mask(
    roi_h: int,
    frame_w: int,
    left_poly: Optional[Tuple[float, ...]],
    right_poly: Optional[Tuple[float, ...]],
    left_detected: bool,
    right_detected: bool,
) -> Optional[Any]:
    """Build a binary mask (uint8) of the lane region in ROI space.

    Returns None when neither boundary is available.
    """
    try:
        import cv2
    except ImportError:
        return None

    if not left_detected and not right_detected:
        return None

    mask = np.zeros((roi_h, frame_w), dtype=np.uint8)
    ys = np.arange(roi_h, dtype=np.float64)

    # If only one side, draw just that boundary line
    if left_detected and left_poly is not None and not right_detected:
        xs = np.clip(
            (left_poly[0] * ys + left_poly[1]).astype(np.int32), 0, frame_w - 1
        )
        pts = np.stack([xs, ys.astype(np.int32)], axis=1)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=4)
        return mask

    if right_detected and right_poly is not None and not left_detected:
        xs = np.clip(
            (right_poly[0] * ys + right_poly[1]).astype(np.int32), 0, frame_w - 1
        )
        pts = np.stack([xs, ys.astype(np.int32)], axis=1)
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=4)
        return mask

    # Both sides: fill the polygon between them
    if left_poly is not None and right_poly is not None:
        lxs = np.clip(
            (left_poly[0] * ys + left_poly[1]).astype(np.int32), 0, frame_w - 1
        )
        rxs = np.clip(
            (right_poly[0] * ys + right_poly[1]).astype(np.int32), 0, frame_w - 1
        )
        left_pts = list(zip(lxs.tolist(), ys.astype(int).tolist()))
        right_pts = list(reversed(list(zip(rxs.tolist(), ys.astype(int).tolist()))))
        polygon = np.array(left_pts + right_pts, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], color=128)
        cv2.polylines(mask, [np.array(left_pts, dtype=np.int32)], False, 255, 3)
        cv2.polylines(mask, [np.array(
            list(zip(rxs.tolist(), ys.astype(int).tolist())), dtype=np.int32
        )], False, 255, 3)

    return mask
