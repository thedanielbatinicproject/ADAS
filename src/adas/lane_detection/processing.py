"""Lane detection processing module.

Provides process_frame() and the stateful LaneProcessor class, both of
which produce a LaneOutput from a raw BGR frame.

Context-adaptive pipeline
-------------------------
The algorithm selects its operating strategy based on the current
ContextState (passed as context_state).  Four main strategies exist:

1. NORMAL_MARKED (default)
   Standard Canny + HoughLinesP on a contrast-enhanced ROI.

2. DEGRADED_MARKED  (rain / fog / night with visible lanes)
   Stronger CLAHE, bilateral filter instead of Gaussian blur, relaxed
   Hough thresholds.

3. UNMARKED_GOOD_VIS / UNMARKED_DEGRADED  (no lane markings)
   Skip line search entirely.  Return a trapezoid LaneOutput that
   represents the forward road area in front of the vehicle.

4. EMERGENCY_OVERRIDE
   Falls through to NORMAL_MARKED strategy with all defaults.

Temporal damping
----------------
LaneProcessor wraps process_frame() and blends new polynomial
coefficients with the previous frame values.  This prevents abrupt
jumps between frames (jitter suppression).  The blend weight and the
maximum allowed per-frame shift in pixels at the bottom of the ROI are
configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import numpy as np

from ..utils.runtime_overrides import apply_dataclass_overrides


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaneOutput:
    """Result of geometric lane detection for one frame.

    Coordinates are in the original (full-frame) pixel space.
    Polynomial coefficients are for x = f(y) in ROI-relative space
    (y=0 at top of ROI, increasing downward).

    Attributes
    ----------
    has_lanes : bool
        True when at least one lane boundary is reliably detected.
    lane_confidence : float
        Overall confidence in [0, 1].
    left_detected : bool
    right_detected : bool
    left_confidence : float
    right_confidence : float
    left_poly : tuple[float, ...] or None
        Coefficients for left boundary in ROI space (linear or quadratic).
    right_poly : tuple[float, ...] or None
    edges : numpy.ndarray or None
        Canny edge map of the ROI (uint8).
    mask : numpy.ndarray or None
        Binary mask of the detected lane region (same size as ROI).
    roi_y1 : int
        Top y-coordinate of the ROI in the original frame.
    roi_y2 : int
        Bottom y-coordinate of the ROI.
    lane_width_px : float or None
        Estimated lane width in pixels at the bottom of the ROI.
    is_trapezoid : bool
        True when the output is a synthetic trapezoid (no real lines).
    """

    has_lanes: bool = False
    lane_confidence: float = 0.0
    left_detected: bool = False
    right_detected: bool = False
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    left_poly: Optional[Tuple[float, ...]] = None
    right_poly: Optional[Tuple[float, ...]] = None
    edges: Optional[Any] = None   # numpy.ndarray
    mask: Optional[Any] = None    # numpy.ndarray
    roi_y1: int = 0
    roi_y2: int = 0
    lane_width_px: Optional[float] = None
    is_trapezoid: bool = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LaneProcessingConfig:
    """Tunable parameters for process_frame().

    Default values are calibrated for DADA-2000 dashcam footage.
    """

    # ROI fractions of frame height
    roi_top: float = 0.38
    roi_bottom: float = 0.90

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
    min_confidence: float = 0.10
    one_side_penalty: float = 0.25

    # Normalisation cap for line length accumulation
    max_accum_length: float = 5000.0

    # Temporal damping (used by LaneProcessor)
    # Weight of the new frame in the blended polynomial (0=no update, 1=raw)
    damping_alpha: float = 0.65
    # Maximum allowed shift of the bottom x-position per frame (pixels).
    # Set to 0 to disable.
    max_shift_px: float = 80.0

    # Trapezoid geometry (fractions of frame width / height)
    trapezoid_bottom_left: float = 0.08
    trapezoid_bottom_right: float = 0.92
    trapezoid_top_left: float = 0.40
    trapezoid_top_right: float = 0.60
    trapezoid_top_frac: float = 0.42

    # Detection corridor constraints to avoid non-road lines (trees, rails)
    min_line_mid_y_frac: float = 0.32
    lane_min_width_frac: float = 0.18
    lane_max_width_frac: float = 0.92
    lane_expected_left_bottom_frac: float = 0.24
    lane_expected_right_bottom_frac: float = 0.76
    lane_expected_left_top_frac: float = 0.42
    lane_expected_right_top_frac: float = 0.58


DEFAULT_PROCESSING_CONFIG = apply_dataclass_overrides(LaneProcessingConfig(), "lane")


# ---------------------------------------------------------------------------
# Stateful processor with temporal damping
# ---------------------------------------------------------------------------

class LaneProcessor:
    """Stateful lane detection processor.

    Wraps process_frame() and applies temporal damping to polynomial
    coefficients across frames.  Create one instance per video and call
    update() for each frame.
    """

    def __init__(self, config: Optional[LaneProcessingConfig] = None) -> None:
        self._config = config or DEFAULT_PROCESSING_CONFIG
        self._prev_left: Optional[Tuple[float, ...]] = None
        self._prev_right: Optional[Tuple[float, ...]] = None

    def reset(self) -> None:
        """Reset temporal state (call when switching videos)."""
        self._prev_left = None
        self._prev_right = None

    def update_config(self, config: LaneProcessingConfig) -> None:
        self._config = config

    def update(
        self,
        frame: Any,
        context_state: Any = None,
    ) -> LaneOutput:
        """Process one frame with temporal smoothing applied."""
        raw = process_frame(frame, context_state, config=self._config)

        if raw.is_trapezoid:
            self._prev_left = None
            self._prev_right = None
            return raw

        roi_h = raw.roi_y2 - raw.roi_y1
        alpha = self._config.damping_alpha
        max_shift = self._config.max_shift_px

        left_poly = _blend_poly(
            raw.left_poly if raw.left_detected else None,
            self._prev_left,
            alpha,
            max_shift,
            roi_h,
        )
        right_poly = _blend_poly(
            raw.right_poly if raw.right_detected else None,
            self._prev_right,
            alpha,
            max_shift,
            roi_h,
        )

        if raw.left_detected and left_poly is not None:
            self._prev_left = left_poly
        elif raw.left_poly is None:
            self._prev_left = None

        if raw.right_detected and right_poly is not None:
            self._prev_right = right_poly
        elif raw.right_poly is None:
            self._prev_right = None

        return _rebuild_with_polys(raw, left_poly, right_poly)


# ---------------------------------------------------------------------------
# Pure functional API
# ---------------------------------------------------------------------------

def process_frame(
    frame: Any,
    context_state: Any = None,
    *,
    config: Optional[LaneProcessingConfig] = None,
) -> LaneOutput:
    """Detect lane boundaries in a dashcam frame.

    Selects an operating strategy based on context_state:
    - UNMARKED_* modes: return a synthetic trapezoid.
    - DEGRADED_MARKED / fog / rain / night: weather-adapted preprocessing.
    - NORMAL_MARKED or no context: standard pipeline.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image, shape (H, W, 3).
    context_state : ContextState, optional
    config : LaneProcessingConfig, optional

    Returns
    -------
    LaneOutput
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

    mode_str = _get_mode_str(context_state)
    weather_str = _get_weather_str(context_state)
    light_str = _get_light_str(context_state)

    # Strategy: no lane markings expected -> synthetic trapezoid
    if mode_str in ("unmarked_good_vis", "unmarked_degraded"):
        return _build_trapezoid(h, w, y1, y2, cfg, mode_str=mode_str)

    # Strategy: degraded conditions -> adapted preprocessing
    use_degraded = (
        mode_str == "degraded_marked"
        or weather_str in ("fog", "rain")
        or light_str == "night"
    )

    roi = frame[y1:y2, :]

    if use_degraded:
        edges = _preprocess_degraded(roi, cfg, weather_str, light_str)
    else:
        edges = _preprocess_normal(roi, cfg)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=_hough_threshold(cfg, use_degraded),
        minLineLength=_hough_min_length(cfg, use_degraded),
        maxLineGap=_hough_max_gap(cfg, use_degraded),
    )

    if lines is None:
        # Always provide a road surface in front of the ego vehicle.
        return _build_trapezoid(
            h,
            w,
            y1,
            y2,
            cfg,
            mode_str=mode_str,
            edges=edges,
        )

    mid_x = w / 2.0
    roi_h = y2 - y1

    left_points: List[Tuple[float, float]] = []
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
        cy = (y1_l + y2_l) / 2.0

        # Focus only on road area in front of the vehicle.
        if cy < roi_h * cfg.min_line_mid_y_frac:
            continue

        exp_left = _expected_lane_x(roi_h, w, float(cy), left=True, cfg=cfg)
        exp_right = _expected_lane_x(roi_h, w, float(cy), left=False, cfg=cfg)
        gate_px = w * 0.22

        if slope < 0 and cx < mid_x and abs(cx - exp_left) <= gate_px:
            left_points.append((float(x1_l), float(y1_l)))
            left_points.append((float(x2_l), float(y2_l)))
            left_len_acc += seg_len
        elif slope > 0 and cx > mid_x and abs(cx - exp_right) <= gate_px:
            right_points.append((float(x1_l), float(y1_l)))
            right_points.append((float(x2_l), float(y2_l)))
            right_len_acc += seg_len

    left_conf = min(1.0, left_len_acc / cfg.max_accum_length)
    right_conf = min(1.0, right_len_acc / cfg.max_accum_length)

    left_poly = _fit_lane_poly(left_points) if len(left_points) >= 4 else None
    right_poly = _fit_lane_poly(right_points) if len(right_points) >= 4 else None

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

    lane_width_px: Optional[float] = None
    if left_poly is not None and right_poly is not None:
        y_bot = float(roi_h - 1)
        lx = _poly_eval(left_poly, y_bot)
        rx = _poly_eval(right_poly, y_bot)
        w_est = abs(rx - lx)
        if 50 < w_est < w * 0.95:
            lane_width_px = w_est

    # If lane geometry is weak or implausible, synthesize perspective road area.
    width_ok = False
    if lane_width_px is not None:
        width_ok = (w * cfg.lane_min_width_frac) <= lane_width_px <= (w * cfg.lane_max_width_frac)
    if (not has_lanes) or (not width_ok):
        return _build_trapezoid(
            h,
            w,
            y1,
            y2,
            cfg,
            mode_str=mode_str,
            edges=edges,
            left_hint=left_poly,
            right_hint=right_poly,
        )

    mask = _build_lane_mask(
        roi_h, w, left_poly, right_poly, left_detected, right_detected
    )

    return LaneOutput(
        has_lanes=has_lanes,
        lane_confidence=lane_confidence,
        left_detected=left_detected,
        right_detected=right_detected,
        left_confidence=left_conf if left_detected else 0.0,
        right_confidence=right_conf if right_detected else 0.0,
        left_poly=left_poly if left_detected else None,
        right_poly=right_poly if right_detected else None,
        edges=edges,
        mask=mask,
        roi_y1=y1,
        roi_y2=y2,
        lane_width_px=lane_width_px,
        is_trapezoid=False,
    )


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_normal(roi: Any, cfg: LaneProcessingConfig) -> Any:
    """Standard preprocessing: CLAHE + Gaussian blur + Canny."""
    import cv2
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip,
        tileGridSize=(cfg.clahe_grid, cfg.clahe_grid),
    )
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    return cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)


def _preprocess_degraded(
    roi: Any,
    cfg: LaneProcessingConfig,
    weather: str,
    light: str,
) -> Any:
    """Weather / night-adapted preprocessing.

    - night:  strong CLAHE to lift dark lanes, lower Canny thresholds.
    - rain:   bilateral filter to remove specular reflection noise.
    - fog:    heavy CLAHE + morphological gradient to recover contrast.
    - glare:  moderate CLAHE + slightly raised Canny thresholds.
    - other:  moderate CLAHE boost.
    """
    import cv2

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if light == "night":
        clip = max(cfg.clahe_clip * 2.0, 3.5)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
        return cv2.Canny(blurred, max(20, cfg.canny_low - 15), max(60, cfg.canny_high - 30))

    if weather == "rain":
        clip = max(cfg.clahe_clip * 1.6, 2.8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        gray = clahe.apply(gray)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.Canny(blurred, cfg.canny_low, cfg.canny_high)

    if weather == "fog":
        clip = max(cfg.clahe_clip * 2.5, 4.0)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        gray = clahe.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        return cv2.Canny(gradient, max(20, cfg.canny_low - 10), cfg.canny_high)

    if weather == "glare":
        clip = cfg.clahe_clip
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
        gray = clahe.apply(gray)
        ksize = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        return cv2.Canny(blurred, cfg.canny_low + 10, cfg.canny_high + 20)

    # Generic degraded: moderate CLAHE boost
    clip = max(cfg.clahe_clip * 1.4, 2.5)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    gray = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    return cv2.Canny(blurred, max(20, cfg.canny_low - 5), cfg.canny_high)


def _hough_threshold(cfg: LaneProcessingConfig, degraded: bool) -> int:
    return max(15, cfg.hough_threshold - 10) if degraded else cfg.hough_threshold


def _hough_min_length(cfg: LaneProcessingConfig, degraded: bool) -> int:
    return max(15, cfg.hough_min_length - 10) if degraded else cfg.hough_min_length


def _hough_max_gap(cfg: LaneProcessingConfig, degraded: bool) -> int:
    return cfg.hough_max_gap + 40 if degraded else cfg.hough_max_gap


# ---------------------------------------------------------------------------
# Trapezoid fallback
# ---------------------------------------------------------------------------

def _build_trapezoid(
    h: int,
    w: int,
    y1: int,
    y2: int,
    cfg: LaneProcessingConfig,
    *,
    mode_str: str = "",
    edges: Any = None,
    left_hint: Optional[Tuple[float, ...]] = None,
    right_hint: Optional[Tuple[float, ...]] = None,
) -> LaneOutput:
    """Build a perspective-adaptive forward-road area.

    This fallback guarantees that the ego vehicle always has a plausible
    road surface directly in front, even when line extraction fails.
    """
    import cv2

    roi_h = y2 - y1
    geom = _estimate_road_geometry(
        h,
        w,
        y1,
        y2,
        cfg,
        edges=edges,
        left_hint=left_hint,
        right_hint=right_hint,
    )
    bx_l = geom["bx_l"]
    bx_r = geom["bx_r"]
    tx_l = geom["tx_l"]
    tx_r = geom["tx_r"]
    top_y_roi = geom["top_y_roi"]
    bot_y_roi = roi_h - 1

    left_poly = _poly_from_three_points(
        float(top_y_roi),
        float(tx_l),
        float((top_y_roi + bot_y_roi) * 0.5),
        float((tx_l + bx_l) * 0.5 + geom["curve_bias"]),
        float(bot_y_roi),
        float(bx_l),
    )
    right_poly = _poly_from_three_points(
        float(top_y_roi),
        float(tx_r),
        float((top_y_roi + bot_y_roi) * 0.5),
        float((tx_r + bx_r) * 0.5 + geom["curve_bias"]),
        float(bot_y_roi),
        float(bx_r),
    )

    lane_width_px = float(abs(bx_r - bx_l))

    mask = np.zeros((roi_h, w), dtype=np.uint8)
    pts = np.array(
        [
            [bx_l, bot_y_roi],
            [bx_r, bot_y_roi],
            [tx_r, top_y_roi],
            [tx_l, top_y_roi],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)

    return LaneOutput(
        has_lanes=True,
        lane_confidence=0.55 if mode_str not in ("unmarked_good_vis", "unmarked_degraded") else 0.5,
        left_detected=True,
        right_detected=True,
        left_confidence=0.55 if mode_str not in ("unmarked_good_vis", "unmarked_degraded") else 0.5,
        right_confidence=0.55 if mode_str not in ("unmarked_good_vis", "unmarked_degraded") else 0.5,
        left_poly=left_poly,
        right_poly=right_poly,
        edges=edges,
        mask=mask,
        roi_y1=y1,
        roi_y2=y2,
        lane_width_px=lane_width_px,
        is_trapezoid=True,
    )


# ---------------------------------------------------------------------------
# Poly / mask helpers
# ---------------------------------------------------------------------------

def _fit_line_poly(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Fit x = a*y + b to a list of (x, y) points."""
    if len(points) < 4:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    if np.std(ys) < 1e-6:
        return None
    coeffs = np.polyfit(ys, xs, 1)
    return (float(coeffs[0]), float(coeffs[1]))


def _fit_lane_poly(points: List[Tuple[float, float]]) -> Optional[Tuple[float, ...]]:
    """Fit x=f(y), preferring a quadratic model for road curvature."""
    if len(points) < 4:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    if np.std(ys) < 1e-6:
        return None
    if len(points) >= 8:
        coeffs = np.polyfit(ys, xs, 2)
        return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    return _fit_line_poly(points)


def _poly_from_two_points(
    y1: float, x1: float, y2: float, x2: float
) -> Tuple[float, float]:
    dy = y2 - y1
    if abs(dy) < 1e-6:
        return (0.0, float(x1))
    a = (x2 - x1) / dy
    b = x1 - a * y1
    return (a, b)


def _poly_from_three_points(
    y1: float,
    x1: float,
    y2: float,
    x2: float,
    y3: float,
    x3: float,
) -> Tuple[float, float, float]:
    ys = np.array([y1, y2, y3], dtype=np.float32)
    xs = np.array([x1, x2, x3], dtype=np.float32)
    coeffs = np.polyfit(ys, xs, 2)
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))


def _poly_eval(poly: Tuple[float, ...], y: float) -> float:
    if len(poly) == 2:
        return float(poly[0] * y + poly[1])
    if len(poly) >= 3:
        return float(poly[0] * y * y + poly[1] * y + poly[2])
    return 0.0


def _to_quadratic(poly: Tuple[float, ...]) -> Tuple[float, float, float]:
    if len(poly) == 2:
        return (0.0, float(poly[0]), float(poly[1]))
    if len(poly) >= 3:
        return (float(poly[0]), float(poly[1]), float(poly[2]))
    return (0.0, 0.0, 0.0)


def _expected_lane_x(
    roi_h: int,
    w: int,
    y: float,
    *,
    left: bool,
    cfg: LaneProcessingConfig,
) -> float:
    if roi_h <= 1:
        return w * (cfg.lane_expected_left_bottom_frac if left else cfg.lane_expected_right_bottom_frac)
    t = max(0.0, min(1.0, y / float(roi_h - 1)))
    if left:
        top = w * cfg.lane_expected_left_top_frac
        bot = w * cfg.lane_expected_left_bottom_frac
    else:
        top = w * cfg.lane_expected_right_top_frac
        bot = w * cfg.lane_expected_right_bottom_frac
    return top + (bot - top) * t


def _estimate_road_geometry(
    h: int,
    w: int,
    y1: int,
    y2: int,
    cfg: LaneProcessingConfig,
    *,
    edges: Any = None,
    left_hint: Optional[Tuple[float, ...]] = None,
    right_hint: Optional[Tuple[float, ...]] = None,
) -> dict:
    roi_h = y2 - y1
    bot_y = max(1, roi_h - 1)

    bx_l = int(w * cfg.trapezoid_bottom_left)
    bx_r = int(w * cfg.trapezoid_bottom_right)
    tx_l = int(w * cfg.trapezoid_top_left)
    tx_r = int(w * cfg.trapezoid_top_right)
    top_y_roi = max(0, int(h * cfg.trapezoid_top_frac) - y1)

    if edges is not None and hasattr(edges, "shape") and edges.size > 0:
        row_scores = edges.mean(axis=1)
        start = max(0, int(roi_h * 0.12))
        end = max(start + 1, int(roi_h * 0.68))
        if end > start:
            local = row_scores[start:end]
            idx = int(np.argmax(local)) + start
            low = int(roi_h * 0.22)
            high = int(roi_h * 0.62)
            top_y_roi = max(low, min(high, idx))

        band_h = max(8, int(roi_h * 0.12))
        bot_band = edges[max(0, roi_h - band_h):roi_h, :]
        top_band = edges[max(0, top_y_roi - band_h // 2):min(roi_h, top_y_roi + band_h // 2 + 1), :]

        bx = np.where(bot_band > 0)[1]
        tx = np.where(top_band > 0)[1]
        if bx.size > 50:
            bx_l = int(np.percentile(bx, 10))
            bx_r = int(np.percentile(bx, 90))
        if tx.size > 30:
            tx_l = int(np.percentile(tx, 35))
            tx_r = int(np.percentile(tx, 65))

    bx_l = max(0, min(w - 2, bx_l))
    bx_r = max(bx_l + 1, min(w - 1, bx_r))
    tx_l = max(0, min(w - 2, tx_l))
    tx_r = max(tx_l + 1, min(w - 1, tx_r))

    # Keep perspective plausible: top must be narrower than bottom.
    b_width = float(max(20, bx_r - bx_l))
    t_width = float(max(10, tx_r - tx_l))
    max_top = b_width * 0.78
    min_top = b_width * 0.32
    if t_width > max_top or t_width < min_top:
        center = (tx_l + tx_r) * 0.5
        t_width = max(min_top, min(max_top, t_width))
        tx_l = int(center - t_width * 0.5)
        tx_r = int(center + t_width * 0.5)

    tx_l = max(0, min(w - 2, tx_l))
    tx_r = max(tx_l + 1, min(w - 1, tx_r))

    curve_bias = 0.0
    if left_hint is not None and right_hint is not None:
        center_top = (_poly_eval(left_hint, float(top_y_roi)) + _poly_eval(right_hint, float(top_y_roi))) * 0.5
        center_bot = (_poly_eval(left_hint, float(bot_y)) + _poly_eval(right_hint, float(bot_y))) * 0.5
        curve_bias = float(np.clip((center_top - center_bot) * 0.25, -w * 0.05, w * 0.05))

    return {
        "bx_l": bx_l,
        "bx_r": bx_r,
        "tx_l": tx_l,
        "tx_r": tx_r,
        "top_y_roi": max(0, min(bot_y - 4, top_y_roi)),
        "curve_bias": curve_bias,
    }


def _build_lane_mask(
    roi_h: int,
    w: int,
    left_poly: Optional[Tuple[float, ...]],
    right_poly: Optional[Tuple[float, ...]],
    left_detected: bool,
    right_detected: bool,
) -> Any:
    """Build a binary lane-area mask for the ROI."""
    try:
        import cv2
    except ImportError:
        return None

    mask = np.zeros((roi_h, w), dtype=np.uint8)
    if not left_detected or not right_detected:
        return mask
    if left_poly is None or right_poly is None:
        return mask

    pts_left: List[Tuple[int, int]] = []
    pts_right: List[Tuple[int, int]] = []
    for y in range(roi_h):
        lx = int(_poly_eval(left_poly, float(y)))
        rx = int(_poly_eval(right_poly, float(y)))
        lx = max(0, min(w - 1, lx))
        rx = max(0, min(w - 1, rx))
        pts_left.append((lx, y))
        pts_right.append((rx, y))

    poly_pts = np.array(pts_left + list(reversed(pts_right)), dtype=np.int32)
    cv2.fillPoly(mask, [poly_pts], 255)
    return mask


# ---------------------------------------------------------------------------
# Temporal damping helper
# ---------------------------------------------------------------------------

def _blend_poly(
    new_poly: Optional[Tuple[float, ...]],
    prev_poly: Optional[Tuple[float, ...]],
    alpha: float,
    max_shift_px: float,
    roi_h: int,
) -> Optional[Tuple[float, ...]]:
    """Exponential blend of new and previous polynomial coefficients."""
    if new_poly is None and prev_poly is None:
        return None
    if new_poly is None:
        return prev_poly
    if prev_poly is None:
        return new_poly

    new_q = _to_quadratic(new_poly)
    prev_q = _to_quadratic(prev_poly)

    blended = tuple(
        alpha * n + (1.0 - alpha) * p
        for n, p in zip(new_q, prev_q)
    )

    if max_shift_px > 0 and roi_h > 0:
        y_bot = float(roi_h - 1)
        prev_bx = _poly_eval(prev_q, y_bot)
        blended_bx = _poly_eval(blended, y_bot)
        shift = blended_bx - prev_bx
        if abs(shift) > max_shift_px:
            direction = 1.0 if shift > 0 else -1.0
            clamped_bx = prev_bx + max_shift_px * direction
            a2, a1, _a0 = blended
            a0 = clamped_bx - (a2 * y_bot * y_bot + a1 * y_bot)
            blended = (a2, a1, a0)

    return blended


def _rebuild_with_polys(
    original: LaneOutput,
    left_poly: Optional[Tuple[float, ...]],
    right_poly: Optional[Tuple[float, ...]],
) -> LaneOutput:
    """Return a copy of original with poly coefficients replaced."""
    roi_h = original.roi_y2 - original.roi_y1
    lane_width_px: Optional[float] = None
    if left_poly is not None and right_poly is not None and roi_h > 0:
        y_bot = float(roi_h - 1)
        lx = _poly_eval(left_poly, y_bot)
        rx = _poly_eval(right_poly, y_bot)
        w_est = abs(rx - lx)
        if w_est > 50:
            lane_width_px = w_est

    return LaneOutput(
        has_lanes=original.has_lanes,
        lane_confidence=original.lane_confidence,
        left_detected=original.left_detected,
        right_detected=original.right_detected,
        left_confidence=original.left_confidence,
        right_confidence=original.right_confidence,
        left_poly=left_poly,
        right_poly=right_poly,
        edges=original.edges,
        mask=original.mask,
        roi_y1=original.roi_y1,
        roi_y2=original.roi_y2,
        lane_width_px=lane_width_px if lane_width_px is not None else original.lane_width_px,
        is_trapezoid=original.is_trapezoid,
    )


# ---------------------------------------------------------------------------
# Context helpers (safe against None)
# ---------------------------------------------------------------------------

def _get_mode_str(context_state: Any) -> str:
    if context_state is None:
        return ""
    mode = getattr(context_state, "mode", None)
    if mode is None:
        return ""
    return str(getattr(mode, "value", mode)).lower()


def _get_weather_str(context_state: Any) -> str:
    if context_state is None:
        return ""
    weather = getattr(context_state, "weather_condition", None)
    if weather is None:
        return ""
    return str(getattr(weather, "value", weather)).lower()


def _get_light_str(context_state: Any) -> str:
    if context_state is None:
        return ""
    light = getattr(context_state, "light_condition", None)
    if light is None:
        return ""
    return str(getattr(light, "value", light)).lower()
