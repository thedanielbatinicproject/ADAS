"""Hough-transform geometric lane detection with RANSAC fitting and Kalman smoothing.

Lane-existence heuristics (does the road have painted markings?) live in
:mod:`adas.context.lane_heuristic` as part of the context package.

This module provides precise geometric lane boundary detection:
- Multi-scale HoughLinesP (3 parameter sets merged and deduplicated)
- Vanishing-point guided left/right line classification
- RANSAC-robust polynomial fitting (quadratic x = f(y) per side)
- 6D Kalman filter tracking state [a, b, c, da, db, dc] per lane
- Sliding-window search seeded by the tracked polynomial
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoughLaneConfig:
    """Tunable parameters for the Hough + RANSAC + Kalman pipeline."""

    # ---- Multi-scale Hough ----
    # Three parameter sets run on the same edge map and merged.
    # Scale A: tight — strong continuous markings (high threshold, long lines)
    scale_a_threshold: int = 45
    scale_a_min_length: int = 40
    scale_a_max_gap: int = 80
    # Scale B: medium — dashed markings at mid range
    scale_b_threshold: int = 25
    scale_b_min_length: int = 20
    scale_b_max_gap: int = 150
    # Scale C: loose — distant or faded dashes
    scale_c_threshold: int = 15
    scale_c_min_length: int = 12
    scale_c_max_gap: int = 200

    # Run a third (looser) Hough pass for very faint or distant markings.
    # Off by default — the two-scale pass is sufficient and much faster.
    use_three_scales: bool = False

    # ---- Slope filter ----
    slope_min: float = 0.20   # tan(~11°) — reject near-horizontal noise
    slope_max: float = 2.80   # tan(~70°) — reject near-vertical noise

    # ---- Vanishing point ----
    # Reject VP candidates whose y exceeds this ROI fraction (below horizon).
    vp_max_y_frac: float = 0.55
    # Minimum candidate intersections required to trust the VP estimate.
    vp_min_candidates: int = 3

    # ---- Corridor filter (fraction of frame width) ----
    # Segments whose centroid falls outside these x-corridors are ignored.
    left_corridor_min: float = 0.00
    left_corridor_max: float = 0.60
    right_corridor_min: float = 0.40
    right_corridor_max: float = 1.00

    # ---- Road-prior geometry gate ----
    # Expected lane positions from perspective model (fractions of frame width).
    expected_left_bottom_frac: float = 0.24
    expected_right_bottom_frac: float = 0.76
    expected_left_top_frac: float = 0.42
    expected_right_top_frac: float = 0.58
    # Segment must have at least one endpoint below this ROI fraction.
    min_seg_bottom_y_frac: float = 0.45
    # Allowed deviation from expected lane x-position (fraction of frame width).
    lane_gate_frac: float = 0.35

    # ---- RANSAC polynomial fitting ----
    ransac_n_iter: int = 20
    ransac_inlier_threshold_px: float = 6.0
    ransac_min_inliers: int = 6
    ransac_degree: int = 2

    # ---- Confidence scoring ----
    # Inlier count that maps to confidence = 1.0 (soft cap via saturation).
    confidence_inlier_norm: float = 14.0
    # Innovation-based penalty: large Kalman corrections reduce confidence.
    # Penalty = exp(-innovation / confidence_innovation_scale).
    confidence_innovation_scale: float = 80.0

    # ---- Kalman filter ----
    kf_process_noise: float = 0.02   # Q diagonal scale
    kf_meas_noise: float = 18.0      # R diagonal scale
    # After this many consecutive frames without a measurement update, reset the KF.
    kf_lost_frames_max: int = 15
    # Innovation gate: if the RANSAC bottom-x deviates more than this from the
    # KF prediction, reject that measurement (keeps the filter clean).
    # Set to 0 to disable.
    kf_max_innovation_px: float = 35.0

    # ---- Polynomial sanity checks ----
    # Reject quadratic fits with |a| above this threshold.  Dashcam lane lines
    # are nearly straight in perspective; a large 'a' value means the fit is
    # chasing noise (and often produces horizontal-looking overlays).
    # For a 300 px ROI: a=0.001 → ~90 px of lateral curvature (generous limit).
    poly_max_quad_coeff: float = 0.0010

    # ---- Sliding window search ----
    sw_n_windows: int = 8
    sw_margin_px: int = 45
    # Minimum non-zero pixels in a window band to accept it.
    sw_min_pixels: int = 4


# ---------------------------------------------------------------------------
# 6D Kalman filter for quadratic lane polynomials
# ---------------------------------------------------------------------------

class PolyKalmanFilter:
    """Drift-safe 3D Kalman filter for quadratic lane polynomial x = a*y² + b*y + c.

    State vector is [a, b, c] only.  We intentionally avoid per-coefficient
    velocity terms because they can introduce artificial drift in stationary
    scenes (dashcam and lane markings static, but tracked lines keep moving).
    """

    def __init__(self, process_noise: float = 0.05, meas_noise: float = 20.0) -> None:
        self._pn = process_noise
        self._mn = meas_noise

        # State: [a, b, c]
        self._x = np.zeros(3, dtype=np.float64)
        # Covariance (start with large uncertainty)
        self._P = np.eye(3, dtype=np.float64) * 1000.0

        # State transition: stationary model x_{k+1} = x_k
        self._F = np.eye(3, dtype=np.float64)

        # Measurement matrix: directly observe [a, b, c]
        self._H = np.eye(3, dtype=np.float64)

        # Process noise covariance
        self._Q = np.eye(3, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self._R = np.eye(3, dtype=np.float64) * meas_noise

        self._initialized = False
        self._lost_frames = 0
        self._last_innovation = 0.0

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def lost_frames(self) -> int:
        return self._lost_frames

    def reset(self) -> None:
        self._x[:] = 0.0
        self._P[:] = 0.0
        np.fill_diagonal(self._P, 1000.0)
        self._initialized = False
        self._lost_frames = 0
        self._last_innovation = 0.0

    def predict(self) -> None:
        if not self._initialized:
            return
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self._lost_frames += 1

    def update(self, coeffs: Tuple[float, float, float]) -> None:
        z = np.array(coeffs, dtype=np.float64)

        if not self._initialized:
            # First measurement: initialize directly
            self._x[:] = z
            np.fill_diagonal(self._P, 100.0)
            self._initialized = True
            self._lost_frames = 0
            self._last_innovation = 0.0
            return

        # Innovation
        y_inn = z - self._H @ self._x
        self._last_innovation = float(np.linalg.norm(y_inn))

        # Innovation covariance
        S = self._H @ self._P @ self._H.T + self._R

        # Kalman gain
        K = self._P @ self._H.T @ np.linalg.inv(S)

        # State + covariance update
        self._x = self._x + K @ y_inn
        I_KH = np.eye(3, dtype=np.float64) - K @ self._H
        self._P = I_KH @ self._P

        self._lost_frames = 0

    def get(self) -> Optional[Tuple[float, float, float]]:
        if not self._initialized:
            return None
        return (float(self._x[0]), float(self._x[1]), float(self._x[2]))

    def innovation(self) -> float:
        return self._last_innovation


# ---------------------------------------------------------------------------
# Stateful lane tracker (owns two KFs)
# ---------------------------------------------------------------------------

class LaneTracker:
    """Stateful lane tracker holding one PolyKalmanFilter per side.

    Create one instance per video (or per LaneProcessor) and call
    reset() when switching to a new video.  Pass the instance to
    detect_lanes_hough() on every frame.
    """

    def __init__(self, cfg: Optional[HoughLaneConfig] = None) -> None:
        _cfg = cfg or HoughLaneConfig()
        self.left_kf = PolyKalmanFilter(_cfg.kf_process_noise, _cfg.kf_meas_noise)
        self.right_kf = PolyKalmanFilter(_cfg.kf_process_noise, _cfg.kf_meas_noise)
        # Per-frame movement tracking for anti-teleport clamp
        self.left_last_bx: float = -1.0
        self.right_last_bx: float = -1.0

    def reset(self) -> None:
        self.left_kf.reset()
        self.right_kf.reset()
        self.left_last_bx = -1.0
        self.right_last_bx = -1.0


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoughLaneResult:
    """Result produced by detect_lanes_hough() for one frame."""

    left_poly: Optional[Tuple[float, float, float]] = None
    right_poly: Optional[Tuple[float, float, float]] = None
    left_conf: float = 0.0
    right_conf: float = 0.0
    left_inliers: int = 0
    right_inliers: int = 0
    vp_x: float = 0.0
    vp_y: float = 0.0
    used_sliding_window: bool = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _multi_scale_hough(
    edges: np.ndarray,
    cfg: HoughLaneConfig,
) -> List[Tuple[int, int, int, int]]:
    """Run HoughLinesP at two (or three) parameter scales and concatenate results.

    No deduplication is performed here — the RANSAC step downstream is
    robust to duplicate/overlapping segments.  Skipping the O(n²) Python
    dedup loop is the single largest performance win.
    """
    try:
        import cv2
    except ImportError:
        return []

    scales = [
        (cfg.scale_a_threshold, cfg.scale_a_min_length, cfg.scale_a_max_gap),
        (cfg.scale_b_threshold, cfg.scale_b_min_length, cfg.scale_b_max_gap),
    ]
    if cfg.use_three_scales:
        scales.append((cfg.scale_c_threshold, cfg.scale_c_min_length, cfg.scale_c_max_gap))

    all_segs: List[Tuple[int, int, int, int]] = []
    for thresh, min_len, max_gap in scales:
        result = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=thresh,
            minLineLength=min_len,
            maxLineGap=max_gap,
        )
        if result is not None:
            for seg in result:
                all_segs.append(tuple(seg[0]))  # type: ignore[arg-type]

    return all_segs


def _line_intersection(
    x1: float, y1: float, x2: float, y2: float,
    x3: float, y3: float, x4: float, y4: float,
) -> Optional[Tuple[float, float]]:
    """Return intersection of two lines, or None if parallel."""
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return (ix, iy)


def _estimate_vanishing_point(
    left_segs: List[Tuple[int, int, int, int]],
    right_segs: List[Tuple[int, int, int, int]],
    roi_h: int,
    w: int,
    cfg: HoughLaneConfig,
) -> Tuple[float, float]:
    """Estimate vanishing point as the median intersection of left/right pairs.

    Only intersections above (smaller y than) vp_max_y_frac * roi_h are
    considered plausible horizon candidates.
    """
    candidates: List[Tuple[float, float]] = []
    max_y = roi_h * cfg.vp_max_y_frac

    # Sample at most 10 from each side to limit O(n²) cost.
    l_sample = left_segs[:10]
    r_sample = right_segs[:10]

    for lx1, ly1, lx2, ly2 in l_sample:
        for rx1, ry1, rx2, ry2 in r_sample:
            pt = _line_intersection(
                float(lx1), float(ly1), float(lx2), float(ly2),
                float(rx1), float(ry1), float(rx2), float(ry2),
            )
            if pt is None:
                continue
            ix, iy = pt
            if 0 <= iy <= max_y and 0 <= ix <= w:
                candidates.append((ix, iy))

    if len(candidates) >= cfg.vp_min_candidates:
        xs = [c[0] for c in candidates]
        ys = [c[1] for c in candidates]
        return (float(np.median(xs)), float(np.median(ys)))

    # Fallback: geometric center of top 20% of ROI
    return (float(w) / 2.0, float(roi_h) * 0.15)


def _classify_segments(
    segs: List[Tuple[int, int, int, int]],
    mid_x: float,
    vp_x: float,
    vp_y: float,
    w: int,
    roi_h: int,
    cfg: HoughLaneConfig,
) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
    """Split segments into left and right lane candidates.

    A segment is considered for a side if:
    1. Its slope sign is consistent with converging toward the vanishing point.
    2. Its centroid-x falls in the expected corridor for that side.
    3. Its centroid-y is in the lower half of the ROI (not sky/horizon noise).
    """
    left_segs: List[Tuple[int, int, int, int]] = []
    right_segs: List[Tuple[int, int, int, int]] = []

    left_min = w * cfg.left_corridor_min
    left_max = w * cfg.left_corridor_max
    right_min = w * cfg.right_corridor_min
    right_max = w * cfg.right_corridor_max
    min_cy = roi_h * 0.15  # ignore segments in the very top (sky)
    min_bottom_y = roi_h * cfg.min_seg_bottom_y_frac
    gate_px = w * cfg.lane_gate_frac

    for x1, y1, x2, y2 in segs:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        slope = dy / dx
        abs_slope = abs(slope)
        if abs_slope < cfg.slope_min or abs_slope > cfg.slope_max:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if cy < min_cy:
            continue

        # Road-prior: keep only segments that extend sufficiently low into ROI.
        if max(y1, y2) < min_bottom_y:
            continue

        # Evaluate expected lane x near the lower endpoint where perspective
        # constraints are strongest.
        if y1 >= y2:
            y_ref = float(y1)
            x_ref = float(x1)
        else:
            y_ref = float(y2)
            x_ref = float(x2)

        exp_left = _expected_lane_x(roi_h, w, y_ref, left=True, cfg=cfg)
        exp_right = _expected_lane_x(roi_h, w, y_ref, left=False, cfg=cfg)

        # Classify by slope sign and corridor
        if slope < 0 and left_min <= cx <= left_max and abs(x_ref - exp_left) <= gate_px:
            left_segs.append((x1, y1, x2, y2))
        elif slope > 0 and right_min <= cx <= right_max and abs(x_ref - exp_right) <= gate_px:
            right_segs.append((x1, y1, x2, y2))

    return left_segs, right_segs


def _fit_poly_ransac(
    points: List[Tuple[float, float]],
    degree: int = 2,
    n_iter: int = 20,
    threshold_px: float = 6.0,
    min_inliers: int = 6,
    max_quad_coeff: float = 0.0010,
) -> Tuple[Optional[Tuple[float, ...]], int]:
    """Fit x = f(y) polynomial via RANSAC.

    Returns (coefficients_tuple, inlier_count).  coefficients are in
    descending degree order (same as numpy.polyfit).
    Returns (None, 0) when fewer than min_inliers points are available.
    """
    if len(points) < min_inliers:
        return None, 0

    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    n = len(points)
    sample_size = degree + 1

    # Precompute ys² once — reused every iteration
    ys2 = ys * ys
    thr2 = threshold_px * threshold_px

    best_inliers = 0
    best_mask: Optional[np.ndarray] = None

    for _ in range(n_iter):
        idx = random.sample(range(n), sample_size)
        try:
            coeffs = np.polyfit(ys[idx], xs[idx], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Direct evaluation (avoids np.polyval overhead)
        if degree == 2:
            diff = coeffs[0] * ys2 + coeffs[1] * ys + coeffs[2] - xs
        else:
            diff = coeffs[0] * ys + coeffs[1] - xs

        mask = diff * diff < thr2
        n_inliers = int(mask.sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_mask = mask

    if best_mask is None or best_inliers < min_inliers:
        return None, 0

    # Refit on all inliers for best accuracy
    try:
        final_coeffs = np.polyfit(ys[best_mask], xs[best_mask], degree)
    except (np.linalg.LinAlgError, ValueError):
        return None, 0

    # Sanity check: clamp unrealistic curvature that causes horizontal-looking lines.
    # If the quadratic term is too large, fall back to a linear fit on the same
    # inliers — a straight line is always safer than a wildly curved one.
    if degree == 2 and abs(float(final_coeffs[0])) > max_quad_coeff:
        try:
            lin = np.polyfit(ys[best_mask], xs[best_mask], 1)
            final_coeffs = np.array([0.0, float(lin[0]), float(lin[1])])
        except (np.linalg.LinAlgError, ValueError):
            return None, 0

    return tuple(float(c) for c in final_coeffs), best_inliers


def _sliding_window_search(
    edges: np.ndarray,
    poly: Tuple[float, float, float],
    roi_h: int,
    w: int,
    n_windows: int = 12,
    margin_px: int = 45,
    min_pixels: int = 4,
) -> List[Tuple[float, float]]:
    """Collect edge pixels that lie near the predicted polynomial curve.

    The ROI is divided into n_windows horizontal bands.  For each band,
    the polynomial is evaluated at the band's centre y to get an expected x,
    then all non-zero pixels within [x - margin_px, x + margin_px] are
    collected.  Bands with fewer than min_pixels hits are skipped to avoid
    pulling the fit off-course with noise.
    """
    points: List[Tuple[float, float]] = []
    band_h = max(1, roi_h // n_windows)

    for i in range(n_windows):
        y_top = i * band_h
        y_bot = min(roi_h, (i + 1) * band_h)
        y_mid = (y_top + y_bot) / 2.0

        a, b, c = poly
        x_center = a * y_mid * y_mid + b * y_mid + c
        x_lo = max(0, int(x_center - margin_px))
        x_hi = min(w, int(x_center + margin_px) + 1)

        band = edges[y_top:y_bot, x_lo:x_hi]
        ys_local, xs_local = np.where(band > 0)

        if len(ys_local) < min_pixels:
            continue

        for yl, xl in zip(ys_local, xs_local):
            points.append((float(xl + x_lo), float(yl + y_top)))

    return points


def _poly_eval_3(poly: Tuple[float, float, float], y: float) -> float:
    return poly[0] * y * y + poly[1] * y + poly[2]


def _expected_lane_x(
    roi_h: int,
    w: int,
    y: float,
    *,
    left: bool,
    cfg: HoughLaneConfig,
) -> float:
    if roi_h <= 1:
        return w * (cfg.expected_left_bottom_frac if left else cfg.expected_right_bottom_frac)
    t = max(0.0, min(1.0, y / float(roi_h - 1)))
    if left:
        top = w * cfg.expected_left_top_frac
        bot = w * cfg.expected_left_bottom_frac
    else:
        top = w * cfg.expected_right_top_frac
        bot = w * cfg.expected_right_bottom_frac
    return top + (bot - top) * t


def _segs_to_points(
    segs: List[Tuple[int, int, int, int]],
) -> List[Tuple[float, float]]:
    """Convert segment endpoint list to (x, y) point list."""
    pts: List[Tuple[float, float]] = []
    for x1, y1, x2, y2 in segs:
        pts.append((float(x1), float(y1)))
        pts.append((float(x2), float(y2)))
    return pts


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def detect_lanes_hough(
    edges: np.ndarray,
    roi_h: int,
    w: int,
    cfg: Optional[HoughLaneConfig] = None,
    tracker: Optional[LaneTracker] = None,
) -> HoughLaneResult:
    """Detect lane boundaries from a Canny edge map using Hough + RANSAC + Kalman.

    Parameters
    ----------
    edges : numpy.ndarray
        Canny edge map of the ROI (uint8, shape roi_h × w).
    roi_h : int
        Height of the ROI in pixels.
    w : int
        Width of the frame in pixels.
    cfg : HoughLaneConfig, optional
        Algorithm parameters; defaults to HoughLaneConfig().
    tracker : LaneTracker, optional
        Stateful Kalman tracker.  When provided, each call does:
        predict → (optional) update → return smoothed polynomial.
        When None the function is stateless (no temporal smoothing).

    Returns
    -------
    HoughLaneResult
    """
    _cfg = cfg or HoughLaneConfig()
    mid_x = w / 2.0

    # ---- 1. Multi-scale Hough ------------------------------------------------
    all_segs = _multi_scale_hough(edges, _cfg)

    # Quick pre-filter by slope so VP estimation uses clean segments
    slope_segs: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in all_segs:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        s = abs(dy / dx)
        if _cfg.slope_min <= s <= _cfg.slope_max:
            slope_segs.append((x1, y1, x2, y2))

    # Rough left/right split for VP estimation only (slope sign + x-side)
    rough_left = [(x1, y1, x2, y2) for x1, y1, x2, y2 in slope_segs
                  if (x2 - x1) != 0 and (y2 - y1) / (x2 - x1) < 0 and (x1 + x2) / 2 < mid_x]
    rough_right = [(x1, y1, x2, y2) for x1, y1, x2, y2 in slope_segs
                   if (x2 - x1) != 0 and (y2 - y1) / (x2 - x1) > 0 and (x1 + x2) / 2 > mid_x]

    # ---- 2. Vanishing point --------------------------------------------------
    vp_x, vp_y = _estimate_vanishing_point(rough_left, rough_right, roi_h, w, _cfg)

    # ---- 3. VP-guided segment classification --------------------------------
    left_segs, right_segs = _classify_segments(
        slope_segs, mid_x, vp_x, vp_y, w, roi_h, _cfg
    )

    # ---- 4 & 5. Per-side: gather points (sliding window or raw endpoints) ---
    used_sw = False

    def _gather_points(
        segs: List[Tuple[int, int, int, int]],
        kf: Optional[PolyKalmanFilter],
    ) -> List[Tuple[float, float]]:
        nonlocal used_sw
        if kf is not None and kf.is_initialized and kf.lost_frames < _cfg.kf_lost_frames_max:
            prev_poly = kf.get()
            if prev_poly is not None:
                pts = _sliding_window_search(
                    edges, prev_poly, roi_h, w,
                    _cfg.sw_n_windows, _cfg.sw_margin_px, _cfg.sw_min_pixels,
                )
                if len(pts) >= _cfg.ransac_min_inliers:
                    used_sw = True
                    return pts
        return _segs_to_points(segs)

    left_kf = tracker.left_kf if tracker is not None else None
    right_kf = tracker.right_kf if tracker is not None else None

    left_pts = _gather_points(left_segs, left_kf)
    right_pts = _gather_points(right_segs, right_kf)

    # ---- 6. RANSAC polynomial fit -------------------------------------------
    left_raw, left_inliers = _fit_poly_ransac(
        left_pts, _cfg.ransac_degree, _cfg.ransac_n_iter,
        _cfg.ransac_inlier_threshold_px, _cfg.ransac_min_inliers,
        _cfg.poly_max_quad_coeff,
    )
    right_raw, right_inliers = _fit_poly_ransac(
        right_pts, _cfg.ransac_degree, _cfg.ransac_n_iter,
        _cfg.ransac_inlier_threshold_px, _cfg.ransac_min_inliers,
        _cfg.poly_max_quad_coeff,
    )

    # Corridor check: left lane must end up left of centre,
    # right lane must end up right of centre. Both must be within the frame
    # (not vanishing to the very edge, which indicates a degenerate fit).
    if left_raw is not None:
        left_bx = _poly_eval_3(left_raw, float(roi_h - 1))
        if left_bx >= w * 0.72 or left_bx < -w * 0.10:
            left_raw, left_inliers = None, 0
    if right_raw is not None:
        right_bx = _poly_eval_3(right_raw, float(roi_h - 1))
        if right_bx <= w * 0.28 or right_bx > w * 1.10:
            right_raw, right_inliers = None, 0

    # ---- 7. Kalman predict → update -----------------------------------------
    def _kf_step(
        kf: Optional[PolyKalmanFilter],
        raw_poly: Optional[Tuple[float, ...]],
        raw_inliers: int,
        lost_max: int,
    ) -> Optional[Tuple[float, float, float]]:
        if kf is None:
            if raw_poly is not None and len(raw_poly) >= 3:
                return (float(raw_poly[0]), float(raw_poly[1]), float(raw_poly[2]))
            return None
        kf.predict()
        if raw_poly is not None and len(raw_poly) >= 3:
            # Innovation gating:
            # - Small innovation  (<= max_inn): accept normally
            # - Large innovation  (> max_inn but <= 3×): reject — noise spike
            # - Very large innovation (> 3×): KF has locked onto the WRONG feature
            #   (e.g. guardrail instead of lane). Reset and accept the new measurement.
            accept = True
            max_inn = _cfg.kf_max_innovation_px
            if max_inn > 0 and kf.is_initialized and roi_h > 1:
                pred = kf.get()
                if pred is not None:
                    y_bot = float(roi_h - 1)
                    pred_bx = _poly_eval_3(pred, y_bot)
                    raw_bx = (float(raw_poly[0]) * y_bot * y_bot
                              + float(raw_poly[1]) * y_bot
                              + float(raw_poly[2]))
                    inn = abs(raw_bx - pred_bx)
                    if inn > max_inn * 3.0:
                        # Filter has drifted far from RANSAC — reset and accept new
                        kf.reset()
                    elif inn > max_inn:
                        accept = False
            if accept:
                kf.update((float(raw_poly[0]), float(raw_poly[1]), float(raw_poly[2])))
        if kf.lost_frames > lost_max:
            kf.reset()
            return None
        est = kf.get()
        if est is None:
            return None

        # Adaptive follow: keep anti-drift behavior, but follow the road faster
        # when we have enough inliers and a clear measurement shift.
        if raw_poly is not None and len(raw_poly) >= 3:
            raw3 = (float(raw_poly[0]), float(raw_poly[1]), float(raw_poly[2]))
            y_bot = float(max(1, roi_h - 1))
            est_bx = _poly_eval_3(est, y_bot)
            raw_bx = _poly_eval_3(raw3, y_bot)
            shift = abs(raw_bx - est_bx)

            # More inliers and larger shift -> follow faster.
            follow = 0.15
            if raw_inliers >= 8:
                follow = 0.22
            if raw_inliers >= 12:
                follow = 0.30
            if shift > 20:
                follow += 0.06
            if shift > 40:
                follow += 0.06
            follow = max(0.10, min(0.40, follow))  # reduced from 0.70

            est = (
                (1.0 - follow) * est[0] + follow * raw3[0],
                (1.0 - follow) * est[1] + follow * raw3[1],
                (1.0 - follow) * est[2] + follow * raw3[2],
            )

        return est

    left_smoothed = _kf_step(left_kf, left_raw, left_inliers, _cfg.kf_lost_frames_max)
    right_smoothed = _kf_step(right_kf, right_raw, right_inliers, _cfg.kf_lost_frames_max)

    # ---- 7b. Post-KF geometry sanity check ---------------------------------
    # If a smoothed poly puts the lane line on the WRONG SIDE of the frame,
    # the KF has drifted onto guardrail / noise. Reset it.
    # Bounds allow partial off-screen lines (near-camera wide lanes).
    if left_smoothed is not None:
        lbx = _poly_eval_3(left_smoothed, float(roi_h - 1))
        if lbx > w * 0.70:   # left line drifted to right half
            if left_kf is not None:
                left_kf.reset()
            left_smoothed = None
            left_inliers = 0

    if right_smoothed is not None:
        rbx = _poly_eval_3(right_smoothed, float(roi_h - 1))
        if rbx < w * 0.30:   # right line drifted to left half
            if right_kf is not None:
                right_kf.reset()
            right_smoothed = None
            right_inliers = 0

    # Crossed lanes: one is definitely wrong — reset lower-confidence side.
    if left_smoothed is not None and right_smoothed is not None:
        lbx = _poly_eval_3(left_smoothed, float(roi_h - 1))
        rbx = _poly_eval_3(right_smoothed, float(roi_h - 1))
        lane_w = rbx - lbx
        # Crossed OR impossibly wide (>85% frame) OR impossibly narrow (<20%)
        bad_width = (lane_w <= 0) or (lane_w > w * 0.85) or (lane_w < w * 0.20)
        if bad_width:
            exp_lbx = w * 0.24
            exp_rbx = w * 0.76
            if abs(lbx - exp_lbx) <= abs(rbx - exp_rbx):
                # right side is more deviated
                if right_kf is not None:
                    right_kf.reset()
                right_smoothed = None
                right_inliers = 0
                if tracker is not None:
                    tracker.right_last_bx = -1.0
            else:
                if left_kf is not None:
                    left_kf.reset()
                left_smoothed = None
                left_inliers = 0
                if tracker is not None:
                    tracker.left_last_bx = -1.0

    # ---- 7c. Slope direction validity ------------------------------------------
    # In dashcam perspective: left line dx/dy < 0 (goes left as y increases),
    # right line dx/dy > 0. A nearly vertical line (slope ≈ 0) is always noise.
    # Threshold -0.25/+0.25: any real lane line in perspective moves ≥0.25 px/py.
    if left_smoothed is not None:
        slope_bot = 2.0 * left_smoothed[0] * (roi_h - 1) + left_smoothed[1]
        if slope_bot > -0.25:  # should be clearly negative for left line
            if left_kf is not None:
                left_kf.reset()
            left_smoothed = None
            left_inliers = 0

    if right_smoothed is not None:
        slope_bot = 2.0 * right_smoothed[0] * (roi_h - 1) + right_smoothed[1]
        if slope_bot < 0.25:  # should be clearly positive for right line
            if right_kf is not None:
                right_kf.reset()
            right_smoothed = None
            right_inliers = 0

    # ---- 7d. Per-frame movement clamp (anti-teleport) --------------------------
    # Limit how many pixels the line can jump between consecutive frames.
    # Adjusting only the c-coefficient keeps the slope intact.
    _max_move = 25.0
    _max_move_confident = 50.0

    if tracker is not None and left_smoothed is not None and tracker.left_last_bx > 0:
        lbx = _poly_eval_3(left_smoothed, float(roi_h - 1))
        delta = lbx - tracker.left_last_bx
        limit = 15.0  # px/frame — tight enough to prevent single-frame teleport
        if abs(delta) > limit:
            direction = 1.0 if delta > 0 else -1.0
            new_bx = tracker.left_last_bx + limit * direction
            left_smoothed = (left_smoothed[0], left_smoothed[1],
                             left_smoothed[2] + (new_bx - lbx))

    if tracker is not None and right_smoothed is not None and tracker.right_last_bx > 0:
        rbx = _poly_eval_3(right_smoothed, float(roi_h - 1))
        delta = rbx - tracker.right_last_bx
        limit = 15.0
        if abs(delta) > limit:
            direction = 1.0 if delta > 0 else -1.0
            new_bx = tracker.right_last_bx + limit * direction
            right_smoothed = (right_smoothed[0], right_smoothed[1],
                              right_smoothed[2] + (new_bx - rbx))

    # Update last_bx for next frame
    if tracker is not None:
        tracker.left_last_bx = (
            _poly_eval_3(left_smoothed, float(roi_h - 1))
            if left_smoothed is not None else -1.0
        )
        tracker.right_last_bx = (
            _poly_eval_3(right_smoothed, float(roi_h - 1))
            if right_smoothed is not None else -1.0
        )

    # ---- 8. Confidence scoring ----------------------------------------------
    def _score(inliers: int, kf: Optional[PolyKalmanFilter]) -> float:
        base = min(1.0, inliers / max(1.0, _cfg.confidence_inlier_norm))
        if kf is not None and kf.is_initialized:
            inn = kf.innovation()
            penalty = math.exp(-inn / max(1.0, _cfg.confidence_innovation_scale))
            return base * (0.6 + 0.4 * penalty)
        return base

    left_conf = _score(left_inliers, left_kf) if left_smoothed is not None else 0.0
    right_conf = _score(right_inliers, right_kf) if right_smoothed is not None else 0.0

    # If only prediction (no fresh inliers) but KF is still valid,
    # carry forward a decayed version of the previous confidence.
    if left_raw is None and left_smoothed is not None:
        left_conf = max(0.0, 0.4 - 0.05 * (left_kf.lost_frames if left_kf else 0))
    if right_raw is None and right_smoothed is not None:
        right_conf = max(0.0, 0.4 - 0.05 * (right_kf.lost_frames if right_kf else 0))

    return HoughLaneResult(
        left_poly=left_smoothed,
        right_poly=right_smoothed,
        left_conf=left_conf,
        right_conf=right_conf,
        left_inliers=left_inliers,
        right_inliers=right_inliers,
        vp_x=vp_x,
        vp_y=vp_y,
        used_sliding_window=used_sw,
    )

