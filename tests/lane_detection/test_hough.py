"""Tests for adas.lane_detection.hough."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from adas.lane_detection.hough import (
    HoughLaneConfig,
    HoughLaneResult,
    LaneTracker,
    PolyKalmanFilter,
    _classify_segments,
    _estimate_vanishing_point,
    _fit_poly_ransac,
    _multi_scale_hough,
    _poly_eval_3,
    _sliding_window_search,
    detect_lanes_hough,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_edge_frame_with_lanes(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic Canny-like edge map with clear left and right lane lines."""
    edges = np.zeros((h, w), dtype=np.uint8)
    # Left boundary: x = -0.5*y + 60 (slope ~-0.5, y in ROI space)
    for y in range(h):
        x = int(-0.5 * y + 60)
        if 0 <= x < w:
            edges[y, max(0, x - 1):min(w, x + 2)] = 255
    # Right boundary: x = 0.5*y + 220 (slope ~+0.5)
    for y in range(h):
        x = int(0.5 * y + 220)
        if 0 <= x < w:
            edges[y, max(0, x - 1):min(w, x + 2)] = 255
    return edges


def _make_lane_bgr(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic BGR frame with white lane lines on dark road."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[int(h * 0.38):, :] = 40  # dark asphalt
    for y in range(int(h * 0.38), h):
        frac = (y - int(h * 0.38)) / (h - int(h * 0.38))
        xl = int(w * 0.08 + frac * w * 0.10)
        xr = int(w * 0.92 - frac * w * 0.10)
        if 0 <= xl < w:
            frame[y, max(0, xl - 2):min(w, xl + 3)] = 255
        if 0 <= xr < w:
            frame[y, max(0, xr - 2):min(w, xr + 3)] = 255
    return frame


# ---------------------------------------------------------------------------
# PolyKalmanFilter tests
# ---------------------------------------------------------------------------

class TestPolyKalmanFilter:
    def test_not_initialized_by_default(self):
        kf = PolyKalmanFilter()
        assert not kf.is_initialized
        assert kf.get() is None

    def test_initializes_on_first_update(self):
        kf = PolyKalmanFilter()
        kf.update((0.01, -0.5, 100.0))
        assert kf.is_initialized
        result = kf.get()
        assert result is not None
        assert abs(result[0] - 0.01) < 1e-6
        assert abs(result[1] - (-0.5)) < 1e-6
        assert abs(result[2] - 100.0) < 1e-6

    def test_converges_to_known_poly(self):
        """Feeding the same measurement repeatedly should converge near it."""
        kf = PolyKalmanFilter(process_noise=0.001, meas_noise=5.0)
        target = (0.001, -0.4, 80.0)
        for _ in range(30):
            kf.predict()
            kf.update(target)
        result = kf.get()
        assert result is not None
        assert abs(result[0] - target[0]) < 0.05
        assert abs(result[1] - target[1]) < 1.0
        assert abs(result[2] - target[2]) < 5.0

    def test_prediction_preserves_estimate(self):
        """Predicting without update should keep state (may drift slightly)."""
        kf = PolyKalmanFilter()
        kf.update((0.0, -0.5, 100.0))
        for _ in range(3):
            kf.predict()
        assert kf.is_initialized
        assert kf.get() is not None
        assert kf.lost_frames == 3

    def test_reset_clears_state(self):
        kf = PolyKalmanFilter()
        kf.update((0.01, -0.3, 50.0))
        kf.reset()
        assert not kf.is_initialized
        assert kf.get() is None
        assert kf.lost_frames == 0

    def test_lost_frames_increments_on_predict(self):
        kf = PolyKalmanFilter()
        kf.update((0.0, 0.0, 60.0))
        assert kf.lost_frames == 0
        kf.predict()
        assert kf.lost_frames == 1
        kf.predict()
        assert kf.lost_frames == 2

    def test_lost_frames_resets_on_update(self):
        kf = PolyKalmanFilter()
        kf.update((0.0, 0.0, 60.0))
        kf.predict()
        kf.predict()
        kf.update((0.0, 0.0, 60.0))
        assert kf.lost_frames == 0


# ---------------------------------------------------------------------------
# RANSAC polynomial fitting tests
# ---------------------------------------------------------------------------

class TestRANSACFit:
    def test_fits_clean_line(self):
        """20 perfectly collinear points should yield a near-perfect fit."""
        pts = [(float(0.5 * y + 60.0), float(y)) for y in range(20)]
        coeffs, n = _fit_poly_ransac(pts, degree=2, n_iter=40,
                                     threshold_px=3.0, min_inliers=6)
        assert coeffs is not None
        assert n >= 18
        # Evaluate at y=10 — should be close to 0.5*10+60=65
        y_test = 10.0
        x_pred = _poly_eval_3(coeffs, y_test)  # type: ignore[arg-type]
        assert abs(x_pred - 65.0) < 3.0

    def test_robust_to_outliers(self):
        """15 inliers + 5 heavy outliers — should still find the inlier line."""
        random.seed(42)
        inlier_pts = [(float(0.5 * y + 60.0), float(y)) for y in range(15)]
        outlier_pts = [(float(random.uniform(200.0, 300.0)), float(random.uniform(0, 100)))
                       for _ in range(5)]
        pts = inlier_pts + outlier_pts
        random.shuffle(pts)
        coeffs, n = _fit_poly_ransac(pts, degree=2, n_iter=80,
                                     threshold_px=8.0, min_inliers=6)
        assert coeffs is not None
        assert n >= 10
        # Inlier line at y=10 → x≈65
        x_pred = _poly_eval_3(coeffs, 10.0)  # type: ignore[arg-type]
        assert abs(x_pred - 65.0) < 10.0

    def test_insufficient_points_returns_none(self):
        pts = [(10.0, float(i)) for i in range(4)]
        coeffs, n = _fit_poly_ransac(pts, degree=2, n_iter=20,
                                     threshold_px=8.0, min_inliers=6)
        assert coeffs is None
        assert n == 0

    def test_returns_three_coefficients_for_degree_2(self):
        pts = [(float(0.3 * y + 50.0), float(y)) for y in range(20)]
        coeffs, _ = _fit_poly_ransac(pts, degree=2, n_iter=40,
                                     threshold_px=3.0, min_inliers=6)
        assert coeffs is not None
        assert len(coeffs) == 3


# ---------------------------------------------------------------------------
# Sliding window tests
# ---------------------------------------------------------------------------

class TestSlidingWindowSearch:
    def test_finds_points_on_known_line(self):
        """Edge pixels placed exactly on a linear poly should all be found."""
        h, w = 120, 200
        edges = np.zeros((h, w), dtype=np.uint8)
        # Linear poly: x = 0.3*y + 50
        poly = (0.0, 0.3, 50.0)
        for y in range(h):
            x = int(0.3 * y + 50.0)
            if 0 <= x < w:
                edges[y, x] = 255
        pts = _sliding_window_search(edges, poly, h, w, n_windows=10,
                                     margin_px=20, min_pixels=1)
        assert len(pts) >= 8  # should find most bands

    def test_empty_edges_returns_empty(self):
        edges = np.zeros((80, 160), dtype=np.uint8)
        poly = (0.0, 0.4, 40.0)
        pts = _sliding_window_search(edges, poly, 80, 160)
        assert pts == []

    def test_no_crash_on_single_band(self):
        edges = np.zeros((5, 100), dtype=np.uint8)
        edges[2, 50] = 255
        poly = (0.0, 0.0, 50.0)
        pts = _sliding_window_search(edges, poly, 5, 100, n_windows=2, margin_px=10, min_pixels=1)
        assert isinstance(pts, list)


# ---------------------------------------------------------------------------
# Vanishing point tests
# ---------------------------------------------------------------------------

class TestVanishingPoint:
    def test_symmetric_converging_lines(self):
        """Left and right lines converging at (160, 20) should give VP near there."""
        # Left: from (20, 100) to (80, 40) — extends toward (160, -20) approx
        # Right: from (300, 100) to (240, 40)
        left_segs = [(20, 100, 80, 40)]
        right_segs = [(300, 100, 240, 40)]
        cfg = HoughLaneConfig(vp_max_y_frac=0.9, vp_min_candidates=1)
        vp_x, vp_y = _estimate_vanishing_point(left_segs, right_segs, 120, 320, cfg)
        # VP should be somewhere near horizontal center, above midpoint
        assert 100 < vp_x < 220
        assert vp_y < 60

    def test_fallback_when_no_valid_candidates(self):
        """Parallel lines produce no intersection → fallback to w/2, roi_h*0.15."""
        # Horizontal segments (slope ≈ 0) — no intersection within ROI
        left_segs = [(10, 50, 100, 50)]
        right_segs = [(150, 50, 250, 50)]
        cfg = HoughLaneConfig(vp_max_y_frac=0.2, vp_min_candidates=3)
        vp_x, vp_y = _estimate_vanishing_point(left_segs, right_segs, 200, 320, cfg)
        assert abs(vp_x - 160.0) < 1e-6
        assert abs(vp_y - 30.0) < 1e-6  # 200 * 0.15


# ---------------------------------------------------------------------------
# Full detect_lanes_hough tests
# ---------------------------------------------------------------------------

class TestDetectLanesHough:
    def test_detects_on_synthetic_edge_map(self):
        """Clear synthetic lanes should be detected with confidence > 0."""
        edges = _make_edge_frame_with_lanes(h=200, w=320)
        result = detect_lanes_hough(edges, 200, 320)
        assert isinstance(result, HoughLaneResult)
        # At least one side should be detected on a clear synthetic frame
        total_inliers = result.left_inliers + result.right_inliers
        assert total_inliers > 0 or (result.left_poly is not None or result.right_poly is not None)

    def test_empty_frame_returns_empty_result(self):
        edges = np.zeros((200, 320), dtype=np.uint8)
        result = detect_lanes_hough(edges, 200, 320)
        assert isinstance(result, HoughLaneResult)
        assert result.left_poly is None
        assert result.right_poly is None
        assert result.left_conf == 0.0
        assert result.right_conf == 0.0

    def test_tracker_provides_smoothing(self):
        """Same frame fed 20× through a LaneTracker should yield stable bottom-x."""
        edges = _make_edge_frame_with_lanes(h=200, w=320)
        cfg = HoughLaneConfig()
        tracker = LaneTracker(cfg)
        roi_h = 200
        w = 320

        left_bx_vals = []
        right_bx_vals = []
        for _ in range(20):
            result = detect_lanes_hough(edges, roi_h, w, cfg, tracker)
            if result.left_poly is not None:
                a, b, c = result.left_poly
                left_bx_vals.append(a * (roi_h - 1) ** 2 + b * (roi_h - 1) + c)
            if result.right_poly is not None:
                a, b, c = result.right_poly
                right_bx_vals.append(a * (roi_h - 1) ** 2 + b * (roi_h - 1) + c)

        if len(left_bx_vals) >= 5:
            assert np.std(left_bx_vals[-10:]) < 5.0, \
                f"Left bottom-x std too high: {np.std(left_bx_vals[-10:]):.2f}"
        if len(right_bx_vals) >= 5:
            assert np.std(right_bx_vals[-10:]) < 5.0, \
                f"Right bottom-x std too high: {np.std(right_bx_vals[-10:]):.2f}"

    def test_tracker_reset_clears_state(self):
        tracker = LaneTracker()
        edges = _make_edge_frame_with_lanes(h=200, w=320)
        for _ in range(5):
            detect_lanes_hough(edges, 200, 320, tracker=tracker)
        tracker.reset()
        assert not tracker.left_kf.is_initialized
        assert not tracker.right_kf.is_initialized

    def test_kalman_predicts_through_missing_frames(self):
        """After tracking is established, blank frames should still yield polys via KF prediction."""
        edges_lane = _make_edge_frame_with_lanes(h=200, w=320)
        edges_blank = np.zeros((200, 320), dtype=np.uint8)
        cfg = HoughLaneConfig(kf_lost_frames_max=5)
        tracker = LaneTracker(cfg)

        # Establish tracking
        for _ in range(8):
            detect_lanes_hough(edges_lane, 200, 320, cfg, tracker)

        # Feed 3 blank frames — KF should still predict
        results = [detect_lanes_hough(edges_blank, 200, 320, cfg, tracker)
                   for _ in range(3)]
        # At least one side should still have a prediction (not None)
        has_any = any(r.left_poly is not None or r.right_poly is not None
                      for r in results)
        assert has_any

    def test_result_is_frozen_dataclass(self):
        result = HoughLaneResult()
        with pytest.raises((AttributeError, TypeError)):
            result.left_conf = 0.9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LaneTracker tests
# ---------------------------------------------------------------------------

class TestLaneTracker:
    def test_default_construction(self):
        tracker = LaneTracker()
        assert not tracker.left_kf.is_initialized
        assert not tracker.right_kf.is_initialized

    def test_reset_clears_both_kfs(self):
        tracker = LaneTracker()
        tracker.left_kf.update((0.0, 0.5, 100.0))
        tracker.right_kf.update((0.0, 0.5, 200.0))
        tracker.reset()
        assert not tracker.left_kf.is_initialized
        assert not tracker.right_kf.is_initialized
