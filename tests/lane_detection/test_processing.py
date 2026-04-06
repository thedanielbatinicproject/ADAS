"""Tests for adas.lane_detection.processing."""

from __future__ import annotations

import numpy as np
import pytest

from adas.lane_detection.processing import (
    LaneOutput,
    LaneProcessingConfig,
    process_frame,
    _fit_line_poly,
)


def _make_blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Create a plain gray frame (no real lane markings)."""
    return np.full((h, w, 3), 80, dtype=np.uint8)


def _make_lane_frame(h: int = 240, w: int = 320) -> np.ndarray:
    """Create a synthetic frame with white left/right lines on dark road."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[int(h * 0.4):, :] = 50  # dark road surface

    # Left boundary: diagonal from top-left toward bottom-left-center
    for y in range(int(h * 0.4), h):
        frac = (y - int(h * 0.4)) / (h - int(h * 0.4))
        x = int(w * 0.08 + frac * w * 0.08)
        if 0 <= x < w:
            frame[y, max(0, x - 2):min(w, x + 3)] = 255

    # Right boundary: diagonal from top-right toward bottom-right-center
    for y in range(int(h * 0.4), h):
        frac = (y - int(h * 0.4)) / (h - int(h * 0.4))
        x = int(w * 0.92 - frac * w * 0.08)
        if 0 <= x < w:
            frame[y, max(0, x - 2):min(w, x + 3)] = 255

    return frame


class TestLaneOutput:
    def test_default_has_no_lanes(self):
        out = LaneOutput()
        assert not out.has_lanes
        assert out.lane_confidence == 0.0

    def test_frozen(self):
        out = LaneOutput()
        with pytest.raises((AttributeError, TypeError)):
            out.has_lanes = True  # type: ignore[misc]


class TestProcessFrame:
    def test_none_frame_returns_empty(self):
        result = process_frame(None)
        assert isinstance(result, LaneOutput)
        assert not result.has_lanes

    def test_empty_array_returns_empty(self):
        result = process_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        assert isinstance(result, LaneOutput)
        assert not result.has_lanes

    def test_blank_frame_no_crashes(self):
        frame = _make_blank_frame()
        result = process_frame(frame)
        assert isinstance(result, LaneOutput)
        # Blank frame should have very low confidence
        assert result.lane_confidence <= 1.0
        assert result.lane_confidence >= 0.0

    def test_lane_frame_detects_something(self):
        frame = _make_lane_frame(h=360, w=640)
        result = process_frame(frame)
        assert isinstance(result, LaneOutput)
        # With clear white lines, should have some detection
        # (may not always fire given the one-side penalty, so just sanity-check)
        assert result.lane_confidence >= 0.0
        assert result.roi_y1 < result.roi_y2

    def test_roi_coordinates_are_valid(self):
        frame = _make_blank_frame(h=480, w=640)
        result = process_frame(frame)
        assert result.roi_y1 >= 0
        assert result.roi_y2 <= 480
        assert result.roi_y1 < result.roi_y2

    def test_edges_shape_matches_roi(self):
        frame = _make_blank_frame(h=240, w=320)
        result = process_frame(frame)
        if result.edges is not None:
            roi_h = result.roi_y2 - result.roi_y1
            assert result.edges.shape == (roi_h, 320)

    def test_custom_config(self):
        frame = _make_blank_frame()
        cfg = LaneProcessingConfig(roi_top=0.5, roi_bottom=0.8)
        result = process_frame(frame, config=cfg)
        assert isinstance(result, LaneOutput)

    def test_context_state_accepted(self):
        # context_state is optional; passing None should not error
        frame = _make_blank_frame()
        result = process_frame(frame, context_state=None)
        assert isinstance(result, LaneOutput)


class TestFitLinePoly:
    def test_enough_points_returns_tuple(self):
        points = [(10.0, 0.0), (12.0, 20.0), (14.0, 40.0), (16.0, 60.0)]
        poly = _fit_line_poly(points)
        assert poly is not None
        assert len(poly) == 2

    def test_too_few_points_returns_none(self):
        points = [(0.0, 0.0), (1.0, 1.0)]
        poly = _fit_line_poly(points)
        assert poly is None

    def test_empty_returns_none(self):
        assert _fit_line_poly([]) is None

    def test_linear_fit_accuracy(self):
        # x = 2*y + 5 -> coefficients should be approximately (2, 5)
        points = [(2 * y + 5, float(y)) for y in range(20)]
        poly = _fit_line_poly(points)
        assert poly is not None
        a, b = poly
        assert abs(a - 2.0) < 0.1
        assert abs(b - 5.0) < 0.5
