"""Tests for adas.ui.overlays and adas.ui.dashboard (no GUI event loop)."""

from __future__ import annotations

import numpy as np
import pytest

from adas.ui.overlays import draw_obstacles, _put_label
from adas.ui.dashboard import draw_stats_panel, draw_stats_overlay, _build_stat_lines
from adas.obstacle_detection.types import DetectedObject


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# draw_obstacles
# ---------------------------------------------------------------------------

class TestDrawObstacles:
    def test_no_obstacles_returns_copy(self):
        frame = _frame()
        result = draw_obstacles(frame, [])
        assert result is not frame
        assert result.shape == frame.shape
        np.testing.assert_array_equal(result, frame)

    def test_one_obstacle_modifies_frame(self):
        frame = _frame(h=300, w=400)
        obj = DetectedObject(bbox=(50, 50, 60, 80), centroid=(80.0, 90.0), track_id=1)
        result = draw_obstacles(frame, [obj])
        assert result.shape == frame.shape
        # Frame should differ from original (box was drawn)
        assert not np.array_equal(result, frame)

    def test_none_frame_returns_none(self):
        result = draw_obstacles(None, [])
        assert result is None

    def test_out_of_bounds_bbox_no_crash(self):
        frame = _frame()
        obj = DetectedObject(bbox=(-10, -10, 500, 500), centroid=(0.0, 0.0), track_id=2)
        result = draw_obstacles(frame, [obj])
        assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# draw_stats_panel
# ---------------------------------------------------------------------------

class TestDrawStatsPanel:
    def test_wider_than_original(self):
        frame = _frame(h=200, w=300)
        stats = {"mode": "NORMAL_MARKED", "fps": 30.0}
        result = draw_stats_panel(frame, stats)
        # Panel is appended to the right, so width increases
        assert result.shape[1] > frame.shape[1]
        assert result.shape[0] == frame.shape[0]

    def test_left_side(self):
        frame = _frame(h=200, w=300)
        stats = {"mode": "TEST"}
        result = draw_stats_panel(frame, stats, side="left")
        assert result.shape[1] > frame.shape[1]

    def test_none_frame_returns_none(self):
        result = draw_stats_panel(None, {})
        assert result is None

    def test_empty_stats_no_crash(self):
        frame = _frame()
        result = draw_stats_panel(frame, {})
        assert result is not None

    def test_full_stats_no_crash(self):
        frame = _frame(h=400, w=600)
        stats = {
            "mode": "NORMAL_MARKED",
            "weather": "clear",
            "light": "day",
            "lane_state": "has_lanes",
            "road_surface": "asphalt_dry",
            "braking_mult": 1.0,
            "visibility_conf": 0.85,
            "ttc": 3.5,
            "risk_score": 0.25,
            "action": "NONE",
            "fps": 29.7,
            "frame_idx": 42,
            "total_frames": 500,
            "annotation_label": "normal",
        }
        result = draw_stats_panel(frame, stats)
        assert result.shape[0] == frame.shape[0]


# ---------------------------------------------------------------------------
# draw_stats_overlay
# ---------------------------------------------------------------------------

class TestDrawStatsOverlay:
    def test_returns_same_array(self):
        frame = _frame()
        original_id = id(frame)
        result = draw_stats_overlay(frame, {"mode": "test"})
        # overlay() returns the same array (modified in-place)
        assert id(result) == original_id

    def test_none_frame_returns_none(self):
        result = draw_stats_overlay(None, {})
        assert result is None


# ---------------------------------------------------------------------------
# _build_stat_lines
# ---------------------------------------------------------------------------

class TestBuildStatLines:
    def test_returns_list_of_tuples(self):
        lines = _build_stat_lines({"mode": "TEST", "fps": 30.0})
        assert isinstance(lines, list)
        for item in lines:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_action_colors_differ(self):
        from adas.ui.dashboard import _BRAKE_COLOR, _WARN_COLOR, _OK_COLOR

        lines_brake = _build_stat_lines({"action": "BRAKE"})
        lines_warn = _build_stat_lines({"action": "WARN"})
        lines_none = _build_stat_lines({"action": "NONE"})

        def get_action_color(lines):
            for text, color in lines:
                if text.startswith("Action:"):
                    return color
            return None

        assert get_action_color(lines_brake) == _BRAKE_COLOR
        assert get_action_color(lines_warn) == _WARN_COLOR
        assert get_action_color(lines_none) == _OK_COLOR
