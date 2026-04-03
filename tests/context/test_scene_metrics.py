"""Tests for adas.context.scene_metrics - metrics computation & visibility."""

import numpy as np
from adas.context.scene_metrics import compute_scene_metrics, estimate_visibility
from adas.context.types import SceneMetrics, VisibilityEstimate


# ------------------------------------------------------- compute_scene_metrics


class TestComputeSceneMetrics:
    def test_black_frame(self, black_frame):
        m = compute_scene_metrics(black_frame)
        assert isinstance(m, SceneMetrics)
        assert m.brightness_mean < 1.0
        assert m.contrast_std < 1.0
        assert m.edge_density < 0.01
        assert m.glare_score == 0.0

    def test_white_frame(self, white_frame):
        m = compute_scene_metrics(white_frame)
        assert m.brightness_mean > 250.0
        assert m.glare_score > 0.5

    def test_gradient_has_contrast(self, gradient_frame):
        m = compute_scene_metrics(gradient_frame)
        assert m.contrast_std > 10.0
        assert m.brightness_mean > 50.0

    def test_noisy_frame_has_edges(self, noisy_frame):
        m = compute_scene_metrics(noisy_frame)
        assert m.edge_density > 0.0

    def test_edges_frame_high_edge_density(self, edges_frame):
        m = compute_scene_metrics(edges_frame)
        assert m.edge_density > 0.01

    def test_empty_frame(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert compute_scene_metrics(empty) == SceneMetrics()

    def test_none_frame(self):
        assert compute_scene_metrics(None) == SceneMetrics()

    def test_roi_isolates_region(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[50:, :, :] = 255  # bottom half white
        m_top = compute_scene_metrics(frame, roi=(0, 0, 100, 50))
        m_bot = compute_scene_metrics(frame, roi=(0, 50, 100, 100))
        assert m_top.brightness_mean < 1.0
        assert m_bot.brightness_mean > 250.0

    def test_grayscale_input(self):
        gray = np.full((100, 100), 128, dtype=np.uint8)
        m = compute_scene_metrics(gray)
        assert m.brightness_mean > 100.0


# -------------------------------------------------------- estimate_visibility


class TestEstimateVisibility:
    def test_black_frame_is_night(self, black_frame):
        m = compute_scene_metrics(black_frame)
        v = estimate_visibility(m)
        assert isinstance(v, VisibilityEstimate)
        assert v.is_night is True
        assert v.is_degraded is True

    def test_white_frame_glare(self, white_frame):
        m = compute_scene_metrics(white_frame)
        v = estimate_visibility(m)
        assert v.is_glare is True

    def test_gradient_has_some_visibility(self, gradient_frame):
        m = compute_scene_metrics(gradient_frame)
        v = estimate_visibility(m)
        assert v.confidence > 0.0

    def test_confidence_in_range(self, noisy_frame):
        m = compute_scene_metrics(noisy_frame)
        v = estimate_visibility(m)
        assert 0.0 <= v.confidence <= 1.0
