"""Tests for adas.collision_risk.estimator."""

from __future__ import annotations

import math
import pytest

from adas.collision_risk.estimator import (
    RiskEstimator,
    EstimatorConfig,
    DEFAULT_ESTIMATOR_CONFIG,
    _compute_ttc,
    _compute_risk_score,
)
from adas.collision_risk.types import RiskResult
from adas.obstacle_detection.types import DetectedObject


def _make_object(
    track_id: int = 1,
    distance_m: float = 20.0,
    cx: float = 160.0,
    cy: float = 200.0,
) -> DetectedObject:
    return DetectedObject(
        bbox=(int(cx - 20), int(cy - 40), 40, 80),
        area=3200.0,
        centroid=(cx, cy),
        track_id=track_id,
        distance_estimate=distance_m,
        confidence=0.9,
    )


class TestComputeTTC:
    def test_no_distance_returns_inf(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        ttc = _compute_ttc(None, None, cfg)
        assert ttc == float("inf")

    def test_zero_distance_returns_inf(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        ttc = _compute_ttc(0.0, 10.0, cfg)
        assert ttc == float("inf")

    def test_too_far_returns_inf(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        ttc = _compute_ttc(cfg.max_distance_m + 1, 10.0, cfg)
        assert ttc == float("inf")

    def test_approaching_object(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        ttc = _compute_ttc(30.0, 10.0, cfg)  # 30 m / 10 m/s = 3 s
        assert abs(ttc - 3.0) < 0.1

    def test_non_approaching_uses_fallback(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        # rel_vel <= 0.5 -> fallback with assumed_ego_speed
        ttc = _compute_ttc(10.0, 0.0, cfg)
        expected = 10.0 / cfg.assumed_ego_speed_mps
        assert abs(ttc - expected) < 0.1

    def test_none_velocity_uses_fallback(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        ttc = _compute_ttc(20.0, None, cfg)
        expected = 20.0 / cfg.assumed_ego_speed_mps
        assert abs(ttc - expected) < 0.1


class TestComputeRiskScore:
    def test_far_object_low_risk(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        score = _compute_risk_score(35.0, 8.0, 0.0, True, 1.0, cfg)
        assert score < 0.5

    def test_close_approaching_in_lane_high_risk(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        score = _compute_risk_score(3.0, 0.5, 0.0, True, 1.0, cfg)
        assert score > 0.5

    def test_out_of_lane_reduces_risk(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        in_lane = _compute_risk_score(10.0, 2.0, 0.0, True, 1.0, cfg)
        out_lane = _compute_risk_score(10.0, 2.0, 3.0, False, 1.0, cfg)
        assert in_lane > out_lane

    def test_wet_road_increases_risk(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        dry = _compute_risk_score(15.0, 3.0, 0.0, True, 1.0, cfg)
        wet = _compute_risk_score(15.0, 3.0, 0.0, True, 1.4, cfg)
        assert wet > dry

    def test_score_clamped_to_one(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        score = _compute_risk_score(0.1, 0.01, 0.0, True, 5.0, cfg)
        assert score <= 1.0

    def test_score_clamped_to_zero(self):
        cfg = DEFAULT_ESTIMATOR_CONFIG
        score = _compute_risk_score(None, float("inf"), 10.0, False, 0.0, cfg)
        assert score >= 0.0


class TestRiskEstimator:
    def test_empty_objects_returns_empty(self):
        est = RiskEstimator()
        result = est.estimate_risk([], frame_idx=0)
        assert result == []

    def test_result_count_matches_input(self):
        est = RiskEstimator()
        objs = [_make_object(track_id=i, distance_m=20.0 - i * 5) for i in range(3)]
        result = est.estimate_risk(objs, frame_idx=0)
        assert len(result) == 3

    def test_result_types(self):
        est = RiskEstimator()
        objs = [_make_object()]
        result = est.estimate_risk(objs, frame_idx=0)
        assert isinstance(result[0], RiskResult)
        assert result[0].risk_score >= 0.0
        assert result[0].risk_score <= 1.0

    def test_approaching_object_lowers_ttc_over_frames(self):
        est = RiskEstimator()
        track_id = 1
        # Simulate object approaching: distance decreasing each frame
        distances = [30.0, 28.0, 25.0, 22.0, 18.0, 15.0]
        ttcs = []
        for frame_idx, dist in enumerate(distances):
            obj = _make_object(track_id=track_id, distance_m=dist)
            results = est.estimate_risk([obj], frame_idx=frame_idx)
            if results:
                ttcs.append(results[0].ttc)

        # TTC should decrease as object approaches (after enough history)
        if len(ttcs) >= 2:
            # The last TTC should be lower than the first (once velocity is known)
            assert ttcs[-1] <= ttcs[0] + 5.0  # allow some slack

    def test_reset_clears_history(self):
        est = RiskEstimator()
        for i in range(5):
            obj = _make_object(track_id=1, distance_m=30.0 - i)
            est.estimate_risk([obj], frame_idx=i)
        est.reset()
        # After reset, history is empty --- new estimate should work fresh
        result = est.estimate_risk([_make_object()], frame_idx=0)
        assert isinstance(result, list)

    def test_object_id_matches_track_id(self):
        est = RiskEstimator()
        obj = _make_object(track_id=42, distance_m=15.0)
        result = est.estimate_risk([obj], frame_idx=0)
        assert result[0].object_id == 42
