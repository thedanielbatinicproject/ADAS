"""Tests for adas.context.lane_state - availability, smoothing, stability."""

from adas.context.lane_state import compute_lane_state
from adas.context.types import LaneDetectionInput, LaneAvailability
from adas.context.defaults import ContextConfig


class TestComputeLaneState:
    def test_no_detection(self):
        ls = compute_lane_state(None)
        assert ls.availability == LaneAvailability.NO_LANES
        assert ls.confidence == 0.0
        assert ls.has_lanes is False

    def test_both_lanes_high_confidence(self):
        ld = LaneDetectionInput(
            left_detected=True,
            right_detected=True,
            left_confidence=0.9,
            right_confidence=0.8,
            lane_width_px=300.0,
        )
        ls = compute_lane_state(ld)
        assert ls.availability == LaneAvailability.HAS_LANES
        assert ls.has_lanes is True
        assert abs(ls.confidence - 0.85) < 1e-6

    def test_one_lane_medium_confidence(self):
        ld = LaneDetectionInput(
            left_detected=True,
            right_detected=False,
            left_confidence=0.7,
            right_confidence=0.0,
        )
        ls = compute_lane_state(ld)
        assert ls.availability == LaneAvailability.DEGRADED_LANES
        assert ls.lanes_degraded is True

    def test_low_confidence_no_lanes(self):
        ld = LaneDetectionInput(
            left_confidence=0.2,
            right_confidence=0.1,
        )
        ls = compute_lane_state(ld)
        assert ls.availability == LaneAvailability.NO_LANES


class TestEMASmoothing:
    def test_smoothing_dampens_drop(self):
        ld_good = LaneDetectionInput(
            left_confidence=0.9,
            right_confidence=0.9,
        )
        ls1 = compute_lane_state(ld_good)
        assert ls1.confidence == 0.9

        ld_zero = LaneDetectionInput(
            left_confidence=0.0,
            right_confidence=0.0,
        )
        ls2 = compute_lane_state(ld_zero, prev_lane_state=ls1)
        # alpha=0.3 → 0.3*0 + 0.7*0.9 = 0.63
        assert abs(ls2.confidence - 0.63) < 0.01

    def test_custom_alpha(self):
        cfg = ContextConfig(lane_ema_alpha=0.5)
        ld1 = LaneDetectionInput(left_confidence=0.8, right_confidence=0.8)
        ls1 = compute_lane_state(ld1, config=cfg)

        ld2 = LaneDetectionInput(left_confidence=0.0, right_confidence=0.0)
        ls2 = compute_lane_state(ld2, prev_lane_state=ls1, config=cfg)
        # 0.5*0 + 0.5*0.8 = 0.4
        assert abs(ls2.confidence - 0.4) < 0.01


class TestStability:
    def test_stable_input(self):
        ld = LaneDetectionInput(left_confidence=0.8, right_confidence=0.8)
        ls1 = compute_lane_state(ld)
        ls2 = compute_lane_state(ld, prev_lane_state=ls1)
        assert ls2.stability > 0.9

    def test_unstable_drop(self):
        ld1 = LaneDetectionInput(left_confidence=0.9, right_confidence=0.9)
        ls1 = compute_lane_state(ld1)
        ld2 = LaneDetectionInput(left_confidence=0.0, right_confidence=0.0)
        ls2 = compute_lane_state(ld2, prev_lane_state=ls1)
        assert ls2.stability < ls1.stability


class TestPassthrough:
    def test_lane_width(self):
        ld = LaneDetectionInput(lane_width_px=350.0)
        ls = compute_lane_state(ld)
        assert ls.lane_width_px == 350.0
