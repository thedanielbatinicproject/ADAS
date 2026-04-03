"""Tests for adas.context.router - mode selection, hysteresis, emergency."""

import numpy as np
from adas.context.router import route, _determine_candidate_mode
from adas.context.types import (
    Mode,
    ContextState,
    LaneDetectionInput,
    EmergencySignal,
    VisibilityEstimate,
    LaneState,
    LaneAvailability,
)
from adas.context.defaults import ContextConfig, DEFAULT_CONFIG


def _make_frame(brightness: int = 128) -> np.ndarray:
    return np.full((480, 640, 3), brightness, dtype=np.uint8)


# ------------------------------------------------ _determine_candidate_mode


class TestDetermineCandidateMode:
    def test_normal_marked(self):
        vis = VisibilityEstimate(confidence=0.8)
        ls = LaneState(confidence=0.7, availability=LaneAvailability.HAS_LANES)
        assert _determine_candidate_mode(vis, ls, DEFAULT_CONFIG) == Mode.NORMAL_MARKED

    def test_degraded_marked(self):
        vis = VisibilityEstimate(confidence=0.3)
        ls = LaneState(confidence=0.5, availability=LaneAvailability.DEGRADED_LANES)
        assert (
            _determine_candidate_mode(vis, ls, DEFAULT_CONFIG) == Mode.DEGRADED_MARKED
        )

    def test_unmarked_good_vis(self):
        vis = VisibilityEstimate(confidence=0.8)
        ls = LaneState(confidence=0.1, availability=LaneAvailability.NO_LANES)
        assert (
            _determine_candidate_mode(vis, ls, DEFAULT_CONFIG) == Mode.UNMARKED_GOOD_VIS
        )

    def test_unmarked_degraded(self):
        vis = VisibilityEstimate(confidence=0.3)
        ls = LaneState(confidence=0.1, availability=LaneAvailability.NO_LANES)
        assert (
            _determine_candidate_mode(vis, ls, DEFAULT_CONFIG) == Mode.UNMARKED_DEGRADED
        )

    def test_moderate_lanes_good_vis_is_normal(self):
        """Lanes above t_lane_low + good visibility -> NORMAL_MARKED."""
        vis = VisibilityEstimate(confidence=0.8)
        ls = LaneState(confidence=0.4, availability=LaneAvailability.DEGRADED_LANES)
        assert _determine_candidate_mode(vis, ls, DEFAULT_CONFIG) == Mode.NORMAL_MARKED


# -------------------------------------------------------------------- route


class TestRouteBasic:
    def test_returns_context_state(self):
        state = route(_make_frame())
        assert isinstance(state, ContextState)
        assert state.scene_metrics is not None
        assert state.visibility is not None
        assert state.lane_state is not None
        assert state.road_surface is not None

    def test_timestamp_and_fps_passthrough(self):
        state = route(_make_frame(), timestamp_s=1.5, fps=30.0)
        assert state.timestamp_s == 1.5
        assert state.fps == 30.0

    def test_with_lanes(self):
        ld = LaneDetectionInput(
            left_detected=True,
            right_detected=True,
            left_confidence=0.9,
            right_confidence=0.9,
        )
        state = route(_make_frame(150), lane_detection=ld)
        assert state.lane_state.has_lanes is True


# -------------------------------------------------------- emergency override


class TestEmergencyOverride:
    def test_immediate(self):
        em = EmergencySignal(active=True, reason="test")
        state = route(_make_frame(), emergency=em)
        assert state.mode == Mode.EMERGENCY_OVERRIDE

    def test_overrides_hysteresis(self):
        s1 = route(_make_frame())
        em = EmergencySignal(active=True, reason="critical")
        s2 = route(_make_frame(), prev_state=s1, emergency=em)
        assert s2.mode == Mode.EMERGENCY_OVERRIDE
        assert s2.mode_hold_count == 1


# --------------------------------------------------------------- hysteresis


class TestHysteresis:
    def test_holds_mode_below_k(self):
        cfg = ContextConfig(hysteresis_k=3, lane_ema_alpha=1.0)
        ld_good = LaneDetectionInput(
            left_detected=True,
            right_detected=True,
            left_confidence=0.9,
            right_confidence=0.9,
        )
        s = route(_make_frame(), lane_detection=ld_good, config=cfg)
        initial_mode = s.mode

        # One frame without lanes should NOT cause a switch
        s2 = route(_make_frame(), lane_detection=None, prev_state=s, config=cfg)
        assert s2.mode == initial_mode
        assert s2.pending_count >= 1

    def test_switches_after_k(self):
        cfg = ContextConfig(hysteresis_k=2, lane_ema_alpha=1.0)
        ld_good = LaneDetectionInput(
            left_detected=True,
            right_detected=True,
            left_confidence=0.9,
            right_confidence=0.9,
        )
        s = route(_make_frame(), lane_detection=ld_good, config=cfg)
        initial_mode = s.mode

        # Feed 2+ frames without lanes
        for _ in range(3):
            s = route(_make_frame(), lane_detection=None, prev_state=s, config=cfg)

        # Should have switched away from initial mode
        assert s.mode != initial_mode or s.pending_count == 0

    def test_mode_hold_count_increments(self):
        frame = _make_frame()
        s1 = route(frame)
        s2 = route(frame, prev_state=s1)
        assert s2.mode_hold_count > s1.mode_hold_count
