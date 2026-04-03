"""Tests for adas.context.types - enums, dataclasses, defaults, immutability."""

import pytest
from adas.context.types import (
    LaneAvailability,
    RoadSurfaceType,
    Mode,
    SceneMetrics,
    VisibilityEstimate,
    LaneDetectionInput,
    LaneState,
    RoadSurfaceHint,
    EmergencySignal,
    ContextState,
)


# -------------------------------------------------------------------- enums


class TestLaneAvailability:
    def test_values(self):
        assert LaneAvailability.HAS_LANES.value == "has_lanes"
        assert LaneAvailability.DEGRADED_LANES.value == "degraded_lanes"
        assert LaneAvailability.NO_LANES.value == "no_lanes"

    def test_member_count(self):
        assert len(LaneAvailability) == 3


class TestRoadSurfaceType:
    def test_values(self):
        assert RoadSurfaceType.ASPHALT_DRY.value == "asphalt_dry"
        assert RoadSurfaceType.ASPHALT_WET.value == "asphalt_wet"
        assert RoadSurfaceType.GRAVEL.value == "gravel"
        assert RoadSurfaceType.UNKNOWN.value == "unknown"


class TestMode:
    def test_all_modes(self):
        assert len(Mode) == 5
        assert Mode.NORMAL_MARKED.value == "normal_marked"
        assert Mode.EMERGENCY_OVERRIDE.value == "emergency_override"


# -------------------------------------------------------------- dataclasses


class TestSceneMetrics:
    def test_defaults(self):
        m = SceneMetrics()
        assert m.brightness_mean == 0.0
        assert m.edge_density == 0.0

    def test_frozen(self):
        m = SceneMetrics()
        with pytest.raises(AttributeError):
            m.brightness_mean = 1.0  # type: ignore[misc]


class TestVisibilityEstimate:
    def test_defaults(self):
        v = VisibilityEstimate()
        assert v.confidence == 0.0
        assert v.is_night is False


class TestLaneDetectionInput:
    def test_defaults(self):
        ld = LaneDetectionInput()
        assert ld.left_detected is False
        assert ld.lane_width_px is None


class TestLaneState:
    def test_defaults(self):
        ls = LaneState()
        assert ls.availability == LaneAvailability.NO_LANES
        assert ls.has_lanes is False


class TestRoadSurfaceHint:
    def test_defaults(self):
        rsh = RoadSurfaceHint()
        assert rsh.surface_type == RoadSurfaceType.UNKNOWN
        assert rsh.confidence == 0.0


class TestEmergencySignal:
    def test_defaults(self):
        es = EmergencySignal()
        assert es.active is False
        assert es.reason == ""


class TestContextState:
    def test_defaults(self):
        cs = ContextState()
        assert cs.mode == Mode.UNMARKED_DEGRADED
        assert cs.braking_multiplier == 1.0
        assert cs.mode_hold_count == 0
        assert cs.pending_mode is None

    def test_frozen(self):
        cs = ContextState()
        with pytest.raises(AttributeError):
            cs.mode = Mode.NORMAL_MARKED  # type: ignore[misc]
