"""Tests for adas.context.road_surface - classification & braking multiplier."""

import numpy as np
from adas.context.road_surface import estimate_road_surface, braking_multiplier
from adas.context.types import RoadSurfaceHint, RoadSurfaceType
from adas.context.defaults import DEFAULT_CONFIG


class TestEstimateRoadSurface:
    def test_uniform_dark_frame(self):
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        hint = estimate_road_surface(frame)
        assert isinstance(hint, RoadSurfaceHint)
        assert hint.surface_type in (
            RoadSurfaceType.ASPHALT_DRY,
            RoadSurfaceType.UNKNOWN,
        )

    def test_bright_frame_is_wet(self):
        frame = np.full((480, 640, 3), 252, dtype=np.uint8)
        hint = estimate_road_surface(frame)
        assert hint.surface_type == RoadSurfaceType.ASPHALT_WET

    def test_noisy_frame(self, noisy_frame):
        hint = estimate_road_surface(noisy_frame)
        assert hint.surface_type in (
            RoadSurfaceType.GRAVEL,
            RoadSurfaceType.UNKNOWN,
        )

    def test_empty_frame(self):
        empty = np.empty((0, 0, 3), dtype=np.uint8)
        assert estimate_road_surface(empty) == RoadSurfaceHint()

    def test_none_frame(self):
        assert estimate_road_surface(None) == RoadSurfaceHint()

    def test_confidence_in_range(self, noisy_frame):
        hint = estimate_road_surface(noisy_frame)
        assert 0.0 <= hint.confidence <= 1.0

    def test_grayscale_input(self):
        gray = np.full((100, 100), 80, dtype=np.uint8)
        hint = estimate_road_surface(gray)
        assert isinstance(hint, RoadSurfaceHint)


class TestBrakingMultiplier:
    def test_dry(self):
        hint = RoadSurfaceHint(
            surface_type=RoadSurfaceType.ASPHALT_DRY,
            confidence=1.0,
        )
        assert braking_multiplier(hint) == DEFAULT_CONFIG.braking_dry

    def test_wet(self):
        hint = RoadSurfaceHint(
            surface_type=RoadSurfaceType.ASPHALT_WET,
            confidence=1.0,
        )
        assert braking_multiplier(hint) == DEFAULT_CONFIG.braking_wet

    def test_gravel(self):
        hint = RoadSurfaceHint(
            surface_type=RoadSurfaceType.GRAVEL,
            confidence=1.0,
        )
        assert braking_multiplier(hint) == DEFAULT_CONFIG.braking_gravel

    def test_unknown(self):
        hint = RoadSurfaceHint(
            surface_type=RoadSurfaceType.UNKNOWN,
            confidence=1.0,
        )
        assert braking_multiplier(hint) == DEFAULT_CONFIG.braking_unknown

    def test_low_confidence_blends_to_fallback(self):
        hint = RoadSurfaceHint(
            surface_type=RoadSurfaceType.ASPHALT_DRY,
            confidence=0.0,
        )
        assert braking_multiplier(hint) == DEFAULT_CONFIG.braking_unknown

    def test_always_gte_one(self):
        for st in RoadSurfaceType:
            for conf in (0.0, 0.5, 1.0):
                hint = RoadSurfaceHint(surface_type=st, confidence=conf)
                assert braking_multiplier(hint) >= 1.0
