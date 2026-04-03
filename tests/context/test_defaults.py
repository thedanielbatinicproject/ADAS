"""Tests for adas.context.defaults - config creation and constraints."""

import pytest
from adas.context.defaults import ContextConfig, DEFAULT_CONFIG


class TestDefaultConfig:
    def test_exists(self):
        assert DEFAULT_CONFIG is not None
        assert isinstance(DEFAULT_CONFIG, ContextConfig)

    def test_thresholds_sane(self):
        cfg = DEFAULT_CONFIG
        assert 0.0 < cfg.t_vis < 1.0
        assert 0.0 < cfg.t_lane < 1.0
        assert 0.0 < cfg.t_lane_low < cfg.t_lane
        assert cfg.hysteresis_k >= 1
        assert cfg.min_fps > 0

    def test_braking_multipliers_ordered(self):
        cfg = DEFAULT_CONFIG
        assert cfg.braking_dry <= cfg.braking_unknown
        assert cfg.braking_unknown <= cfg.braking_wet
        assert cfg.braking_wet <= cfg.braking_gravel

    def test_visibility_weights_sum_to_one(self):
        cfg = DEFAULT_CONFIG
        total = cfg.w_contrast + cfg.w_blur + cfg.w_edge + cfg.w_glare
        assert abs(total - 1.0) < 1e-9

    def test_frozen(self):
        with pytest.raises(AttributeError):
            DEFAULT_CONFIG.t_vis = 0.9  # type: ignore[misc]


class TestCustomConfig:
    def test_override_fields(self):
        cfg = ContextConfig(t_vis=0.7, hysteresis_k=10)
        assert cfg.t_vis == 0.7
        assert cfg.hysteresis_k == 10
        # other defaults unchanged
        assert cfg.t_lane == 0.6

    def test_different_instances_independent(self):
        a = ContextConfig(t_vis=0.1)
        b = ContextConfig(t_vis=0.9)
        assert a.t_vis != b.t_vis
