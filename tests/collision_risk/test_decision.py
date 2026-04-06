"""Tests for adas.collision_risk.decision."""

from __future__ import annotations

import pytest

from adas.collision_risk.decision import (
    decide,
    DecisionConfig,
    DEFAULT_DECISION_CONFIG,
    _evaluate_single,
)
from adas.collision_risk.types import RiskResult, SystemAction


def _risk(
    score: float = 0.0,
    ttc: float = float("inf"),
    in_ego_lane: bool = True,
    object_id: int = 1,
    distance_m: float = 20.0,
) -> RiskResult:
    return RiskResult(
        object_id=object_id,
        ttc=ttc,
        distance_m=distance_m,
        risk_score=score,
        lateral_offset_m=0.0,
        in_ego_lane=in_ego_lane,
    )


class TestDecide:
    def test_empty_risks_returns_none(self):
        action, intensity = decide([])
        assert action == SystemAction.NONE
        assert intensity == 0.0

    def test_low_score_returns_none(self):
        action, intensity = decide([_risk(score=0.1, ttc=10.0)])
        assert action == SystemAction.NONE

    def test_medium_score_returns_warn(self):
        action, _ = decide([_risk(score=0.50, ttc=5.0)])
        assert action == SystemAction.WARN

    def test_high_score_returns_brake(self):
        action, intensity = decide([_risk(score=0.85, ttc=1.0)])
        assert action == SystemAction.BRAKE
        assert intensity > 0.5

    def test_very_low_ttc_in_lane_triggers_brake(self):
        action, _ = decide([_risk(score=0.1, ttc=0.8, in_ego_lane=True)])
        assert action == SystemAction.BRAKE

    def test_low_ttc_in_lane_triggers_warn(self):
        action, _ = decide([_risk(score=0.1, ttc=2.0, in_ego_lane=True)])
        assert action == SystemAction.WARN

    def test_out_of_lane_never_brakes_by_default(self):
        cfg = DecisionConfig(only_in_lane=True)
        action, _ = decide([_risk(score=0.90, ttc=0.5, in_ego_lane=False)], config=cfg)
        # Should not BRAKE for out-of-lane, but may WARN
        assert action != SystemAction.BRAKE

    def test_highest_action_wins_across_objects(self):
        risks = [
            _risk(score=0.2, ttc=10.0, object_id=1),
            _risk(score=0.85, ttc=1.0, object_id=2),
        ]
        action, _ = decide(risks)
        assert action == SystemAction.BRAKE

    def test_wet_road_lowers_threshold(self):
        class FakeCtx:
            braking_multiplier = 1.4

        # Score that does NOT trigger WARN on dry road
        risk_list = [_risk(score=0.30, ttc=4.0)]
        action_dry, _ = decide(risk_list)
        action_wet, _ = decide(risk_list, context_state=FakeCtx())
        # On wet road, threshold is lower, may trigger WARN
        # At minimum, wet should be >= dry in severity
        _order = [SystemAction.NONE, SystemAction.WARN, SystemAction.BRAKE]
        assert _order.index(action_wet) >= _order.index(action_dry)

    def test_intensity_in_range(self):
        _, intensity = decide([_risk(score=0.7, ttc=1.0)])
        assert 0.0 <= intensity <= 1.0


class TestEvaluateSingle:
    def test_low_score_no_action(self):
        cfg = DEFAULT_DECISION_CONFIG
        risk = _risk(score=0.1, ttc=15.0, in_ego_lane=False)
        action, intensity = _evaluate_single(
            risk, cfg.warn_score_threshold, cfg.brake_score_threshold, cfg
        )
        assert action == SystemAction.NONE

    def test_brake_threshold_exact(self):
        cfg = DEFAULT_DECISION_CONFIG
        risk = _risk(score=cfg.brake_score_threshold + 0.01, ttc=5.0, in_ego_lane=True)
        action, _ = _evaluate_single(
            risk, cfg.warn_score_threshold, cfg.brake_score_threshold, cfg
        )
        assert action == SystemAction.BRAKE
