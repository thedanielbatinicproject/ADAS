"""Collision risk decision module.

Maps a list of RiskResult objects + ContextState into a SystemAction.

Decision logic:
  1. Find the highest risk score among all in-lane obstacles.
  2. Apply TTC-based urgency override (very low TTC -> BRAKE regardless of score).
  3. Apply braking_multiplier from road surface (wet/gravel increases sensitivity).
  4. Return NONE / WARN / BRAKE based on configurable thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from .types import RiskResult, SystemAction
from ..utils.runtime_overrides import apply_dataclass_overrides


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DecisionConfig:
    """Thresholds for converting risk scores into SystemActions.

    Attributes
    ----------
    warn_score_threshold : float
        risk_score >= this -> at least WARN.
    brake_score_threshold : float
        risk_score >= this -> BRAKE.
    ttc_warn_s : float
        TTC <= this (for in-lane object) -> at least WARN.
    ttc_brake_s : float
        TTC <= this (for in-lane object) -> BRAKE.
    road_penalty_warn : float
        Additional score bonus per unit of (braking_multiplier - 1.0).
        Models increased stopping distance on wet/gravel roads.
    only_in_lane : bool
        If True, only objects with in_ego_lane=True trigger BRAKE.
        WARN can still be raised for out-of-lane objects with high score.
    """

    warn_score_threshold: float = 0.35
    brake_score_threshold: float = 0.65
    ttc_warn_s: float = 3.0
    ttc_brake_s: float = 1.5
    road_penalty_warn: float = 0.08
    only_in_lane: bool = True


DEFAULT_DECISION_CONFIG = apply_dataclass_overrides(DecisionConfig(), "decision")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def decide(
    risks: List[RiskResult],
    context_state: Any = None,
    config: Optional[DecisionConfig] = None,
) -> Tuple[SystemAction, float]:
    """Decide system action from risk estimates.

    Parameters
    ----------
    risks : list[RiskResult]
        Output of RiskEstimator.estimate_risk().
    context_state : ContextState, optional
        Used to read braking_multiplier and mode.
    config : DecisionConfig, optional
        Thresholds. Falls back to DEFAULT_DECISION_CONFIG.

    Returns
    -------
    action : SystemAction
    intensity : float
        Intensity of the action in [0, 1]. Useful for graded warning sounds.
    """
    cfg = config or DEFAULT_DECISION_CONFIG

    braking_mult = 1.0
    if context_state is not None and hasattr(context_state, "braking_multiplier"):
        braking_mult = float(context_state.braking_multiplier)

    # Road surface penalty: on wet/gravel, lower the effective threshold
    road_penalty = cfg.road_penalty_warn * max(0.0, braking_mult - 1.0)

    effective_warn = max(0.05, cfg.warn_score_threshold - road_penalty)
    effective_brake = max(0.10, cfg.brake_score_threshold - road_penalty)

    if not risks:
        return SystemAction.NONE, 0.0

    best_action = SystemAction.NONE
    best_intensity = 0.0

    for risk in risks:
        action, intensity = _evaluate_single(risk, effective_warn, effective_brake, cfg)
        if action == SystemAction.BRAKE:
            return SystemAction.BRAKE, max(intensity, best_intensity)
        if action == SystemAction.WARN and best_action != SystemAction.BRAKE:
            best_action = SystemAction.WARN
            best_intensity = max(intensity, best_intensity)

    return best_action, best_intensity


def _evaluate_single(
    risk: RiskResult,
    warn_threshold: float,
    brake_threshold: float,
    cfg: DecisionConfig,
) -> Tuple[SystemAction, float]:
    """Evaluate a single RiskResult and return (action, intensity)."""
    score = risk.risk_score
    ttc = risk.ttc
    in_lane = risk.in_ego_lane

    # TTC-based hard override for in-lane obstacles
    if in_lane:
        if ttc <= cfg.ttc_brake_s:
            intensity = _score_to_intensity(score)
            return SystemAction.BRAKE, max(intensity, 0.8)
        if ttc <= cfg.ttc_warn_s:
            intensity = _score_to_intensity(score)
            return SystemAction.WARN, max(intensity, 0.5)

    # Score-based decision
    if cfg.only_in_lane and not in_lane:
        # Out-of-lane objects: only warn, never brake
        if score >= warn_threshold:
            return SystemAction.WARN, _score_to_intensity(score) * 0.6
        return SystemAction.NONE, 0.0

    if score >= brake_threshold:
        return SystemAction.BRAKE, _score_to_intensity(score)
    if score >= warn_threshold:
        return SystemAction.WARN, _score_to_intensity(score)
    return SystemAction.NONE, 0.0


def _score_to_intensity(score: float) -> float:
    """Map a risk score to an intensity value in [0, 1]."""
    return max(0.0, min(1.0, score))
