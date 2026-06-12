"""Collision risk estimator.

Converts DetectedObject list + lane geometry + context into RiskResult per
obstacle.

The TTC estimate uses a simple kinematic model:
    TTC = distance_m / relative_velocity_mps

where relative_velocity is derived from the change in distance_estimate
between the current frame and a rolling buffer of recent frames for the
same track ID. When velocity cannot be estimated (first appearance, or
distance unavailable), a conservative fallback based on distance alone
is used to derive the risk score.

Risk score combines:
  - Proximity: how close (inverse of distance, normalized)
  - TTC rating: how urgent the time-to-collision is
  - Lateral position: higher weight for obstacles in the ego lane
  - braking_multiplier from ContextState (wet/gravel road penalty)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .types import RiskResult
from ..obstacle_detection.types import DetectedObject
from ..utils.runtime_overrides import apply_dataclass_overrides


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EstimatorConfig:
    """Tunable parameters for the risk estimator.

    Attributes
    ----------
    ttc_warning_s : float
        TTC below this triggers WARN level.
    ttc_brake_s : float
        TTC below this triggers BRAKE level.
    max_ttc_s : float
        TTC above this is treated as no risk.
    max_distance_m : float
        Distance beyond this is ignored (too far for ADAS intervention).
    proximity_weight : float
        Weight of distance score in the combined risk score.
    ttc_weight : float
        Weight of TTC score in the combined risk score.
    lateral_weight : float
        Weight of lateral position in the combined risk score.
    in_lane_distance_fraction : float
        Fraction of estimated lane width within which an object is
        considered "in ego lane". Default 0.5 = within half lane width.
    assumed_ego_speed_mps : float
        Fallback ego speed (m/s) when no GPS/OBD is available.
        Used only for TTC when relative velocity cannot be computed.
        Default 10 m/s ~ 36 km/h (conservative urban speed).
    velocity_buffer_size : int
        Number of recent frames to keep per track for velocity estimation.
    """

    ttc_warning_s: float = 3.0
    ttc_brake_s: float = 1.5
    max_ttc_s: float = 10.0
    max_distance_m: float = 40.0
    proximity_weight: float = 0.35
    ttc_weight: float = 0.50
    lateral_weight: float = 0.15
    in_lane_distance_fraction: float = 0.5
    assumed_ego_speed_mps: float = 10.0
    velocity_buffer_size: int = 8
    min_distance_delta_m: float = 0.4
    max_relative_speed_mps: float = 30.0
    risk_ema_alpha: float = 0.35
    distance_ema_alpha: float = 0.35
    max_distance_jump_m: float = 4.0


DEFAULT_ESTIMATOR_CONFIG = apply_dataclass_overrides(EstimatorConfig(), "estimator")


# ---------------------------------------------------------------------------
# Estimator class (stateful: maintains velocity buffer per track)
# ---------------------------------------------------------------------------

class RiskEstimator:
    """Stateful collision risk estimator.

    Maintains a per-track distance history to compute relative velocity.
    Create one instance per video and call estimate_risk() each frame.
    """

    def __init__(self, config: Optional[EstimatorConfig] = None) -> None:
        self._config = config or DEFAULT_ESTIMATOR_CONFIG
        # track_id -> deque of (frame_idx, distance_m)
        self._distance_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self._distance_ema: Dict[int, float] = {}
        self._risk_ema: Dict[int, float] = {}

    def reset(self) -> None:
        """Clear history (call when switching to a new video)."""
        self._distance_history.clear()
        self._distance_ema.clear()
        self._risk_ema.clear()

    def estimate_risk(
        self,
        objects: List[DetectedObject],
        lane_output: Any = None,
        context_state: Any = None,
        frame_idx: int = -1,
    ) -> List[RiskResult]:
        """Estimate collision risk for each detected obstacle.

        Parameters
        ----------
        objects : list[DetectedObject]
            Tracked obstacles from SimpleTracker.update().
        lane_output : LaneOutput, optional
            Lane geometry for lateral offset computation.
        context_state : ContextState, optional
            Used to apply braking_multiplier to risk scores.
        frame_idx : int
            Current frame index.

        Returns
        -------
        list[RiskResult]
        """
        braking_mult = 1.0
        lane_width_px: Optional[float] = None

        if context_state is not None and hasattr(context_state, "braking_multiplier"):
            braking_mult = float(context_state.braking_multiplier)

        if lane_output is not None:
            if hasattr(lane_output, "lane_width_px") and lane_output.lane_width_px:
                lane_width_px = float(lane_output.lane_width_px)

        results: List[RiskResult] = []

        for obj in objects:
            dist_m = obj.distance_estimate
            track_id = obj.track_id

            # Smooth noisy distance estimates per track to reduce TTC/risk jitter.
            if dist_m is not None and track_id >= 0:
                d_cur = float(dist_m)
                prev_d = self._distance_ema.get(track_id)
                if prev_d is not None:
                    max_jump = max(0.5, self._config.max_distance_jump_m)
                    d_cur = max(prev_d - max_jump, min(prev_d + max_jump, d_cur))
                    a_d = self._config.distance_ema_alpha
                    d_cur = (a_d * d_cur) + ((1.0 - a_d) * prev_d)
                self._distance_ema[track_id] = d_cur
                dist_m = d_cur

            # Update distance history
            if dist_m is not None and track_id >= 0:
                history = self._distance_history[track_id]
                history.append((frame_idx, float(dist_m)))
                max_buf = self._config.velocity_buffer_size
                if len(history) > max_buf:
                    self._distance_history[track_id] = history[-max_buf:]

            # Estimate relative velocity from history
            rel_vel_mps = self._estimate_velocity(track_id, frame_idx)

            # Compute TTC
            ttc = _compute_ttc(dist_m, rel_vel_mps, self._config)

            # Lateral position
            lateral_offset_m, in_ego_lane = _compute_lateral(
                obj, lane_output, lane_width_px, self._config
            )

            # Risk score
            raw_risk_score = _compute_risk_score(
                dist_m, ttc, lateral_offset_m, in_ego_lane,
                braking_mult, self._config
            )

            prev = self._risk_ema.get(track_id)
            if prev is None:
                risk_score = raw_risk_score
            else:
                a = self._config.risk_ema_alpha
                risk_score = (a * raw_risk_score) + ((1.0 - a) * prev)
            self._risk_ema[track_id] = risk_score

            results.append(RiskResult(
                object_id=track_id,
                ttc=ttc,
                distance_m=dist_m if dist_m is not None else float("inf"),
                risk_score=risk_score,
                lateral_offset_m=lateral_offset_m,
                in_ego_lane=in_ego_lane,
            ))

        # Prune history for tracks that are no longer visible
        active_ids = {obj.track_id for obj in objects}
        stale = [tid for tid in self._distance_history if tid not in active_ids]
        for tid in stale:
            del self._distance_history[tid]
            self._distance_ema.pop(tid, None)
            self._risk_ema.pop(tid, None)

        return results

    def _estimate_velocity(self, track_id: int, current_frame: int) -> Optional[float]:
        """Estimate relative approach velocity (m/s) from distance history.

        Returns None when insufficient history is available.
        Positive value means the obstacle is approaching.
        """
        history = self._distance_history.get(track_id, [])
        if len(history) < 3:
            return None

        vel_samples: List[float] = []
        for (f0, d0), (f1, d1) in zip(history[:-1], history[1:]):
            df = f1 - f0
            if df <= 0:
                continue
            delta_d = d0 - d1  # positive => approaching
            if abs(delta_d) < self._config.min_distance_delta_m:
                continue
            dt = df / 30.0
            if dt <= 1e-6:
                continue
            v = delta_d / dt
            if 0.0 < v <= self._config.max_relative_speed_mps:
                vel_samples.append(v)

        if not vel_samples:
            return None

        vel_samples.sort()
        mid = len(vel_samples) // 2
        if len(vel_samples) % 2 == 1:
            return float(vel_samples[mid])
        return float((vel_samples[mid - 1] + vel_samples[mid]) * 0.5)


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def _compute_ttc(
    distance_m: Optional[float],
    rel_vel_mps: Optional[float],
    cfg: EstimatorConfig,
) -> float:
    """Compute TTC in seconds."""
    if distance_m is None or distance_m <= 0:
        return float("inf")
    if distance_m > cfg.max_distance_m:
        return float("inf")

    if rel_vel_mps is not None and rel_vel_mps > 0.5:
        # Obstacle is approaching
        ttc = distance_m / rel_vel_mps
    else:
        # Unknown/non-approaching relative velocity: use conservative fallback
        # based on assumed ego speed so TTC remains finite and comparable.
        ego_v = max(0.1, float(cfg.assumed_ego_speed_mps))
        ttc = distance_m / ego_v

    return min(float(ttc), cfg.max_ttc_s * 2)


def _compute_lateral(
    obj: DetectedObject,
    lane_output: Any,
    lane_width_px: Optional[float],
    cfg: EstimatorConfig,
) -> Tuple[float, bool]:
    """Compute lateral offset and in-lane flag.

    Returns
    -------
    lateral_offset_m : float
        Positive = right of center. Rough estimate using pixel-to-meter
        scaling from lane_width_px (assumes ~3.5 m lane width standard).
    in_ego_lane : bool
    """
    cx_px = obj.centroid[0]
    in_ego_lane = False
    lateral_offset_m = 0.0

    if lane_output is None:
        return lateral_offset_m, in_ego_lane

    # Estimate lane center x from polynomials at the obstacle's y position
    if (
        hasattr(lane_output, "left_poly")
        and hasattr(lane_output, "right_poly")
        and lane_output.left_poly is not None
        and lane_output.right_poly is not None
    ):
        y_px = float(obj.centroid[1]) - float(getattr(lane_output, "roi_y1", 0))
        lx = lane_output.left_poly[0] * y_px + lane_output.left_poly[1]
        rx = lane_output.right_poly[0] * y_px + lane_output.right_poly[1]
        lane_cx_px = (lx + rx) / 2.0
        width_px = abs(rx - lx)

        if width_px > 10:
            # Convert pixel offset to meters using assumed 3.5 m lane width
            px_per_m = width_px / 3.5
            lateral_offset_m = (cx_px - lane_cx_px) / px_per_m
            in_lane_threshold_px = width_px * cfg.in_lane_distance_fraction
            in_ego_lane = abs(cx_px - lane_cx_px) < in_lane_threshold_px

    elif lane_width_px is not None and lane_width_px > 10:
        # Rough center heuristic: assume lane center is at frame center
        in_ego_lane = True
        lateral_offset_m = 0.0

    return lateral_offset_m, in_ego_lane


def _compute_risk_score(
    distance_m: Optional[float],
    ttc: float,
    lateral_offset_m: float,
    in_ego_lane: bool,
    braking_mult: float,
    cfg: EstimatorConfig,
) -> float:
    """Combine proximity, TTC, and lateral position into [0, 1] risk score."""
    # Proximity score: 1.0 at 0 m, 0.0 at max_distance_m
    if distance_m is None or distance_m >= cfg.max_distance_m:
        prox_score = 0.0
    else:
        prox_score = 1.0 - distance_m / cfg.max_distance_m
        prox_score = max(0.0, min(1.0, prox_score))

    # TTC score: 1.0 at 0 s, 0.0 at max_ttc_s
    if ttc >= cfg.max_ttc_s:
        ttc_score = 0.0
    else:
        ttc_score = 1.0 - ttc / cfg.max_ttc_s
        ttc_score = max(0.0, min(1.0, ttc_score))

    # Lateral score: higher weight if in ego lane
    if in_ego_lane:
        lat_score = 1.0
    else:
        # Decay with lateral distance from lane center
        lat_score = max(0.0, 1.0 - abs(lateral_offset_m) / 2.0)

    score = (
        cfg.proximity_weight * prox_score
        + cfg.ttc_weight * ttc_score
        + cfg.lateral_weight * lat_score
    )

    # Apply braking multiplier from road surface (wet/gravel = more risk)
    # Clamp to [0, 1]
    score = score * braking_mult
    return max(0.0, min(1.0, score))
