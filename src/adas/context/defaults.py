"""
Tunable thresholds and constants for the context package.

All numeric parameters are collected in :class:`ContextConfig` so that
experiments, tests, and deployment can override them in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextConfig:
    """All tunable thresholds and constants used by ``adas.context``."""

    # ---- visibility thresholds ----
    t_vis: float = 0.5
    t_night_brightness: float = 50.0
    t_night_edge_density: float = 0.02
    t_glare: float = 0.15

    # ---- visibility combination weights (sum → 1.0) ----
    w_contrast: float = 0.30
    w_blur: float = 0.30
    w_edge: float = 0.25
    w_glare: float = 0.15

    # ---- metric normalisation ranges ----
    contrast_min: float = 5.0
    contrast_max: float = 60.0
    blur_var_min: float = 10.0
    blur_var_max: float = 500.0
    edge_density_min: float = 0.01
    edge_density_max: float = 0.15

    # ---- lane thresholds ----
    t_lane: float = 0.6
    t_lane_low: float = 0.3
    lane_ema_alpha: float = 0.3

    # ---- hysteresis ----
    hysteresis_k: int = 5

    # ---- road surface ----
    default_roi_fraction: float = 0.4
    braking_dry: float = 1.0
    braking_wet: float = 1.4
    braking_gravel: float = 2.0
    braking_unknown: float = 1.2
    t_specular: float = 0.05
    t_roughness: float = 300.0
    t_uniformity: float = 30.0

    # ---- Canny thresholds (edge-density computation) ----
    canny_low: int = 50
    canny_high: int = 150

    # ---- glare pixel threshold ----
    glare_pixel_threshold: int = 250

    # ---- FPS ----
    min_fps: float = 25.0


DEFAULT_CONFIG = ContextConfig()
