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
    # Night detection — dual-path:
    #   Primary:   p25 < t_night_p25  → 75 %+ of scene is dark  (most reliable)
    #   Secondary: p05 < t_night_p05  AND  mean < t_night_mean_max  (fallback)
    # Daytime p25 is typically > 70; night street-lit scenes p25 ≈ 20–50.
    t_night_p25: float = 55.0
    t_night_p05: float = 45.0
    t_night_mean_max: float = 90.0
    # Day override: if top-5% pixels are bright, sky/sun is in frame → daytime.
    # Lowered from 210 → 160: overcast sky registers p95 ≈ 150-190, not 210.
    # Night scenes with many streetlamps still have p95 ≈ 130-180 but their
    # mean brightness stays low; the secondary mean guard handles that case.
    t_day_p95: float = 160.0
    # Secondary day override: if mean brightness > this, the scene is too
    # uniformly bright to be night (overcast day mean ≈ 60-90; night ≈ 35-65).
    t_day_mean: float = 65.0
    t_glare: float = 0.15

    # ---- visibility combination weights (must sum to 1.0) ----
    w_contrast: float = 0.20
    w_blur: float = 0.30
    w_edge: float = 0.38
    w_glare: float = 0.12

    # ---- metric normalisation ranges ----
    contrast_min: float = 5.0
    contrast_max: float = 90.0
    blur_var_min: float = 10.0
    blur_var_max: float = 500.0
    edge_density_min: float = 0.01
    edge_density_max: float = 0.15

    # ---- scene ROI ----
    # Use only the top fraction of the frame for scene-quality analysis.
    # This excludes the car interior / dashboard that is visible in the
    # bottom portion of dashcam footage and would otherwise inflate the
    # contrast score even in foggy or night conditions.
    scene_roi_top_fraction: float = 0.65

    # ---- lane thresholds ----
    t_lane: float = 0.6
    t_lane_low: float = 0.3
    lane_ema_alpha: float = 0.3

    # ---- hysteresis ----
    hysteresis_k: int = 5

    # ---- road surface ----
    default_roi_fraction: float = 0.35   # fraction of frame height to use
    # Exclude bottom margin (dashboard/hood) from road-surface analysis.
    road_roi_bottom_margin: float = 0.20
    braking_dry: float = 1.0
    braking_wet: float = 1.4
    braking_gravel: float = 2.0
    braking_unknown: float = 1.2
    t_roughness: float = 300.0
    t_uniformity: float = 30.0

    # ---- Hough lane detection ----
    lane_roi_top: float = 0.35     # top of lane-search ROI (fraction of frame height)
    lane_roi_bottom: float = 0.88  # bottom of lane-search ROI (excludes dashboard)
    # Higher threshold = require more collinear edge pixels = fewer false positives.
    # Real white markings on asphalt produce 60–150+ votes; noise 20–40.
    lane_hough_threshold: int = 55   # was 25
    lane_min_length: int = 50        # was 25 — short segments are noise
    lane_max_gap: int = 80           # was 60 — allow more gap for dashed markings
    lane_slope_min: float = 0.3    # tan(~17°) – filter near-horizontal noise
    lane_slope_max: float = 4.0    # tan(~76°) – filter near-vertical noise

    # ---- Canny thresholds (edge-density computation) ----
    canny_low: int = 50
    canny_high: int = 150

    # ---- glare pixel threshold ----
    # Lowered from 250 → 220: wet-road reflections at night register at 200–240,
    # not 250.  Dry road under headlights stays mostly below 200.
    glare_pixel_threshold: int = 220

    # ---- road-surface specular threshold ----
    # Lowered from 0.03 → 0.015: rain on asphalt at night creates subtler
    # reflective patches; lower threshold catches them earlier.
    t_specular: float = 0.015

    # ---- FPS ----
    min_fps: float = 25.0


DEFAULT_CONFIG = ContextConfig()
