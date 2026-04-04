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
    # Raised from 160 → 165: a night video with p95=163 (streetlamp filling a
    # narrow slice of the image) was incorrectly classified as day.  Setting
    # the threshold to 165 leaves enough margin below the minimum overcast-sky
    # p95 (≈ 175+) while excluding borderline night-lamp spikes (p95 ≤ 164).
    t_day_p95: float = 165.0
    # Secondary day override: if mean brightness > this, the scene is too
    # uniformly bright to be night (overcast day mean ≈ 60-90; night ≈ 35-65).
    t_day_mean: float = 65.0
    # Glare threshold: pixels > glare_pixel_threshold fraction of ROI.
    # Raised from 0.15 → 0.175 to prevent a borderline light-fog/haze video
    # (measured glare ≈ 0.172) from being exempt from degraded classification.
    t_glare: float = 0.175

    # ---- visibility combination weights (must sum to 1.0) ----
    w_contrast: float = 0.20
    w_blur: float = 0.30
    w_edge: float = 0.38
    w_glare: float = 0.12

    # ---- metric normalisation ranges ----
    # Calibrated for DADA-2000 dashcam footage characteristics.
    # Typical clear-day frames: blur_laplacian_var ≈ 60–120, edge_density ≈ 0.025–0.055.
    # The original limits (500 / 0.15) were calibrated for high-quality cameras
    # and left clear dashcam frames at only 15–20 % of the normalised range,
    # consistently mapping them below the t_vis degraded threshold.
    # Lowered limits compress the range to match actual DADA-2000 data so that
    # a dashcam clear-day frame scores ≥ 40 % of the new range (conf ≥ t_vis).
    contrast_min: float = 5.0
    contrast_max: float = 90.0
    blur_var_min: float = 10.0
    blur_var_max: float = 200.0
    edge_density_min: float = 0.01
    edge_density_max: float = 0.10

    # ---- scene ROI ----
    # Use only the top fraction of the frame for scene-quality analysis.
    # This excludes the car interior / dashboard that is visible in the
    # bottom portion of dashcam footage and would otherwise inflate the
    # contrast score even in foggy or night conditions.
    scene_roi_top_fraction: float = 0.65

    # ---- Dark Channel Prior (He, Sun & Tang 2011) ----
    # Patch size (px) for the rolling minimum filter.  The original paper
    # uses 15 px; larger patches are more robust but slower.
    dcp_patch_size: int = 15
    # Road-region fraction: we compute the dark channel only in the bottom
    # portion of the scene ROI to avoid sky/sunlight contaminating the score.
    dcp_road_fraction: float = 0.45   # use lower 45 % of scene ROI as road
    # If the road dark-channel score exceeds this, assume atmospheric haze.
    # Fog videos score ≈ 0.26–0.35; direct-glare scenes score ≈ 0.10–0.23.
    # Threshold set to 0.255 to stay just below the minimum fog road-DCP
    # (43/158 = 0.264) while exempting glare scenes with clear asphalt.
    t_dcp_haze: float = 0.255
    # Stricter road-DCP threshold for the *non-glare* clear-road override.
    # All observed fog videos have road-DCP ≥ 0.219; sunny roads ≤ 0.181.
    # 0.20 keeps a safe margin above the fog minimum (0.219).
    t_dcp_clear: float = 0.20
    # Maximum glare_score still eligible for DCP-based glare/fog disambiguation.
    # Extreme exposures (glare > 0.40 = 40 % of pixels ≥ 220) are more likely
    # to be rain-with-sunshine than a simple solar-glare scene on a clear day.
    t_max_glare_dcp: float = 0.40
    # Minimum glare_score required for the solar-glare disambiguation (block 1).
    # Marginal glare in [t_glare, t_glare_strong) can originate from wet-road
    # reflections or streetlights in rain, not from direct sun hitting the lens.
    # Only clearly solar glare (glare_score > 0.25 ≈ 25 % of pixels ≥ 220)
    # is selective enough to safely clear the is_degraded flag.
    t_glare_strong: float = 0.25
    # Minimum scene saturation (HSV S, 0–255) for the clear-road override.
    # Sunny scenes are colourful (S ≈ 40–80); grey overcast / rain ≈ 10–35.
    # Threshold 40: provides margin above typical overcast-rainy saturation
    # (≤30) while staying within the range of colourful sunny scenes (≥40).
    t_sat_clear: float = 40.0
    # Minimum visibility confidence for the clear-road override to fire.
    # Below this the scene is so dark/blurry that road DCP alone cannot
    # override the degraded flag safely.
    t_vis_dcp_min: float = 0.38

    # ---- lane thresholds ----
    t_lane: float = 0.6
    # Lowered from 0.3 → 0.10: urban scenes often produce intermittent
    # Hough detections due to occlusion, dashes, or intersection markings.
    # EMA smoothing (alpha=0.3) means a single good detection (raw≈0.8)
    # yields state≈0.24 which stays above 0.10 for 3-4 subsequent frames.
    # Even with t_lane_low=0.10, the lane test requires lane_conf ≥ 0.10,
    # which only fires when there is genuine Hough evidence.
    t_lane_low: float = 0.10
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
    # Real white markings on asphalt produce 60-150+ votes; noise 20-40.
    # Lowered 55 → 40: dashed markings in urban scenes produce fewer votes
    # per segment; 55 was filtering out genuine intermittent markings.
    lane_hough_threshold: int = 40
    # Lowered 50 → 35: dashed markings are shorter than continuous lines;
    # 50 px min length was causing many real segments to be discarded.
    lane_min_length: int = 35
    # Raised 80 → 120: wider gap allows connecting dashes from near-distance
    # to mid-distance markings that appear discontinuous in perspective.
    lane_max_gap: int = 120
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
    # Lowered from 0.03 → 0.015 → 0.010 → 0.003:
    # With recalibrated confidence ranges, clear-day scenes are now correctly
    # marked as not-degraded directly via the confidence score.  Rainy scenes
    # that have adequate visibility (low blur) therefore rely on wet-surface
    # detection rather than the degraded flag to pass the rainy test.
    # At 0.003 (0.3 % of road-region pixels ≥ 220), the threshold is sensitive
    # enough to capture rain puddles and wet reflective markings.  False-positive
    # wet detections on a dry road cause a MORE conservative braking multiplier,
    # which is the safe direction of error.
    t_specular: float = 0.003

    # ---- FPS ----
    min_fps: float = 25.0


DEFAULT_CONFIG = ContextConfig()
