"""
Scene-quality metrics and visibility estimation (heuristic, frame-level).

All computations are deterministic and based on classical image-processing
operations (luminance statistics, Laplacian variance, Canny edge ratio, …).
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np

from .types import SceneMetrics, VisibilityEstimate
from .defaults import ContextConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------- math


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo))


# -------------------------------------------------------------------- public


def compute_scene_metrics(
    frame: Any,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    config: Optional[ContextConfig] = None,
) -> SceneMetrics:
    """Compute image-quality / visibility metrics from a BGR frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image, shape ``(H, W, 3)``.
    roi : ``(x1, y1, x2, y2)``, optional
        Region of interest in pixel coordinates.  If *None* the full frame
        is used.
    config : ContextConfig, optional
        Override thresholds.  Falls back to :data:`DEFAULT_CONFIG`.
    """
    import cv2

    cfg = config or DEFAULT_CONFIG

    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return SceneMetrics()

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if roi is not None:
        x1, y1, x2, y2 = roi
        frame = frame[y1:y2, x1:x2]
        if frame.size == 0:
            return SceneMetrics()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness_mean = float(np.mean(gray))
    brightness_p05 = float(np.percentile(gray, 5))
    brightness_p25 = float(np.percentile(gray, 25))
    brightness_p95 = float(np.percentile(gray, 95))
    contrast_std = float(np.std(gray))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_laplacian_var = float(np.var(laplacian))

    edges = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    edge_density = float(np.count_nonzero(edges)) / max(gray.size, 1)

    saturation_mean = float(np.mean(hsv[:, :, 1]))

    glare_score = float(np.count_nonzero(gray >= cfg.glare_pixel_threshold)) / max(
        gray.size, 1
    )

    return SceneMetrics(
        brightness_mean=brightness_mean,
        brightness_p05=brightness_p05,
        brightness_p25=brightness_p25,
        brightness_p95=brightness_p95,
        contrast_std=contrast_std,
        blur_laplacian_var=blur_laplacian_var,
        edge_density=edge_density,
        saturation_mean=saturation_mean,
        glare_score=glare_score,
    )


def estimate_visibility(
    metrics: SceneMetrics,
    *,
    config: Optional[ContextConfig] = None,
) -> VisibilityEstimate:
    """Derive a visibility assessment from raw :class:`SceneMetrics`.

    The ``confidence`` value is a weighted combination of normalised
    contrast, blur-sharpness, edge-density and inverse-glare scores.
    """
    cfg = config or DEFAULT_CONFIG

    norm_contrast = _normalize(
        metrics.contrast_std,
        cfg.contrast_min,
        cfg.contrast_max,
    )
    norm_blur = _normalize(
        metrics.blur_laplacian_var,
        cfg.blur_var_min,
        cfg.blur_var_max,
    )
    norm_edge = _normalize(
        metrics.edge_density,
        cfg.edge_density_min,
        cfg.edge_density_max,
    )
    norm_glare = 1.0 - _clamp(metrics.glare_score / max(cfg.t_glare, 1e-9))

    confidence = (
        cfg.w_contrast * norm_contrast
        + cfg.w_blur * norm_blur
        + cfg.w_edge * norm_edge
        + cfg.w_glare * norm_glare
    )
    confidence = _clamp(confidence)

    # Night detection:
    #   Day override (primary)   – p95 very bright  → sky/sun visible → day
    #   Day override (secondary) – mean high enough  → scene too bright for night
    #     Overcast day:  mean ≈ 60-90,  p95 ≈ 150-200
    #     Night + lamps: mean ≈ 35-65,  p95 ≈ 130-180
    #   Night condition fires only when BOTH day overrides are false.
    is_day_override = (
        metrics.brightness_p95 > cfg.t_day_p95
        or metrics.brightness_mean > cfg.t_day_mean
    )
    is_night = (
        not is_day_override
        and (
            metrics.brightness_p25 < cfg.t_night_p25
            or (
                metrics.brightness_p05 < cfg.t_night_p05
                and metrics.brightness_mean < cfg.t_night_mean_max
            )
        )
    )
    is_degraded = confidence < cfg.t_vis
    is_glare = metrics.glare_score > cfg.t_glare

    return VisibilityEstimate(
        confidence=confidence,
        is_night=is_night,
        is_degraded=is_degraded,
        is_glare=is_glare,
    )
