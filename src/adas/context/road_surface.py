"""
Heuristic road-surface estimation and braking-distance multiplier.

Analyses the bottom region of the frame (where the road is visible) to
guess whether the surface is dry asphalt, wet asphalt, gravel, or unknown.
The result is a *probabilistic hint* - downstream code must treat low
confidence as a reason to fall back to conservative defaults.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np

from .types import RoadSurfaceHint, RoadSurfaceType
from .defaults import ContextConfig, DEFAULT_CONFIG


# ------------------------------------------------------------------ internal


def _extract_roi(
    frame: Any,
    roi: Optional[Tuple[int, int, int, int]],
    default_fraction: float,
) -> Any:
    """Return the region of interest from *frame*.

    Parameters
    ----------
    roi : ``(x1, y1, x2, y2)`` or None
        Explicit ROI.  If *None*, the bottom ``default_fraction`` of the
        frame is used (this is where the road surface is typically visible
        in a dashcam perspective).
    """
    if roi is not None:
        x1, y1, x2, y2 = roi
        return frame[y1:y2, x1:x2]
    h = frame.shape[0]
    y_start = int(h * (1.0 - default_fraction))
    return frame[y_start:, :]


# -------------------------------------------------------------------- public


def estimate_road_surface(
    frame: Any,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    config: Optional[ContextConfig] = None,
) -> RoadSurfaceHint:
    """Heuristic road-surface classification from the bottom region of a frame.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image.
    roi : ``(x1, y1, x2, y2)``, optional
        Custom ROI.  Default: bottom :pyattr:`ContextConfig.default_roi_fraction`
        of the frame.
    config : ContextConfig, optional
    """
    import cv2

    cfg = config or DEFAULT_CONFIG

    if frame is None or not hasattr(frame, "size") or frame.size == 0:
        return RoadSurfaceHint()

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    region = _extract_roi(frame, roi, cfg.default_roi_fraction)
    if region.size == 0:
        return RoadSurfaceHint()

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # ---- texture roughness (Laplacian variance) ----
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    roughness = float(np.var(laplacian))

    # ---- colour uniformity (std of grayscale in ROI) ----
    uniformity_std = float(np.std(gray))

    # ---- specular highlights (fraction of near-white pixels) ----
    specular = float(np.count_nonzero(gray >= cfg.glare_pixel_threshold)) / max(
        gray.size, 1
    )

    # ---- classification ----
    if specular > cfg.t_specular:
        surface = RoadSurfaceType.ASPHALT_WET
        margin = (specular - cfg.t_specular) / max(cfg.t_specular, 1e-9)
        confidence = min(0.5 + 0.5 * margin, 1.0)
    elif roughness > cfg.t_roughness and uniformity_std > cfg.t_uniformity:
        surface = RoadSurfaceType.GRAVEL
        margin_r = (roughness - cfg.t_roughness) / max(cfg.t_roughness, 1e-9)
        margin_u = (uniformity_std - cfg.t_uniformity) / max(cfg.t_uniformity, 1e-9)
        confidence = min(0.5 + 0.25 * (margin_r + margin_u), 1.0)
    elif roughness <= cfg.t_roughness and uniformity_std <= cfg.t_uniformity:
        surface = RoadSurfaceType.ASPHALT_DRY
        margin_r = (cfg.t_roughness - roughness) / max(cfg.t_roughness, 1e-9)
        margin_u = (cfg.t_uniformity - uniformity_std) / max(cfg.t_uniformity, 1e-9)
        confidence = min(0.5 + 0.25 * (margin_r + margin_u), 1.0)
    else:
        surface = RoadSurfaceType.UNKNOWN
        confidence = 0.3

    return RoadSurfaceHint(surface_type=surface, confidence=confidence)


def braking_multiplier(
    surface_hint: RoadSurfaceHint,
    *,
    config: Optional[ContextConfig] = None,
) -> float:
    """Return a braking-distance multiplier for the estimated surface.

    The multiplier is blended between the surface-specific value and the
    *unknown* fallback according to ``surface_hint.confidence``, so a
    low-confidence estimate naturally converges towards the conservative
    default.

    Returns
    -------
    float
        Value ≥ 1.0 (multiply nominal braking / TTC threshold).
    """
    cfg = config or DEFAULT_CONFIG

    lookup = {
        RoadSurfaceType.ASPHALT_DRY: cfg.braking_dry,
        RoadSurfaceType.ASPHALT_WET: cfg.braking_wet,
        RoadSurfaceType.GRAVEL: cfg.braking_gravel,
        RoadSurfaceType.UNKNOWN: cfg.braking_unknown,
    }

    specific = lookup.get(surface_hint.surface_type, cfg.braking_unknown)
    fallback = cfg.braking_unknown

    return (
        surface_hint.confidence * specific + (1.0 - surface_hint.confidence) * fallback
    )
