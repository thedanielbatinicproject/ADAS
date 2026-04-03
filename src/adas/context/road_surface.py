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
    bottom_margin: float = 0.0,
) -> Any:
    """Return the region of interest from *frame*.

    Parameters
    ----------
    roi : ``(x1, y1, x2, y2)`` or None
        Explicit ROI.  If *None*, the rows between
        ``h*(1 - default_fraction - bottom_margin)`` and
        ``h*(1 - bottom_margin)`` are used — this is the road
        surface visible through the windscreen, excluding the dashboard.
    """
    if roi is not None:
        x1, y1, x2, y2 = roi
        return frame[y1:y2, x1:x2]
    h = frame.shape[0]
    y_end = int(h * (1.0 - bottom_margin))
    y_start = int(h * (1.0 - default_fraction - bottom_margin))
    y_start = max(y_start, 0)
    y_end = max(y_end, y_start + 1)
    return frame[y_start:y_end, :]


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

    region = _extract_roi(frame, roi, cfg.default_roi_fraction, cfg.road_roi_bottom_margin)
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
    # Priority: WET (specular highlights) > GRAVEL (high roughness) > DRY (default).
    # UNKNOWN is reserved only for situations where the ROI gives no usable data.
    # A smooth road is DRY by default — we only override with positive evidence.
    if specular > cfg.t_specular:
        surface = RoadSurfaceType.ASPHALT_WET
        margin = (specular - cfg.t_specular) / max(cfg.t_specular, 1e-9)
        confidence = min(0.5 + 0.5 * margin, 1.0)
    elif roughness > cfg.t_roughness:
        surface = RoadSurfaceType.GRAVEL
        margin_r = (roughness - cfg.t_roughness) / max(cfg.t_roughness, 1e-9)
        confidence = min(0.5 + 0.4 * margin_r, 1.0)
    else:
        # Default: smooth surface → dry asphalt.
        # Confidence scales with how far below roughness threshold we are
        # (very smooth = very confident dry).
        surface = RoadSurfaceType.ASPHALT_DRY
        margin_r = (cfg.t_roughness - roughness) / max(cfg.t_roughness, 1e-9)
        confidence = min(0.45 + 0.45 * margin_r, 0.90)

    # Confidence cap when ROI is very dark (night / underpass):
    # hard to distinguish wet from dry without visible specular patterns.
    roi_brightness = float(np.mean(gray))
    if roi_brightness < 40.0:
        confidence = min(confidence, 0.40)
    elif roi_brightness < 70.0:
        confidence = min(confidence, 0.65)

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
