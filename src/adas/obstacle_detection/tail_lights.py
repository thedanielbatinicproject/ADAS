"""Tail-light based vehicle detector.

Detects vehicles by finding paired red/white tail-light blobs.
Works even when MOG2 fails (vehicles moving at same speed as ego).

Algorithm
---------
1. Extract red-light mask from HSV (H near 0/180°, high S+V).
2. Find blobs within a sensible size range.
3. Pair blobs that are:
   - At similar vertical position (same car row).
   - Horizontally separated by a distance consistent with a car width.
4. Estimate distance from the blob pair vertical size using pinhole model:
   distance = focal_px * assumed_light_height_m / light_blob_height_px
5. Estimate relative speed across frames from distance change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import DetectedObject


@dataclass
class TailLightConfig:
    # ROI (same fractions as main detector)
    roi_top: float = 0.25
    roi_bottom: float = 0.80

    # Red HSV thresholds — two ranges cover the full red hue wrap-around
    red_h_lo1: int = 0
    red_h_hi1: int = 12
    red_h_lo2: int = 168
    red_h_hi2: int = 180
    red_s_min: int = 90
    red_v_min: int = 90

    # Blob size constraints (px)
    min_blob_area: float = 12.0
    max_blob_area: float = 4000.0
    max_blob_aspect: float = 5.0   # width/height

    # Pairing: how similar must the blob centres be vertically?
    pair_max_y_diff_frac: float = 0.06  # fraction of ROI height
    # Minimum and maximum light separation consistent with a real car
    pair_min_sep_frac: float = 0.04   # fraction of frame width
    pair_max_sep_frac: float = 0.60   # fraction of frame width

    # Distance estimation
    focal_length_px: float = 700.0
    # Known real-world separation between tail lights of a car (m)
    assumed_light_sep_m: float = 1.4
    # Car body bbox expansion factor around the light pair
    car_height_expand: float = 3.5    # total car height ≈ 3.5× light blob height
    car_width_expand: float = 1.15    # car body slightly wider than light sep

    # Minimum confidence score
    min_confidence: float = 0.45


DEFAULT_TAIL_LIGHT_CONFIG = TailLightConfig()


def detect_tail_light_vehicles(
    frame: Any,
    roi_y1: int,
    roi_y2: int,
    cfg: TailLightConfig,
    frame_idx: int = -1,
) -> List[DetectedObject]:
    """Detect vehicles via paired tail-light blobs.

    Parameters
    ----------
    frame : numpy.ndarray
        Full BGR frame.
    roi_y1, roi_y2 : int
        Vertical ROI bounds (full-frame coords).
    cfg : TailLightConfig
    frame_idx : int

    Returns
    -------
    list[DetectedObject]  — one entry per paired tail-light (= one vehicle).
    """
    try:
        import cv2
    except ImportError:
        return []

    if frame is None or not hasattr(frame, "shape"):
        return []

    h, w = frame.shape[:2]
    roi = frame[roi_y1:roi_y2, :]
    roi_h = roi_y2 - roi_y1
    if roi_h <= 0 or w <= 0:
        return []

    # --- Build red mask ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(
        hsv,
        np.array([cfg.red_h_lo1, cfg.red_s_min, cfg.red_v_min], dtype=np.uint8),
        np.array([cfg.red_h_hi1, 255, 255], dtype=np.uint8),
    )
    red2 = cv2.inRange(
        hsv,
        np.array([cfg.red_h_lo2, cfg.red_s_min, cfg.red_v_min], dtype=np.uint8),
        np.array([cfg.red_h_hi2, 255, 255], dtype=np.uint8),
    )
    red_mask = cv2.bitwise_or(red1, red2)

    # Small morphological cleanup to connect fragmented light pixels
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, k, iterations=2)

    # --- Find blobs ---
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    blobs: List[Tuple[int, int, int, int, float]] = []  # (cx, cy, w, h, area)
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < cfg.min_blob_area or area > cfg.max_blob_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bh == 0:
            continue
        asp = bw / bh
        if asp > cfg.max_blob_aspect:
            continue
        blobs.append((bx + bw // 2, by + bh // 2, bw, bh, area))

    if len(blobs) < 2:
        return []

    results: List[DetectedObject] = []
    used = set()

    pair_max_dy = int(roi_h * cfg.pair_max_y_diff_frac)
    pair_min_dx = int(w * cfg.pair_min_sep_frac)
    pair_max_dx = int(w * cfg.pair_max_sep_frac)

    # Sort blobs left-to-right for pairing
    blobs_sorted = sorted(blobs, key=lambda b: b[0])

    for i, (cx_i, cy_i, bw_i, bh_i, area_i) in enumerate(blobs_sorted):
        if i in used:
            continue
        best_j: Optional[int] = None
        best_score = float("inf")

        for j, (cx_j, cy_j, bw_j, bh_j, area_j) in enumerate(blobs_sorted):
            if j <= i or j in used:
                continue
            dx = abs(cx_j - cx_i)
            dy = abs(cy_j - cy_i)
            if dy > pair_max_dy:
                continue
            if dx < pair_min_dx or dx > pair_max_dx:
                continue
            # Prefer pairs where blobs are similar size
            size_ratio = max(area_i, area_j) / max(0.1, min(area_i, area_j))
            if size_ratio > 8.0:
                continue
            score = dy + abs(size_ratio - 1.0) * 20
            if score < best_score:
                best_score = score
                best_j = j

        if best_j is None:
            continue

        used.add(i)
        used.add(best_j)

        cx_j, cy_j, bw_j, bh_j, area_j = blobs_sorted[best_j]

        # Build car bounding box from light pair
        light_sep_px = abs(cx_j - cx_i)
        light_h_avg = (bh_i + bh_j) / 2.0

        car_w = int(light_sep_px * cfg.car_width_expand)
        car_h = int(max(light_h_avg * cfg.car_height_expand, light_sep_px * 0.5))

        pair_cx = (cx_i + cx_j) // 2
        pair_cy = (cy_i + cy_j) // 2

        # Bottom of car ≈ bottom of light blobs + small margin
        car_bottom_roi = max(cy_i, cy_j) + int(bh_i * 0.5)
        car_top_roi = max(0, car_bottom_roi - car_h)
        car_left = max(0, pair_cx - car_w // 2)
        car_right = min(w - 1, pair_cx + car_w // 2)
        car_bw = car_right - car_left
        car_bh = car_bottom_roi - car_top_roi

        if car_bw <= 0 or car_bh <= 0:
            continue

        # Full-frame coords
        full_y = car_top_roi + roi_y1

        # Distance from light separation using pinhole: d = f * real_sep / pixel_sep
        dist_est: Optional[float] = None
        if light_sep_px > 4:
            dist_est = round(cfg.focal_length_px * cfg.assumed_light_sep_m / light_sep_px, 1)

        # Confidence: higher when blobs are well-matched and symmetric
        size_ratio = max(area_i, area_j) / max(0.1, min(area_i, area_j))
        dy_norm = abs(cy_j - cy_i) / max(1, pair_max_dy)
        confidence = max(0.0, min(1.0, cfg.min_confidence + 0.4 * (1.0 - dy_norm) * (1.0 / max(1.0, size_ratio))))

        if confidence < cfg.min_confidence:
            continue

        results.append(DetectedObject(
            bbox=(car_left, full_y, car_bw, car_bh),
            area=float(car_bw * car_bh),
            centroid=(float(car_left + car_bw / 2), float(full_y + car_bh / 2)),
            track_id=-1,
            distance_estimate=dist_est,
            confidence=confidence,
            frame_idx=frame_idx,
        ))

    return results
