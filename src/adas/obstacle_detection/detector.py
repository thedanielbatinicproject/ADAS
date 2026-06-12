"""Classical CV obstacle detector using background subtraction and contours.

Context-adaptive pipeline
--------------------------
The detector adjusts its operating parameters based on the current
ContextState to improve detection under varying conditions:

- Night (light_condition=NIGHT):
    Lower MOG2 variance threshold (less filtering of slow-moving shadows),
    larger morphological kernel to merge fragmented blobs.

- Rain (weather_condition=RAIN):
    Higher learning rate so the background model adapts to streaming water
    and moving reflections.  Smaller min_area to catch partially occluded
    vehicles.

- Fog (weather_condition=FOG):
    Reduced min_area and confidence threshold because low-contrast scenes
    produce weak foreground responses.

- Degraded mode (DEGRADED_MARKED / UNMARKED_DEGRADED):
    Combined adaptation: relaxed area filter, higher learning rate.

- Good-visibility unmarked road (UNMARKED_GOOD_VIS):
    Wider ROI (start higher in frame) because without lane boundaries
    larger portions of the forward scene are valid detection areas.

- Normal (NORMAL_MARKED, CLEAR, DAY):
    Default parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple

from .types import DetectedObject
from ..utils.runtime_overrides import apply_dataclass_overrides


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Tunable parameters for detect_obstacles().

    Attributes
    ----------
    roi_top : float
        Top boundary of detection ROI as a fraction of frame height.
    roi_bottom : float
        Bottom boundary of detection ROI as a fraction of frame height.
    min_area : float
        Minimum contour area in pixels to be considered an obstacle.
    max_area_fraction : float
        Maximum contour area as a fraction of the ROI area.
    max_aspect_ratio : float
        Maximum width/height ratio.
    min_aspect_ratio : float
        Minimum width/height ratio.
    mog2_history : int
        Number of frames for MOG2 background model history.
    mog2_var_threshold : float
        Variance threshold for MOG2 foreground detection.
    mog2_learning_rate : float
        MOG2 learning rate (0 = no update, 1 = full update per frame).
    morph_kernel_size : int
        Size of the morphological kernel for noise removal.
    min_confidence : float
        Minimum confidence score to include detection in results.
    focal_length_px : float
        Approximate focal length in pixels for distance estimation.
    assumed_object_height_m : float
        Typical obstacle height in meters (car height ~ 1.5 m).
    """

    roi_top: float = 0.30
    roi_bottom: float = 0.90
    min_area: float = 280.0
    max_area_fraction: float = 0.25
    max_aspect_ratio: float = 6.0
    min_aspect_ratio: float = 0.2
    mog2_history: int = 300
    mog2_var_threshold: float = 60.0
    mog2_learning_rate: float = 0.005
    foreground_threshold: int = 150
    morph_kernel_size: int = 5
    min_confidence: float = 0.18
    focal_length_px: float = 700.0
    assumed_object_height_m: float = 1.5
    min_bbox_width_px: int = 10
    min_bbox_height_px: int = 14
    max_bbox_area_fraction: float = 0.20
    max_bbox_width_fraction: float = 0.62
    max_bbox_height_fraction: float = 0.55
    near_camera_bottom_fraction: float = 0.78
    min_lane_overlap: float = 0.05
    min_lane_mask_coverage: float = 0.02
    merge_iou_threshold: float = 0.10
    merge_gap_px: int = 20
    merge_x_overlap_min: float = 0.45
    merge_vertical_gap_px: int = 60
    min_fill_ratio: float = 0.10
    min_solidity: float = 0.12
    max_detections: int = 20
    # Suppress detections while MOG2 background model is building up.
    # During these frames MOG2 is still updated but outputs are discarded.
    warmup_frames: int = 30


DEFAULT_DETECTOR_CONFIG = apply_dataclass_overrides(DetectorConfig(), "obstacle")


# ---------------------------------------------------------------------------
# Context-based config selector
# ---------------------------------------------------------------------------

def _config_for_context(
    base: DetectorConfig,
    context_state: Any,
) -> DetectorConfig:
    """Return a (possibly modified) config tailored to the current context.

    The returned config is a shallow copy of base with some fields overridden.
    base itself is never modified.
    """
    if context_state is None:
        return base

    mode_str = _get_attr_str(context_state, "mode")
    weather_str = _get_attr_str(context_state, "weather_condition")
    light_str = _get_attr_str(context_state, "light_condition")

    # Shallow dataclass copy preserving all fields/overrides.
    cfg = replace(base)

    if light_str == "night":
        # Night: more sensitive foreground detection
        cfg.mog2_var_threshold = max(20.0, base.mog2_var_threshold * 0.6)
        cfg.morph_kernel_size = base.morph_kernel_size + 2
        cfg.min_area = max(200.0, base.min_area * 0.7)
        cfg.min_confidence = max(0.2, base.min_confidence - 0.05)

    if weather_str == "rain":
        # Rain: faster background adaptation to handle reflections
        cfg.mog2_learning_rate = min(0.05, base.mog2_learning_rate * 4.0)
        cfg.min_area = max(250.0, base.min_area * 0.8)
        cfg.morph_kernel_size = base.morph_kernel_size + 2

    if weather_str == "fog":
        # Fog: relax confidence and area thresholds
        cfg.min_area = max(200.0, base.min_area * 0.6)
        cfg.min_confidence = max(0.18, base.min_confidence - 0.10)
        cfg.mog2_var_threshold = max(25.0, base.mog2_var_threshold * 0.7)

    if mode_str in ("degraded_marked", "unmarked_degraded"):
        cfg.mog2_learning_rate = min(0.04, cfg.mog2_learning_rate * 3.0)
        cfg.min_area = max(200.0, cfg.min_area * 0.75)

    if mode_str in ("unmarked_good_vis", "unmarked_degraded"):
        # No lane guidance: widen search area upward
        cfg.roi_top = max(0.20, base.roi_top - 0.10)

    return cfg


def _get_attr_str(context_state: Any, attr: str) -> str:
    val = getattr(context_state, attr, None)
    if val is None:
        return ""
    return str(getattr(val, "value", val)).lower()


# ---------------------------------------------------------------------------
# Detector class (stateful: maintains background model per video)
# ---------------------------------------------------------------------------

class Detector:
    """Stateful obstacle detector wrapping a MOG2 background subtractor.

    Create one instance per video and call detect() for each frame in order.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self._config = config or DEFAULT_DETECTOR_CONFIG
        self._bg_subtractor: Any = None
        self._frame_idx = 0
        self._init_bg_subtractor()

    def _init_bg_subtractor(self) -> None:
        try:
            import cv2
            cfg = self._config
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=cfg.mog2_history,
                varThreshold=cfg.mog2_var_threshold,
                detectShadows=True,
            )
        except ImportError:
            self._bg_subtractor = None

    def reset(self) -> None:
        """Reset the background model (call when switching videos)."""
        self._frame_idx = 0
        self._init_bg_subtractor()

    def update_config(self, config: DetectorConfig) -> None:
        """Replace the base config and recreate the background model."""
        self._config = config
        self._init_bg_subtractor()

    def detect(
        self,
        frame: Any,
        lane_output: Any = None,
        context_state: Any = None,
    ) -> List[DetectedObject]:
        """Detect obstacles in the given frame.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR uint8 image, shape (H, W, 3).
        lane_output : LaneOutput, optional
            Lane detection result.  Used to restrict the ROI to the lane
            interior when available.
        context_state : ContextState, optional
            Current system context.  Used to adapt detection parameters.

        Returns
        -------
        list[DetectedObject]
        """
        # Derive a per-frame config based on context
        effective_cfg = _config_for_context(self._config, context_state)

        result = detect_obstacles(
            frame,
            lane_output=lane_output,
            context_state=context_state,
            bg_subtractor=self._bg_subtractor,
            frame_idx=self._frame_idx,
            config=effective_cfg,
        )
        self._frame_idx += 1
        # While MOG2 is still building its background model the foreground mask
        # contains almost everything.  Suppress output until warmup is done.
        if self._frame_idx <= effective_cfg.warmup_frames:
            return []
        return result


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------

def detect_obstacles(
    frame: Any,
    lane_output: Any = None,
    context_state: Any = None,
    bg_subtractor: Any = None,
    frame_idx: int = -1,
    config: Optional[DetectorConfig] = None,
) -> List[DetectedObject]:
    """Detect moving obstacles in a dashcam frame (functional API).

    Prefer using the Detector class which maintains background model state.
    This function accepts an external bg_subtractor so you can share state.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR uint8 image.
    lane_output : LaneOutput, optional
        Used to restrict the search to the lane ROI.
    context_state : ContextState, optional
        Used to adapt the effective config if no explicit config is given.
    bg_subtractor : cv2.BackgroundSubtractor, optional
        Pre-created background subtractor with existing history.
    frame_idx : int
        Frame index for annotating results.
    config : DetectorConfig, optional
        Tunable parameters.  When None and context_state is provided,
        an adapted config derived from DEFAULT_DETECTOR_CONFIG is used.
    """
    try:
        import cv2
    except ImportError:
        return []

    if config is None:
        config = _config_for_context(DEFAULT_DETECTOR_CONFIG, context_state)

    cfg = config

    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    y1 = int(h * cfg.roi_top)
    y2 = int(h * cfg.roi_bottom)
    if y2 <= y1:
        return []

    roi = frame[y1:y2, :]
    roi_area = (y2 - y1) * w

    if bg_subtractor is None:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history,
            varThreshold=cfg.mog2_var_threshold,
            detectShadows=True,
        )

    fg_mask = bg_subtractor.apply(roi, learningRate=cfg.mog2_learning_rate)

    fg_thr = int(max(80, min(240, cfg.foreground_threshold)))
    _, fg_mask = cv2.threshold(fg_mask, fg_thr, 255, cv2.THRESH_BINARY)

    kernel_size = max(1, cfg.morph_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)
    # Join fragmented pieces (person head/body, vehicle parts) into one blob.
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    results: List[DetectedObject] = []
    max_area = roi_area * cfg.max_area_fraction
    candidates: List[Tuple[Tuple[int, int, int, int], float]] = []

    lane_mask = None
    lane_overlap_enabled = False
    lane_roi_y1 = 0  # lane-mask top in full-frame pixel coords
    if lane_output is not None and hasattr(lane_output, "mask"):
        lm = getattr(lane_output, "mask", None)
        if lm is not None and hasattr(lm, "shape") and lm.size > 0:
            lane_mask = lm
            lane_roi_y1 = int(getattr(lane_output, "roi_y1", 0))
            # Coverage check: is there enough lane area to trust the mask?
            lane_cov = float((lm > 0).sum()) / float(max(1, lm.size))
            lane_overlap_enabled = lane_cov >= cfg.min_lane_mask_coverage

    roi_h = max(1, y2 - y1)
    max_bbox_area = roi_area * cfg.max_bbox_area_fraction
    max_bbox_w = int(w * cfg.max_bbox_width_fraction)
    max_bbox_h = int(roi_h * cfg.max_bbox_height_fraction)
    near_bottom_y = int(roi_h * cfg.near_camera_bottom_fraction)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bh == 0:
            continue

        if bw < cfg.min_bbox_width_px or bh < cfg.min_bbox_height_px:
            continue

        aspect = bw / bh
        if aspect > cfg.max_aspect_ratio or aspect < cfg.min_aspect_ratio:
            continue

        bbox_area = float(max(1, bw * bh))
        is_giant = (
            bbox_area > max_bbox_area
            or bw > max_bbox_w
            or bh > max_bbox_h
        )
        near_camera = (by + bh) >= near_bottom_y
        if is_giant and not near_camera:
            continue

        fill_ratio = area / bbox_area
        if fill_ratio < cfg.min_fill_ratio:
            continue
        if is_giant and fill_ratio < max(cfg.min_fill_ratio, 0.18):
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 1e-6 else 0.0
        if solidity < cfg.min_solidity:
            continue

        # If lane mask is available, keep detections that overlap the drivable
        # area; this removes many side-scene false positives (trees/branches).
        if lane_mask is not None and lane_overlap_enabled:
            # Convert obstacle-ROI-relative coords to lane-mask-relative coords.
            # obstacle full-frame y = by + y1 (detector ROI top)
            # lane mask y = full-frame y - lane_roi_y1
            lm_y1r = max(0, (by + y1) - lane_roi_y1)
            lm_y2r = min(lane_mask.shape[0], (by + bh + y1) - lane_roi_y1)
            x1r = max(0, bx)
            x2r = min(lane_mask.shape[1], bx + bw)
            cx = bx + (bw * 0.5)
            central_view = (0.22 * w) <= cx <= (0.78 * w)  # narrowed from 0.15/0.85
            if lm_y2r > lm_y1r and x2r > x1r and lm_y1r < lane_mask.shape[0]:
                patch = lane_mask[lm_y1r:lm_y2r, x1r:x2r]
                overlap = float((patch > 0).sum()) / float(max(1, patch.size))
                if overlap < cfg.min_lane_overlap and not central_view:
                    continue
            elif not central_view:
                continue

        # Hard geometric boundary check using lane polynomials.
        # Rejects objects clearly outside the lane even when mask overlap passes.
        # Only applied when lane detection is confident and not a synthetic trapezoid.
        if (
            lane_output is not None
            and not getattr(lane_output, 'is_trapezoid', True)
            and getattr(lane_output, 'lane_confidence', 0.0) >= 0.45
            and getattr(lane_output, 'left_poly', None) is not None
            and getattr(lane_output, 'right_poly', None) is not None
        ):
            full_cy = float(by + y1 + bh * 0.5)
            full_cx = float(bx + bw * 0.5)
            lane_roi_y1_geo = int(getattr(lane_output, 'roi_y1', 0))
            lane_roi_y2_geo = int(getattr(lane_output, 'roi_y2', h))
            lane_y = full_cy - lane_roi_y1_geo
            lane_roi_h_geo = max(1, lane_roi_y2_geo - lane_roi_y1_geo)
            if 0 <= lane_y <= lane_roi_h_geo:
                lp = lane_output.left_poly
                rp = lane_output.right_poly
                left_x = lp[0] * lane_y**2 + lp[1] * lane_y + lp[2]
                right_x = rp[0] * lane_y**2 + rp[1] * lane_y + rp[2]
                margin = 70.0  # px: allow objects partially overlapping lane edge
                if full_cx < left_x - margin or full_cx > right_x + margin:
                    continue

        candidates.append(((bx, by, bw, bh), area))

    merged = _merge_candidate_boxes(
        candidates,
        iou_thr=cfg.merge_iou_threshold,
        gap_px=cfg.merge_gap_px,
        x_overlap_thr=cfg.merge_x_overlap_min,
        vertical_gap_px=cfg.merge_vertical_gap_px,
    )

    for (bx, by, bw, bh), area in merged:
        merged_bbox_area = float(max(1, bw * bh))
        merged_is_giant = (
            merged_bbox_area > max_bbox_area
            or bw > max_bbox_w
            or bh > max_bbox_h
        )
        merged_near_camera = (by + bh) >= near_bottom_y
        if merged_is_giant and not merged_near_camera:
            continue

        full_bx = bx
        full_by = by + y1
        cx = float(full_bx + bw / 2)
        cy = float(full_by + bh / 2)

        fill_ratio = area / float(bw * bh) if (bw * bh) > 0 else 0.0
        fill_ratio = max(0.0, min(1.0, fill_ratio))
        if merged_is_giant and fill_ratio < 0.25:
            continue
        norm_area = min(1.0, area / max(1.0, max_area))
        confidence = 0.5 * fill_ratio + 0.5 * norm_area
        confidence = max(0.0, min(1.0, confidence))

        if confidence < cfg.min_confidence:
            continue

        distance_est: Optional[float] = None
        if bh > 0:
            distance_est = (cfg.focal_length_px * cfg.assumed_object_height_m) / float(bh)
            distance_est = round(distance_est, 1)

        results.append(DetectedObject(
            bbox=(full_bx, full_by, bw, bh),
            area=area,
            centroid=(cx, cy),
            track_id=-1,
            distance_estimate=distance_est,
            confidence=confidence,
            frame_idx=frame_idx,
        ))

    if len(results) > cfg.max_detections:
        # Keep the strongest obstacles and drop tiny clutter.
        results.sort(key=lambda o: (o.confidence, o.area), reverse=True)
        results = results[:cfg.max_detections]

    return results


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _bbox_union(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def _bbox_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    gap_x = max(0, max(ax, bx) - min(ax2, bx2))
    gap_y = max(0, max(ay, by) - min(ay2, by2))
    return gap_x, gap_y


def _bbox_x_overlap_ratio(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax, _ay, aw, _ah = a
    bx, _by, bw, _bh = b
    ax2 = ax + aw
    bx2 = bx + bw
    inter = max(0, min(ax2, bx2) - max(ax, bx))
    denom = float(max(1, min(aw, bw)))
    return float(inter) / denom


def _merge_candidate_boxes(
    candidates: List[Tuple[Tuple[int, int, int, int], float]],
    *,
    iou_thr: float,
    gap_px: int,
    x_overlap_thr: float,
    vertical_gap_px: int,
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    if not candidates:
        return []

    merged = candidates[:]
    changed = True
    while changed:
        changed = False
        out: List[Tuple[Tuple[int, int, int, int], float]] = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            box_i, area_i = merged[i]
            cur_box = box_i
            cur_area = area_i
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                box_j, area_j = merged[j]
                iou = _bbox_iou(cur_box, box_j)
                gx, gy = _bbox_gap(cur_box, box_j)
                x_ov = _bbox_x_overlap_ratio(cur_box, box_j)
                # Merge when highly overlapping, very close, or vertically stacked
                # with substantial x-overlap (head/body or split vehicle blob).
                should_merge = (
                    iou >= iou_thr
                    or (gx <= gap_px and gy <= gap_px)
                    or (gx <= gap_px and gy <= vertical_gap_px and x_ov >= x_overlap_thr)
                )
                if should_merge:
                    cur_box = _bbox_union(cur_box, box_j)
                    cur_area = cur_area + area_j
                    used[j] = True
                    changed = True

            out.append((cur_box, cur_area))

        merged = out

    return merged
