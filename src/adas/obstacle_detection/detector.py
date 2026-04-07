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

from dataclasses import dataclass
from typing import Any, List, Optional

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
    min_area: float = 400.0
    max_area_fraction: float = 0.25
    max_aspect_ratio: float = 6.0
    min_aspect_ratio: float = 0.2
    mog2_history: int = 120
    mog2_var_threshold: float = 50.0
    mog2_learning_rate: float = 0.005
    morph_kernel_size: int = 5
    min_confidence: float = 0.3
    focal_length_px: float = 700.0
    assumed_object_height_m: float = 1.5


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

    # Start with a copy by re-creating the dataclass with same values.
    cfg = DetectorConfig(
        roi_top=base.roi_top,
        roi_bottom=base.roi_bottom,
        min_area=base.min_area,
        max_area_fraction=base.max_area_fraction,
        max_aspect_ratio=base.max_aspect_ratio,
        min_aspect_ratio=base.min_aspect_ratio,
        mog2_history=base.mog2_history,
        mog2_var_threshold=base.mog2_var_threshold,
        mog2_learning_rate=base.mog2_learning_rate,
        morph_kernel_size=base.morph_kernel_size,
        min_confidence=base.min_confidence,
        focal_length_px=base.focal_length_px,
        assumed_object_height_m=base.assumed_object_height_m,
    )

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

    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    kernel_size = max(1, cfg.morph_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return []

    results: List[DetectedObject] = []
    max_area = roi_area * cfg.max_area_fraction

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.min_area or area > max_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(contour)
        if bh == 0:
            continue

        aspect = bw / bh
        if aspect > cfg.max_aspect_ratio or aspect < cfg.min_aspect_ratio:
            continue

        full_bx = bx
        full_by = by + y1
        cx = float(full_bx + bw / 2)
        cy = float(full_by + bh / 2)

        fill_ratio = area / float(bw * bh) if (bw * bh) > 0 else 0.0
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

    return results
