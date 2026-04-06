"""Classical CV obstacle detector using background subtraction and contours.

The algorithm:
  1. Extract the road / forward-view ROI using lane boundaries when available.
  2. Apply a background subtractor (MOG2) to detect moving objects.
  3. Apply morphological operations to clean up the foreground mask.
  4. Find contours and filter by area, aspect ratio, and position.
  5. Estimate distance from bounding-box height using a simple pinhole model.

This is an intentionally simple classical approach. It works well on the
DADA-2000 dataset where the dashcam is static and objects move relative to
the background.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .types import DetectedObject


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class DetectorConfig:
    """Tunable parameters for detect_obstacles().

    Attributes
    ----------
    roi_top : float
        Top boundary of detection ROI as a fraction of frame height.
        Should be set to exclude sky and keep roads/intersections.
    roi_bottom : float
        Bottom boundary of detection ROI as a fraction of frame height.
        Should exclude the dashboard.
    min_area : float
        Minimum contour area in pixels to be considered an obstacle.
    max_area_fraction : float
        Maximum contour area as a fraction of the ROI area. Larger blobs
        are likely background artifacts.
    max_aspect_ratio : float
        Maximum width/height ratio. Very wide boxes are road markings.
    min_aspect_ratio : float
        Minimum width/height ratio. Very tall thin boxes are posts; allow.
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


DEFAULT_DETECTOR_CONFIG = DetectorConfig()


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class Detector:
    """Stateful obstacle detector wrapping a MOG2 background subtractor.

    The detector keeps its background model across frames, so it should be
    created once per video and reused for each frame in sequence.

    Usage
    -----
    detector = Detector()
    for frame in frames:
        objects = detector.detect(frame, lane_output=lane_out, context_state=ctx)
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
            Lane detection result. Used to restrict the detection ROI to
            the lane interior when available.
        context_state : ContextState, optional
            Current system context. Not used in the base implementation
            but kept for future adaptive tuning.

        Returns
        -------
        list[DetectedObject]
            Detected obstacles in full-frame pixel coordinates.
        """
        result = detect_obstacles(
            frame,
            lane_output=lane_output,
            context_state=context_state,
            bg_subtractor=self._bg_subtractor,
            frame_idx=self._frame_idx,
            config=self._config,
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
        Not used currently; reserved for future adaptive behavior.
    bg_subtractor : cv2.BackgroundSubtractor, optional
        Pre-created background subtractor with existing history.
    frame_idx : int
        Frame index for annotating results.
    config : DetectorConfig, optional
        Tunable parameters.

    Returns
    -------
    list[DetectedObject]
    """
    try:
        import cv2
    except ImportError:
        return []

    cfg = config or DEFAULT_DETECTOR_CONFIG

    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        return []

    h, w = frame.shape[:2]
    y1 = int(h * cfg.roi_top)
    y2 = int(h * cfg.roi_bottom)
    if y2 <= y1:
        return []

    roi = frame[y1:y2, :]
    roi_area = (y2 - y1) * w

    # Background subtraction
    if bg_subtractor is None:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history,
            varThreshold=cfg.mog2_var_threshold,
            detectShadows=True,
        )

    cfg_lr = cfg.mog2_learning_rate
    fg_mask = bg_subtractor.apply(roi, learningRate=cfg_lr)

    # Threshold: remove shadows (127) and keep foreground (255)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel_size = max(1, cfg.morph_kernel_size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        # Convert to full-frame coordinates
        full_bx = bx
        full_by = by + y1
        cx = float(full_bx + bw / 2)
        cy = float(full_by + bh / 2)

        # Simple confidence: larger, more compact contours score higher
        fill_ratio = area / float(bw * bh) if (bw * bh) > 0 else 0.0
        norm_area = min(1.0, area / max(1.0, max_area))
        confidence = 0.5 * fill_ratio + 0.5 * norm_area
        confidence = max(0.0, min(1.0, confidence))

        if confidence < cfg.min_confidence:
            continue

        # Distance heuristic: object_height_px -> distance via pinhole model
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
