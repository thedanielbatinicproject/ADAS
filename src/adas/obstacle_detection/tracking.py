"""Simple IoU-based multi-object tracker.

Assigns stable track IDs to detections across frames using the Hungarian
algorithm (or a greedy IoU match when scipy is unavailable).

Usage
-----
tracker = SimpleTracker(iou_threshold=0.30, max_missing=10)
objects_with_ids = tracker.update(raw_detections)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .types import DetectedObject


# ---------------------------------------------------------------------------
# Tracker state
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    """Internal track state."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    area: float = 0.0
    distance_estimate: float | None = None
    confidence: float = 1.0
    frame_idx: int = -1
    missing_frames: int = 0
    age: int = 1  # frames since creation


class SimpleTracker:
    """Greedy IoU-based tracker with a fixed-ID budget.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to match a detection to an existing track.
    max_missing : int
        Number of consecutive frames a track can be missing before it
        is pruned.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missing: int = 8,
        coast_frames: int = 3,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missing = max_missing
        self._coast_frames = max(0, int(coast_frames))
        self._tracks: Dict[int, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        """Clear all tracks (call when starting a new video)."""
        self._tracks = {}
        self._next_id = 1

    def update(self, detections: List[DetectedObject]) -> List[DetectedObject]:
        """Match detections to existing tracks and assign IDs.

        Parameters
        ----------
        detections : list[DetectedObject]
            Raw detections from detector.detect() (track_id may be -1).

        Returns
        -------
        list[DetectedObject]
            Same detections with track_id filled in.
        """
        if not detections:
            result: List[DetectedObject] = []
            dead = []
            for tid, track in self._tracks.items():
                track.missing_frames += 1
                if track.missing_frames <= self._coast_frames:
                    result.append(self._track_to_object(track, coasting=True))
                if track.missing_frames >= self._max_missing:
                    dead.append(tid)
            for tid in dead:
                del self._tracks[tid]
            return result

        track_ids = list(self._tracks.keys())
        track_bboxes = [self._tracks[tid].bbox for tid in track_ids]
        det_bboxes = [d.bbox for d in detections]

        # Build IoU matrix (len(tracks) x len(detections))
        if track_ids:
            iou_matrix = _iou_matrix(track_bboxes, det_bboxes)
            matched_track_idx, matched_det_idx = _greedy_match(
                iou_matrix, self._iou_threshold
            )
        else:
            matched_track_idx = []
            matched_det_idx = []

        matched_track_set = set(matched_track_idx)
        matched_det_set = set(matched_det_idx)

        # Update matched tracks
        result: List[DetectedObject] = []
        for t_idx, d_idx in zip(matched_track_idx, matched_det_idx):
            tid = track_ids[t_idx]
            track = self._tracks[tid]
            track.bbox = detections[d_idx].bbox
            track.area = detections[d_idx].area
            track.distance_estimate = detections[d_idx].distance_estimate
            track.confidence = detections[d_idx].confidence
            track.frame_idx = detections[d_idx].frame_idx
            track.missing_frames = 0
            track.age += 1
            det = detections[d_idx]
            result.append(DetectedObject(
                bbox=det.bbox,
                area=det.area,
                centroid=det.centroid,
                track_id=tid,
                distance_estimate=det.distance_estimate,
                confidence=det.confidence,
                frame_idx=det.frame_idx,
            ))

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_set:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = _Track(
                    track_id=new_id,
                    bbox=det.bbox,
                    area=det.area,
                    distance_estimate=det.distance_estimate,
                    confidence=det.confidence,
                    frame_idx=det.frame_idx,
                )
                result.append(DetectedObject(
                    bbox=det.bbox,
                    area=det.area,
                    centroid=det.centroid,
                    track_id=new_id,
                    distance_estimate=det.distance_estimate,
                    confidence=det.confidence,
                    frame_idx=det.frame_idx,
                ))

        # Increment missing counter for unmatched tracks; prune dead ones
        dead = []
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_track_set:
                track = self._tracks[tid]
                track.missing_frames += 1
                if track.missing_frames <= self._coast_frames:
                    result.append(self._track_to_object(track, coasting=True))
                if track.missing_frames >= self._max_missing:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        return result

    def _track_to_object(self, track: _Track, coasting: bool) -> DetectedObject:
        x, y, w, h = track.bbox
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        conf = float(track.confidence)
        if coasting and track.missing_frames > 0:
            conf *= max(0.2, 0.8 ** track.missing_frames)
        return DetectedObject(
            bbox=track.bbox,
            area=track.area if track.area > 0 else float(w * h),
            centroid=(cx, cy),
            track_id=track.track_id,
            distance_estimate=track.distance_estimate,
            confidence=max(0.0, min(1.0, conf)),
            frame_idx=track.frame_idx,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """Compute IoU of two bboxes in (x, y, w, h) format."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(
    tracks: List[Tuple[int, int, int, int]],
    dets: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Build an IoU matrix of shape (len(tracks), len(dets))."""
    mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            mat[i, j] = _iou(t, d)
    return mat


def _greedy_match(
    iou_matrix: np.ndarray,
    threshold: float,
) -> Tuple[List[int], List[int]]:
    """Greedy assignment: highest IoU first, each track/det used at most once."""
    matched_tracks: List[int] = []
    matched_dets: List[int] = []
    if iou_matrix.size == 0:
        return matched_tracks, matched_dets

    used_tracks = set()
    used_dets = set()

    # Flatten and sort by IoU descending
    indices = np.dstack(np.unravel_index(
        np.argsort(iou_matrix, axis=None)[::-1],
        iou_matrix.shape,
    ))[0]

    for t_idx, d_idx in indices:
        if iou_matrix[t_idx, d_idx] < threshold:
            break
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        matched_tracks.append(int(t_idx))
        matched_dets.append(int(d_idx))
        used_tracks.add(t_idx)
        used_dets.add(d_idx)

    return matched_tracks, matched_dets
