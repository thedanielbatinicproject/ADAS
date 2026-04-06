"""Obstacle detection evaluation metrics.

Provides IoU-based precision/recall/F1 evaluation for detected objects
against ground-truth bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import DetectedObject


@dataclass
class DetectionEvalResult:
    """Aggregate evaluation result for obstacle detection."""

    n_frames: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    iou_threshold: float = 0.5


def evaluate_detections(
    predicted: List[DetectedObject],
    ground_truth: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> dict:
    """Evaluate predicted bounding boxes against ground-truth boxes.

    Parameters
    ----------
    predicted : list[DetectedObject]
        Detected objects for one frame.
    ground_truth : list[tuple[int, int, int, int]]
        Ground-truth bounding boxes in (x, y, w, h) format.
    iou_threshold : float
        Minimum IoU to count as a true positive.

    Returns
    -------
    dict
        Keys: tp, fp, fn, precision, recall, f1.
    """
    tp = 0
    fp = 0
    fn = 0

    matched_gt = set()

    for det in predicted:
        best_iou = 0.0
        best_gt_idx = -1
        for g_idx, gt_bbox in enumerate(ground_truth):
            if g_idx in matched_gt:
                continue
            iou_val = _iou(det.bbox, gt_bbox)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt_idx = g_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate_evaluation(
    per_frame_results: List[dict],
) -> DetectionEvalResult:
    """Aggregate per-frame evaluation dicts into a summary.

    Parameters
    ----------
    per_frame_results : list[dict]
        Each element is the output of evaluate_detections() for one frame.

    Returns
    -------
    DetectionEvalResult
    """
    result = DetectionEvalResult(n_frames=len(per_frame_results))
    for r in per_frame_results:
        result.tp += r.get("tp", 0)
        result.fp += r.get("fp", 0)
        result.fn += r.get("fn", 0)

    if result.tp + result.fp > 0:
        result.precision = result.tp / (result.tp + result.fp)
    if result.tp + result.fn > 0:
        result.recall = result.tp / (result.tp + result.fn)
    if result.precision + result.recall > 0:
        result.f1 = (
            2 * result.precision * result.recall
            / (result.precision + result.recall)
        )

    return result


def _iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
