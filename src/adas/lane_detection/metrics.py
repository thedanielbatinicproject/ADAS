"""Lane detection evaluation metrics.

Provides functions for benchmarking LaneOutput against ground-truth labels.
Useful for regression tests and offline evaluation on annotated frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .processing import LaneOutput


@dataclass
class LaneEvalResult:
    """Summary of lane detection evaluation on a set of frames."""

    n_frames: int = 0
    n_has_lanes_pred: int = 0
    n_has_lanes_gt: int = 0
    tp: int = 0   # predicted has_lanes=True, GT has_lanes=True
    fp: int = 0   # predicted has_lanes=True, GT has_lanes=False
    fn: int = 0   # predicted has_lanes=False, GT has_lanes=True
    tn: int = 0   # predicted has_lanes=False, GT has_lanes=False
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_confidence: float = 0.0


def evaluate_detection(
    lane_output: LaneOutput,
    gt_has_lanes: bool,
) -> Dict[str, Any]:
    """Evaluate a single frame prediction against a ground-truth label.

    Parameters
    ----------
    lane_output : LaneOutput
        Output from process_frame().
    gt_has_lanes : bool
        Ground truth: True if lane markings are present in this frame.

    Returns
    -------
    dict
        Keys: predicted, ground_truth, correct, confidence.
    """
    predicted = lane_output.has_lanes
    return {
        "predicted": predicted,
        "ground_truth": gt_has_lanes,
        "correct": predicted == gt_has_lanes,
        "confidence": lane_output.lane_confidence,
    }


def evaluate_batch(
    outputs: List[LaneOutput],
    ground_truths: List[bool],
) -> LaneEvalResult:
    """Aggregate evaluation over a list of frames.

    Parameters
    ----------
    outputs : list[LaneOutput]
    ground_truths : list[bool]
        Must be the same length as outputs.

    Returns
    -------
    LaneEvalResult
    """
    if len(outputs) != len(ground_truths):
        raise ValueError(
            f"outputs and ground_truths must have the same length, "
            f"got {len(outputs)} vs {len(ground_truths)}"
        )

    result = LaneEvalResult(n_frames=len(outputs))
    confidence_sum = 0.0

    for out, gt in zip(outputs, ground_truths):
        pred = out.has_lanes
        confidence_sum += out.lane_confidence
        if gt:
            result.n_has_lanes_gt += 1
        if pred:
            result.n_has_lanes_pred += 1
        if pred and gt:
            result.tp += 1
        elif pred and not gt:
            result.fp += 1
        elif not pred and gt:
            result.fn += 1
        else:
            result.tn += 1

    if result.n_frames > 0:
        result.mean_confidence = confidence_sum / result.n_frames
    if result.tp + result.fp > 0:
        result.precision = result.tp / (result.tp + result.fp)
    if result.tp + result.fn > 0:
        result.recall = result.tp / (result.tp + result.fn)
    if result.precision + result.recall > 0:
        result.f1 = (
            2 * result.precision * result.recall / (result.precision + result.recall)
        )

    return result
