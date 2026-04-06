"""Collision risk evaluation metrics.

Evaluation of the ADAS risk/action predictions against ground-truth
accident labels from the DADA-2000 annotation CSV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .types import SystemAction


@dataclass
class RiskEvalResult:
    """Aggregate evaluation of SystemAction predictions vs ground truth."""

    n_frames: int = 0
    n_positive: int = 0   # GT: danger frames (accident or pre-accident)
    n_negative: int = 0   # GT: normal frames
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    false_alarm_rate: float = 0.0  # FP / (FP + TN)


def evaluate_action(
    predicted: SystemAction,
    gt_label: str,
) -> Dict[str, Any]:
    """Evaluate a single frame prediction against a ground-truth label.

    Parameters
    ----------
    predicted : SystemAction
        System action chosen by decide().
    gt_label : str
        One of: "normal", "abnormal", "accident_frame".
        "abnormal" and "accident_frame" are treated as positive/dangerous.

    Returns
    -------
    dict with keys: tp, fp, fn, tn, predicted_positive, gt_positive.
    """
    gt_positive = gt_label in ("abnormal", "accident_frame")
    pred_positive = predicted != SystemAction.NONE

    tp = int(pred_positive and gt_positive)
    fp = int(pred_positive and not gt_positive)
    fn = int(not pred_positive and gt_positive)
    tn = int(not pred_positive and not gt_positive)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "predicted_positive": pred_positive,
        "gt_positive": gt_positive,
    }


def aggregate_evaluation(
    per_frame_results: List[Dict[str, Any]],
) -> RiskEvalResult:
    """Aggregate per-frame evaluation into a summary.

    Parameters
    ----------
    per_frame_results : list[dict]
        Each element is the output of evaluate_action() for one frame.

    Returns
    -------
    RiskEvalResult
    """
    result = RiskEvalResult(n_frames=len(per_frame_results))
    for r in per_frame_results:
        result.tp += r.get("tp", 0)
        result.fp += r.get("fp", 0)
        result.fn += r.get("fn", 0)
        result.tn += r.get("tn", 0)

    result.n_positive = result.tp + result.fn
    result.n_negative = result.fp + result.tn

    if result.tp + result.fp > 0:
        result.precision = result.tp / (result.tp + result.fp)
    if result.tp + result.fn > 0:
        result.recall = result.tp / (result.tp + result.fn)
    if result.precision + result.recall > 0:
        result.f1 = (
            2 * result.precision * result.recall
            / (result.precision + result.recall)
        )
    if result.fp + result.tn > 0:
        result.false_alarm_rate = result.fp / (result.fp + result.tn)

    return result
