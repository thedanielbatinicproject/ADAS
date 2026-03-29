"""
DADA2000 dataset reader: simple access to video IDs, annotations, and paths.

Features:
- Parse DADA2000_video_annotations.csv (CSV, delimiter=';')
- Get all video IDs and their annotations
- Map video ID to folder/video path
- Get annotation for a video or a frame
"""
import os
import csv
from typing import Optional, Dict, Any, List

def load_annotations(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load DADA2000_video_annotations.csv and return a dict by video_id.
    """
    annotations: Dict[str, Dict[str, Any]] = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        try:
            next(reader)
        except StopIteration:
            return {}
        for row in reader:
            if not row:
                continue
            vid = row[0].strip() if len(row) > 0 else ""
            if not vid:
                continue
            def _safe_int(v):
                try:
                    return int(v)
                except Exception:
                    return None
            entry: Dict[str, Any] = {"video_id": vid}
            entry["weather"] = row[1].strip() if len(row) > 1 else None
            entry["light"] = row[2].strip() if len(row) > 2 else None
            entry["scenes"] = row[3].strip() if len(row) > 3 else None
            entry["linear"] = row[4].strip() if len(row) > 4 else None
            entry["type"] = row[5].strip() if len(row) > 5 else None
            entry["accident_flag"] = _safe_int(row[6]) if len(row) > 6 else None
            entry["abnormal_start_frame"] = _safe_int(row[7]) if len(row) > 7 else None
            entry["accident_frame"] = _safe_int(row[8]) if len(row) > 8 else None
            entry["abnormal_end_frame"] = _safe_int(row[9]) if len(row) > 9 else None
            entry["total_frames"] = _safe_int(row[10]) if len(row) > 10 else None
            # intervals (11..15)
            intervals = []
            for pos in range(11, 16):
                if len(row) > pos and row[pos].strip() != "":
                    intervals.append(_safe_int(row[pos].strip()))
                else:
                    intervals.append(None)
            entry["intervals"] = intervals
            entry["texts"] = row[16].strip() if len(row) > 16 else None
            entry["causes"] = row[17].strip() if len(row) > 17 else None
            entry["measures"] = row[18].strip() if len(row) > 18 else None
            annotations[vid] = entry
    return annotations

def get_all_video_ids(annotations: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return all video IDs from the annotations dict.
    """
    return list(annotations.keys())

def get_annotation_for_video(annotations: Dict[str, Dict[str, Any]], video_id: str) -> Optional[Dict[str, Any]]:
    """
    Return annotation for the given video_id.
    """
    return annotations.get(str(video_id))

def get_annotation_for_frame(annotations: Dict[str, Dict[str, Any]], video_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
    """
    Return annotation for a frame (adds label: 'normal', 'abnormal', 'accident_frame').
    """
    ann = get_annotation_for_video(annotations, video_id)
    if not ann:
        return None
    label = "normal"
    af = ann.get("accident_frame")
    bs = ann.get("abnormal_start_frame")
    be = ann.get("abnormal_end_frame")
    try:
        if af is not None and af >= 0 and frame_idx == af:
            label = "accident_frame"
        elif bs is not None and be is not None and bs <= frame_idx <= be:
            label = "abnormal"
    except Exception:
        pass
    out = dict(ann)
    out.update({"frame_idx": int(frame_idx), "label": label})
    return out

def get_video_path(dataset_root: str, video_id: str) -> Optional[str]:
    """
    Return absolute path to folder/video for the given video_id (recursive search).
    """
    best_match = None
    max_depth = -1
    for root, dirs, files in os.walk(dataset_root):
        for d in dirs:
            if d == video_id or d.zfill(3) == video_id or d.zfill(4) == video_id:
                full_path = os.path.abspath(os.path.join(root, d))
                depth = full_path.count(os.sep)
                if depth > max_depth:
                    best_match = full_path
                    max_depth = depth
        for f in files:
            name, ext = os.path.splitext(f)
            if name == video_id or name.zfill(3) == video_id or name.zfill(4) == video_id:
                full_path = os.path.abspath(os.path.join(root, f))
                depth = full_path.count(os.sep)
                if depth > max_depth:
                    best_match = full_path
                    max_depth = depth
    return best_match
