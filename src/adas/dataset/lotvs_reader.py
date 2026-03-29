"""
LOT V S / DADA specific reader and annotation bridge.

This module provides:
- loading train/val/test lists (train_file.json, val_file.json, test_file.json)
- loading/parsing DADA2000_video_annotations.csv (semicolon-delimited)
- resolving dataset entries (from train lists or video ids) to filesystem paths
- convenience iterators to traverse records

Usage examples:
    lists = load_indexed_lists(dataset_root)
    ann = load_video_annotations(dataset_root=dataset_root)
    path = resolve_record_path(dataset_root, lists['train'][0])
    ann_frame = get_annotation_for_frame(ann, '001', 63)
"""
from __future__ import annotations

import os
import json
import csv
import glob
import logging
from typing import Optional, Tuple, Dict, Any, List, Iterable

from src.adas.dataset import parser

logger = logging.getLogger(__name__)

# Module-level simple cache for annotations to avoid reloading repeatedly
_ANNOTATIONS_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_indexed_lists(root: str) -> Dict[str, Optional[List[Any]]]:
    """
    Look for train_file.json / val_file.json / test_file.json in `root`.
    Returns dict with keys present (train/val/test) mapping to loaded JSON content.
    """
    out: Dict[str, Optional[List[Any]]] = {}
    for name in ("train_file.json", "val_file.json", "test_file.json"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            try:
                out[name.split("_")[0]] = _load_json(p)
            except Exception as e:
                logger.warning("Failed to load %s: %s", p, e)
                out[name.split("_")[0]] = None
    return out


def _entry_to_components(entry) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Convert an entry (as used in train_file.json) into (category, folder, cc).
    Handles shapes like [[cat, folder], cc], flat lists, dicts, or a string id/path.
    """
    if isinstance(entry, list) and len(entry) >= 1:
        first = entry[0]
        if isinstance(first, list) and len(first) >= 2:
            return str(first[0]), str(first[1]), str(entry[1]) if len(entry) > 1 else None
        if len(entry) >= 3:
            return str(entry[0]), str(entry[1]), str(entry[2])
    if isinstance(entry, dict):
        return (str(entry.get("category")) if entry.get("category") else None,
                str(entry.get("folder")) if entry.get("folder") else None,
                str(entry.get("cc")) if entry.get("cc") else None)
    if isinstance(entry, str):
        # string may be id or path
        return None, None, entry
    return None, None, None


def _find_dir_by_name(root: str, name: str) -> Optional[str]:
    """
    Find (first) directory under root whose basename equals `name` or its zero-padded variant.
    Uses glob with recursive search.
    """
    if not name:
        return None
    candidates = [name, name.zfill(3), name.zfill(4)]
    for cand in candidates:
        pattern = os.path.join(root, "**", cand)
        matches = glob.glob(pattern, recursive=True)
        for m in matches:
            if os.path.isdir(m):
                return os.path.abspath(m)
    return None


def resolve_record_path(dataset_root: str, entry) -> Optional[str]:
    """
    Resolve an entry (from train_file.json or a video id) to an existing record path.
    Tries a set of sensible candidate paths for DADA layout:
      dataset_root/<category>/<folder>/<cc>
      dataset_root/<category>/<folder>
      dataset_root/<category>/<cc>
      dataset_root/**/<cc> (recursive search)
      dataset_root/<entry> (if string)
    Returns absolute path or None.
    """
    cat, folder, cc = _entry_to_components(entry)
    candidates: List[str] = []

    if cat and folder and cc:
        candidates.extend([
            os.path.join(dataset_root, cat, folder, cc),
            os.path.join(dataset_root, cat, folder),
            os.path.join(dataset_root, cat, cc),
        ])
    elif cat and cc:
        candidates.append(os.path.join(dataset_root, cat, cc))
    elif isinstance(entry, str):
        candidates.append(os.path.join(dataset_root, entry))
        candidates.append(os.path.join(dataset_root, entry.zfill(3)))
        candidates.append(os.path.join(dataset_root, entry.zfill(4)))

    # quick check of candidates
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)

    # fallback: try to locate directory by video id (cc)
    if cc:
        found = _find_dir_by_name(dataset_root, cc)
        if found:
            return found

    # if entry was a bare id (first component None), try to find it directly
    if isinstance(entry, str):
        found = _find_dir_by_name(dataset_root, entry)
        if found:
            return found

    # as last resort, try parser.find_records discovery and match basename
    for rid, rtype, path, meta in parser.find_records(dataset_root):
        base = os.path.basename(path)
        if base == str(cc) or base == str(entry):
            return path

    return None


def load_video_annotations(csv_path: Optional[str] = None, dataset_root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load DADA2000_video_annotations.csv and return a dict keyed by video id string (e.g. '001').

    If csv_path is None, tries multiple likely locations relative to dataset_root and CWD.
    Caches loaded annotation dict in module-level cache.
    """
    key = csv_path or (dataset_root or "")
    if key in _ANNOTATIONS_CACHE:
        return _ANNOTATIONS_CACHE[key]

    candidates = []
    if csv_path:
        candidates.append(csv_path)
    if dataset_root:
        candidates.append(os.path.join(dataset_root, "data", "raw", "DADA2000_video_annotations.csv"))
        candidates.append(os.path.join(dataset_root, "DADA2000_video_annotations.csv"))
    candidates.append(os.path.join("data", "raw", "DADA2000_video_annotations.csv"))
    candidates.append("DADA2000_video_annotations.csv")

    csv_file = None
    for c in candidates:
        if c and os.path.exists(c):
            csv_file = c
            break

    if not csv_file:
        logger.info("load_video_annotations: CSV not found in candidates; returning empty dict")
        _ANNOTATIONS_CACHE[key] = {}
        return {}

    annotations: Dict[str, Dict[str, Any]] = {}
    # CSV is semicolon delimited and may contain quoted fields
    with open(csv_file, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        try:
            header = next(reader)
        except StopIteration:
            _ANNOTATIONS_CACHE[key] = {}
            return {}
        # normalize header
        header = [h.strip() for h in header]
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
            # map known positions from provided sample; be defensive for missing columns
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
            # intervals (11..15) - keep as list
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

            annotations[str(vid)] = entry

    _ANNOTATIONS_CACHE[key] = annotations
    logger.info("load_video_annotations: loaded %d entries from %s", len(annotations), csv_file)
    return annotations


def get_video_annotation(annotations: Dict[str, Dict[str, Any]], video_id: str) -> Optional[Dict[str, Any]]:
    """Return loaded annotation dict for the given video_id (or None)."""
    if not annotations:
        return None
    return annotations.get(str(video_id))


def get_annotation_for_frame(annotations: Dict[str, Dict[str, Any]], video_id: str, frame_idx: int) -> Optional[Dict[str, Any]]:
    """
    Return per-frame view of the video-level annotation.
    Adds fields:
      - frame_idx
      - label: one of ('accident_frame','abnormal','normal')
    """
    ann = get_video_annotation(annotations, video_id)
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
    out = dict(ann)  # shallow copy
    out.update({"frame_idx": int(frame_idx), "label": label})
    return out


def iter_train_records(dataset_root: str) -> Iterable[str]:
    """
    Yield resolved record paths for entries present in train_file.json (if available).
    Falls back to discovery via parser.find_records.
    """
    lists = load_indexed_lists(dataset_root)
    train = lists.get("train")
    if train:
        for entry in train:
            p = resolve_record_path(dataset_root, entry)
            if p:
                yield p
    else:
        for _, rtype, path, _ in parser.find_records(dataset_root):
            yield path


def get_record_by_index(dataset_root: str, global_idx: int) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Resolve a 0-based global index across train/val/test concatenated lists into (path, meta).
    """
    lists = load_indexed_lists(dataset_root)
    concat: List[Any] = []
    for k in ("train", "val", "test"):
        if lists.get(k):
            concat.extend(lists[k])
    if not concat:
        return None
    if global_idx < 0 or global_idx >= len(concat):
        return None
    entry = concat[global_idx]
    p = resolve_record_path(dataset_root, entry)
    if p:
        meta = parser.record_metadata(p)
        return p, {"entry": entry, **meta}
    return None


def find_record_by_video_id(dataset_root: str, video_id: str) -> Optional[str]:
    """
    Find a record directory given a video id (tries index lists first, then recursive search).
    """
    # first try index lists
    lists = load_indexed_lists(dataset_root)
    for k in ("train", "val", "test"):
        for entry in lists.get(k, []) or []:
            cat, folder, cc = _entry_to_components(entry)
            if cc and (cc == video_id or (cc.isdigit() and str(int(cc)) == str(int(video_id)))):
                p = resolve_record_path(dataset_root, entry)
                if p:
                    return p
    # fallback: try to find by video_id as directory name
    found = _find_dir_by_name(dataset_root, video_id)
    if found:
        return found

    # as last resort, try parser.find_records discovery and match basename
    for rid, rtype, path, meta in parser.find_records(dataset_root):
        base = os.path.basename(path)
        if base == str(video_id) or (video_id.isdigit() and base == str(int(video_id))):
            return path

    return None
