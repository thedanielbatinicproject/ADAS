"""Annotation parsing and unified access helpers (JSON/CSV/TXT)."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

from . import parser


def parse_annotation_file(path: str, delimiter: str = ";") -> Dict[str, Any]:
    """Parse annotation file and return normalized structure.

    Returns:
        {
            "annotation_path": <path>,
            "format": "json" | "csv" | "txt" | "unknown",
            "data": <parsed payload>
        }
    """
    ext = os.path.splitext(path)[1].lower()
    out: Dict[str, Any] = {
        "annotation_path": os.path.abspath(path),
        "format": "unknown",
        "data": None,
    }

    if ext == ".json":
        out["format"] = "json"
        with open(path, "r", encoding="utf-8") as fh:
            out["data"] = json.load(fh)
        return out

    if ext == ".csv":
        out["format"] = "csv"
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            out["data"] = [dict(row) for row in reader]
        return out

    if ext == ".txt":
        out["format"] = "txt"
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            out["data"] = [line.rstrip("\n") for line in fh]
        return out

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        out["data"] = fh.read()
    return out


def get_annotation(record_path_or_id: str, explicit_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get annotation via explicit path or parser heuristics."""
    ann_path = explicit_path
    if ann_path is None:
        ann_path = parser.find_annotation_for_record(record_path_or_id)
    if ann_path is None:
        return None
    return parse_annotation_file(ann_path)


def extract_labels(annotation: Dict[str, Any], label_keys: Optional[List[str]] = None) -> List[Any]:
    """Extract potential labels from normalized annotation payload."""
    label_keys = label_keys or ["label", "labels", "class", "category", "cause", "causes"]
    data = annotation.get("data")

    if isinstance(data, dict):
        return [data[k] for k in label_keys if k in data]
    if isinstance(data, list):
        labels: List[Any] = []
        for row in data:
            if isinstance(row, dict):
                for k in label_keys:
                    if k in row:
                        labels.append(row[k])
        return labels
    return []
