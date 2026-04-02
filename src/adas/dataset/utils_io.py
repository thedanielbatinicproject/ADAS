"""I/O helpers for robust dataset operations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence


def normalize_path(path: str) -> str:
    """Normalize path separators and remove redundant segments."""
    return os.path.normpath(path).replace("\\", "/")


def ensure_dir(path: str) -> str:
    """Create directory if missing and return normalized absolute path."""
    os.makedirs(path, exist_ok=True)
    return normalize_path(os.path.abspath(path))


def safe_imread(path: str):
    """Read image with OpenCV safely.

    Returns None if cv2 is unavailable, path does not exist, or read fails.
    """
    if not os.path.exists(path):
        return None
    try:
        import cv2
    except Exception:
        return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def file_checksum(path: str, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    """Compute checksum for a file."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str, expected: str, algorithm: str = "sha256") -> bool:
    """Verify file checksum."""
    return file_checksum(path, algorithm=algorithm).lower() == expected.lower()


def export_jsonl_sharded(
    rows: Sequence[Mapping],
    out_dir: str,
    shard_size: int,
    prefix: str = "shard",
) -> List[str]:
    """Export rows to JSONL shards and return written file paths."""
    if shard_size <= 0:
        raise ValueError("shard_size must be > 0")

    out_root = Path(ensure_dir(out_dir))
    paths: List[str] = []

    for i in range(0, len(rows), shard_size):
        shard_idx = i // shard_size
        shard_path = out_root / f"{prefix}_{shard_idx:05d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as fh:
            for row in rows[i : i + shard_size]:
                fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        paths.append(normalize_path(str(shard_path)))

    return paths
