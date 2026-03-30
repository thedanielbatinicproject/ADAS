"""
Minimal generic dataset parser (lazy, generator-based).

Purpose
-------
Lightweight, robust helpers to discover records (image sequences or videos),
iterate frames lazily, load frames, and query nearby annotation files.

Design goals
------------
- Work with large datasets without loading everything into memory.
- Be tolerant to different directory layouts (video files or folders with frames).
- Return simple "frame refs" that downstream code (loader/wrappers) can use.
- Avoid heavy side-effects on import (no cv2 at module import time).

API (high level)
-----------------
- find_records(root) -> yields (record_id, record_type, path, meta)
- iter_frames(record_path, n_frames_hint=None) -> yields (frame_idx, frame_ref)
- get_frame(frame_ref) -> numpy BGR image or None
- find_annotation_for_record(path_or_record_id) -> optional annotation file path
- get_annotation(path_or_record_id) -> parsed JSON or {'annotation_path': path}
- record_metadata(record_path) -> dict with category, folder, cc, n_frames
- infer_framerate(path) -> fps or None
"""

from __future__ import annotations

import os
import glob
import json
import logging
from typing import Generator, Tuple, Optional, Dict, Iterable

# Configure module logger (parent app may reconfigure)
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _is_image_file(path: str, exts: Iterable[str] = IMAGE_EXTS) -> bool:
    return os.path.splitext(path)[1].lower() in exts


def _is_video_file(path: str, exts: Iterable[str] = VIDEO_EXTS) -> bool:
    return os.path.splitext(path)[1].lower() in exts


def find_records(root: str) -> Generator[Tuple[str, str, str, Dict], None, None]:
    """
    Discover candidate records under `root`.

    Yields tuples:
      (record_id, record_type, path, meta)

    record_type: 'image_seq' | 'video' | 'unknown'
    meta: best-effort dict possibly containing 'n_images' or other info

    Heuristics:
    - If top-level contains video files -> yield each video as a record.
    - Walk directories: any directory that contains >=1 image file is yielded as an image_seq.
    - If a directory contains a video file (directly), yield that video as a record.
    """
    root = os.path.abspath(root)
    if not os.path.exists(root):
        logger.warning("find_records: root does not exist: %s", root)
        return

    # top-level videos
    for ext in VIDEO_EXTS:
        for v in glob.glob(os.path.join(root, f"*{ext}")):
            rid = os.path.splitext(os.path.basename(v))[0]
            yield (rid, "video", v, {})

    # walk directories (shallow first to capture dataset layouts like category/folder/seq)
    for dirpath, dirnames, filenames in os.walk(root):
        # If any video file directly in dir -> yield them
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if _is_video_file(fp):
                rid = os.path.relpath(fp, root)
                yield (rid, "video", fp, {})
        # If directory contains image files -> yield as image_seq
        n_imgs = 0
        for f in filenames:
            if _is_image_file(f):
                n_imgs += 1
                if n_imgs >= 1:
                    break
        if n_imgs > 0:
            rid = os.path.relpath(dirpath, root)
            yield (
                rid,
                "image_seq",
                dirpath,
                {"n_images": len([f for f in filenames if _is_image_file(f)])},
            )
        # continue walking - don't special-case depth

    # Nothing else to do; function ends


def iter_frames(
    record_path: str, n_frames_hint: Optional[int] = None
) -> Generator[Tuple[int, str], None, None]:
    """
    Lazily iterate frames for a given record_path.

    - If record_path is a directory with image files, yields (idx, image_path) sorted by filename.
    - If record_path is a video file, yields (idx, "video_path::frame::idx") for idx in [0, frame_count-1].
      Use get_frame(...) to materialize a frame from a video frame-ref.

    n_frames_hint: optional int. If provided for directories, and files are numerically named
    with zero-padded indices, callers can implement faster generators. This implementation
    will still list files to find actual names (safer).
    """
    if not os.path.exists(record_path):
        return

    if os.path.isdir(record_path):
        # gather image files lazily
        imgs = []
        for ext in IMAGE_EXTS:
            imgs.extend(glob.glob(os.path.join(record_path, f"*{ext}")))
        imgs = sorted(imgs)
        for i, p in enumerate(imgs):
            yield (i, p)
    elif os.path.isfile(record_path) and _is_video_file(record_path):
        # video: query frame count via cv2 if available
        try:
            import cv2  # local import to avoid heavy dependency at module import time
        except Exception:
            logger.error(
                "iter_frames: cv2 not available to read video frames for %s",
                record_path,
            )
            return
        cap = cv2.VideoCapture(record_path)
        if not cap.isOpened():
            logger.warning("iter_frames: failed to open video: %s", record_path)
            cap.release()
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        for i in range(total):
            yield (i, f"{record_path}::frame::{i}")
    else:
        # unknown file type - nothing to yield
        return


def get_frame(frame_ref: str):
    """
    Materialize a frame from a frame_ref.

    frame_ref forms supported:
    - path/to/image.jpg -> returns cv2 BGR image (ndarray) or None
    - /abs/path/to/video.mp4::frame::123 -> reads frame index 123 from video and returns image or None

    Returns:
      ndarray (BGR) or None on failure.
    """
    try:
        import cv2
    except Exception:
        raise RuntimeError("opencv (cv2) is required for get_frame")

    if isinstance(frame_ref, str) and "::frame::" in frame_ref:
        video_path, _, idx_s = frame_ref.rpartition("::frame::")
        try:
            idx = int(idx_s)
        except Exception:
            logger.error("get_frame: invalid frame index in ref %s", frame_ref)
            return None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("get_frame: cannot open video %s", video_path)
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logger.debug("get_frame: failed to read frame %d from %s", idx, video_path)
            return None
        return frame
    else:
        # image file
        if not os.path.exists(frame_ref):
            logger.debug("get_frame: image path not found: %s", frame_ref)
            return None
        img = cv2.imread(frame_ref, cv2.IMREAD_COLOR)
        if img is None:
            logger.debug("get_frame: cv2 failed to read image: %s", frame_ref)
        return img


def find_annotation_for_record(path_or_record_id: str) -> Optional[str]:
    """
    Heuristic search for annotation files near the record.

    Looks for common names:
      <basename>.json | <basename>.csv | annotations.json | annotations.csv | *_annotation.json
    Looks in record folder and its parent.
    """
    if os.path.isdir(path_or_record_id):
        folder = path_or_record_id
        base = os.path.basename(path_or_record_id)
    else:
        folder = os.path.dirname(path_or_record_id)
        base = os.path.splitext(os.path.basename(path_or_record_id))[0]

    candidates = [
        os.path.join(folder, base + ".json"),
        os.path.join(folder, base + ".csv"),
        os.path.join(folder, "annotations.json"),
        os.path.join(folder, "annotations.csv"),
    ]
    # wildcard candidate (e.g., DADA annotation files)
    for p in glob.glob(os.path.join(folder, "*annotation*.json")):
        candidates.append(p)
    for c in candidates:
        if c and os.path.exists(c):
            return c

    # try parent folder
    parent = os.path.dirname(folder)
    for c in candidates:
        c2 = os.path.join(parent, os.path.basename(c))
        if os.path.exists(c2):
            return c2

    return None


def get_annotation(path_or_record_id: str) -> Optional[Dict]:
    """
    Return parsed annotation if JSON, or a dict with 'annotation_path' for other formats.
    """
    p = find_annotation_for_record(path_or_record_id)
    if not p:
        return None
    if p.lower().endswith(".json"):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning("get_annotation: failed to parse json %s : %s", p, e)
            return {"annotation_path": p}
    if p.lower().endswith(".csv"):
        import csv

        data = []
        try:
            with open(p, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter=";")
                for row in reader:
                    data.append(dict(row))
            return {"annotation_path": p, "data": data}
        except Exception as e:
            logger.warning("get_annotation: failed to parse csv %s : %s", p, e)
            return {"annotation_path": p}
    # Other types: return path for now
    return {"annotation_path": p}


def record_metadata(record_path: str, dataset_root: str = None) -> Dict:
    """
    Best-effort metadata for a record_path.

    Returns keys:
      - path (abs)
      - n_frames (int)  (0 if unknown)
      - category, folder, cc (best-effort from path components)
    """
    meta: Dict = {}
    record_path = os.path.abspath(record_path)
    meta["path"] = record_path
    meta["n_frames"] = 0

    if os.path.isdir(record_path):
        # count image files (cheap-ish, required to know length)
        cnt = 0
        for _ in iter_frames(record_path):
            cnt += 1
        meta["n_frames"] = cnt
    elif os.path.isfile(record_path) and _is_video_file(record_path):
        try:
            import cv2

            cap = cv2.VideoCapture(record_path)
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                meta["n_frames"] = total
            cap.release()
        except Exception:
            meta["n_frames"] = 0

    # Odredi category kao prvi poddirektorij relativno na dataset_root
    if dataset_root is not None:
        try:
            rel = os.path.relpath(record_path, dataset_root)
            rel_parts = rel.split(os.sep)
            meta["category"] = rel_parts[0] if len(rel_parts) > 1 else None
            meta["folder"] = rel_parts[1] if len(rel_parts) > 2 else None
            meta["cc"] = (
                rel_parts[2]
                if len(rel_parts) > 3
                else rel_parts[-1] if rel_parts else None
            )
        except Exception:
            meta["category"] = None
            meta["folder"] = None
            meta["cc"] = None
    else:
        # fallback heuristika
        parts = record_path.split(os.sep)
        if len(parts) >= 3:
            meta["category"] = parts[-3]
            meta["folder"] = parts[-2]
            meta["cc"] = parts[-1]
        else:
            meta["category"] = None
            meta["folder"] = None
            meta["cc"] = parts[-1] if parts else None

    return meta


def infer_framerate(path: str) -> Optional[float]:
    """
    If path is a video file, attempt to return FPS using cv2.
    For image sequences this function returns None (unknown).
    """
    if os.path.isfile(path) and _is_video_file(path):
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                return None
        finally:
            return None
