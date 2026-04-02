"""SQLite indexer for dataset records and optional DADA2000 annotations."""

from __future__ import annotations

import csv
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .parser import find_records, record_metadata

DEFAULT_INDEX_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../data/processed/index.db")
)
INDEX_VERSION = 2


@dataclass(frozen=True)
class AnnotationKey:
    category_id: int
    video_id: int


def _warn(msg: str) -> None:
    print(f"\033[33m[WARN]\033[0m {msg}")


def _info(msg: str) -> None:
    print(f"[indexer] {msg}", flush=True)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return int(s)
    except Exception:
        return None


def _extract_category_video_ids(record_id: str, record_path: str, dataset_root: str) -> Tuple[Optional[int], Optional[int]]:
    record_id_parts = [p for p in record_id.replace("\\", "/").split("/") if p]

    def _parse(parts: List[str]) -> Tuple[Optional[int], Optional[int]]:
        if len(parts) < 2:
            return None, None
        category = _safe_int(parts[0])
        video_str = os.path.splitext(parts[1])[0]
        video = _safe_int(video_str)
        return category, video

    category_id, video_id = _parse(record_id_parts)
    if category_id is not None and video_id is not None:
        return category_id, video_id

    rel = os.path.relpath(record_path, dataset_root).replace("\\", "/")
    rel_parts = [p for p in rel.split("/") if p]
    return _parse(rel_parts)


def _load_annotations_csv(csv_path: str) -> Dict[AnnotationKey, Dict[str, Any]]:
    annotations: Dict[AnnotationKey, Dict[str, Any]] = {}

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh, delimiter=";")
        try:
            next(reader)
        except StopIteration:
            return {}

        for row_no, row in enumerate(reader, start=2):
            if not row:
                continue

            video_id = _safe_int(row[0] if len(row) > 0 else None)
            category_id = _safe_int(row[5] if len(row) > 5 else None)
            if video_id is None or category_id is None:
                _warn(f"Invalid annotation key at CSV row {row_no}, skipping.")
                continue

            key = AnnotationKey(category_id=category_id, video_id=video_id)
            if key in annotations:
                _warn(
                    "Duplicate annotation key in CSV for "
                    f"(category_id={category_id}, video_id={video_id}) at row {row_no}; keeping first."
                )
                continue

            annotations[key] = {
                "video_raw": row[0] if len(row) > 0 else None,
                "weather": _safe_int(row[1] if len(row) > 1 else None),
                "light": _safe_int(row[2] if len(row) > 2 else None),
                "scenes": _safe_int(row[3] if len(row) > 3 else None),
                "linear": _safe_int(row[4] if len(row) > 4 else None),
                "type": _safe_int(row[5] if len(row) > 5 else None),
                "accident_occurred": _safe_int(row[6] if len(row) > 6 else None),
                "abnormal_start_frame": _safe_int(row[7] if len(row) > 7 else None),
                "accident_frame": _safe_int(row[8] if len(row) > 8 else None),
                "abnormal_end_frame": _safe_int(row[9] if len(row) > 9 else None),
                "total_frames": _safe_int(row[10] if len(row) > 10 else None),
                "interval_0_tai": _safe_int(row[11] if len(row) > 11 else None),
                "interval_tai_tco": _safe_int(row[12] if len(row) > 12 else None),
                "interval_tai_tae": _safe_int(row[13] if len(row) > 13 else None),
                "interval_tco_tae": _safe_int(row[14] if len(row) > 14 else None),
                "interval_tae_end": _safe_int(row[15] if len(row) > 15 else None),
                "texts": row[16] if len(row) > 16 else None,
                "causes": row[17] if len(row) > 17 else None,
                "measures": row[18] if len(row) > 18 else None,
            }

    return annotations


def _create_schema(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            record_id TEXT PRIMARY KEY,
            category TEXT,
            path TEXT,
            n_frames INTEGER,
            maps_path TEXT,
            category_id INTEGER,
            video_id INTEGER,
            annotation_status TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            category_id INTEGER NOT NULL,
            video_id INTEGER NOT NULL,
            video_raw TEXT,
            weather INTEGER,
            light INTEGER,
            scenes INTEGER,
            linear INTEGER,
            type INTEGER,
            accident_occurred INTEGER,
            abnormal_start_frame INTEGER,
            accident_frame INTEGER,
            abnormal_end_frame INTEGER,
            total_frames INTEGER,
            interval_0_tai INTEGER,
            interval_tai_tco INTEGER,
            interval_tai_tae INTEGER,
            interval_tco_tae INTEGER,
            interval_tae_end INTEGER,
            texts TEXT,
            causes TEXT,
            measures TEXT,
            PRIMARY KEY (category_id, video_id)
        )
        """
    )


def build_index(
    dataset_root: str,
    index_path: Optional[str] = None,
    overwrite: bool = False,
    annotations_csv_path: Optional[str] = None,
    progress_interval_sec: int = 10,
) -> None:
    """Build SQLite index for records and optional annotations.

    - Record key for annotation join: (category_id, video_id)
    - Atomic write: create temp DB and os.replace at end
    - Logs progress every `progress_interval_sec` seconds
    """
    if index_path is None:
        index_path = DEFAULT_INDEX_PATH

    dataset_root = os.path.abspath(dataset_root)
    if not os.path.exists(dataset_root):
        _warn(f"dataset_root does not exist: {dataset_root}")

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    tmp_index_path = index_path + ".tmp"
    if overwrite and os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(tmp_index_path):
        os.remove(tmp_index_path)

    progress_state: Dict[str, Any] = {
        "stage": "initializing",
        "scanned": 0,
        "inserted": 0,
        "current_record": None,
    }
    stop_event = threading.Event()

    def _heartbeat() -> None:
        while not stop_event.wait(progress_interval_sec):
            _info(
                "progress: "
                f"stage={progress_state['stage']}, "
                f"scanned={progress_state['scanned']}, "
                f"inserted={progress_state['inserted']}, "
                f"current_record={progress_state['current_record']}"
            )

    heartbeat_thread: Optional[threading.Thread] = None
    if progress_interval_sec > 0:
        heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()

    annotations: Dict[AnnotationKey, Dict[str, Any]] = {}
    if annotations_csv_path:
        if os.path.exists(annotations_csv_path):
            progress_state["stage"] = "loading_annotations"
            _info(f"Loading annotations CSV: {annotations_csv_path}")
            annotations = _load_annotations_csv(annotations_csv_path)
            _info(f"Annotations loaded: {len(annotations)}")
        else:
            _warn(f"annotations CSV not found: {annotations_csv_path}; continuing without annotations")

    conn = sqlite3.connect(tmp_index_path)
    start_ts = time.time()

    record_key_to_id: Dict[AnnotationKey, str] = {}
    duplicate_record_keys: set[AnnotationKey] = set()
    discovered_record_keys: set[AnnotationKey] = set()

    try:
        c = conn.cursor()
        _create_schema(c)
        c.execute("DELETE FROM records")
        c.execute("DELETE FROM annotations")

        scanned = 0
        inserted = 0
        progress_state["stage"] = "scanning_records"
        for record_id, record_type, path, _ in find_records(dataset_root):
            scanned += 1
            progress_state["scanned"] = scanned
            progress_state["current_record"] = record_id
            meta = record_metadata(path, dataset_root)
            category_id, video_id = _extract_category_video_ids(record_id, path, dataset_root)

            key: Optional[AnnotationKey] = None
            if category_id is not None and video_id is not None:
                key = AnnotationKey(category_id=category_id, video_id=video_id)
                discovered_record_keys.add(key)
                if key in record_key_to_id:
                    duplicate_record_keys.add(key)
                else:
                    record_key_to_id[key] = record_id

            c.execute(
                """
                INSERT OR REPLACE INTO records (
                    record_id, category, path, n_frames, maps_path, category_id, video_id, annotation_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    meta.get("category"),
                    meta.get("path"),
                    meta.get("n_frames"),
                    meta.get("maps_path"),
                    category_id,
                    video_id,
                    None,
                ),
            )
            inserted += 1
            progress_state["inserted"] = inserted

        if duplicate_record_keys:
            for key in sorted(duplicate_record_keys, key=lambda x: (x.category_id, x.video_id)):
                _warn(
                    "Duplicate dataset key detected, annotation join will be skipped for "
                    f"(category_id={key.category_id}, video_id={key.video_id})."
                )

        if annotations:
            progress_state["stage"] = "writing_annotations"
            for key, ann in annotations.items():
                c.execute(
                    """
                    INSERT OR REPLACE INTO annotations (
                        category_id, video_id, video_raw, weather, light, scenes, linear, type,
                        accident_occurred, abnormal_start_frame, accident_frame, abnormal_end_frame,
                        total_frames, interval_0_tai, interval_tai_tco, interval_tai_tae,
                        interval_tco_tae, interval_tae_end, texts, causes, measures
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        key.category_id,
                        key.video_id,
                        ann.get("video_raw"),
                        ann.get("weather"),
                        ann.get("light"),
                        ann.get("scenes"),
                        ann.get("linear"),
                        ann.get("type"),
                        ann.get("accident_occurred"),
                        ann.get("abnormal_start_frame"),
                        ann.get("accident_frame"),
                        ann.get("abnormal_end_frame"),
                        ann.get("total_frames"),
                        ann.get("interval_0_tai"),
                        ann.get("interval_tai_tco"),
                        ann.get("interval_tai_tae"),
                        ann.get("interval_tco_tae"),
                        ann.get("interval_tae_end"),
                        ann.get("texts"),
                        ann.get("causes"),
                        ann.get("measures"),
                    ),
                )

        missing_annotation_count = 0
        matched_count = 0
        ambiguous_count = 0
        progress_state["stage"] = "matching_annotations"
        progress_state["current_record"] = None

        c.execute("SELECT record_id, category_id, video_id FROM records")
        rows = c.fetchall()
        for record_id, category_id, video_id in rows:
            if category_id is None or video_id is None:
                continue

            key = AnnotationKey(category_id=int(category_id), video_id=int(video_id))
            if key in duplicate_record_keys:
                c.execute(
                    "UPDATE records SET annotation_status = ? WHERE record_id = ?",
                    ("ambiguous_record_key", record_id),
                )
                ambiguous_count += 1
                continue

            if key in annotations:
                c.execute(
                    "UPDATE records SET annotation_status = ? WHERE record_id = ?",
                    ("matched", record_id),
                )
                matched_count += 1
            else:
                c.execute(
                    "UPDATE records SET annotation_status = ? WHERE record_id = ?",
                    ("missing_annotation", record_id),
                )
                missing_annotation_count += 1
                _warn(
                    f"Missing annotation for record '{record_id}' "
                    f"(category_id={key.category_id}, video_id={key.video_id})"
                )

        missing_record_count = 0
        progress_state["stage"] = "finding_orphan_annotations"
        for key in sorted(annotations.keys(), key=lambda x: (x.category_id, x.video_id)):
            if key not in discovered_record_keys:
                missing_record_count += 1
                _warn(
                    "Missing dataset record for annotation "
                    f"(category_id={key.category_id}, video_id={key.video_id})"
                )

        conn.commit()
        progress_state["stage"] = "finalizing"

        elapsed = time.time() - start_ts
        _info(
            "build summary: "
            f"records={inserted}, annotations={len(annotations)}, matched={matched_count}, "
            f"missing_annotation={missing_annotation_count}, missing_record={missing_record_count}, "
            f"ambiguous_record_key={ambiguous_count}, elapsed_sec={elapsed:.1f}"
        )
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
        conn.close()

    os.replace(tmp_index_path, index_path)
    try:
        os.utime(index_path, None)
    except Exception:
        pass


def is_index_fresh(dataset_root: str, index_path: Optional[str] = None) -> bool:
    """Check if index exists and is newer than the latest file in dataset_root."""
    if index_path is None:
        index_path = DEFAULT_INDEX_PATH
    if not os.path.exists(index_path):
        return False

    index_mtime = os.path.getmtime(index_path)
    for root, dirs, files in os.walk(dataset_root):
        for name in files + dirs:
            fpath = os.path.join(root, name)
            if os.path.getmtime(fpath) >= index_mtime:
                return False
    return True


def get_record(index_path: Optional[str], record_id: str) -> Optional[Dict[str, Any]]:
    """Return metadata for a given record_id from the index."""
    if index_path is None:
        index_path = DEFAULT_INDEX_PATH
    conn = sqlite3.connect(index_path)
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT record_id, category, path, n_frames, maps_path, category_id, video_id, annotation_status
            FROM records
            WHERE record_id = ?
            """,
            (record_id,),
        )
        row = c.fetchone()
        if not row:
            return None
        keys = [
            "record_id",
            "category",
            "path",
            "n_frames",
            "maps_path",
            "category_id",
            "video_id",
            "annotation_status",
        ]
        return dict(zip(keys, row))
    finally:
        conn.close()


def list_records(
    index_path: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List all (or filtered) records from the index."""
    if index_path is None:
        index_path = DEFAULT_INDEX_PATH
    conn = sqlite3.connect(index_path)
    try:
        c = conn.cursor()
        if category:
            c.execute(
                """
                SELECT record_id, category, path, n_frames, maps_path, category_id, video_id, annotation_status
                FROM records
                WHERE category = ?
                """,
                (category,),
            )
        else:
            c.execute(
                """
                SELECT record_id, category, path, n_frames, maps_path, category_id, video_id, annotation_status
                FROM records
                """
            )
        rows = c.fetchall()
        keys = [
            "record_id",
            "category",
            "path",
            "n_frames",
            "maps_path",
            "category_id",
            "video_id",
            "annotation_status",
        ]
        return [dict(zip(keys, row)) for row in rows]
    finally:
        conn.close()


def get_record_with_annotation(index_path: Optional[str], record_id: str) -> Optional[Dict[str, Any]]:
    """Return record joined with annotation row (if available)."""
    if index_path is None:
        index_path = DEFAULT_INDEX_PATH
    conn = sqlite3.connect(index_path)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT r.*, a.*
            FROM records r
            LEFT JOIN annotations a
              ON a.category_id = r.category_id
             AND a.video_id = r.video_id
            WHERE r.record_id = ?
            """,
            (record_id,),
        )
        row = c.fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()
