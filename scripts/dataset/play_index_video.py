#!/usr/bin/env python3
"""Play a dataset record selected from index.db by category_id and video_id."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.dataset import indexer, parser  # noqa: E402

C_RESET = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"


def ctext(text: str, color: str) -> str:
    return f"{color}{text}{C_RESET}"


def get_records_by_key(index_path: str, category_id: int, video_id: int) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(index_path)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT record_id, category, path, n_frames, maps_path, category_id, video_id, annotation_status
            FROM records
            WHERE category_id = ? AND video_id = ?
            ORDER BY record_id ASC
            """,
            (category_id, video_id),
        )
        return [dict(r) for r in c.fetchall()]
    finally:
        conn.close()


def get_annotation_by_key(index_path: str, category_id: int, video_id: int) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(index_path)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT *
            FROM annotations
            WHERE category_id = ? AND video_id = ?
            """,
            (category_id, video_id),
        )
        row = c.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def print_annotation_details(annotation: Optional[Dict[str, Any]]) -> None:
    if annotation is None:
        print(ctext("[WARN] No annotation found for this (category_id, video_id).", C_YELLOW))
        return

    print(ctext("\n=== Annotation Details ===", C_CYAN))
    ordered_keys = [
        "category_id",
        "video_id",
        "weather",
        "light",
        "scenes",
        "linear",
        "type",
        "accident_occurred",
        "abnormal_start_frame",
        "accident_frame",
        "abnormal_end_frame",
        "total_frames",
        "interval_0_tai",
        "interval_tai_tco",
        "interval_tai_tae",
        "interval_tco_tae",
        "interval_tae_end",
        "texts",
        "causes",
        "measures",
    ]
    for k in ordered_keys:
        v = annotation.get(k)
        if v is None:
            print(f"{ctext(k + ':', C_MAGENTA)} NULL")
        else:
            print(f"{ctext(k + ':', C_MAGENTA)} {v}")


def frame_label(frame_idx: int, annotation: Optional[Dict[str, Any]]) -> str:
    if annotation is None:
        return "unknown"
    af = annotation.get("accident_frame")
    bs = annotation.get("abnormal_start_frame")
    be = annotation.get("abnormal_end_frame")

    if isinstance(af, int) and af >= 0 and frame_idx == af:
        return "accident_frame"
    if isinstance(bs, int) and isinstance(be, int) and bs <= frame_idx <= be:
        return "abnormal"
    return "normal"


def main() -> int:
    p = argparse.ArgumentParser(description="Play video/sequence from index.db by category and video id")
    p.add_argument("--index-path", default=indexer.DEFAULT_INDEX_PATH, help="Path to index.db")
    p.add_argument("--category-id", type=int, required=True, help="Category id (e.g. 1..61)")
    p.add_argument("--video-id", type=int, required=True, help="Video id within category (e.g. 1, 2, 3)")
    p.add_argument("--delay-ms", type=int, default=30, help="Delay between frames in ms")
    p.add_argument("--window-name", default="DADA2000 Player", help="OpenCV window name")
    args = p.parse_args()

    if not os.path.exists(args.index_path):
        print(ctext(f"[ERROR] Index file not found: {args.index_path}", C_RED))
        return 2

    records = get_records_by_key(args.index_path, args.category_id, args.video_id)
    if not records:
        print(
            ctext(
                f"[ERROR] No record found for category_id={args.category_id}, video_id={args.video_id}",
                C_RED,
            )
        )
        return 3

    if len(records) > 1:
        print(
            ctext(
                f"[WARN] Multiple records found ({len(records)}). Using first: {records[0]['record_id']}",
                C_YELLOW,
            )
        )

    record = records[0]
    annotation = get_annotation_by_key(args.index_path, args.category_id, args.video_id)

    print(
        ctext(
            f"[INFO] Playing record_id={record['record_id']} path={record['path']} status={record['annotation_status']}",
            C_GREEN,
        )
    )
    print_annotation_details(annotation)

    try:
        import cv2
    except Exception:
        print(ctext("[ERROR] OpenCV (cv2) is required for player window.", C_RED))
        return 4

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    shown = 0
    start = time.time()
    for idx, frame_ref in parser.iter_frames(record["path"]):
        frame = parser.get_frame(frame_ref)
        if frame is None:
            continue

        lbl = frame_label(idx, annotation)
        color = (0, 255, 0) if lbl == "normal" else (0, 255, 255) if lbl == "abnormal" else (0, 0, 255)
        overlay = f"cat={args.category_id} vid={args.video_id} frame={idx} label={lbl}"
        cv2.putText(frame, overlay, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(args.window_name, frame)
        shown += 1

        key = cv2.waitKey(max(args.delay_ms, 1)) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    elapsed = max(time.time() - start, 1e-6)
    fps = shown / elapsed
    print(ctext(f"[INFO] Playback finished. frames_shown={shown}, avg_fps={fps:.2f}", C_GREEN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
