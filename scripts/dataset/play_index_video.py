#!/usr/bin/env python3
"""Play a dataset record selected from index.db by category_id and video_id."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.dataset import indexer, parser  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

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


def pick_record(records: List[Dict[str, Any]], record_type: str) -> Dict[str, Any]:
    """Return the record whose path ends with `record_type`, or the first if not found."""
    for r in records:
        if os.path.basename(r["path"].rstrip("/\\")) == record_type:
            return r
    print(
        ctext(
            f"[WARN] No record with type '{record_type}' found. "
            f"Available: {[os.path.basename(r['path'].rstrip('/')) for r in records]}. "
            f"Using first: {records[0]['record_id']}",
            C_YELLOW,
        )
    )
    return records[0]


# ── UI constants ────────────────────────────────────────────────────────────
_CTRL_H = 56          # height of the button strip below the video
_BTN_W = 76           # width of each button
_BTN_PAD_Y = 8        # top/bottom padding inside the strip
_BG = (25, 25, 25)
_BTN_BG = (55, 55, 55)
_BTN_BORDER = (90, 90, 90)
_BTN_TXT = (230, 230, 230)
_LABEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "normal": (0, 220, 0),
    "abnormal": (0, 220, 220),
    "accident_frame": (0, 0, 230),
    "unknown": (150, 150, 150),
}
_BTN_ACTIONS = [("<<", "first"), ("<", "prev"), ("||", "pause"), (">", "next"), (">>", "last")]


def _draw_controls(width: int, paused: bool, pos: int, total: int) -> np.ndarray:
    strip = np.full((_CTRL_H, width, 3), _BG, dtype=np.uint8)
    n = len(_BTN_ACTIONS)
    spacing = max((width - n * _BTN_W) // (n + 1), 6)
    bh = _CTRL_H - _BTN_PAD_Y * 2
    for i, (label, action) in enumerate(_BTN_ACTIONS):
        if action == "pause":
            label = ">" if paused else "||"
        bx = spacing + i * (_BTN_W + spacing)
        by = _BTN_PAD_Y
        cv2.rectangle(strip, (bx, by), (bx + _BTN_W, by + bh), _BTN_BG, -1)
        cv2.rectangle(strip, (bx, by), (bx + _BTN_W, by + bh), _BTN_BORDER, 1)
        ts, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = bx + (_BTN_W - ts[0]) // 2
        ty = by + (bh + ts[1]) // 2
        cv2.putText(strip, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BTN_TXT, 2)

    counter = f"frame {pos + 1} / {total}"
    ts, _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(strip, counter, (width - ts[0] - 10, _CTRL_H // 2 + ts[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, _BTN_TXT, 1)
    return strip


def _hit_button(click_x: int, click_y: int, frame_h: int, frame_w: int) -> Optional[str]:
    """Return action name if (click_x, click_y) lands on a button in the control strip."""
    cy = click_y - frame_h
    if cy < _BTN_PAD_Y or cy > _CTRL_H - _BTN_PAD_Y:
        return None
    n = len(_BTN_ACTIONS)
    spacing = max((frame_w - n * _BTN_W) // (n + 1), 6)
    for i, (_, action) in enumerate(_BTN_ACTIONS):
        bx = spacing + i * (_BTN_W + spacing)
        if bx <= click_x <= bx + _BTN_W:
            return action
    return None


def _put_overlay(frame: np.ndarray, cat: int, vid: int, pos: int, total: int,
                 lbl: str, paused: bool) -> None:
    color = _LABEL_COLORS.get(lbl, (150, 150, 150))
    text = f"cat={cat} vid={vid}  frame={pos + 1}/{total}  [{lbl}]"
    if paused:
        text += "  || PAUSED ||"
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    hint = "SPACE=pause  A/<= prev  D/>= next  Q=quit  |  click buttons below"
    cv2.putText(frame, hint, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(frame, hint, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


def main() -> int:
    p = argparse.ArgumentParser(description="Play video/sequence from index.db by category and video id")
    p.add_argument("--index-path", default=indexer.DEFAULT_INDEX_PATH, help="Path to index.db")
    p.add_argument("--category-id", type=int, required=True, help="Category id (e.g. 1..61)")
    p.add_argument("--video-id", type=int, required=True, help="Video id within category (e.g. 1, 2, 3)")
    p.add_argument("--delay-ms", type=int, default=30, help="Delay between frames in ms (0 = manual step)")
    p.add_argument("--record-type", default="images", help="Which subfolder to play (default: images)")
    p.add_argument("--window-name", default="DADA2000 Player", help="OpenCV window name")
    args = p.parse_args()

    if cv2 is None:
        print(ctext("[ERROR] OpenCV (cv2) is required for player window.", C_RED))
        return 4

    if not os.path.exists(args.index_path):
        print(ctext(f"[ERROR] Index file not found: {args.index_path}", C_RED))
        return 2

    records = get_records_by_key(args.index_path, args.category_id, args.video_id)
    if not records:
        print(ctext(f"[ERROR] No record found for category_id={args.category_id}, video_id={args.video_id}", C_RED))
        return 3

    record = pick_record(records, args.record_type)
    annotation = get_annotation_by_key(args.index_path, args.category_id, args.video_id)

    print(ctext(f"[INFO] path={record['path']}  status={record['annotation_status']}", C_GREEN))
    print_annotation_details(annotation)
    print(ctext("\nControls: SPACE=pause  A/<= prev  D/>= next  Q=quit  | buttons & frame slider in UI\n", C_CYAN),
          flush=True)

    print(ctext("[INFO] Loading frame list...", C_GREEN), flush=True)
    frame_refs: List[Tuple[int, Any]] = list(parser.iter_frames(record["path"]))
    if not frame_refs:
        print(ctext("[ERROR] No frames found in record path.", C_RED))
        return 5
    total = len(frame_refs)
    print(ctext(f"[INFO] {total} frames found. Starting playback...", C_GREEN), flush=True)

    WIN = args.window_name
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    default_w = 1000
    default_h = int(default_w * 9 / 16) + _CTRL_H
    cv2.resizeWindow(WIN, default_w, default_h)

    # Shared mutable state for trackbar/mouse callbacks
    _st: Dict[str, Any] = {
        "seek": None,       # frame index to jump to (set by callbacks)
        "toggle_pause": False,
        "fh": 480,          # frame image height (updated each render)
        "fw": 640,          # frame image width
        "tb_from_code": False,  # guard: suppress seek when we set trackbar ourselves
    }

    def on_trackbar(val: int) -> None:
        if not _st["tb_from_code"]:
            _st["seek"] = val

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        action = _hit_button(x, y, _st["fh"], _st["fw"])
        if action is None:
            return
        if action == "first":
            _st["seek"] = 0
        elif action == "prev":
            _st["seek"] = max(0, _st.get("cur_pos", 0) - 1)
        elif action == "pause":
            _st["toggle_pause"] = True
        elif action == "next":
            _st["seek"] = min(total - 1, _st.get("cur_pos", 0) + 1)
        elif action == "last":
            _st["seek"] = total - 1

    cv2.createTrackbar("Frame", WIN, 0, max(total - 1, 1), on_trackbar)
    cv2.setMouseCallback(WIN, on_mouse)

    pos = 0
    paused = False
    shown = 0
    start = time.time()
    _resized_to_frame = False

    while True:
        # Apply pending seeks/toggles from callbacks
        if _st["seek"] is not None:
            pos = max(0, min(total - 1, int(_st["seek"])))
            _st["seek"] = None
            paused = True  # entering step mode on any seek
        if _st["toggle_pause"]:
            paused = not paused
            _st["toggle_pause"] = False

        _st["cur_pos"] = pos

        # Load frame
        idx, frame_ref = frame_refs[pos]
        frame = parser.get_frame(frame_ref)
        if frame is None:
            if not paused:
                pos = min(total - 1, pos + 1)
            continue

        h, w = frame.shape[:2]
        _st["fh"] = h
        _st["fw"] = w

        if not _resized_to_frame:
            cv2.resizeWindow(WIN, w, h + _CTRL_H)
            _resized_to_frame = True

        lbl = frame_label(idx, annotation)
        _put_overlay(frame, args.category_id, args.video_id, pos, total, lbl, paused)

        controls = _draw_controls(w, paused, pos, total)
        cv2.imshow(WIN, np.vstack([frame, controls]))

        # Sync trackbar position (guard against re-triggering callback)
        current_tb = cv2.getTrackbarPos("Frame", WIN)
        if current_tb != pos:
            _st["tb_from_code"] = True
            cv2.setTrackbarPos("Frame", WIN, pos)
            _st["tb_from_code"] = False

        shown += 1

        wait = 1 if paused or args.delay_ms == 0 else max(args.delay_ms, 1)
        key = cv2.waitKey(wait) & 0xFF

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (83, ord("d")):   # right arrow / D — step forward
            pos = min(total - 1, pos + 1)
            paused = True
        elif key in (81, ord("a")):   # left arrow / A — step back
            pos = max(0, pos - 1)
            paused = True
        elif not paused:
            pos += 1
            if pos >= total:
                pos = total - 1
                paused = True  # freeze at last frame

    cv2.destroyAllWindows()
    elapsed = max(time.time() - start, 1e-6)
    print(ctext(f"\n[INFO] Playback finished. frames_shown={shown}, avg_fps={shown / elapsed:.2f}", C_GREEN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

