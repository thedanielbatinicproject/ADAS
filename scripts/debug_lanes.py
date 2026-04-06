#!/usr/bin/env python3
"""Debug lane detection on a single DADA-2000 video.

Shows the raw Canny edges, the fitted polynomial boundaries, and the
lane mask side-by-side in an OpenCV window. No obstacle detection or
risk estimation is run.

Usage
-----
python scripts/debug_lanes.py --category-id 1 --video-id 1
python scripts/debug_lanes.py --category-id 1 --video-id 1 --show-edges
python scripts/debug_lanes.py --category-id 1 --video-id 1 --max-frames 200
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Suppress Qt font-dir warnings before cv2 import
if not os.environ.get("QT_QPA_FONTDIR"):
    for _fd in ("/usr/share/fonts", "/usr/share/fonts/truetype"):
        if os.path.isdir(_fd):
            os.environ["QT_QPA_FONTDIR"] = _fd
            break

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from adas.dataset import parser  # noqa: E402
from adas.lane_detection import process_frame, draw_lanes, draw_edges  # noqa: E402
from adas.lane_detection.processing import LaneProcessingConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lane detection debug viewer.")
    p.add_argument("--category-id", type=int, default=1)
    p.add_argument("--video-id", type=int, default=1)
    p.add_argument(
        "--index-path", default="data/processed/index.db",
        help="Path to index.db.",
    )
    p.add_argument(
        "--dataset-root", default="data/raw/DADA2000",
        help="Path to DADA-2000 root.",
    )
    p.add_argument(
        "--delay-ms", type=int, default=33,
        help="cv2.waitKey delay in ms per frame (default: 33 ~ 30 fps).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames.",
    )
    p.add_argument(
        "--show-edges", action="store_true",
        help="Overlay Canny edges on the frame.",
    )
    p.add_argument(
        "--max-width", type=int, default=1280,
        help="Max display width in pixels.",
    )
    p.add_argument(
        "--ui-backend", choices=["dpg", "cv2"], default="dpg",
        help="UI backend (default: dpg). 'cv2' for legacy OpenCV window.",
    )
    return p.parse_args()


def _load_record(index_path: str, category_id: int, video_id: int):
    if not os.path.exists(index_path):
        return None
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM records WHERE category_id = ? AND video_id = ? LIMIT 1",
        (category_id, video_id),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row is not None else None


def _resize(frame: np.ndarray, max_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    return cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    args = _parse_args()

    index_path = args.index_path
    dataset_root = args.dataset_root
    if not os.path.isabs(index_path):
        index_path = os.path.join(PROJECT_ROOT, index_path)
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(PROJECT_ROOT, dataset_root)

    record = _load_record(index_path, args.category_id, args.video_id)
    if record is None:
        print(f"[ERROR] No record found for category_id={args.category_id}, video_id={args.video_id}")
        sys.exit(1)

    record_path = record["path"]
    print(f"[debug_lanes] Playing: {record_path}")

    # Build frame list (iter_frames yields (idx, ref) tuples)
    frame_items = list(parser.iter_frames(record_path))
    if not frame_items:
        print("[ERROR] No frames found.")
        sys.exit(1)
    if args.max_frames is not None:
        frame_items = frame_items[: args.max_frames]

    lane_cfg = LaneProcessingConfig()
    win = "Lane Detection Debug"

    def _overlay(display, _idx):
        lane_out = process_frame(display, config=lane_cfg)
        if args.show_edges:
            display = draw_edges(display, lane_out, alpha=0.45)
        display = draw_lanes(display, lane_out)
        conf_text = (
            f"lane_conf={lane_out.lane_confidence:.2f} "
            f"L={lane_out.left_confidence:.2f} "
            f"R={lane_out.right_confidence:.2f} "
            f"has_lanes={lane_out.has_lanes}"
        )
        cv2.putText(
            display, conf_text, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        if lane_out.lane_width_px is not None:
            cv2.putText(
                display, f"lane_width={lane_out.lane_width_px:.0f}px",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1, cv2.LINE_AA,
            )
        return display

    if args.ui_backend == "dpg":
        from adas.ui.player import create_player, run_player_loop
        player = create_player("dpg", window_name=win, max_display_width=args.max_width)
        fps = (1000.0 / args.delay_ms) if args.delay_ms > 0 else 0.0
        frame_count = run_player_loop(
            frame_items, parser.get_frame, player=player,
            overlay_fn=_overlay, target_fps=fps,
        )
    else:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        paused = False
        frame_count = 0
        for idx, frame_ref in frame_items:
            frame = parser.get_frame(frame_ref)
            if frame is None:
                continue
            display = frame.copy()
            display = _overlay(display, idx)
            display = _resize(display, args.max_width)
            cv2.imshow(win, display)
            key = cv2.waitKey(1 if paused else args.delay_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord(" "), ord("p")):
                paused = not paused
            frame_count += 1
        cv2.destroyAllWindows()

    print(f"[debug_lanes] Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
