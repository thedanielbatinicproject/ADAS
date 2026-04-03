#!/usr/bin/env python3
"""Run context analysis on every N-th frame of a dataset video.

Usage
-----
python scripts/context/analyze_video.py \
    --index-path data/processed/index.db \
    --category-id 1 \
    --video-id 1 \
    --every-n 10
"""
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

from adas.dataset import parser  # noqa: E402
from adas.context import (  # noqa: E402
    route,
    ContextConfig,
    ContextState,
    Mode,
)

# ── ANSI colours ────────────────────────────────────────────────────────────
C_RESET = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"

_MODE_COLORS = {
    Mode.NORMAL_MARKED: C_GREEN,
    Mode.DEGRADED_MARKED: C_YELLOW,
    Mode.UNMARKED_GOOD_VIS: C_CYAN,
    Mode.UNMARKED_DEGRADED: C_RED,
    Mode.EMERGENCY_OVERRIDE: C_RED + C_BOLD,
}


def _c(text: str, color: str) -> str:
    return f"{color}{text}{C_RESET}"


# ── Index query ─────────────────────────────────────────────────────────────


def get_records_by_key(
    index_path: str,
    category_id: int,
    video_id: int,
) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(index_path)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """
            SELECT record_id, category, path, n_frames, maps_path,
                   category_id, video_id, annotation_status
            FROM records
            WHERE category_id = ? AND video_id = ?
            ORDER BY record_id ASC
            """,
            (category_id, video_id),
        )
        return [dict(r) for r in c.fetchall()]
    finally:
        conn.close()


def pick_record(
    records: List[Dict[str, Any]],
    record_type: str,
) -> Dict[str, Any]:
    for r in records:
        if os.path.basename(r["path"].rstrip("/\\")) == record_type:
            return r
    return records[0]


# ── Pretty printing ─────────────────────────────────────────────────────────


def _header() -> str:
    cols = (
        f"{'Frame':>6}",
        f"{'Mode':<22}",
        f"{'Vis':>5}",
        f"{'Night':>5}",
        f"{'Glare':>5}",
        f"{'LaneConf':>8}",
        f"{'Lanes':<16}",
        f"{'Surface':<14}",
        f"{'Brake':>6}",
        f"{'Bright':>7}",
        f"{'Contr':>6}",
        f"{'Blur':>8}",
        f"{'Edges':>6}",
    )
    return " | ".join(cols)


def _format_state(frame_idx: int, s: ContextState) -> str:
    vis = s.visibility
    lm = s.lane_state
    rs = s.road_surface
    sm = s.scene_metrics

    mode_color = _MODE_COLORS.get(s.mode, "")
    mode_str = _c(f"{s.mode.value:<22}", mode_color)

    cols = (
        f"{frame_idx:>6}",
        mode_str,
        f"{vis.confidence:>5.2f}" if vis else f"{'?':>5}",
        f"{'Y' if vis and vis.is_night else 'N':>5}",
        f"{'Y' if vis and vis.is_glare else 'N':>5}",
        f"{lm.confidence:>8.3f}" if lm else f"{'?':>8}",
        f"{lm.availability.value:<16}" if lm else f"{'?':<16}",
        f"{rs.surface_type.value:<14}" if rs else f"{'?':<14}",
        f"{s.braking_multiplier:>6.2f}",
        f"{sm.brightness_mean:>7.1f}" if sm else f"{'?':>7}",
        f"{sm.contrast_std:>6.1f}" if sm else f"{'?':>6}",
        f"{sm.blur_laplacian_var:>8.1f}" if sm else f"{'?':>8}",
        f"{sm.edge_density:>6.3f}" if sm else f"{'?':>6}",
    )
    return " | ".join(cols)


# ── Summary ─────────────────────────────────────────────────────────────────


def _print_summary(
    states: List[ContextState],
    total_frames: int,
    elapsed: float,
) -> None:
    print(f"\n{_c('═' * 72, C_CYAN)}")
    print(_c("  SUMMARY", C_BOLD))
    print(_c("═" * 72, C_CYAN))

    n = len(states)
    if n == 0:
        print("  No frames analysed.")
        return

    fps_actual = n / elapsed if elapsed > 0 else 0.0
    print(f"  Frames analysed : {n} / {total_frames}")
    print(f"  Wall time       : {elapsed:.2f} s  ({fps_actual:.1f} frames/s)")

    # mode distribution
    mode_counts: Dict[Mode, int] = {}
    for s in states:
        mode_counts[s.mode] = mode_counts.get(s.mode, 0) + 1
    print(f"\n  {'Mode':<24} {'Count':>6} {'%':>6}")
    print(f"  {'─' * 38}")
    for mode in Mode:
        cnt = mode_counts.get(mode, 0)
        pct = 100.0 * cnt / n
        color = _MODE_COLORS.get(mode, "")
        print(f"  {_c(mode.value, color):<35} {cnt:>6} {pct:>5.1f}%")

    # averages
    avg_vis = sum(s.visibility.confidence for s in states if s.visibility) / n
    avg_lane = sum(s.lane_state.confidence for s in states if s.lane_state) / n
    avg_brake = sum(s.braking_multiplier for s in states) / n
    avg_bright = (
        sum(s.scene_metrics.brightness_mean for s in states if s.scene_metrics) / n
    )

    print(f"\n  Avg visibility confidence : {avg_vis:.3f}")
    print(f"  Avg lane confidence       : {avg_lane:.3f}")
    print(f"  Avg braking multiplier    : {avg_brake:.2f}")
    print(f"  Avg brightness            : {avg_bright:.1f}")

    # mode switches
    switches = sum(1 for a, b in zip(states, states[1:]) if a.mode != b.mode)
    print(f"  Mode switches             : {switches}")

    print(_c("═" * 72, C_CYAN))


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Context analysis on every N-th frame of a dataset video.",
    )
    p.add_argument(
        "--index-path",
        default=os.path.join(PROJECT_ROOT, "data", "processed", "index.db"),
        help="Path to index.db (default: data/processed/index.db)",
    )
    p.add_argument(
        "--category-id", type=int, required=True, help="Category (folder number)"
    )
    p.add_argument(
        "--video-id", type=int, required=True, help="Video ID within category"
    )
    p.add_argument(
        "--every-n",
        type=int,
        default=10,
        help="Analyse every N-th frame (default: 10)",
    )
    p.add_argument(
        "--record-type",
        default="images",
        help="Record type to select (default: images)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many analysed frames (0 = all)",
    )

    args = p.parse_args()

    # ── Validate paths ──────────────────────────────────────────────────
    if not os.path.exists(args.index_path):
        print(_c(f"[ERROR] Index not found: {args.index_path}", C_RED))
        print(
            "  Run:  python scripts/dataset/build_index.py --dataset-root data/raw/DADA2000"
        )
        sys.exit(1)

    # ── Find record ─────────────────────────────────────────────────────
    records = get_records_by_key(args.index_path, args.category_id, args.video_id)
    if not records:
        print(
            _c(
                f"[ERROR] No records for category_id={args.category_id}, video_id={args.video_id}",
                C_RED,
            )
        )
        sys.exit(1)

    record = pick_record(records, args.record_type)
    rec_path = record["path"]
    print(_c(f"Record : {record['record_id']}", C_CYAN))
    print(_c(f"Path   : {rec_path}", C_DIM))

    if not os.path.exists(rec_path):
        print(_c(f"[ERROR] Record path does not exist: {rec_path}", C_RED))
        sys.exit(1)

    # ── Enumerate frames ────────────────────────────────────────────────
    all_frames = list(parser.iter_frames(rec_path))
    total = len(all_frames)
    if total == 0:
        print(_c("[ERROR] No frames found in record.", C_RED))
        sys.exit(1)

    print(f"Total frames: {total},  analysing every {args.every_n}-th\n")
    print(_c(_header(), C_BOLD))
    print("─" * 130)

    # ── Analyse ─────────────────────────────────────────────────────────
    cfg = ContextConfig()
    prev_state: Optional[ContextState] = None
    states: List[ContextState] = []
    analysed = 0
    t0 = time.monotonic()

    for frame_idx, frame_ref in all_frames:
        if frame_idx % args.every_n != 0:
            continue

        frame = parser.get_frame(frame_ref)
        if frame is None:
            print(f"{frame_idx:>6} | {_c('SKIPPED (frame load failed)', C_YELLOW)}")
            continue

        ts = frame_idx / max(cfg.min_fps, 1.0)
        state = route(
            frame,
            timestamp_s=ts,
            fps=cfg.min_fps,
            prev_state=prev_state,
            config=cfg,
        )
        prev_state = state
        states.append(state)
        analysed += 1

        print(_format_state(frame_idx, state))

        if args.max_frames > 0 and analysed >= args.max_frames:
            break

    elapsed = time.monotonic() - t0
    _print_summary(states, total, elapsed)


if __name__ == "__main__":
    main()
