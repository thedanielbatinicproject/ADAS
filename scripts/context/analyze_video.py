#!/usr/bin/env python3
"""Run context analysis on every N-th frame of a dataset video.

Modes
-----
- **Terminal** (default): prints a table of context metrics per analysed frame.
- **GUI** (``--gui``): full video player with context-overlay panel.
  The overlay persists between analysis frames and updates on every N-th frame.

Usage
-----
# terminal-only
python scripts/context/analyze_video.py \\
    --index-path data/processed/index.db \\
    --category-id 1 --video-id 1 --every-n 10

# GUI player with overlay
python scripts/context/analyze_video.py \\
    --index-path data/processed/index.db \\
    --category-id 1 --video-id 1 --every-n 10 --gui
"""
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

from adas.dataset import parser  # noqa: E402
from adas.context import (  # noqa: E402
    route,
    ContextConfig,
    ContextState,
    Mode,
)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

# GPU backend detection
_USE_CUDA = False
_HAS_OPENCL = False
if cv2 is not None:
    try:
        _USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        _USE_CUDA = False
    try:
        _HAS_OPENCL = cv2.ocl.haveOpenCL()
        if _HAS_OPENCL:
            cv2.ocl.setUseOpenCL(True)
    except Exception:
        _HAS_OPENCL = False

# ── ANSI colours ────────────────────────────────────────────────────────────
C_RESET = "\033[0m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_MAGENTA = "\033[35m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"

_MODE_ANSI = {
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
    index_path: str, category_id: int, video_id: int
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


def get_annotation_by_key(
    index_path: str, category_id: int, video_id: int
) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(index_path)
    try:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            "SELECT * FROM annotations WHERE category_id = ? AND video_id = ?",
            (category_id, video_id),
        )
        row = c.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def pick_record(records: List[Dict[str, Any]], record_type: str) -> Dict[str, Any]:
    for r in records:
        if os.path.basename(r["path"].rstrip("/\\")) == record_type:
            return r
    return records[0]


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


# ── Terminal-mode helpers ───────────────────────────────────────────────────


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
    mode_color = _MODE_ANSI.get(s.mode, "")
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


def _print_summary(
    states: List[ContextState], total_frames: int, elapsed: float
) -> None:
    print(f"\n{_c('=' * 72, C_CYAN)}")
    print(_c("  SUMMARY", C_BOLD))
    print(_c("=" * 72, C_CYAN))
    n = len(states)
    if n == 0:
        print("  No frames analysed.")
        return
    fps_actual = n / elapsed if elapsed > 0 else 0.0
    print(f"  Frames analysed : {n} / {total_frames}")
    print(f"  Wall time       : {elapsed:.2f} s  ({fps_actual:.1f} frames/s)")
    mode_counts: Dict[Mode, int] = {}
    for s in states:
        mode_counts[s.mode] = mode_counts.get(s.mode, 0) + 1
    print(f"\n  {'Mode':<24} {'Count':>6} {'%':>6}")
    print(f"  {'-' * 38}")
    for mode in Mode:
        cnt = mode_counts.get(mode, 0)
        pct = 100.0 * cnt / n
        color = _MODE_ANSI.get(mode, "")
        print(f"  {_c(mode.value, color):<35} {cnt:>6} {pct:>5.1f}%")
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
    switches = sum(1 for a, b in zip(states, states[1:]) if a.mode != b.mode)
    print(f"  Mode switches             : {switches}")
    print(_c("=" * 72, C_CYAN))


# =====================================================================
# GUI mode - full player with context overlay
# =====================================================================

# ── UI constants ────────────────────────────────────────────────────────────
_CTRL_H = 56
_BTN_W = 76
_BTN_PAD_Y = 8
_BG = (25, 25, 25)
_BTN_BG = (55, 55, 55)
_BTN_BORDER = (90, 90, 90)
_BTN_TXT = (230, 230, 230)
_LABEL_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "normal": (0, 220, 0),
    "abnormal": (0, 220, 220),
    "accident_frame": (0, 0, 230),
    "unknown": (150, 150, 150),
}
_BTN_ACTIONS = [
    ("<<", "first"),
    ("<", "prev"),
    ("||", "pause"),
    (">", "next"),
    (">>", "last"),
]

_MODE_BGR: Dict[Mode, Tuple[int, int, int]] = {
    Mode.NORMAL_MARKED: (0, 200, 0),
    Mode.DEGRADED_MARKED: (0, 200, 200),
    Mode.UNMARKED_GOOD_VIS: (200, 180, 0),
    Mode.UNMARKED_DEGRADED: (0, 60, 200),
    Mode.EMERGENCY_OVERRIDE: (0, 0, 255),
}

_OVL_H = 160  # height of the context overlay panel


# ── GPU helpers ─────────────────────────────────────────────────────────────


def _gpu_resize(img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    if img.shape[1] == new_w and img.shape[0] == new_h:
        return img
    if _USE_CUDA:
        try:
            g = cv2.cuda_GpuMat()
            g.upload(img)
            g = cv2.cuda.resize(g, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return g.download()
        except Exception:
            pass
    if _HAS_OPENCL:
        try:
            u = cv2.UMat(img)
            r = cv2.resize(u, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return r.get()  # type: ignore[union-attr]
        except Exception:
            pass
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _resize_for_display(frame: np.ndarray, max_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    return _gpu_resize(frame, max_w, int(h * max_w / w))


# ── Drawing helpers ─────────────────────────────────────────────────────────


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
    cv2.putText(
        strip,
        counter,
        (width - ts[0] - 10, _CTRL_H // 2 + ts[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        _BTN_TXT,
        1,
    )
    return strip


def _hit_button(
    click_x: int, click_y: int, ctrl_top_y: int, frame_w: int
) -> Optional[str]:
    """Return action if click landed on a button, else None."""
    cy = click_y - ctrl_top_y
    if cy < _BTN_PAD_Y or cy > _CTRL_H - _BTN_PAD_Y:
        return None
    n = len(_BTN_ACTIONS)
    spacing = max((frame_w - n * _BTN_W) // (n + 1), 6)
    for i, (_, action) in enumerate(_BTN_ACTIONS):
        bx = spacing + i * (_BTN_W + spacing)
        if bx <= click_x <= bx + _BTN_W:
            return action
    return None


def _draw_context_overlay(
    width: int,
    ctx: Optional[ContextState],
    analysed_frame: Optional[int],
    every_n: int,
) -> np.ndarray:
    """Draw a dark panel with context state details."""
    panel = np.full((_OVL_H, width, 3), (30, 30, 30), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if ctx is None:
        cv2.putText(
            panel,
            "Context: waiting for first analysis...",
            (12, 28),
            font,
            0.55,
            (150, 150, 150),
            1,
        )
        cv2.putText(
            panel,
            f"(analysing every {every_n} frames)",
            (12, 52),
            font,
            0.45,
            (120, 120, 120),
            1,
        )
        return panel

    vis = ctx.visibility
    lm = ctx.lane_state
    rs = ctx.road_surface
    sm = ctx.scene_metrics

    # Row 1: mode
    mode_color = _MODE_BGR.get(ctx.mode, (200, 200, 200))
    cv2.putText(panel, "MODE:", (12, 24), font, 0.5, (160, 160, 160), 1)
    cv2.putText(panel, ctx.mode.value.upper(), (70, 24), font, 0.6, mode_color, 2)
    if analysed_frame is not None:
        tag = f"[ctx @ frame {analysed_frame}]"
        cv2.putText(panel, tag, (width - 250, 24), font, 0.45, (120, 120, 120), 1)

    # Row 2: visibility
    y2 = 50
    if vis:
        flags = []
        if vis.is_night:
            flags.append("NIGHT")
        if vis.is_glare:
            flags.append("GLARE")
        if vis.is_degraded:
            flags.append("DEGRADED")
        flag_str = ", ".join(flags) if flags else "OK"
        vis_color = (0, 200, 0) if not vis.is_degraded else (0, 140, 220)
        cv2.putText(panel, "Visibility:", (12, y2), font, 0.45, (160, 160, 160), 1)
        cv2.putText(panel, f"{vis.confidence:.2f}", (110, y2), font, 0.5, vis_color, 1)
        cv2.putText(panel, flag_str, (180, y2), font, 0.45, vis_color, 1)

    # Row 3: lane state
    y3 = 76
    if lm:
        lane_color = (
            (0, 200, 0)
            if lm.has_lanes
            else (0, 180, 200) if lm.lanes_degraded else (0, 80, 200)
        )
        cv2.putText(panel, "Lanes:", (12, y3), font, 0.45, (160, 160, 160), 1)
        cv2.putText(
            panel,
            f"{lm.availability.value}  conf={lm.confidence:.2f}  stab={lm.stability:.2f}",
            (72, y3),
            font,
            0.45,
            lane_color,
            1,
        )
        if lm.lane_width_px is not None:
            cv2.putText(
                panel,
                f"w={lm.lane_width_px:.0f}px",
                (400, y3),
                font,
                0.45,
                lane_color,
                1,
            )

    # Row 4: road surface + braking
    y4 = 102
    if rs:
        surf_color = (180, 180, 180)
        cv2.putText(panel, "Surface:", (12, y4), font, 0.45, (160, 160, 160), 1)
        cv2.putText(
            panel,
            f"{rs.surface_type.value}  conf={rs.confidence:.2f}",
            (90, y4),
            font,
            0.45,
            surf_color,
            1,
        )
        cv2.putText(
            panel,
            f"brake x{ctx.braking_multiplier:.2f}",
            (350, y4),
            font,
            0.45,
            (0, 160, 220),
            1,
        )

    # Row 5: raw metrics
    y5 = 128
    if sm:
        cv2.putText(panel, "Metrics:", (12, y5), font, 0.45, (160, 160, 160), 1)
        metrics_str = (
            f"bright={sm.brightness_mean:.0f}  "
            f"contr={sm.contrast_std:.1f}  "
            f"blur={sm.blur_laplacian_var:.0f}  "
            f"edges={sm.edge_density:.3f}  "
            f"sat={sm.saturation_mean:.0f}  "
            f"glare={sm.glare_score:.3f}"
        )
        cv2.putText(panel, metrics_str, (90, y5), font, 0.4, (140, 140, 140), 1)

    # Row 6: hysteresis
    y6 = 150
    hyst_str = f"hold={ctx.mode_hold_count}"
    if ctx.pending_mode is not None:
        hyst_str += f"  pending={ctx.pending_mode.value}({ctx.pending_count})"
    cv2.putText(panel, "Hysteresis:", (12, y6), font, 0.4, (100, 100, 100), 1)
    cv2.putText(panel, hyst_str, (110, y6), font, 0.4, (100, 100, 100), 1)

    return panel


def _put_top_overlay(
    frame: np.ndarray,
    cat: int,
    vid: int,
    pos: int,
    total: int,
    lbl: str,
    paused: bool,
) -> None:
    color = _LABEL_COLORS_BGR.get(lbl, (150, 150, 150))
    text = f"cat={cat} vid={vid}  frame={pos + 1}/{total}  [{lbl}]"
    if paused:
        text += "  || PAUSED ||"
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
    cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    hint = "SPACE=pause  A/<= prev  D/>= next  Q=quit  |  click buttons below"
    cv2.putText(frame, hint, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
    cv2.putText(
        frame, hint, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1
    )


# ── GUI main loop ──────────────────────────────────────────────────────────


def _run_gui(
    args: argparse.Namespace,
    record: Dict[str, Any],
    annotation: Optional[Dict[str, Any]],
    frame_refs: List[Tuple[int, Any]],
) -> int:
    if cv2 is None:
        print(_c("[ERROR] OpenCV (cv2) is required for --gui mode.", C_RED))
        return 4

    total = len(frame_refs)
    cfg = ContextConfig()
    every_n = args.every_n

    # Context state persists between analysis frames
    cur_ctx: Optional[ContextState] = None
    prev_ctx: Optional[ContextState] = None
    ctx_frame_idx: Optional[int] = None

    # Suppress Qt font warnings
    if not os.environ.get("QT_QPA_FONTDIR"):
        for _fd in ("/usr/share/fonts", "/usr/share/fonts/truetype"):
            if os.path.isdir(_fd):
                os.environ["QT_QPA_FONTDIR"] = _fd
                break

    WIN = args.window_name
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    init_w = min(args.max_width, 1280)
    cv2.resizeWindow(WIN, init_w, int(init_w * 9 / 16) + _OVL_H + _CTRL_H)

    _st: Dict[str, Any] = {
        "seek": None,
        "toggle_pause": False,
        "fh": 480,
        "fw": 640,
        "tb_from_code": False,
    }

    def on_trackbar(val: int) -> None:
        if not _st["tb_from_code"]:
            _st["seek"] = val

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ctrl_top = _st["fh"] + _OVL_H
        action = _hit_button(x, y, ctrl_top, _st["fw"])
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
    _canvas: Optional[np.ndarray] = None
    _ctrl_cache: Optional[np.ndarray] = None
    _ctrl_key: Tuple[bool, int] = (False, -1)
    _ovl_cache: Optional[np.ndarray] = None
    _ovl_key: Optional[int] = None
    _next_frame_t = time.perf_counter()

    while True:
        if _st["seek"] is not None:
            pos = max(0, min(total - 1, int(_st["seek"])))
            _st["seek"] = None
            paused = True
        if _st["toggle_pause"]:
            paused = not paused
            _st["toggle_pause"] = False

        _st["cur_pos"] = pos

        idx, frame_ref = frame_refs[pos]
        frame = parser.get_frame(frame_ref)
        if frame is None:
            if not paused:
                pos = min(total - 1, pos + 1)
            continue

        frame = _resize_for_display(frame, args.max_width)
        h, w = frame.shape[:2]
        _st["fh"] = h
        _st["fw"] = w

        if not _resized_to_frame:
            cv2.resizeWindow(WIN, w, h + _OVL_H + _CTRL_H)
            _resized_to_frame = True

        # ── Context analysis on every N-th frame ────────────────────────
        if idx % every_n == 0:
            ts = idx / max(cfg.min_fps, 1.0)
            cur_ctx = route(
                frame,
                timestamp_s=ts,
                fps=cfg.min_fps,
                prev_state=prev_ctx,
                config=cfg,
            )
            prev_ctx = cur_ctx
            ctx_frame_idx = idx
            _ovl_key = None  # force overlay redraw

        # ── Draw frame overlay ──────────────────────────────────────────
        lbl = frame_label(idx, annotation)
        _put_top_overlay(
            frame, args.category_id, args.video_id, pos, total, lbl, paused
        )

        # ── Assemble canvas: frame + context panel + controls ───────────
        canvas_h = h + _OVL_H + _CTRL_H
        if _canvas is None or _canvas.shape[0] != canvas_h or _canvas.shape[1] != w:
            _canvas = np.empty((canvas_h, w, 3), dtype=np.uint8)

        _canvas[:h] = frame

        # Context overlay (redraw only when context changes)
        if _ovl_key != id(cur_ctx):
            _ovl_cache = _draw_context_overlay(w, cur_ctx, ctx_frame_idx, every_n)
            _ovl_key = id(cur_ctx)
        _canvas[h : h + _OVL_H] = _ovl_cache

        # Control strip
        ck = (paused, pos)
        if ck != _ctrl_key or _ctrl_cache is None:
            _ctrl_cache = _draw_controls(w, paused, pos, total)
            _ctrl_key = ck
        _canvas[h + _OVL_H :] = _ctrl_cache

        cv2.imshow(WIN, _canvas)

        # Sync trackbar
        current_tb = cv2.getTrackbarPos("Frame", WIN)
        if current_tb != pos:
            _st["tb_from_code"] = True
            cv2.setTrackbarPos("Frame", WIN, pos)
            _st["tb_from_code"] = False

        shown += 1

        key = cv2.waitKey(1) & 0xFF
        if not paused and args.delay_ms > 0:
            now = time.perf_counter()
            sleep_s = _next_frame_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            _next_frame_t = max(
                time.perf_counter(), _next_frame_t + args.delay_ms / 1000.0
            )
        else:
            _next_frame_t = time.perf_counter()

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (83, ord("d")):
            pos = min(total - 1, pos + 1)
            paused = True
        elif key in (81, ord("a")):
            pos = max(0, pos - 1)
            paused = True
        elif not paused:
            pos += 1
            if pos >= total:
                pos = total - 1
                paused = True

    cv2.destroyAllWindows()
    elapsed = max(time.time() - start, 1e-6)
    print(
        _c(
            f"\n[INFO] Playback finished. frames_shown={shown}, "
            f"avg_fps={shown / elapsed:.2f}",
            C_GREEN,
        )
    )
    return 0


# =====================================================================
# CLI entry point
# =====================================================================


def main() -> int:
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
        help="Stop after this many analysed frames (0 = all, terminal mode only)",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Open GUI player with context overlay (requires cv2 + display)",
    )
    p.add_argument(
        "--delay-ms",
        type=int,
        default=33,
        help="GUI: delay between frames in ms (default: 33 ~ 30 FPS)",
    )
    p.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="GUI: max display width (default: 1280)",
    )
    p.add_argument(
        "--window-name",
        default="Context Analyzer",
        help="GUI: OpenCV window name",
    )
    args = p.parse_args()

    # ── Validate paths ──────────────────────────────────────────────────
    if not os.path.exists(args.index_path):
        print(_c(f"[ERROR] Index not found: {args.index_path}", C_RED))
        print(
            "  Run:  python scripts/dataset/build_index.py "
            "--dataset-root data/raw/DADA2000"
        )
        return 1

    # ── Find record ─────────────────────────────────────────────────────
    records = get_records_by_key(args.index_path, args.category_id, args.video_id)
    if not records:
        print(
            _c(
                f"[ERROR] No records for category_id={args.category_id}, "
                f"video_id={args.video_id}",
                C_RED,
            )
        )
        return 1

    record = pick_record(records, args.record_type)
    rec_path = record["path"]
    print(_c(f"Record : {record['record_id']}", C_CYAN))
    print(_c(f"Path   : {rec_path}", C_DIM))

    if not os.path.exists(rec_path):
        print(_c(f"[ERROR] Record path does not exist: {rec_path}", C_RED))
        return 1

    # ── Enumerate frames ────────────────────────────────────────────────
    print(_c("[INFO] Loading frame list...", C_GREEN), flush=True)
    all_frames: List[Tuple[int, Any]] = list(parser.iter_frames(rec_path))
    total = len(all_frames)
    if total == 0:
        print(_c("[ERROR] No frames found in record.", C_RED))
        return 1

    print(_c(f"[INFO] {total} frames, analysing every {args.every_n}-th", C_GREEN))

    # ── GUI or terminal mode ────────────────────────────────────────────
    if args.gui:
        annotation = get_annotation_by_key(
            args.index_path, args.category_id, args.video_id
        )
        return _run_gui(args, record, annotation, all_frames)

    # ── Terminal mode ───────────────────────────────────────────────────
    print()
    print(_c(_header(), C_BOLD))
    print("-" * 130)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
