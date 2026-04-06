"""Generic video player loop.

Provides a reusable play-loop that works with any backend (dpg or cv2)
and replaces the boilerplate event loops in individual scripts.

Usage
-----
    from adas.ui.player import create_player, run_player_loop

    player = create_player("dpg", window_name="My Viewer")
    run_player_loop(
        frame_items,          # [(idx, ref), ...]
        parser.get_frame,     # ref -> BGR ndarray | None
        player=player,
        overlay_fn=my_draw,   # (frame, idx) -> frame
    )
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import UICommand, UIState


def create_player(
    backend: str = "dpg",
    *,
    window_name: str = "Player",
    max_display_width: int = 1280,
) -> Any:
    """Factory for UI player backends.

    Parameters
    ----------
    backend : str
        ``"dpg"`` for Dear PyGui (GPU, default), ``"cv2"`` for OpenCV.
    """
    if backend == "dpg":
        from .backend_dpg import DpgPlayer
        return DpgPlayer(
            window_name=window_name, max_display_width=max_display_width,
        )
    if backend == "cv2":
        from .backend_cv2 import Cv2Player
        return Cv2Player(
            window_name=window_name, max_display_width=max_display_width,
        )
    if backend == "none":
        return None
    raise ValueError(f"Unknown UI backend: {backend!r}")


def _precache_frames(
    frame_items: List[Tuple[int, Any]],
    get_frame_fn: Callable,
) -> Dict[int, np.ndarray]:
    """Load all frames into memory upfront.

    Returns a dict mapping list-index -> BGR ndarray.
    Prints a progress bar to stderr.
    """
    total = len(frame_items)
    cache: Dict[int, np.ndarray] = {}
    for i, (idx, ref) in enumerate(frame_items):
        frame = get_frame_fn(ref)
        if frame is not None:
            cache[i] = frame
        if (i + 1) % 50 == 0 or i + 1 == total:
            pct = 100.0 * (i + 1) / total
            print(
                f"\r[cache] Loading frames: {i + 1}/{total} ({pct:.0f}%)",
                end="", file=sys.stderr, flush=True,
            )
    if total > 0:
        print(file=sys.stderr)  # newline after progress
    return cache


def run_player_loop(
    frame_items: List[Tuple[int, Any]],
    get_frame_fn: Callable,
    *,
    player: Any,
    overlay_fn: Optional[Callable] = None,
    stats_fn: Optional[Callable] = None,
    frame_label_fn: Optional[Callable] = None,
    target_fps: float = 30.0,
    precache: bool = True,
) -> int:
    """Run a generic playback loop with seek, pause, and step support.

    Parameters
    ----------
    frame_items : list of (idx, ref)
        Pre-built list from ``list(parser.iter_frames(path))``.
    get_frame_fn : callable(ref) -> ndarray | None
        Materializes a frame from its reference.
    player : Cv2Player | DpgPlayer
        An instantiated (but not necessarily ``setup()``-ed) player.
    overlay_fn : (frame, idx) -> frame, optional
        Draw overlays on the display copy before showing.
    stats_fn : (idx) -> dict, optional
        Return stats dict for the DPG native dashboard panel.
    frame_label_fn : (idx) -> str, optional
        Return ground-truth label for the frame index.
    target_fps : float
        Target playback FPS. 0 = unlimited.
    precache : bool
        If True (default), load all frames into RAM before playback.
        Eliminates per-frame disk I/O for smooth seeking and playback.

    Returns
    -------
    int
        Number of frames shown.
    """
    total = len(frame_items)
    if total == 0 or player is None:
        return 0

    # Pre-cache all frames into RAM if requested
    frame_cache: Optional[Dict[int, np.ndarray]] = None
    if precache:
        frame_cache = _precache_frames(frame_items, get_frame_fn)

    ui_state = UIState(
        is_playing=True, current_frame_idx=0, total_frames=total,
    )
    spf = (1.0 / target_fps) if target_fps > 0 else 0.0

    cursor = 0
    last = total - 1
    shown = 0

    while 0 <= cursor <= last:
        t0 = time.monotonic()

        idx, ref = frame_items[cursor]
        if frame_cache is not None:
            frame = frame_cache.get(cursor)
        else:
            frame = get_frame_fn(ref)
        if frame is None:
            cursor += 1
            continue

        display = frame.copy()
        if overlay_fn is not None:
            display = overlay_fn(display, idx)

        ui_state.current_frame_idx = cursor
        label = frame_label_fn(idx) if frame_label_fn else ""

        # Native stats panel (DpgPlayer)
        if stats_fn is not None and hasattr(player, "update_stats"):
            player.update_stats(stats_fn(idx))

        cmd = player.show_frame(display, ui_state, annotation_label=label)

        next_cursor = cursor + 1 if ui_state.is_playing else cursor

        if cmd == UICommand.QUIT:
            break
        elif cmd in (UICommand.PLAY, UICommand.PAUSE):
            ui_state.is_playing = not ui_state.is_playing
            next_cursor = cursor + 1 if ui_state.is_playing else cursor
        elif cmd == UICommand.STEP_FWD:
            ui_state.is_playing = False
            next_cursor = min(last, cursor + 1)
        elif cmd == UICommand.STEP_BACK:
            ui_state.is_playing = False
            next_cursor = max(0, cursor - 1)
        elif cmd == UICommand.SEEK_TO and ui_state.seek_target_frame is not None:
            next_cursor = max(0, min(last, int(ui_state.seek_target_frame)))
            ui_state.seek_target_frame = None

        # Pause loop
        if not ui_state.is_playing and cmd == UICommand.NONE:
            while True:
                cmd2 = player.show_frame(display, ui_state, annotation_label=label)
                if cmd2 == UICommand.QUIT:
                    player.close()
                    return shown
                if cmd2 in (UICommand.PLAY, UICommand.PAUSE):
                    ui_state.is_playing = True
                    next_cursor = min(last, cursor + 1)
                    break
                if cmd2 == UICommand.STEP_FWD:
                    next_cursor = min(last, cursor + 1)
                    break
                if cmd2 == UICommand.STEP_BACK:
                    next_cursor = max(0, cursor - 1)
                    break
                if cmd2 == UICommand.SEEK_TO and ui_state.seek_target_frame is not None:
                    next_cursor = max(0, min(last, int(ui_state.seek_target_frame)))
                    ui_state.seek_target_frame = None
                    break
                time.sleep(0.016)

        cursor = next_cursor
        shown += 1

        # Frame timing
        elapsed = time.monotonic() - t0
        if spf > 0 and elapsed < spf and ui_state.is_playing:
            time.sleep(spf - elapsed)

    player.close()
    return shown
