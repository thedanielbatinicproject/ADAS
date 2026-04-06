"""OpenCV UI backend (Cv2Player).

Provides a minimal interactive player using cv2.imshow.
Draws a button strip at the bottom of the window, a frame trackbar, and
reads keyboard input to produce UICommand values.

Key bindings:
  Space / P   -> toggle play/pause
  Right arrow -> step forward one frame
  Left arrow  -> step backward one frame
  Q / Esc     -> quit

This backend is intentionally simple and has no layout dependencies.
It can be swapped for a Qt backend later without changing scenario/runner.py.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from .types import UICommand, UIState


# ---------------------------------------------------------------------------
# Layout constants (same style as play_index_video.py)
# ---------------------------------------------------------------------------

_CTRL_H = 56          # height of the button strip below the video
_BTN_W = 76           # width of each button
_BTN_PAD_Y = 8        # top/bottom padding inside the strip
_BG = (25, 25, 25)    # dark background
_FG = (200, 200, 200) # default button text color
_BTN_ACTIVE = (60, 60, 60)
_BTN_HOVER_NOT_IMPL = (40, 40, 40)

_BUTTONS = [
    {"label": "<<",    "cmd": UICommand.STEP_BACK},
    {"label": "Play",  "cmd": UICommand.PLAY},
    {"label": "Pause", "cmd": UICommand.PAUSE},
    {"label": ">>",    "cmd": UICommand.STEP_FWD},
    {"label": "Quit",  "cmd": UICommand.QUIT},
]

_TRACKBAR_NAME = "Frame"
_WIN_NAME = "ADAS Scenario"


class Cv2Player:
    """OpenCV-based player for ADAS scenario.

    Parameters
    ----------
    window_name : str
        Name of the cv2 window.
    max_display_width : int
        If the frame is wider than this, it will be scaled down for display.
    """

    def __init__(
        self,
        window_name: str = _WIN_NAME,
        max_display_width: int = 1280,
    ) -> None:
        try:
            import cv2
            self._cv2 = cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required for Cv2Player") from exc

        self._win = window_name
        self._max_w = max_display_width
        self._initialized = False
        self._last_trackbar_pos = -1
        self._ignore_trackbar_event = False
        self._pending_seek: Optional[int] = None
        self._pending_cmd: UICommand = UICommand.NONE
        self._button_rects: List[Tuple[int, int, int, int, UICommand]] = []
        self._last_display_h = 0
        self._trackbar_enabled = False

    def setup(self, total_frames: int) -> None:
        """Create the window and trackbar. Call once before the main loop."""
        cv2 = self._cv2
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._on_mouse)
        if total_frames > 1:
            cv2.createTrackbar(_TRACKBAR_NAME, self._win, 0, total_frames - 1, self._on_trackbar)
            self._trackbar_enabled = True
        self._initialized = True
        self._last_trackbar_pos = 0

    def show_frame(
        self,
        frame: Any,
        ui_state: UIState,
        annotation_label: str = "",
    ) -> UICommand:
        """Display a frame and return any UICommand from keyboard input.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR image to display (already composited with overlays/dashboard).
        ui_state : UIState
            Current player state. Used to update the trackbar.
        annotation_label : str
            Optional ground-truth label shown in the title bar.

        Returns
        -------
        UICommand
        """
        cv2 = self._cv2
        if not self._initialized:
            self.setup(ui_state.total_frames)

        display = _scale_to_width(frame, self._max_w)
        h, w = display.shape[:2]
        self._last_display_h = h

        # Append button strip
        strip, button_rects = _draw_button_strip(w, ui_state)
        self._button_rects = button_rects
        composite = np.vstack([display, strip])

        cv2.imshow(self._win, composite)

        # Update trackbar position
        if self._trackbar_enabled and ui_state.total_frames > 1:
            self._last_trackbar_pos = ui_state.current_frame_idx
            self._ignore_trackbar_event = True
            cv2.setTrackbarPos(_TRACKBAR_NAME, self._win, ui_state.current_frame_idx)
            self._ignore_trackbar_event = False

        # Window title
        label_part = f"  [{annotation_label}]" if annotation_label else ""
        status = "PLAYING" if ui_state.is_playing else "PAUSED"
        title = (
            f"{self._win} | Frame {ui_state.current_frame_idx}/{ui_state.total_frames - 1}"
            f" | {status}{label_part}"
        )
        cv2.setWindowTitle(self._win, title)

        # Mouse click command has priority.
        if self._pending_cmd != UICommand.NONE:
            cmd = self._pending_cmd
            self._pending_cmd = UICommand.NONE
            return cmd

        # Slider seek command.
        if self._pending_seek is not None:
            ui_state.seek_target_frame = int(self._pending_seek)
            self._pending_seek = None
            return UICommand.SEEK_TO

        # Read key (waitKeyEx preserves special keys such as arrows).
        key = cv2.waitKeyEx(1)
        return _map_key(key)

    def close(self) -> None:
        """Destroy the player window."""
        try:
            self._cv2.destroyWindow(self._win)
        except Exception:
            pass

    def _on_trackbar(self, pos: int) -> None:
        """Trackbar callback. Records user seek target frame."""
        if self._ignore_trackbar_event:
            return
        self._pending_seek = int(pos)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Mouse callback for button clicks in the bottom control strip."""
        cv2 = self._cv2
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Buttons are drawn in the strip appended under the video frame.
        local_y = y - self._last_display_h
        if local_y < 0:
            return

        for bx, by, bw, bh, cmd in self._button_rects:
            if bx <= x <= bx + bw and by <= local_y <= by + bh:
                self._pending_cmd = cmd
                return


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scale_to_width(frame: Any, max_w: int) -> Any:
    """Scale frame down if wider than max_w, preserving aspect ratio."""
    import cv2
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    new_w = max_w
    new_h = int(h * max_w / w)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _draw_button_strip(width: int, ui_state: UIState) -> Tuple[Any, List[Tuple[int, int, int, int, UICommand]]]:
    """Build the button strip and return button rectangles for hit testing."""
    import cv2

    strip = np.full((_CTRL_H, width, 3), _BG, dtype=np.uint8)
    rects: List[Tuple[int, int, int, int, UICommand]] = []
    n = len(_BUTTONS)
    total_btn_w = n * _BTN_W
    x_start = max(0, (width - total_btn_w) // 2)

    for i, btn in enumerate(_BUTTONS):
        bx = x_start + i * _BTN_W
        by = _BTN_PAD_Y
        bw = _BTN_W - 4
        bh = _CTRL_H - 2 * _BTN_PAD_Y
        rects.append((bx, by, bw, bh, btn["cmd"]))

        # Highlight active state
        if btn["cmd"] == UICommand.PLAY and ui_state.is_playing:
            bg = (30, 90, 30)
        elif btn["cmd"] == UICommand.PAUSE and not ui_state.is_playing:
            bg = (30, 30, 90)
        else:
            bg = _BTN_ACTIVE

        cv2.rectangle(strip, (bx, by), (bx + bw, by + bh), bg, -1)
        cv2.rectangle(strip, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)

        label = btn["label"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        tx = bx + (bw - tw) // 2
        ty = by + (bh + th) // 2
        cv2.putText(strip, label, (tx, ty), font, fs, _FG, 1, cv2.LINE_AA)

    # Frame counter on the right
    counter = f"{ui_state.current_frame_idx} / {max(0, ui_state.total_frames - 1)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(counter, font, 0.45, 1)
    cv2.putText(
        strip, counter,
        (width - tw - 10, _CTRL_H // 2 + 6),
        font, 0.45, (160, 160, 160), 1, cv2.LINE_AA,
    )

    return strip, rects


def _map_key(key: int) -> UICommand:
    """Map a cv2 key code to UICommand."""
    if key in (-1, 255, 0xFF):
        return UICommand.NONE
    if key in (ord("q"), 27):   # q or Esc
        return UICommand.QUIT
    if key in (ord(" "), ord("p"), ord("P")):
        return UICommand.PLAY  # caller toggles play/pause
    if key in (ord("d"), ord("D"), 83, 2555904, 65363):
        return UICommand.STEP_FWD
    if key in (ord("a"), ord("A"), 81, 2424832, 65361):
        return UICommand.STEP_BACK
    return UICommand.NONE
