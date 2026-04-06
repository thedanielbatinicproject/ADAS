"""Dear PyGui UI backend (DpgPlayer).

Uses OpenGL-based rendering via Dear PyGui for:
- GPU-accelerated frame texture display (no per-pixel X11 bitmap transfer)
- Resolution-independent crisp text via native DPG widgets
- Responsive slider, buttons, and keyboard with proper event handling

Manual render mode: the caller drives the loop by calling show_frame()
repeatedly, each of which renders exactly one DPG frame.

Key bindings (same as Cv2Player):
  Space / P   -> toggle play/pause
  Right / D   -> step forward one frame
  Left / A    -> step backward one frame
  Q / Esc     -> quit

Requires: pip install dearpygui
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .types import UICommand, UIState


class DpgPlayer:
    """Dear PyGui-based player for ADAS scenario.

    Drop-in replacement for Cv2Player with GPU-accelerated rendering
    and native UI widgets.  Text is rendered independently of the frame
    texture, so it remains crisp at any window size.

    Parameters
    ----------
    window_name : str
        Viewport title.
    max_display_width : int
        Initial viewport width hint (DPG viewport is freely resizable).
    """

    def __init__(
        self,
        window_name: str = "ADAS Scenario",
        max_display_width: int = 1920,
    ) -> None:
        try:
            import dearpygui.dearpygui as dpg
            self._dpg = dpg
        except ImportError as exc:
            raise ImportError(
                "dearpygui is required for DpgPlayer. "
                "Install with: pip install dearpygui"
            ) from exc

        self._win_name = window_name
        self._max_w = max_display_width
        self._initialized = False
        self._pending_cmd = UICommand.NONE
        self._pending_seek: Optional[int] = None
        self._suppress_slider = False
        self._tex_tag: Optional[int] = None
        self._tex_w = 0
        self._tex_h = 0
        self._total_frames = 0

    # ------------------------------------------------------------------
    # Public API (mirrors Cv2Player)
    # ------------------------------------------------------------------

    def setup(self, total_frames: int) -> None:
        """Create the DPG viewport, window layout, texture, and handlers."""
        dpg = self._dpg
        self._total_frames = total_frames

        dpg.create_context()

        # ── Viewport: half primary monitor width, centered ─────────────
        # VcXsrv reports a combined virtual desktop; assume primary monitor
        # is ~1920 px wide and position the window roughly in its centre.
        # An explicit y offset keeps the OS title bar visible.
        vp_w = min(self._max_w, 960)
        vp_h = int(vp_w * 9 / 16) + 340
        x_center = max(0, (1920 - vp_w) // 2)
        dpg.create_viewport(
            title=self._win_name,
            width=vp_w,
            height=vp_h,
            x_pos=x_center,
            y_pos=50,
            decorated=True,
        )

        # -- Texture (placeholder, resized on first frame) ----------------
        with dpg.texture_registry():
            pw, ph = 640, 480
            self._tex_w, self._tex_h = pw, ph
            self._tex_tag = dpg.add_dynamic_texture(
                pw, ph, [0.0] * (pw * ph * 4),
            )

        # -- Main window filling the viewport -----------------------------
        with dpg.window(tag="main_win", no_title_bar=True):
            # Status line
            dpg.add_text("", tag="status_text", color=(200, 200, 200))
            dpg.add_separator()

            # Video frame (GPU texture)
            dpg.add_image(self._tex_tag, tag="frame_img")

            dpg.add_separator()

            # Frame slider
            dpg.add_slider_int(
                tag="frame_slider",
                label="",
                min_value=0,
                max_value=max(total_frames - 1, 1),
                default_value=0,
                width=-1,
                callback=self._on_slider,
            )

            # Control buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="  <<  ",
                    callback=lambda: self._set_cmd(UICommand.STEP_BACK),
                )
                dpg.add_button(
                    label=" Play ",
                    callback=lambda: self._set_cmd(UICommand.PLAY),
                )
                dpg.add_button(
                    label=" Pause ",
                    callback=lambda: self._set_cmd(UICommand.PAUSE),
                )
                dpg.add_button(
                    label="  >>  ",
                    callback=lambda: self._set_cmd(UICommand.STEP_FWD),
                )
                dpg.add_button(
                    label=" Quit ",
                    callback=lambda: self._set_cmd(UICommand.QUIT),
                )
                dpg.add_spacer(width=20)
                dpg.add_text(
                    "0 / 0", tag="frame_counter", color=(160, 160, 160),
                )

            dpg.add_separator()

            # Dashboard — card grid
            with dpg.collapsing_header(
                label="Dashboard", default_open=True, tag="stats_header",
            ):
                # Will be populated dynamically in update_stats()
                dpg.add_group(tag="dashboard_grid")

        dpg.set_primary_window("main_win", True)

        # -- Card themes for dashboard ------------------------------------
        self._card_themes: Dict[str, int] = {}
        _card_defs = {
            "default":  ((40, 40, 48), (70, 70, 80)),     # bg, border
            "good":     ((20, 55, 30), (40, 140, 60)),     # green
            "warn":     ((60, 55, 15), (200, 180, 30)),    # yellow
            "danger":   ((65, 20, 20), (220, 50, 50)),     # red
            "brake":    ((70, 15, 15), (255, 40, 40)),     # bright red
            "info":     ((20, 35, 60), (50, 110, 190)),    # blue
        }
        for name, (bg, border) in _card_defs.items():
            with dpg.theme() as t:
                with dpg.theme_component(dpg.mvChildWindow):
                    dpg.add_theme_color(dpg.mvThemeCol_ChildBg, bg)
                    dpg.add_theme_color(dpg.mvThemeCol_Border, border)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                    dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize, 2)
            self._card_themes[name] = t
        self._dashboard_built = False

        # -- Keyboard handlers --------------------------------------------
        with dpg.handler_registry():
            dpg.add_key_press_handler(
                dpg.mvKey_Spacebar,
                callback=lambda: self._set_cmd(UICommand.PLAY),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_P,
                callback=lambda: self._set_cmd(UICommand.PLAY),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Right,
                callback=lambda: self._set_cmd(UICommand.STEP_FWD),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_D,
                callback=lambda: self._set_cmd(UICommand.STEP_FWD),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Left,
                callback=lambda: self._set_cmd(UICommand.STEP_BACK),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_A,
                callback=lambda: self._set_cmd(UICommand.STEP_BACK),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Q,
                callback=lambda: self._set_cmd(UICommand.QUIT),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_Escape,
                callback=lambda: self._set_cmd(UICommand.QUIT),
            )

        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._initialized = True

    def show_frame(
        self,
        frame: Any,
        ui_state: UIState,
        annotation_label: str = "",
    ) -> UICommand:
        """Upload *frame* to the GPU texture, render one DPG frame, return command.

        Parameters
        ----------
        frame : numpy.ndarray
            BGR uint8 image (overlays already composited by the caller).
        ui_state : UIState
            Current player state.
        annotation_label : str
            Ground-truth label shown in the status bar.

        Returns
        -------
        UICommand
        """
        dpg = self._dpg

        if not self._initialized:
            self.setup(ui_state.total_frames)

        if not dpg.is_dearpygui_running():
            return UICommand.QUIT

        # Upload frame to GPU texture
        self._upload_frame(frame)

        # Auto-scale image widget to viewport width
        vp_w = dpg.get_viewport_client_width()
        if self._tex_w > 0 and vp_w > 0:
            scale = vp_w / self._tex_w
            dpg.configure_item(
                "frame_img", width=vp_w, height=int(self._tex_h * scale),
            )

        # Update slider (suppress callback to avoid re-seek)
        self._suppress_slider = True
        dpg.set_value("frame_slider", ui_state.current_frame_idx)
        self._suppress_slider = False

        # Update text displays
        status = "PLAYING" if ui_state.is_playing else "PAUSED"
        max_f = max(0, ui_state.total_frames - 1)
        label_str = f"  [{annotation_label}]" if annotation_label else ""
        dpg.set_value(
            "status_text",
            f"Frame {ui_state.current_frame_idx}/{max_f} | {status}{label_str}",
        )
        dpg.set_value(
            "frame_counter",
            f"{ui_state.current_frame_idx} / {max_f}",
        )

        # Render one DPG frame
        dpg.render_dearpygui_frame()

        # Return pending command
        if self._pending_cmd != UICommand.NONE:
            cmd = self._pending_cmd
            self._pending_cmd = UICommand.NONE
            return cmd

        if self._pending_seek is not None:
            ui_state.seek_target_frame = int(self._pending_seek)
            self._pending_seek = None
            return UICommand.SEEK_TO

        return UICommand.NONE

    def update_stats(self, stats: Dict[str, Any]) -> None:
        """Update the dashboard as a grid of themed cards.

        Cards are colour-coded:
        - **danger/brake** (red): action == BRAKE or risk_score > 0.7
        - **warn** (yellow): action == WARN or risk_score > 0.3
        - **good** (green): normal / safe values
        - **info** (blue): neutral informational data
        """
        dpg = self._dpg
        if not self._initialized:
            return

        # ── Derive card entries from stats ──────────────────────────────
        action_raw = str(stats.get("action", "")).upper()
        risk_raw = stats.get("risk_score")
        risk_val = float(risk_raw) if risk_raw is not None else 0.0

        # Global alert level used to tint relevant cards
        if "BRAKE" in action_raw or risk_val > 0.7:
            alert = "brake"
        elif "WARN" in action_raw or risk_val > 0.3:
            alert = "warn"
        else:
            alert = "good"

        _CARD_MAP: List[tuple] = [
            # (stat_key, display_label, theme_fn)
            ("mode", "Mode", lambda v: "info"),
            ("weather", "Weather", lambda v: "warn" if v and "RAIN" in str(v).upper() else "info"),
            ("light", "Light", lambda v: "warn" if v and "NIGHT" in str(v).upper() else "info"),
            ("lane_state", "Lanes", lambda v: "good" if v and "BOTH" in str(v).upper() else "warn"),
            ("road_surface", "Road Surface", lambda v: "warn" if v and "WET" in str(v).upper() else "info"),
            ("visibility_conf", "Visibility", lambda v: "good" if v and float(v) > 0.6 else "warn"),
            ("braking_mult", "Braking", lambda v: "warn" if v and float(v) > 1.0 else "good"),
            ("risk_score", "Risk Score", lambda _: alert),
            ("action", "Action", lambda _: alert),
            ("fps", "FPS", lambda _: "info"),
            ("annotation_label", "GT Label", lambda v: "danger" if v and "accident" in str(v).lower() else ("warn" if v and "abnormal" in str(v).lower() else "info")),
            # TTC at the end so its appearance/disappearance does not shift
            # other cards.
            ("ttc", "TTC (s)", lambda v: "danger" if v and float(v) < 2.0 else ("warn" if v and float(v) < 4.0 else "good")),
        ]

        # Collect visible cards: canonical + extras
        cards: List[tuple] = []  # (label, value_str, theme_name)
        seen: set = set()
        for key, label, theme_fn in _CARD_MAP:
            val = stats.get(key)
            seen.add(key)
            if val is None:
                # Always reserve a slot so the grid stays stable.
                cards.append((label, "\u2014", "default"))
                continue
            val_s = f"{val:.2f}" if isinstance(val, float) else str(val)
            try:
                th = theme_fn(val_s)
            except Exception:
                th = "default"
            cards.append((label, val_s, th))
        # Extra keys
        for key, val in stats.items():
            if key in seen:
                continue
            label = key.replace("_", " ").title()
            val_s = f"{val:.2f}" if isinstance(val, float) else str(val)
            cards.append((label, val_s, "default"))

        # ── Rebuild grid ────────────────────────────────────────────────
        # We delete and recreate children each update.  DPG is fast enough
        # because we only have ~12–15 cards.
        dpg.delete_item("dashboard_grid", children_only=True)

        COLS = 4
        vp_w = dpg.get_viewport_client_width() or 800
        card_w = max(100, (vp_w - 40) // COLS - 8)
        card_h = 56

        row_items = [cards[i:i + COLS] for i in range(0, len(cards), COLS)]
        for row in row_items:
            with dpg.group(horizontal=True, parent="dashboard_grid"):
                for label, val_s, theme_name in row:
                    with dpg.child_window(width=card_w, height=card_h, border=True, no_scrollbar=True) as cw:
                        dpg.add_text(label, color=(160, 160, 180))
                        dpg.add_text(val_s, color=(255, 255, 255))
                    theme_id = self._card_themes.get(theme_name, self._card_themes["default"])
                    dpg.bind_item_theme(cw, theme_id)

    def close(self) -> None:
        """Destroy the DPG context and viewport."""
        try:
            self._dpg.destroy_context()
        except Exception:
            pass
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upload_frame(self, frame: Any) -> None:
        """Convert BGR uint8 -> RGBA float32 and upload to GPU texture."""
        dpg = self._dpg
        h, w = frame.shape[:2]

        # Recreate texture if frame dimensions changed
        if w != self._tex_w or h != self._tex_h:
            dpg.delete_item(self._tex_tag)
            with dpg.texture_registry():
                self._tex_tag = dpg.add_dynamic_texture(
                    w, h, [0.0] * (w * h * 4),
                )
            dpg.configure_item("frame_img", texture_tag=self._tex_tag)
            self._tex_w, self._tex_h = w, h

        # Vectorized numpy conversion (fast)
        frame_f = frame.astype(np.float32) * (1.0 / 255.0)
        rgba = np.empty((h, w, 4), dtype=np.float32)
        rgba[:, :, 0] = frame_f[:, :, 2]  # R from BGR channel 2
        rgba[:, :, 1] = frame_f[:, :, 1]  # G
        rgba[:, :, 2] = frame_f[:, :, 0]  # B from BGR channel 0
        rgba[:, :, 3] = 1.0

        dpg.set_value(self._tex_tag, rgba.ravel())

    def _set_cmd(self, cmd: UICommand) -> None:
        self._pending_cmd = cmd

    def _on_slider(self, sender: Any, value: Any) -> None:
        if not self._suppress_slider:
            self._pending_seek = int(value)
