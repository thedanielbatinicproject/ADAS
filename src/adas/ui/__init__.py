"""
UI package for ADAS.

Handles all display (overlays, dashboard, player controls) and audio feedback.
No detection logic lives here.

Sub-modules:
  types.py       - UICommand, UIState
  backend_cv2.py - Cv2Player (OpenCV-based interactive player)
  overlays.py    - draw_lanes(), draw_obstacles(), draw_risk()
  dashboard.py   - draw_stats_panel(), draw_stats_overlay()
  audio.py       - play_warning_beep(), play_brake_beep()
"""

from .types import UICommand, UIState
from .backend_cv2 import Cv2Player
from .overlays import draw_lanes, draw_obstacles, draw_risk
from .dashboard import draw_stats_panel, draw_stats_overlay
from .audio import play_warning_beep, play_brake_beep
from .player import create_player, run_player_loop

try:
    from .backend_dpg import DpgPlayer
except ImportError:
    DpgPlayer = None  # type: ignore[assignment,misc]

__all__ = [
    # types
    "UICommand",
    "UIState",
    # player
    "Cv2Player",
    # overlays
    "draw_lanes",
    "draw_obstacles",
    "draw_risk",
    # dashboard
    "draw_stats_panel",
    "draw_stats_overlay",
    # audio
    "play_warning_beep",
    "play_brake_beep",
    # player
    "create_player",
    "run_player_loop",
]
