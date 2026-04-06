"""UI types: commands and player state."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class UICommand(enum.Enum):
    """Commands produced by the UI backend when the user presses a key."""

    PLAY = "play"
    PAUSE = "pause"
    STEP_FWD = "step_fwd"
    STEP_BACK = "step_back"
    SEEK_TO = "seek_to"     # requires seek_target_frame to be set
    QUIT = "quit"
    NONE = "none"           # no command issued this frame


@dataclass
class UIState:
    """Mutable state of the UI player.

    Attributes
    ----------
    is_playing : bool
        True when video is playing, False when paused.
    current_frame_idx : int
        Current frame index.
    total_frames : int
        Total number of frames in the current record.
    speed : float
        Playback speed multiplier (1.0 = real-time).
    seek_target_frame : int or None
        When a SEEK_TO command is issued, this is the target frame index.
    """

    is_playing: bool = True
    current_frame_idx: int = 0
    total_frames: int = 0
    speed: float = 1.0
    seek_target_frame: Optional[int] = None
