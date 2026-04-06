"""Scenario event logging.

Structured events are printed to stdout and optionally written to a log file.
Events are intentionally simple dicts so they can be serialized to JSON.
"""

from __future__ import annotations

import enum
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


class EventType(enum.Enum):
    """Types of scenario events."""

    MODE_CHANGE = "mode_change"
    EMERGENCY = "emergency"
    WARN = "warn"
    BRAKE = "brake"
    VIDEO_START = "video_start"
    VIDEO_END = "video_end"
    QUIT = "quit"
    INFO = "info"


@dataclass
class ScenarioEvent:
    """A single logged scenario event.

    Attributes
    ----------
    event_type : EventType
    frame_idx : int
    timestamp_s : float
    details : dict
        Arbitrary key-value pairs with event-specific context.
    wall_time : float
        Epoch time when the event was created.
    """

    event_type: EventType
    frame_idx: int
    timestamp_s: float
    details: Dict[str, Any]
    wall_time: float = 0.0

    def __post_init__(self) -> None:
        if self.wall_time == 0.0:
            self.wall_time = time.time()


def log_event(event: ScenarioEvent, *, log_file: Optional[str] = None) -> None:
    """Print event to stdout and optionally append to a JSONL log file.

    Parameters
    ----------
    event : ScenarioEvent
    log_file : str, optional
        Path to a JSONL file. Each event is appended as one JSON line.
    """
    ts = event.timestamp_s
    etype = event.event_type.value.upper()
    details_str = " ".join(f"{k}={v}" for k, v in event.details.items())
    print(f"[{etype}] frame={event.frame_idx} t={ts:.2f}s {details_str}")

    if log_file:
        record = {
            "event_type": event.event_type.value,
            "frame_idx": event.frame_idx,
            "timestamp_s": ts,
            "wall_time": event.wall_time,
            "details": event.details,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
