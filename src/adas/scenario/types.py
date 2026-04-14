"""Scenario data types.

Frozen dataclasses used by the scenario runner and the master-panel UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class ScenarioConfig:
    """All parameters needed to run one ADAS scenario."""

    category_id: int = 1
    video_id: int = 1
    dataset_root: str = "data/raw/DADA2000"
    index_path: str = "data/processed/index.db"
    ui_backend: str = "dpg"
    target_fps: float = 30.0
    max_frames: Optional[int] = None
    context_interval: int = 5
    enable_audio: bool = True
    show_dashboard: bool = True
    show_lanes: bool = True
    show_obstacles: bool = True
    show_risk: bool = True


@dataclass
class FrameResult:
    """Result produced by the pipeline for a single frame."""

    frame_idx: int = 0
    timestamp_s: float = 0.0
    lane_output: Any = None
    obstacles: List[Any] = field(default_factory=list)
    risks: List[Any] = field(default_factory=list)
    action: Any = None
    action_intensity: float = 0.0
    context_state: Any = None
    annotation_label: str = "unknown"
