"""Scenario types: configuration and per-frame results."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass(frozen=True)
class ScenarioConfig:
    """All configuration needed to run one ADAS scenario.

    Attributes
    ----------
    dataset_root : str
        Path to the DADA-2000 root directory.
    index_path : str
        Path to the index.db SQLite file.
    category_id : int
        Category ID to play (e.g. 1 = rear-end collision).
    video_id : int
        Video ID within the category.
    target_fps : float
        Frames to process per second (wall-clock).
        Set to 0 to process as fast as possible.
    context_interval : int
        Number of frames between full context route() calls.
        Between calls the previous ContextState is reused.
        Lower = more accurate, higher = faster.
    ui_backend : str
        Which UI backend to use. Supported: "cv2", "none".
        "none" runs without any display (useful for tests/batch).
    enable_audio : bool
        Whether to play warning beeps when risk is detected.
    max_frames : int or None
        Maximum number of frames to process. None = whole video.
    show_dashboard : bool
        Whether to render the stats panel next to the video.
    show_lanes : bool
        Whether to draw lane overlay.
    show_obstacles : bool
        Whether to draw obstacle bounding boxes.
    show_risk : bool
        Whether to draw risk level indicators.
    annotations_csv : str or None
        Path to DADA2000_video_annotations.csv for ground-truth labels.
    seed : int
        Random seed (used if sampling is needed).
    """

    dataset_root: str = "data/raw/DADA2000"
    index_path: str = "data/processed/index.db"
    category_id: int = 1
    video_id: int = 1
    target_fps: float = 30.0
    context_interval: int = 5
    ui_backend: str = "cv2"
    enable_audio: bool = True
    max_frames: Optional[int] = None
    show_dashboard: bool = True
    show_lanes: bool = True
    show_obstacles: bool = True
    show_risk: bool = True
    annotations_csv: Optional[str] = None
    seed: int = 42


@dataclass(frozen=True)
class FrameResult:
    """Result of processing one frame through the full ADAS pipeline.

    Attributes
    ----------
    frame_idx : int
        Index of the frame within the video.
    timestamp_s : float
        Estimated timestamp in seconds (frame_idx / fps).
    lane_output : object or None
        LaneOutput from lane detection.
    obstacles : list
        List of DetectedObject (tracked).
    risks : list
        List of RiskResult per obstacle.
    action : object
        SystemAction chosen by the decision module.
    action_intensity : float
        Intensity of the action in [0, 1].
    context_state : object or None
        ContextState from the context router.
    annotation_label : str
        Ground-truth annotation label ("normal", "abnormal", "accident_frame").
    """

    frame_idx: int = 0
    timestamp_s: float = 0.0
    lane_output: Any = None
    obstacles: List[Any] = field(default_factory=list)
    risks: List[Any] = field(default_factory=list)
    action: Any = None
    action_intensity: float = 0.0
    context_state: Any = None
    annotation_label: str = "unknown"
