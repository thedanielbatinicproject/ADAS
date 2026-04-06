"""
Scenario package.

Handles orchestration of the full ADAS pipeline for one video:
  dataset -> lane_detection -> obstacle_detection -> collision_risk -> UI -> audio

Sub-modules:
  types.py   - ScenarioConfig, FrameResult
  runner.py  - run_scenario(config) -> None
  events.py  - ScenarioEvent, EventType, log_event()
"""

from .types import ScenarioConfig, FrameResult
from .runner import run_scenario
from .events import ScenarioEvent, EventType, log_event

__all__ = [
    "run_scenario",
    "ScenarioConfig",
    "FrameResult",
    "ScenarioEvent",
    "EventType",
    "log_event",
]
