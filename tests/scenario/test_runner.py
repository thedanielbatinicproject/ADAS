"""Tests for adas.scenario.runner (dry run without GUI or real dataset)."""

from __future__ import annotations

import types
from typing import Iterator, List, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from adas.scenario.types import ScenarioConfig, FrameResult
from adas.scenario.events import ScenarioEvent, EventType, log_event


# ---------------------------------------------------------------------------
# Tests for ScenarioConfig
# ---------------------------------------------------------------------------

class TestScenarioConfig:
    def test_defaults(self):
        cfg = ScenarioConfig()
        assert cfg.category_id == 1
        assert cfg.video_id == 1
        assert cfg.ui_backend == "cv2"
        assert cfg.target_fps == 30.0
        assert cfg.context_interval == 5

    def test_frozen(self):
        cfg = ScenarioConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.category_id = 99  # type: ignore[misc]

    def test_custom_values(self):
        cfg = ScenarioConfig(
            category_id=3,
            video_id=5,
            ui_backend="none",
            target_fps=0,
            max_frames=100,
        )
        assert cfg.category_id == 3
        assert cfg.ui_backend == "none"
        assert cfg.max_frames == 100


# ---------------------------------------------------------------------------
# Tests for FrameResult
# ---------------------------------------------------------------------------

class TestFrameResult:
    def test_defaults(self):
        r = FrameResult()
        assert r.frame_idx == 0
        assert r.obstacles == []
        assert r.risks == []
        assert r.annotation_label == "unknown"

    def test_frozen(self):
        r = FrameResult()
        with pytest.raises((AttributeError, TypeError)):
            r.frame_idx = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests for log_event
# ---------------------------------------------------------------------------

class TestLogEvent:
    def test_prints_without_crash(self, capsys):
        event = ScenarioEvent(
            event_type=EventType.WARN,
            frame_idx=42,
            timestamp_s=1.4,
            details={"intensity": 0.7},
        )
        log_event(event)
        captured = capsys.readouterr()
        assert "WARN" in captured.out
        assert "42" in captured.out

    def test_writes_to_log_file(self, tmp_path):
        import json
        log_path = str(tmp_path / "events.jsonl")
        event = ScenarioEvent(
            event_type=EventType.BRAKE,
            frame_idx=10,
            timestamp_s=0.33,
            details={"intensity": 0.9},
        )
        log_event(event, log_file=log_path)

        with open(log_path) as f:
            line = f.readline()
        record = json.loads(line)
        assert record["event_type"] == "brake"
        assert record["frame_idx"] == 10


# ---------------------------------------------------------------------------
# Dry-run test for run_scenario (mocked dataset + no GUI)
# ---------------------------------------------------------------------------

def _fake_frames(n: int = 10):
    """Yield fake (frame_idx, frame_ref) tuples with BGR numpy frames."""
    for i in range(n):
        yield (i, f"fake://frame/{i}")


def _fake_get_frame(frame_ref):
    """Return a synthetic BGR frame for any ref."""
    return np.zeros((180, 320, 3), dtype=np.uint8)


class TestRunScenarioDryRun:
    """Smoke test: run_scenario with no-GUI and mocked dataset components."""

    def test_headless_run_no_crash(self, tmp_path, monkeypatch):
        """Run the scenario with ui_backend='none' and mocked index + parser."""
        import adas.scenario.runner as runner_mod
        import adas.dataset.parser as real_parser

        # Mock _load_record to return a fake record
        fake_record = {
            "record_id": 1,
            "category_id": 1,
            "video_id": 1,
            "path": "/fake/path/video",
            "n_frames": 10,
            "category": "1",
            "annotation_status": "ok",
        }
        monkeypatch.setattr(runner_mod, "_load_record", lambda cfg: fake_record)
        monkeypatch.setattr(runner_mod, "_load_annotation", lambda cfg: None)

        # Mock parser.iter_frames and parser.get_frame
        def fake_iter(path):
            return _fake_frames(10)

        monkeypatch.setattr(real_parser, "iter_frames", fake_iter)
        monkeypatch.setattr(real_parser, "get_frame", _fake_get_frame)

        # Patch route() to return a minimal context state
        from adas.context.types import (
            ContextState, Mode, WeatherCondition, LightCondition,
            LaneState, LaneAvailability, RoadSurfaceHint, RoadSurfaceType,
            SceneMetrics, VisibilityEstimate,
        )
        fake_ctx = ContextState(
            mode=Mode.NORMAL_MARKED,
            braking_multiplier=1.0,
            weather_condition=WeatherCondition.CLEAR,
            light_condition=LightCondition.DAY,
        )

        monkeypatch.setattr("adas.context.route", lambda *a, **kw: fake_ctx)
        monkeypatch.setattr(
            "adas.context.lane_heuristic.detect_lanes_heuristic",
            lambda frame, config=None: MagicMock(
                left_detected=False, right_detected=False,
                left_confidence=0.0, right_confidence=0.0,
            )
        )

        cfg = ScenarioConfig(
            category_id=1,
            video_id=1,
            dataset_root="/fake",
            index_path=str(tmp_path / "index.db"),
            ui_backend="none",
            enable_audio=False,
            max_frames=5,
            show_dashboard=False,
        )

        # Should complete without exceptions
        runner_mod.run_scenario(cfg)
