"""Main scenario runner.

Orchestrates the full ADAS pipeline for one video:
  dataset frame source -> lane_detection -> obstacle_detection
  -> collision_risk estimator -> decision -> context -> UI -> audio

Entry point: run_scenario(config: ScenarioConfig) -> None
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .types import ScenarioConfig, FrameResult
from .events import ScenarioEvent, EventType, log_event


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_scenario(config: ScenarioConfig, *, log_file: Optional[str] = None) -> None:
    """Run the full ADAS pipeline for one video.

    Parameters
    ----------
    config : ScenarioConfig
        All configuration for this run.
    log_file : str, optional
        Path to a JSONL file for event logging. None = no file logging.
    """
    _ensure_src_on_path()

    from adas.dataset import parser, indexer
    from adas.context.defaults import DEFAULT_CONFIG
    from adas.lane_detection import process_frame
    from adas.obstacle_detection.detector import Detector
    from adas.obstacle_detection.tracking import SimpleTracker
    from adas.collision_risk.estimator import RiskEstimator
    from adas.collision_risk.decision import decide
    from adas.collision_risk.types import SystemAction

    # ---- Load record from index ----
    record = _load_record(config)
    if record is None:
        print(
            f"[ERROR] No record found for category_id={config.category_id}, "
            f"video_id={config.video_id} in {config.index_path}"
        )
        return

    record_path = record["path"]
    frame_source_path = _resolve_frame_source_path(record_path)
    n_frames = int(record.get("n_frames") or 0)

    # ---- Load annotation ----
    annotation = _load_annotation(config)

    log_event(ScenarioEvent(
        event_type=EventType.VIDEO_START,
        frame_idx=0,
        timestamp_s=0.0,
        details={
            "category_id": config.category_id,
            "video_id": config.video_id,
            "n_frames": n_frames,
            "path": record_path,
            "frame_source": frame_source_path,
        },
    ), log_file=log_file)

    # ---- Frame iterator ----
    frame_iter = _iter_frames_lazy(frame_source_path, parser)
    if frame_iter is None:
        print(f"[ERROR] Could not build frame iterator for: {frame_source_path}")
        return
    frame_items = list(frame_iter)
    if not frame_items:
        print(f"[ERROR] No frames found at: {frame_source_path}")
        return

    # Trust discovered frame count for UI seek/slider range.
    n_frames = len(frame_items)

    # ---- Pipeline components ----
    ctx_config = DEFAULT_CONFIG
    prev_ctx_state: Any = None
    detector = Detector()
    tracker = SimpleTracker()
    risk_estimator = RiskEstimator()

    # ---- UI setup ----
    player = None
    if config.ui_backend == "cv2":
        from adas.ui.backend_cv2 import Cv2Player
        from adas.ui.types import UIState
        player = Cv2Player(window_name=f"ADAS | cat={config.category_id} vid={config.video_id}")
        ui_state = UIState(
            is_playing=True,
            current_frame_idx=0,
            total_frames=n_frames,
        )

    # ---- Timing ----
    target_spf = (1.0 / config.target_fps) if config.target_fps > 0 else 0.0

    # ---- State for logging ----
    prev_mode: Any = None
    last_warn_frame = -999
    last_brake_frame = -999
    warn_cooldown = 30  # frames between audio alerts

    # ---- Main loop ----
    frame_count = 0
    quit_requested = False

    cursor = 0
    last_cursor = n_frames - 1

    while 0 <= cursor <= last_cursor:
        if config.max_frames is not None and frame_count >= config.max_frames:
            break

        frame_idx, frame_ref = frame_items[cursor]

        t_loop_start = time.monotonic()

        # ---- Get frame ----
        frame = parser.get_frame(frame_ref)
        if frame is None:
            cursor += 1
            continue

        timestamp_s = frame_idx / max(1.0, config.target_fps)

        # ---- Context (every N frames or first frame) ----
        if frame_idx % config.context_interval == 0 or prev_ctx_state is None:
            from adas.context import route
            from adas.context.lane_heuristic import detect_lanes_heuristic
            lane_input = detect_lanes_heuristic(frame, config=ctx_config)
            prev_ctx_state = route(
                frame,
                lane_detection=lane_input,
                timestamp_s=timestamp_s,
                fps=config.target_fps,
                emergency=None,
                prev_state=prev_ctx_state,
                config=ctx_config,
            )

        ctx_state = prev_ctx_state

        # Log mode changes
        if prev_mode is not None and ctx_state.mode != prev_mode:
            log_event(ScenarioEvent(
                event_type=EventType.MODE_CHANGE,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                details={"from": prev_mode.value, "to": ctx_state.mode.value},
            ), log_file=log_file)
        prev_mode = ctx_state.mode

        # ---- Lane detection ----
        from adas.lane_detection import process_frame
        lane_output = process_frame(frame, ctx_state)

        # ---- Obstacle detection + tracking ----
        raw_detections = detector.detect(frame, lane_output=lane_output, context_state=ctx_state)
        tracked = tracker.update(raw_detections)

        # ---- Collision risk ----
        risks = risk_estimator.estimate_risk(
            tracked, lane_output=lane_output, context_state=ctx_state, frame_idx=frame_idx
        )
        action, intensity = decide(risks, context_state=ctx_state)

        # ---- Audio ----
        if config.enable_audio:
            from adas.ui.audio import play_warning_beep, play_brake_beep
            from adas.collision_risk.types import SystemAction
            if action == SystemAction.BRAKE and frame_idx - last_brake_frame > warn_cooldown:
                play_brake_beep()
                last_brake_frame = frame_idx
            elif action == SystemAction.WARN and frame_idx - last_warn_frame > warn_cooldown:
                play_warning_beep()
                last_warn_frame = frame_idx

        # Log significant risk events
        if action == SystemAction.BRAKE:
            log_event(ScenarioEvent(
                event_type=EventType.BRAKE,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                details={"intensity": round(intensity, 2)},
            ), log_file=log_file)
        elif action == SystemAction.WARN:
            log_event(ScenarioEvent(
                event_type=EventType.WARN,
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                details={"intensity": round(intensity, 2)},
            ), log_file=log_file)

        # ---- Ground-truth label ----
        annotation_label = _get_frame_label(annotation, frame_idx)

        # ---- Compose frame result ----
        result = FrameResult(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            lane_output=lane_output,
            obstacles=tracked,
            risks=risks,
            action=action,
            action_intensity=intensity,
            context_state=ctx_state,
            annotation_label=annotation_label,
        )

        # ---- UI ----
        if player is not None:
            from adas.ui.overlays import draw_lanes, draw_obstacles, draw_risk
            from adas.ui.dashboard import draw_stats_panel
            from adas.ui.types import UICommand

            display_frame = frame.copy()

            if config.show_lanes:
                display_frame = draw_lanes(display_frame, lane_output)
            if config.show_obstacles:
                display_frame = draw_obstacles(display_frame, tracked)
            if config.show_risk:
                display_frame = draw_risk(display_frame, risks, ctx_state)

            if config.show_dashboard:
                best_ttc = min((r.ttc for r in risks), default=float("inf"))
                best_risk = max((r.risk_score for r in risks), default=0.0)
                stats = _build_stats_dict(
                    ctx_state, action, frame_idx, n_frames,
                    best_ttc, best_risk, annotation_label, frame_count,
                    t_loop_start,
                )
                display_frame = draw_stats_panel(display_frame, stats)

            ui_state.current_frame_idx = frame_idx
            if n_frames > 0:
                ui_state.total_frames = n_frames

            cmd = player.show_frame(display_frame, ui_state, annotation_label=annotation_label)

            next_cursor = min(last_cursor, cursor + 1)

            # Handle UI commands
            if cmd == UICommand.QUIT:
                quit_requested = True
                break
            elif cmd in (UICommand.PLAY, UICommand.PAUSE):
                ui_state.is_playing = not ui_state.is_playing
            elif cmd == UICommand.STEP_FWD:
                ui_state.is_playing = False
                next_cursor = min(last_cursor, cursor + 1)
            elif cmd == UICommand.STEP_BACK:
                ui_state.is_playing = False
                next_cursor = max(0, cursor - 1)
            elif cmd == UICommand.SEEK_TO and ui_state.seek_target_frame is not None:
                next_cursor = max(0, min(last_cursor, int(ui_state.seek_target_frame)))
                ui_state.seek_target_frame = None

            # If paused, wait until play is pressed
            if not ui_state.is_playing:
                while True:
                    cmd2 = player.show_frame(display_frame, ui_state)
                    if cmd2 == UICommand.QUIT:
                        quit_requested = True
                        break
                    if cmd2 in (UICommand.PLAY, UICommand.PAUSE, UICommand.STEP_FWD):
                        if cmd2 == UICommand.STEP_FWD:
                            next_cursor = min(last_cursor, cursor + 1)
                            break  # advance one frame
                        ui_state.is_playing = True
                        next_cursor = min(last_cursor, cursor + 1)
                        break
                    if cmd2 == UICommand.STEP_BACK:
                        next_cursor = max(0, cursor - 1)
                        break
                    if cmd2 == UICommand.SEEK_TO and ui_state.seek_target_frame is not None:
                        next_cursor = max(0, min(last_cursor, int(ui_state.seek_target_frame)))
                        ui_state.seek_target_frame = None
                        break
                    import time as _t
                    _t.sleep(0.016)
                if quit_requested:
                    break

            cursor = next_cursor
        else:
            cursor += 1

        # ---- Timing control ----
        elapsed = time.monotonic() - t_loop_start
        if target_spf > 0 and elapsed < target_spf:
            time.sleep(target_spf - elapsed)

        frame_count += 1

    # ---- Cleanup ----
    if player is not None:
        player.close()

    if quit_requested:
        log_event(ScenarioEvent(
            event_type=EventType.QUIT,
            frame_idx=frame_count,
            timestamp_s=frame_count / max(1.0, config.target_fps),
            details={"frames_processed": frame_count},
        ), log_file=log_file)
    else:
        log_event(ScenarioEvent(
            event_type=EventType.VIDEO_END,
            frame_idx=frame_count,
            timestamp_s=frame_count / max(1.0, config.target_fps),
            details={"frames_processed": frame_count},
        ), log_file=log_file)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    """Add src/ to sys.path if not already present."""
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    src_root = os.path.join(project_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def _load_record(config: ScenarioConfig) -> Optional[Dict[str, Any]]:
    """Load record metadata from index.db."""
    if not os.path.exists(config.index_path):
        return None
    try:
        conn = sqlite3.connect(config.index_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM records WHERE category_id = ? AND video_id = ? LIMIT 1",
            (config.category_id, config.video_id),
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row is not None else None
    except sqlite3.Error:
        return None


def _load_annotation(config: ScenarioConfig) -> Optional[Dict[str, Any]]:
    """Load DADA-2000 annotation for the configured video."""
    try:
        conn = sqlite3.connect(config.index_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM annotations WHERE category_id = ? AND video_id = ? LIMIT 1",
            (config.category_id, config.video_id),
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row is not None else None
    except sqlite3.Error:
        return None


def _resolve_frame_source_path(record_path: str) -> str:
    """Resolve a display-friendly frame source path.

    DADA records may point to auxiliary folders such as fixation/maps/seg.
    For scenario playback, prefer sibling images/ when it exists.
    """
    if not record_path:
        return record_path

    norm = os.path.normpath(record_path)
    base = os.path.basename(norm).lower()
    if base in {"fixation", "maps", "seg", "semantic"}:
        parent = os.path.dirname(norm)
        candidate = os.path.join(parent, "images")
        if os.path.isdir(candidate):
            print(f"[INFO] Using image frames from: {candidate} (instead of {record_path})")
            return candidate

    return record_path


def _iter_frames_lazy(record_path: str, parser: Any) -> Optional[Iterator[Tuple[int, Any]]]:
    """Return a normalized iterator of (frame_idx, frame_ref)."""
    try:
        raw_iter = parser.iter_frames(record_path)
        if raw_iter is None:
            return None

        def _normalized() -> Iterator[Tuple[int, Any]]:
            fallback_idx = 0
            for item in raw_iter:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], int)
                ):
                    yield item[0], item[1]
                else:
                    yield fallback_idx, item
                fallback_idx += 1

        return _normalized()
    except Exception as exc:
        print(f"[WARN] iter_frames failed: {exc}")
        return None


def _get_frame_label(annotation: Optional[Dict[str, Any]], frame_idx: int) -> str:
    """Derive ground-truth label for a frame from annotation metadata."""
    if annotation is None:
        return "unknown"

    af = annotation.get("accident_frame")
    bs = annotation.get("abnormal_start_frame")
    be = annotation.get("abnormal_end_frame")

    if isinstance(af, int) and af >= 0 and frame_idx == af:
        return "accident_frame"
    if isinstance(bs, int) and isinstance(be, int) and bs <= frame_idx <= be:
        return "abnormal"
    return "normal"


def _build_stats_dict(
    ctx_state: Any,
    action: Any,
    frame_idx: int,
    n_frames: int,
    best_ttc: float,
    best_risk: float,
    annotation_label: str,
    frame_count: int,
    t_loop_start: float,
) -> Dict[str, Any]:
    """Build stats dict for dashboard rendering."""
    elapsed = time.monotonic() - t_loop_start
    fps_est = 1.0 / elapsed if elapsed > 0.001 else 0.0

    stats: Dict[str, Any] = {
        "frame_idx": frame_idx,
        "total_frames": n_frames,
        "fps": round(fps_est, 1),
        "action": action.value if hasattr(action, "value") else str(action),
        "annotation_label": annotation_label,
        "ttc": best_ttc if best_ttc < float("inf") else None,
        "risk_score": round(best_risk, 3),
    }

    if ctx_state is not None:
        stats["mode"] = ctx_state.mode.value if hasattr(ctx_state.mode, "value") else str(ctx_state.mode)
        stats["weather"] = ctx_state.weather_condition.value if hasattr(ctx_state, "weather_condition") else "N/A"
        stats["light"] = ctx_state.light_condition.value if hasattr(ctx_state, "light_condition") else "N/A"
        stats["braking_mult"] = round(float(ctx_state.braking_multiplier), 2)

        ls = getattr(ctx_state, "lane_state", None)
        if ls is not None:
            avail = ls.availability if hasattr(ls, "availability") else None
            stats["lane_state"] = avail.value if hasattr(avail, "value") else str(avail)

        rs = getattr(ctx_state, "road_surface", None)
        if rs is not None:
            st = rs.surface_type if hasattr(rs, "surface_type") else None
            stats["road_surface"] = st.value if hasattr(st, "value") else str(st)

        vis = getattr(ctx_state, "visibility", None)
        if vis is not None:
            stats["visibility_conf"] = round(float(vis.confidence), 2)

    return stats
