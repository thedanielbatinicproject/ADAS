#!/usr/bin/env python
"""CLI entry-point for running an ADAS scenario.

Normal mode
-----------
    python scripts/run_scenario.py --category-id 1 --video-id 3

Streaming mode  (used by the configurator window)
--------------------------------------------------
    python scripts/run_scenario.py --category-id 1 --video-id 3 \
        --stream-dir data/processed/configurator_stream --loop --log-every 30

In streaming mode the script writes ``latest.jpg`` and ``status.json``
to *stream-dir* every frame so the host GUI can display a live preview
without needing OpenCV locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _ensure_src_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_root = os.path.join(project_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an ADAS scenario")
    p.add_argument("--category-id", type=int, required=True)
    p.add_argument("--video-id", type=int, required=True)
    p.add_argument("--dataset-root", default="data/raw/DADA2000")
    p.add_argument("--index-path", default="data/processed/index.db")
    p.add_argument("--ui-backend", default="dpg", choices=["dpg", "cv2", "none"])
    p.add_argument("--target-fps", type=float, default=30.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--context-interval", type=int, default=5)
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--no-lanes", action="store_true")
    p.add_argument("--no-obstacles", action="store_true")
    p.add_argument("--no-risk", action="store_true")
    # Streaming / configurator mode
    p.add_argument("--stream-dir", default=None, help="Directory for latest.jpg + status.json (enables streaming mode)")
    p.add_argument("--loop", action="store_true", help="Loop video indefinitely (streaming mode)")
    p.add_argument("--log-every", type=int, default=0, help="Print a log line every N frames (0 = every frame)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------

def _run_streaming(args: argparse.Namespace) -> None:
    """Headless pipeline that writes annotated frames to disk."""
    _ensure_src_on_path()

    import cv2
    from adas.dataset import parser
    from adas.context.defaults import DEFAULT_CONFIG
    from adas.context.service import ContextService
    from adas.lane_detection import LaneProcessor
    from adas.obstacle_detection.detector import Detector
    from adas.obstacle_detection.tracking import SimpleTracker
    from adas.collision_risk.estimator import RiskEstimator
    from adas.collision_risk.decision import decide
    from adas.collision_risk.types import SystemAction
    from adas.ui.overlays import draw_lanes, draw_obstacles, draw_risk
    from adas.utils.runtime_overrides import load_runtime_overrides

    stream_dir = args.stream_dir
    os.makedirs(stream_dir, exist_ok=True)
    latest_path = os.path.join(stream_dir, "latest.jpg")
    status_path = os.path.join(stream_dir, "status.json")
    tmp_path = os.path.join(stream_dir, ".latest_tmp.jpg")

    # --- Load record --------------------------------------------------
    import sqlite3
    index_path = args.index_path
    if not os.path.exists(index_path):
        print(f"[ERROR] index not found: {index_path}", flush=True)
        return
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM records WHERE category_id = ? AND video_id = ? LIMIT 1",
        (args.category_id, args.video_id),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        print(f"[ERROR] record not found cat={args.category_id} vid={args.video_id}", flush=True)
        return
    record = dict(row)
    record_path = record["path"]

    # Prefer images/ sibling when a non-image folder is referenced
    norm = os.path.normpath(record_path)
    base = os.path.basename(norm).lower()
    if base in {"fixation", "maps", "seg", "semantic"}:
        candidate = os.path.join(os.path.dirname(norm), "images")
        if os.path.isdir(candidate):
            record_path = candidate

    # --- Build frame list ---------------------------------------------
    raw_iter = parser.iter_frames(record_path)
    if raw_iter is None:
        print(f"[ERROR] no frames at {record_path}", flush=True)
        return

    frame_items = []
    fidx = 0
    for item in raw_iter:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
            frame_items.append(item)
        else:
            frame_items.append((fidx, item))
        fidx += 1

    if not frame_items:
        print(f"[ERROR] empty frame list for {record_path}", flush=True)
        return

    n_frames = len(frame_items)
    print(f"[stream] {n_frames} frames loaded from {record_path}", flush=True)

    # --- Pre-cache frames into RAM ------------------------------------
    frame_cache = {}
    for ci, (_fi, _fr) in enumerate(frame_items):
        img = parser.get_frame(_fr)
        if img is not None:
            frame_cache[ci] = img
        if (ci + 1) % 50 == 0 or ci + 1 == n_frames:
            pct = 100.0 * (ci + 1) / n_frames
            print(f"\r[cache] {ci + 1}/{n_frames} ({pct:.0f}%)", end="", file=sys.stderr, flush=True)
    if n_frames > 0:
        print(file=sys.stderr)

    # --- Pipeline components ------------------------------------------
    ctx_service = ContextService(config=DEFAULT_CONFIG, context_interval=args.context_interval)
    ctx_service.start()
    lane_processor = LaneProcessor()
    detector = Detector()
    tracker = SimpleTracker()
    risk_estimator = RiskEstimator()

    target_spf = (1.0 / args.target_fps) if args.target_fps > 0 else 0.0
    log_every = max(1, args.log_every) if args.log_every > 0 else 1

    iteration = 0
    prev_overrides: dict = {}
    control_path = os.path.join(stream_dir, "control.json")
    paused = False
    need_rerender = False  # True when step/param change needs a fresh render while paused
    try:
        while True:
            cursor = 0
            while cursor < n_frames:
                t0 = time.monotonic()

                # -- Read control commands from GUI --
                ctrl_cmd, ctrl_data = _read_control(control_path)
                if ctrl_cmd == "pause":
                    paused = True
                elif ctrl_cmd == "play":
                    paused = False
                elif ctrl_cmd == "prev":
                    paused = True
                    cursor = max(0, cursor - 1)
                    need_rerender = True
                elif ctrl_cmd == "next":
                    paused = True
                    cursor = min(n_frames - 1, cursor + 1)
                    need_rerender = True
                elif ctrl_cmd == "seek":
                    paused = True
                    target = ctrl_data.get("target", 0)
                    if target == -1:
                        cursor = n_frames - 1
                    else:
                        cursor = max(0, min(n_frames - 1, int(target)))
                    need_rerender = True

                if paused:
                    time.sleep(0.05)
                    # Re-check control during pause
                    ctrl2, ctrl2_data = _read_control(control_path)
                    if ctrl2 == "play":
                        paused = False
                    elif ctrl2 == "prev":
                        cursor = max(0, cursor - 1)
                        need_rerender = True
                    elif ctrl2 == "next":
                        cursor = min(n_frames - 1, cursor + 1)
                        need_rerender = True
                    elif ctrl2 == "seek":
                        target = ctrl2_data.get("target", 0)
                        if target == -1:
                            cursor = n_frames - 1
                        else:
                            cursor = max(0, min(n_frames - 1, int(target)))
                        need_rerender = True
                    # Check if overrides changed
                    new_ov = load_runtime_overrides()
                    if new_ov != prev_overrides:
                        prev_overrides = _apply_overrides(new_ov, lane_processor, detector, risk_estimator, prev_overrides)
                        need_rerender = True
                    # Only re-render when frame/params changed (avoids destroying tracker/MOG2)
                    if need_rerender:
                        _render_frame(cursor, frame_items, frame_cache, args, lane_processor,
                                      detector, tracker, risk_estimator, ctx_service, decide,
                                      draw_lanes, draw_obstacles, draw_risk, cv2,
                                      tmp_path, latest_path, status_path, iteration, n_frames, log_every)
                        need_rerender = False
                    continue

                # -- Reload overrides each frame so GUI changes apply live --
                overrides = load_runtime_overrides()
                prev_overrides = _apply_overrides(overrides, lane_processor, detector, risk_estimator, prev_overrides)

                _render_frame(cursor, frame_items, frame_cache, args, lane_processor,
                              detector, tracker, risk_estimator, ctx_service, decide,
                              draw_lanes, draw_obstacles, draw_risk, cv2,
                              tmp_path, latest_path, status_path, iteration, n_frames, log_every)

                elapsed = time.monotonic() - t0
                if target_spf > 0 and elapsed < target_spf:
                    time.sleep(target_spf - elapsed)

                cursor += 1

            iteration += 1
            if not args.loop:
                break
            # Reset stateful components for next loop
            lane_processor.reset()
            tracker = SimpleTracker()
            risk_estimator = RiskEstimator()

    except KeyboardInterrupt:
        print("[stream] interrupted", flush=True)
    finally:
        ctx_service.stop()
        lane_processor.reset()
        print("[stream] done", flush=True)


def _read_control(control_path: str) -> tuple:
    """Read and consume a control command from the GUI. Returns (command, data_dict)."""
    try:
        if not os.path.exists(control_path):
            return ("", {})
        with open(control_path, "r") as fh:
            data = json.load(fh)
        os.remove(control_path)
        return (data.get("command", ""), data)
    except Exception:
        return ("", {})


def _render_frame(cursor, frame_items, frame_cache, args, lane_processor,
                  detector, tracker, risk_estimator, ctx_service, decide,
                  draw_lanes, draw_obstacles, draw_risk, cv2,
                  tmp_path, latest_path, status_path, iteration, n_frames, log_every):
    """Run pipeline on one frame and write output to stream files."""
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    frame_idx, _ref = frame_items[cursor]
    frame = frame_cache.get(cursor)
    if frame is None:
        return

    ts = frame_idx / max(1.0, args.target_fps)

    ctx_service.push_frame(frame, frame_idx, timestamp_s=ts, fps=args.target_fps)
    ctx_state = ctx_service.get_state()

    lane_output = lane_processor.update(frame, ctx_state)
    raw_det = detector.detect(frame, lane_output=lane_output, context_state=ctx_state)
    tracked = tracker.update(raw_det)
    risks = risk_estimator.estimate_risk(tracked, lane_output=lane_output, context_state=ctx_state, frame_idx=frame_idx)
    action, intensity = decide(risks, context_state=ctx_state)

    display = frame.copy()
    if not args.no_lanes:
        display = draw_lanes(display, lane_output)
    if not args.no_obstacles:
        display = draw_obstacles(display, tracked)
    if not args.no_risk:
        display = draw_risk(display, risks, ctx_state)

    cv2.imwrite(tmp_path, display)
    try:
        os.replace(tmp_path, latest_path)
    except OSError:
        os.makedirs(os.path.dirname(latest_path), exist_ok=True)
        try:
            import shutil
            shutil.copy2(tmp_path, latest_path)
        except OSError:
            pass

    status = {
        "frame_idx": frame_idx,
        "total_frames": n_frames,
        "iteration": iteration,
        "action": action.value if hasattr(action, "value") else str(action),
        "intensity": round(intensity, 3),
        "n_risks": len(risks),
        "ts": round(ts, 3),
    }
    with open(status_path, "w") as fh:
        json.dump(status, fh)

    action_str = status['action']
    if action_str not in ('none', 'None', ''):
        print(f"[stream] iter={iteration} frame={frame_idx}/{n_frames} action={action_str}", flush=True)


def _apply_overrides(overrides: dict, lane_processor, detector, risk_estimator,
                     prev_overrides: dict) -> dict:
    """Hot-apply runtime overrides to pipeline component *instances*.

    Returns the current overrides dict (to be passed as prev_overrides next call).
    Only logs when values actually change compared to prev_overrides.
    """
    import dataclasses

    changed_any = False

    lane_ov = overrides.get("lane")
    if isinstance(lane_ov, dict) and lane_ov:
        try:
            from adas.lane_detection.processing import LaneProcessingConfig
            cur = lane_processor._config
            new_kw = {k: type(getattr(cur, k))(v) for k, v in lane_ov.items() if hasattr(cur, k)}
            new_cfg = dataclasses.replace(cur, **new_kw)
            if new_cfg != cur:
                lane_processor.update_config(new_cfg)
                diff = {k: v for k, v in new_kw.items() if getattr(cur, k) != getattr(new_cfg, k)}
                if diff != (prev_overrides.get("_lane_diff") or {}):
                    for k, v in diff.items():
                        print(f"[override] lane.{k} = {v}  (was {getattr(cur, k)})", flush=True)
                    changed_any = True
        except Exception as exc:
            print(f"[override] lane apply error: {exc}", flush=True)

    obs_ov = overrides.get("obstacle")
    if isinstance(obs_ov, dict) and obs_ov:
        try:
            from adas.obstacle_detection.detector import DetectorConfig
            cur = detector._config
            new_kw = {k: type(getattr(cur, k))(v) for k, v in obs_ov.items() if hasattr(cur, k)}
            new_cfg = dataclasses.replace(cur, **new_kw)
            if new_cfg != cur:
                # Set config directly to avoid resetting MOG2 background model
                detector._config = new_cfg
                diff = {k: v for k, v in new_kw.items() if getattr(cur, k) != getattr(new_cfg, k)}
                if diff != (prev_overrides.get("_obs_diff") or {}):
                    for k, v in diff.items():
                        print(f"[override] obstacle.{k} = {v}  (was {getattr(cur, k)})", flush=True)
                    changed_any = True
        except Exception as exc:
            print(f"[override] obstacle apply error: {exc}", flush=True)

    est_ov = overrides.get("estimator")
    if isinstance(est_ov, dict) and est_ov:
        try:
            from adas.collision_risk.estimator import EstimatorConfig
            cur = risk_estimator._config
            new_kw = {k: type(getattr(cur, k))(v) for k, v in est_ov.items() if hasattr(cur, k)}
            new_cfg = dataclasses.replace(cur, **new_kw)
            if new_cfg != cur:
                risk_estimator._config = new_cfg
                diff = {k: v for k, v in new_kw.items() if getattr(cur, k) != getattr(new_cfg, k)}
                if diff != (prev_overrides.get("_est_diff") or {}):
                    for k, v in diff.items():
                        print(f"[override] estimator.{k} = {v}  (was {getattr(cur, k)})", flush=True)
                    changed_any = True
        except Exception as exc:
            print(f"[override] estimator apply error: {exc}", flush=True)

    dec_ov = overrides.get("decision")
    if isinstance(dec_ov, dict) and dec_ov:
        try:
            import adas.collision_risk.decision as _decmod
            from adas.collision_risk.decision import DecisionConfig
            cur = _decmod.DEFAULT_DECISION_CONFIG
            new_kw = {k: type(getattr(cur, k))(v) for k, v in dec_ov.items() if hasattr(cur, k)}
            new_cfg = dataclasses.replace(cur, **new_kw)
            if new_cfg != cur:
                _decmod.DEFAULT_DECISION_CONFIG = new_cfg
                diff = {k: v for k, v in new_kw.items() if getattr(cur, k) != getattr(new_cfg, k)}
                if diff != (prev_overrides.get("_dec_diff") or {}):
                    for k, v in diff.items():
                        print(f"[override] decision.{k} = {v}  (was {getattr(cur, k)})", flush=True)
                    changed_any = True
        except Exception as exc:
            print(f"[override] decision apply error: {exc}", flush=True)

    if changed_any:
        print("[override] parameters applied successfully", flush=True)

    return overrides


# ---------------------------------------------------------------------------
# Normal mode
# ---------------------------------------------------------------------------

def _run_normal(args: argparse.Namespace) -> None:
    _ensure_src_on_path()
    from adas.scenario.types import ScenarioConfig
    from adas.scenario.runner import run_scenario

    cfg = ScenarioConfig(
        category_id=args.category_id,
        video_id=args.video_id,
        dataset_root=args.dataset_root,
        index_path=args.index_path,
        ui_backend=args.ui_backend,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        context_interval=args.context_interval,
        enable_audio=not args.no_audio,
        show_dashboard=not args.no_dashboard,
        show_lanes=not args.no_lanes,
        show_obstacles=not args.no_obstacles,
        show_risk=not args.no_risk,
    )
    run_scenario(cfg)


def main() -> None:
    args = _parse_args()
    if args.stream_dir:
        _run_streaming(args)
    else:
        _run_normal(args)


if __name__ == "__main__":
    main()
