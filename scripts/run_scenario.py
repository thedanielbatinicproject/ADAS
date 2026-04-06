#!/usr/bin/env python3
"""Main ADAS scenario runner script.

Builds ScenarioConfig from CLI args and runs the full pipeline:
  dataset frames -> lane_detection -> obstacle_detection -> collision_risk
  -> context -> UI (cv2) -> audio

Usage
-----
python scripts/run_scenario.py --category-id 1 --video-id 1
python scripts/run_scenario.py --category-id 1 --video-id 1 --no-audio --no-dashboard
python scripts/run_scenario.py --category-id 1 --video-id 1 --ui-backend none --max-frames 100
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.scenario import run_scenario, ScenarioConfig  # noqa: E402


def _check_pulseaudio() -> bool:
    """Return True if PulseAudio TCP server is reachable on the host."""
    host = os.environ.get("PULSE_SERVER", "tcp:host.docker.internal:4713")
    # Parse host:port from PULSE_SERVER (format: tcp:host:port)
    parts = host.replace("tcp:", "").rsplit(":", 1)
    pa_host = parts[0] if parts else "host.docker.internal"
    pa_port = int(parts[1]) if len(parts) > 1 else 4713
    try:
        with socket.create_connection((pa_host, pa_port), timeout=1):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the ADAS scenario pipeline on one DADA-2000 video.",
    )
    p.add_argument(
        "--category-id", type=int, default=1,
        help="Category ID (default: 1).",
    )
    p.add_argument(
        "--video-id", type=int, default=1,
        help="Video ID within the category (default: 1).",
    )
    p.add_argument(
        "--dataset-root", default="data/raw/DADA2000",
        help="Path to the DADA-2000 root directory (default: data/raw/DADA2000).",
    )
    p.add_argument(
        "--index-path", default="data/processed/index.db",
        help="Path to index.db (default: data/processed/index.db).",
    )
    p.add_argument(
        "--ui-backend", choices=["dpg", "cv2", "none"], default="dpg",
        help="UI backend (default: dpg). Use 'cv2' for legacy OpenCV or 'none' for headless.",
    )
    p.add_argument(
        "--target-fps", type=float, default=30.0,
        help="Target playback FPS (default: 30). 0 = unlimited.",
    )
    p.add_argument(
        "--context-interval", type=int, default=5,
        help="Frames between full context route() calls (default: 5).",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many frames. Default: whole video.",
    )
    p.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio feedback.",
    )
    p.add_argument(
        "--audio", action="store_true", default=False,
        help="Explicitly enable audio (default). Checks PulseAudio connectivity on startup.",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable the stats dashboard panel.",
    )
    p.add_argument(
        "--no-lanes", action="store_true",
        help="Disable lane overlay.",
    )
    p.add_argument(
        "--no-obstacles", action="store_true",
        help="Disable obstacle overlay.",
    )
    p.add_argument(
        "--no-risk", action="store_true",
        help="Disable risk overlay.",
    )
    p.add_argument(
        "--log-file", default=None,
        help="Path to a JSONL file for event logging.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Audio: --audio is default-on; --no-audio disables it
    enable_audio = not args.no_audio

    if enable_audio:
        if _check_pulseaudio():
            print("[run_scenario] PulseAudio server detected — audio enabled.")
        else:
            print(
                "[run_scenario] WARNING: PulseAudio server not reachable on host.\n"
                "  Audio will fall back to terminal bell (may be silent).\n"
                "  To enable audio, run on Windows host:  scripts\\start_pulseaudio.bat"
            )

    # Resolve paths relative to project root
    dataset_root = args.dataset_root
    index_path = args.index_path
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(PROJECT_ROOT, dataset_root)
    if not os.path.isabs(index_path):
        index_path = os.path.join(PROJECT_ROOT, index_path)

    config = ScenarioConfig(
        dataset_root=dataset_root,
        index_path=index_path,
        category_id=args.category_id,
        video_id=args.video_id,
        target_fps=args.target_fps,
        context_interval=args.context_interval,
        ui_backend=args.ui_backend,
        enable_audio=enable_audio,
        max_frames=args.max_frames,
        show_dashboard=not args.no_dashboard,
        show_lanes=not args.no_lanes,
        show_obstacles=not args.no_obstacles,
        show_risk=not args.no_risk,
    )

    print(
        f"[run_scenario] Starting: cat={config.category_id} vid={config.video_id} "
        f"ui={config.ui_backend} fps={config.target_fps}"
    )

    run_scenario(config, log_file=args.log_file)


if __name__ == "__main__":
    main()
