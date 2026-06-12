#!/usr/bin/env python
"""CLI wrapper around adas.dataset.video_loader.extract_video_to_frames.

Usage
-----
    python scripts/dataset/extract_video.py \
        --video-path data/temp/<uuid>/input.mp4 \
        --output-dir data/temp/<uuid>/frames

Paths are relative to the project root (/app inside the container).
Progress is printed as: PROGRESS:<done>/<total>
"""

from __future__ import annotations

import argparse
import os
import sys


def _ensure_src_on_path() -> None:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_root = os.path.join(project_root, "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract video frames to JPEG images")
    p.add_argument("--video-path", required=True, help="Path to source video file")
    p.add_argument("--output-dir", required=True, help="Directory to write frames into")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality (1-100)")
    args = p.parse_args()

    _ensure_src_on_path()

    from adas.dataset.video_loader import extract_video_to_frames

    def _progress(done: int, total: int) -> None:
        print(f"PROGRESS:{done}/{total}", flush=True)

    try:
        n = extract_video_to_frames(
            video_path=args.video_path,
            output_dir=args.output_dir,
            progress_callback=_progress,
            jpeg_quality=args.quality,
        )
        print(f"DONE:{n}", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"ERROR:{exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
