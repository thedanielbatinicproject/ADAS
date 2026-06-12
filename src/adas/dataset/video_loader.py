"""Video file loader: extracts frames from popular video container formats.

Supports: mp4, mov, mkv, avi, wmv, webm, flv, m4v, ts, mts, 3gp, ogv

Extracted frames are written as JPEG images to a caller-specified directory so
they can be used with the standard parser.iter_frames() / parser.get_frame()
pipeline unchanged.

Typical usage
-------------
    from adas.dataset.video_loader import extract_video_to_frames

    n = extract_video_to_frames(
        video_path="/path/to/clip.mp4",
        output_dir="/app/data/temp/my_uuid",
        progress_callback=lambda done, total: print(f"{done}/{total}"),
    )
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Set

# Video container extensions we accept
SUPPORTED_EXTENSIONS: Set[str] = {
    ".mp4", ".mov", ".mkv", ".avi", ".wmv",
    ".webm", ".flv", ".m4v", ".ts", ".mts",
    ".3gp", ".ogv", ".mpg", ".mpeg",
}


def is_supported_video(path: str) -> bool:
    """Return True if *path* has a supported video file extension."""
    _, ext = os.path.splitext(path)
    return ext.lower() in SUPPORTED_EXTENSIONS


def extract_video_to_frames(
    video_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    jpeg_quality: int = 90,
) -> int:
    """Extract all frames from *video_path* into JPEG images in *output_dir*.

    Parameters
    ----------
    video_path : str
        Absolute path to the input video file.
    output_dir : str
        Directory to write frame images into.  Created if it does not exist.
    progress_callback : callable(done: int, total: int), optional
        Called after each frame is written.  *total* may be 0 when the video
        container does not report its frame count up-front.
    jpeg_quality : int
        JPEG compression quality (1-100).  Default 90.

    Returns
    -------
    int
        Number of frames extracted.

    Raises
    ------
    ImportError
        If OpenCV (cv2) is not available.
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If the video file cannot be opened by OpenCV.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV (cv2) is required for video frame extraction."
        ) from exc

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    # Try to get total frame count for progress reporting (may be 0 / -1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 0:
        total = 0

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            filename = os.path.join(output_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame, encode_params)
            frame_idx += 1
            if progress_callback is not None:
                progress_callback(frame_idx, total)
    finally:
        cap.release()

    return frame_idx
