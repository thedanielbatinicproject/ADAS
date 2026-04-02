#!/usr/bin/env python3
"""Export random or top-k sample frames from dataset records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.dataset import parser, sampler  # noqa: E402
from adas.dataset.utils_io import ensure_dir, normalize_path  # noqa: E402


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _export_image(src_path: str, dst_path: str) -> None:
    shutil.copy2(src_path, dst_path)


def _export_video_frame(frame_ref: str, dst_path: str) -> bool:
    img = parser.get_frame(frame_ref)
    if img is None:
        return False
    try:
        import cv2
    except Exception:
        return False
    return bool(cv2.imwrite(dst_path, img))


def _pick_records(dataset_root: str, mode: str, n: int, seed: int | None) -> List[Dict]:
    records: List[Dict] = []
    for record_id, record_type, record_path, _ in parser.find_records(dataset_root):
        records.append(
            {
                "record_id": record_id,
                "record_type": record_type,
                "record_path": record_path,
            }
        )

    if mode == "topk":
        return records[:n]
    return sampler.random_sample(records, n=n, seed=seed)


def main() -> int:
    p = argparse.ArgumentParser(description="Export sample frames from dataset")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=10, help="Number of records to sample")
    p.add_argument(
        "--mode",
        choices=["random", "topk"],
        default="random",
        help="Sampling mode",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--frames-per-record",
        type=int,
        default=3,
        help="How many frames to export per selected record",
    )
    args = p.parse_args()

    out_dir = ensure_dir(args.out_dir)
    selected = _pick_records(args.dataset_root, args.mode, args.n, args.seed)

    manifest: List[Dict] = []
    export_count = 0

    for record in selected:
        record_id = record["record_id"]
        record_path = record["record_path"]

        frame_iter = parser.iter_frames(record_path)
        local_count = 0
        for frame_idx, frame_ref in frame_iter:
            if local_count >= args.frames_per_record:
                break

            dst_name = f"{_safe_name(record_id)}__{frame_idx:06d}.jpg"
            dst_path = os.path.join(out_dir, dst_name)

            ok = False
            if isinstance(frame_ref, str) and "::frame::" in frame_ref:
                ok = _export_video_frame(frame_ref, dst_path)
            else:
                _export_image(frame_ref, dst_path)
                ok = True

            if ok:
                export_count += 1
                local_count += 1
                manifest.append(
                    {
                        "record_id": record_id,
                        "frame_idx": frame_idx,
                        "frame_ref": frame_ref,
                        "output_path": normalize_path(dst_path),
                    }
                )

    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    print(f"Selected records: {len(selected)}")
    print(f"Exported frames: {export_count}")
    print(f"Manifest: {normalize_path(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
