#!/usr/bin/env python3
"""Run parser or DADA reader and print a quick preview."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.dataset import lotvs_reader, parser  # noqa: E402


def run_generic_parser(dataset_root: str, limit: int) -> int:
    count = 0
    for record_id, record_type, record_path, _ in parser.find_records(dataset_root):
        meta = parser.record_metadata(record_path, dataset_root=dataset_root)
        print(
            json.dumps(
                {
                    "record_id": record_id,
                    "record_type": record_type,
                    "path": record_path,
                    "n_frames": meta.get("n_frames"),
                    "category": meta.get("category"),
                },
                ensure_ascii=False,
            )
        )
        count += 1
        if count >= limit:
            break

    print(f"Total previewed records: {count}")
    return 0


def run_lotvs(dataset_root: str, csv_path: str, limit: int) -> int:
    annotations = lotvs_reader.load_annotations(csv_path)
    video_ids = lotvs_reader.get_all_video_ids(annotations)[:limit]

    for video_id in video_ids:
        ann = lotvs_reader.get_annotation_for_video(annotations, video_id)
        path = lotvs_reader.get_video_path(dataset_root, video_id)
        print(
            json.dumps(
                {
                    "video_id": video_id,
                    "path": path,
                    "accident_frame": ann.get("accident_frame") if ann else None,
                    "total_frames": ann.get("total_frames") if ann else None,
                },
                ensure_ascii=False,
            )
        )

    print(f"Total previewed videos: {len(video_ids)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Run dataset parser preview")
    p.add_argument("--dataset-root", required=True, help="Dataset root folder")
    p.add_argument(
        "--backend",
        choices=["parser", "lotvs_reader"],
        default="parser",
        help="Reader backend",
    )
    p.add_argument(
        "--csv-path",
        default="",
        help="Path to DADA2000 annotations CSV (required for lotvs_reader)",
    )
    p.add_argument("--limit", type=int, default=10, help="Preview limit")
    args = p.parse_args()

    if args.backend == "parser":
        return run_generic_parser(args.dataset_root, args.limit)

    if not args.csv_path:
        raise SystemExit("--csv-path is required when --backend lotvs_reader")
    return run_lotvs(args.dataset_root, args.csv_path, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
