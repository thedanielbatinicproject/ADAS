#!/usr/bin/env python3
"""Build local SQLite index for dataset metadata."""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from adas.dataset import indexer  # noqa: E402


def _is_index_fresh_with_progress(
    dataset_root: str,
    index_path: str,
    progress_sec: int,
) -> bool:
    if not os.path.exists(index_path):
        print("[build_index] Freshness check: index does not exist.", flush=True)
        return False

    index_mtime = os.path.getmtime(index_path)
    checked = 0
    last = time.time()

    for root, _dirs, files in os.walk(dataset_root):
        for name in files:
            checked += 1
            fpath = os.path.join(root, name)
            try:
                if os.path.getmtime(fpath) >= index_mtime:
                    print(
                        f"[build_index] Freshness check: newer file found ({fpath}).",
                        flush=True,
                    )
                    return False
            except FileNotFoundError:
                # File may disappear during scan; treat as not fresh to be safe.
                return False

            now = time.time()
            if progress_sec > 0 and now - last >= progress_sec:
                print(
                    f"[build_index] Freshness check progress: files_checked={checked}, current_dir={root}",
                    flush=True,
                )
                last = now

    print(
        f"[build_index] Freshness check complete: files_checked={checked}, index is fresh.",
        flush=True,
    )
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="Build dataset index")
    p.add_argument("--dataset-root", required=True, help="Dataset root directory")
    p.add_argument(
        "--index-path",
        default=indexer.DEFAULT_INDEX_PATH,
        help="Target SQLite index path",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing index",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index appears fresh",
    )
    p.add_argument(
        "--check-freshness",
        action="store_true",
        help="Run freshness check before build (can be slow on very large datasets)",
    )
    p.add_argument(
        "--annotations-csv",
        default="",
        help="Optional path to DADA2000_video_annotations.csv",
    )
    p.add_argument(
        "--progress-sec",
        type=int,
        default=10,
        help="Progress print interval in seconds",
    )
    args = p.parse_args()

    print(f"[build_index] dataset_root={args.dataset_root}", flush=True)
    print(f"[build_index] index_path={args.index_path}", flush=True)

    annotations_csv = args.annotations_csv
    if not annotations_csv:
        candidate = os.path.join(
            os.path.dirname(os.path.abspath(args.dataset_root)),
            "DADA2000_video_annotations.csv",
        )
        if os.path.exists(candidate):
            annotations_csv = candidate
            print(f"Auto-detected annotations CSV: {annotations_csv}", flush=True)

    if args.force:
        print("[build_index] --force enabled: skipping freshness check.", flush=True)
    elif args.check_freshness:
        print("[build_index] Running freshness check...", flush=True)
        if _is_index_fresh_with_progress(
            args.dataset_root,
            args.index_path,
            args.progress_sec,
        ):
            print("Index is fresh. Skipping rebuild.", flush=True)
            return 0
    else:
        print(
            "[build_index] Freshness check skipped (default) for speed on large datasets.",
            flush=True,
        )

    print("[build_index] Starting index build...", flush=True)

    indexer.build_index(
        args.dataset_root,
        args.index_path,
        overwrite=args.overwrite,
        annotations_csv_path=annotations_csv or None,
        progress_interval_sec=args.progress_sec,
    )
    records = indexer.list_records(args.index_path)
    print(f"Index built: {args.index_path}", flush=True)
    print(f"Records indexed: {len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
