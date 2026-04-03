#!/usr/bin/env python3
"""Sample N random (category_id, video_id) pairs per annotation condition.

Conditions covered
------------------
  Weather : sunny, rainy, snowy, foggy
  Light   : day, night
  Lane    : lanes present (linear=arterials/curve), no clear lanes (intersection/T-jcn/ramp)

Only records that have actual image files on disk are included.

Output is a plain table suitable for copy-pasting into
``analyze_video.py --category-id ... --video-id ...`` calls.

Usage
-----
    python scripts/dataset/sample_conditions.py
    python scripts/dataset/sample_conditions.py --n 5 --seed 7
    python scripts/dataset/sample_conditions.py --index data/processed/index.db \\
        --dataset-root data/raw/DADA2000
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DEFAULT_INDEX = os.path.join(PROJECT_ROOT, "data", "processed", "index.db")
DEFAULT_DATASET_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "DADA2000")

# ---------------------------------------------------------------------------
# Annotation codes
# ---------------------------------------------------------------------------
WEATHER_CODES = {1: "sunny", 2: "rainy", 3: "snowy", 4: "foggy"}
LIGHT_CODES = {1: "day", 2: "night"}

# linear(arterials=1, curve=2, intersection=3, T-junction=4, ramp=5)
# 1,2 → lane markings clearly present on straight/curved roads
# 3,4,5 → intersection / T-junction / ramp — lanes unclear or absent
LANE_PRESENT_VALUES = {1, 2}
LANE_ABSENT_VALUES = {3, 4, 5}

LANE_CODES = {
    "lanes_present": LANE_PRESENT_VALUES,
    "no_clear_lanes": LANE_ABSENT_VALUES,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_path(dataset_root: str, category_id: int, video_id: int) -> Optional[str]:
    """Return the images directory path for a record if it exists on disk."""
    base = os.path.join(dataset_root, str(category_id), f"{video_id:03d}")
    images = os.path.join(base, "images")
    if os.path.isdir(images):
        return images
    if os.path.isdir(base):
        return base
    return None


def _query(conn: sqlite3.Connection, where_sql: str, params: tuple = ()) -> List[dict]:
    c = conn.cursor()
    c.execute(
        f"SELECT category_id, video_id, weather, light, linear, scenes "  # noqa: S608
        f"FROM annotations WHERE {where_sql}",
        params,
    )
    cols = [d[0] for d in c.description]
    return [dict(zip(cols, row)) for row in c.fetchall()]


def _sample(
    rows: List[dict],
    n: int,
    dataset_root: str,
    rng: random.Random,
) -> List[dict]:
    """Keep only on-disk records, shuffle, return first n."""
    on_disk = []
    for r in rows:
        path = _record_path(dataset_root, r["category_id"], r["video_id"])
        if path is not None:
            r["path"] = path
            on_disk.append(r)
    rng.shuffle(on_disk)
    return on_disk[:n]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

def build_groups(
    conn: sqlite3.Connection,
    dataset_root: str,
    n: int,
    rng: random.Random,
) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}

    # --- Weather ---
    for code, label in WEATHER_CODES.items():
        rows = _query(conn, "weather = ?", (code,))
        groups[f"weather_{label}"] = _sample(rows, n, dataset_root, rng)

    # --- Light ---
    for code, label in LIGHT_CODES.items():
        rows = _query(conn, "light = ?", (code,))
        groups[f"light_{label}"] = _sample(rows, n, dataset_root, rng)

    # --- Lanes (linear column) ---
    for label, values in LANE_CODES.items():
        placeholders = ",".join("?" * len(values))
        rows = _query(conn, f"linear IN ({placeholders})", tuple(values))
        groups[f"lane_{label}"] = _sample(rows, n, dataset_root, rng)

    return groups


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

_SCENE_NAMES = {1: "highway", 2: "tunnel", 3: "mountain", 4: "urban", 5: "rural"}
_LINEAR_NAMES = {1: "arterials", 2: "curve", 3: "intersection", 4: "T-junction", 5: "ramp"}


def _fmt_row(r: dict) -> str:
    weather = WEATHER_CODES.get(r.get("weather", 0), "?")
    light = LIGHT_CODES.get(r.get("light", 0), "?")
    scene = _SCENE_NAMES.get(r.get("scenes", 0), "?")
    road = _LINEAR_NAMES.get(r.get("linear", 0), "?")
    return (
        f"  cat={r['category_id']:>3d}  vid={r['video_id']:>3d}"
        f"  [{weather:<6s} {light:<5s} {scene:<9s} {road}]"
        f"  {r['path']}"
    )


def print_groups(groups: Dict[str, List[dict]], n: int) -> None:
    group_order = [
        ("weather_sunny",       "Weather: sunny"),
        ("weather_rainy",       "Weather: rainy"),
        ("weather_snowy",       "Weather: snowy"),
        ("weather_foggy",       "Weather: foggy"),
        ("light_day",           "Light:   day"),
        ("light_night",         "Light:   night"),
        ("lane_lanes_present",  "Lanes:   present  (arterials / curve)"),
        ("lane_no_clear_lanes", "Lanes:   absent   (intersection / T-junction / ramp)"),
    ]

    for key, title in group_order:
        rows = groups.get(key, [])
        print(f"\n{'─' * 72}")
        print(f"  {title}  ({len(rows)}/{n} on disk)")
        print(f"{'─' * 72}")
        if rows:
            for r in rows:
                print(_fmt_row(r))
        else:
            print("  (none found)")

    # Convenience: print analyze_video.py commands for manual inspection
    print(f"\n{'═' * 72}")
    print("  Quick-run commands (analyze_video.py):")
    print(f"{'═' * 72}")
    shown: set = set()
    for key, _ in group_order:
        rows = groups.get(key, [])
        for r in rows[:2]:               # first 2 of each group
            pair = (r["category_id"], r["video_id"])
            if pair in shown:
                continue
            shown.add(pair)
            print(
                f"  python scripts/context/analyze_video.py"
                f" --index-path data/processed/index.db"
                f" --category-id {r['category_id']}"
                f" --video-id {r['video_id']} --gui"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample N random records per annotation condition from DADA-2000 index.",
    )
    parser.add_argument(
        "--n", type=int, default=10,
        help="Number of samples per condition (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--index-path", default=DEFAULT_INDEX,
        help=f"Path to index.db (default: {DEFAULT_INDEX}).",
    )
    parser.add_argument(
        "--dataset-root", default=DEFAULT_DATASET_ROOT,
        help=f"Path to DADA2000 root directory (default: {DEFAULT_DATASET_ROOT}).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.index_path):
        parser.error(f"Index not found: {args.index_path}")
    if not os.path.isdir(args.dataset_root):
        parser.error(f"Dataset root not found: {args.dataset_root}")

    conn = sqlite3.connect(args.index_path)
    rng = random.Random(args.seed)

    print(f"Sampling up to {args.n} records per condition  (seed={args.seed})")
    groups = build_groups(conn, args.dataset_root, args.n, rng)
    print_groups(groups, args.n)
    conn.close()


if __name__ == "__main__":
    main()
