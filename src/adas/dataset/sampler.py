"""Sampling utilities for dataset records and frame indices."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def random_sample(items: Sequence[T], n: int, seed: int | None = None) -> List[T]:
    """Return up to `n` random unique items from `items`.

    If `n` is larger than the number of items, all items are returned in random order.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if len(items) == 0 or n == 0:
        return []

    rng = random.Random(seed)
    if n >= len(items):
        out = list(items)
        rng.shuffle(out)
        return out
    return rng.sample(list(items), n)


def stratified_sample(
    items: Sequence[T],
    by: str | Callable[[T], str],
    n_per_group: int | None = None,
    seed: int | None = None,
) -> Dict[str, List[T]]:
    """Stratified sampling grouped by category/key.

    Args:
        items: Sequence of items.
        by: Either key name for dict-like items or a callable returning group key.
        n_per_group: Optional maximum number of samples per group. If None, returns all.
        seed: Optional RNG seed.

    Returns:
        Dict[group_key, sampled_items].
    """
    groups: Dict[str, List[T]] = defaultdict(list)

    if callable(by):
        key_fn = by
    else:
        def key_fn(x: T) -> str:
            if isinstance(x, dict):
                return str(x.get(by, "unknown"))
            return str(getattr(x, by, "unknown"))

    for item in items:
        groups[key_fn(item)].append(item)

    rng = random.Random(seed)
    out: Dict[str, List[T]] = {}
    for key, group_items in groups.items():
        if n_per_group is None or n_per_group >= len(group_items):
            sampled = list(group_items)
            rng.shuffle(sampled)
            out[key] = sampled
        else:
            out[key] = rng.sample(group_items, n_per_group)
    return out


def sequence_sampler(
    total_frames: int,
    length: int,
    stride: int = 1,
    step: int | None = None,
) -> List[List[int]]:
    """Generate frame-index sequences (windows).

    Example:
        total_frames=10, length=4, stride=2 ->
        [[0,2,4,6], [1,3,5,7], [2,4,6,8], [3,5,7,9]]

    Args:
        total_frames: Number of available frames.
        length: Number of frames in each sequence.
        stride: Distance between consecutive indices inside a sequence.
        step: Shift between sequence starts. If None, defaults to 1.
    """
    if total_frames < 0:
        raise ValueError("total_frames must be >= 0")
    if length <= 0:
        raise ValueError("length must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    step = 1 if step is None else step
    if step <= 0:
        raise ValueError("step must be > 0")

    if total_frames == 0:
        return []

    last_start = total_frames - 1 - (length - 1) * stride
    if last_start < 0:
        return []

    sequences: List[List[int]] = []
    for start in range(0, last_start + 1, step):
        seq = [start + i * stride for i in range(length)]
        sequences.append(seq)
    return sequences
