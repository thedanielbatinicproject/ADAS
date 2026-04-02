import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

from adas.dataset import sampler


def test_random_sample_basic_and_seed_reproducible():
    items = [1, 2, 3, 4, 5]
    s1 = sampler.random_sample(items, n=3, seed=42)
    s2 = sampler.random_sample(items, n=3, seed=42)
    assert s1 == s2
    assert len(s1) == 3
    assert len(set(s1)) == 3


def test_random_sample_n_larger_than_items():
    items = [1, 2]
    out = sampler.random_sample(items, n=10, seed=1)
    assert sorted(out) == [1, 2]


def test_random_sample_empty_and_zero():
    assert sampler.random_sample([], n=5) == []
    assert sampler.random_sample([1, 2], n=0) == []


def test_random_sample_negative_n_raises():
    with pytest.raises(ValueError):
        sampler.random_sample([1], n=-1)


def test_stratified_sample_by_dict_key_with_limit():
    items = [
        {"category": "a", "id": 1},
        {"category": "a", "id": 2},
        {"category": "b", "id": 3},
    ]
    out = sampler.stratified_sample(items, by="category", n_per_group=1, seed=7)
    assert set(out.keys()) == {"a", "b"}
    assert len(out["a"]) == 1
    assert len(out["b"]) == 1


def test_stratified_sample_by_callable_all_items():
    items = ["x1", "x2", "y1"]
    out = sampler.stratified_sample(items, by=lambda x: x[0], n_per_group=None, seed=3)
    assert set(out.keys()) == {"x", "y"}
    assert sorted(out["x"]) == ["x1", "x2"]
    assert out["y"] == ["y1"]


def test_sequence_sampler_basic():
    out = sampler.sequence_sampler(total_frames=10, length=4, stride=2)
    assert out == [[0, 2, 4, 6], [1, 3, 5, 7], [2, 4, 6, 8], [3, 5, 7, 9]]


def test_sequence_sampler_with_step():
    out = sampler.sequence_sampler(total_frames=10, length=3, stride=1, step=2)
    assert out == [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]]


def test_sequence_sampler_invalid_arguments_and_edge_cases():
    with pytest.raises(ValueError):
        sampler.sequence_sampler(total_frames=-1, length=2)
    with pytest.raises(ValueError):
        sampler.sequence_sampler(total_frames=10, length=0)
    with pytest.raises(ValueError):
        sampler.sequence_sampler(total_frames=10, length=2, stride=0)
    with pytest.raises(ValueError):
        sampler.sequence_sampler(total_frames=10, length=2, step=0)

    assert sampler.sequence_sampler(total_frames=0, length=2) == []
    assert sampler.sequence_sampler(total_frames=3, length=5, stride=1) == []
