import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

from adas.dataset import loader_wrappers


def test_iter_frame_samples_with_transform_and_metadata(monkeypatch):
    monkeypatch.setattr(
        loader_wrappers.parser,
        "find_records",
        lambda _root: iter([("1/001", "image_seq", "/tmp/fake", {})]),
    )
    monkeypatch.setattr(
        loader_wrappers.parser,
        "record_metadata",
        lambda _path, dataset_root=None: {"category": "1", "n_frames": 2},
    )
    monkeypatch.setattr(
        loader_wrappers.parser,
        "iter_frames",
        lambda _path: iter([(0, "f0"), (1, "f1")]),
    )
    monkeypatch.setattr(
        loader_wrappers.parser,
        "get_frame",
        lambda frame_ref: None if frame_ref == "f1" else "img0",
    )

    samples = list(
        loader_wrappers.iter_frame_samples(
            "root",
            transform=lambda x: f"tx:{x}",
            with_metadata=True,
        )
    )

    assert len(samples) == 1
    s = samples[0]
    assert s["image"] == "tx:img0"
    assert s["record_id"] == "1/001"
    assert s["metadata"]["category"] == "1"


def test_iter_frame_samples_without_metadata(monkeypatch):
    monkeypatch.setattr(
        loader_wrappers.parser,
        "find_records",
        lambda _root: iter([("1/001", "image_seq", "/tmp/fake", {})]),
    )
    monkeypatch.setattr(loader_wrappers.parser, "record_metadata", lambda *_a, **_k: {})
    monkeypatch.setattr(loader_wrappers.parser, "iter_frames", lambda _path: iter([(0, "f0")]))
    monkeypatch.setattr(loader_wrappers.parser, "get_frame", lambda _ref: "img")

    sample = next(loader_wrappers.iter_frame_samples("root", with_metadata=False))
    assert "metadata" not in sample


def test_frame_iterable_delegates_iterator(monkeypatch):
    monkeypatch.setattr(
        loader_wrappers,
        "iter_frame_samples",
        lambda dataset_root, transform=None: iter([{"record_id": "x", "image": 1}]),
    )
    it = loader_wrappers.FrameIterable("root")
    got = list(iter(it))
    assert got == [{"record_id": "x", "image": 1}]


def test_torch_frame_dataset_behavior(monkeypatch):
    # Works in both environments:
    # - torch installed: dataset is usable
    # - torch missing: constructor raises RuntimeError by design
    monkeypatch.setattr(
        loader_wrappers.parser,
        "find_records",
        lambda _root: iter([("1/001", "image_seq", "/tmp/fake", {})]),
    )
    monkeypatch.setattr(
        loader_wrappers.parser,
        "record_metadata",
        lambda _path, dataset_root=None: {"category": "1", "n_frames": 1},
    )
    monkeypatch.setattr(loader_wrappers.parser, "iter_frames", lambda _path: iter([(0, "f0")]))
    monkeypatch.setattr(loader_wrappers.parser, "get_frame", lambda _ref: "img")

    try:
        ds = loader_wrappers.TorchFrameDataset("root", transform=lambda x: f"tx:{x}")
    except RuntimeError as e:
        assert "Torch is not installed" in str(e)
        return

    assert len(ds) == 1
    item = ds[0]
    assert item["image"] == "tx:img"


def test_torch_frame_dataset_getitem_raises_on_missing_frame(monkeypatch):
    monkeypatch.setattr(
        loader_wrappers.parser,
        "find_records",
        lambda _root: iter([("1/001", "image_seq", "/tmp/fake", {})]),
    )
    monkeypatch.setattr(loader_wrappers.parser, "record_metadata", lambda *_a, **_k: {})
    monkeypatch.setattr(loader_wrappers.parser, "iter_frames", lambda _path: iter([(0, "f0")]))
    monkeypatch.setattr(loader_wrappers.parser, "get_frame", lambda _ref: None)

    try:
        ds = loader_wrappers.TorchFrameDataset("root")
    except RuntimeError as e:
        assert "Torch is not installed" in str(e)
        return

    with pytest.raises(RuntimeError):
        _ = ds[0]
