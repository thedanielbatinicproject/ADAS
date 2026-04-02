import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

from adas.dataset import utils_io


def test_normalize_path_and_ensure_dir(tmp_path):
    p = utils_io.normalize_path("a//b\\c/../d")
    assert "/" in p

    out_dir = tmp_path / "x" / "y"
    ensured = utils_io.ensure_dir(str(out_dir))
    assert os.path.isdir(out_dir)
    assert ensured.endswith("/x/y")


def test_safe_imread_missing_path_returns_none():
    assert utils_io.safe_imread("/no/such/file.jpg") is None


def test_safe_imread_with_mocked_cv2(monkeypatch, tmp_path):
    img_path = tmp_path / "img.jpg"
    img_path.write_text("dummy")

    fake_cv2 = SimpleNamespace(IMREAD_COLOR=1, imread=lambda path, flag: {"path": path, "flag": flag})
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    out = utils_io.safe_imread(str(img_path))
    assert out == {"path": str(img_path), "flag": 1}


def test_file_checksum_and_verify(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")

    expected = hashlib.sha256(b"hello").hexdigest()
    got = utils_io.file_checksum(str(f))
    assert got == expected
    assert utils_io.verify_checksum(str(f), expected)
    assert not utils_io.verify_checksum(str(f), "deadbeef")


def test_export_jsonl_sharded(tmp_path):
    rows = [{"id": i} for i in range(5)]
    paths = utils_io.export_jsonl_sharded(rows, str(tmp_path / "out"), shard_size=2, prefix="part")

    assert len(paths) == 3
    for p in paths:
        assert os.path.exists(p)

    with open(paths[0], "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert lines == [{"id": 0}, {"id": 1}]


def test_export_jsonl_sharded_invalid_shard_size(tmp_path):
    with pytest.raises(ValueError):
        utils_io.export_jsonl_sharded([], str(tmp_path), shard_size=0)
