import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

from adas.dataset import annotation


def test_parse_annotation_file_json_csv_txt_unknown(tmp_path):
    j = tmp_path / "a.json"
    j.write_text(json.dumps({"label": "car"}), encoding="utf-8")
    out_j = annotation.parse_annotation_file(str(j))
    assert out_j["format"] == "json"
    assert out_j["data"]["label"] == "car"

    c = tmp_path / "a.csv"
    c.write_text("label;id\ncar;1\n", encoding="utf-8")
    out_c = annotation.parse_annotation_file(str(c))
    assert out_c["format"] == "csv"
    assert out_c["data"][0]["label"] == "car"

    t = tmp_path / "a.txt"
    t.write_text("line1\nline2\n", encoding="utf-8")
    out_t = annotation.parse_annotation_file(str(t))
    assert out_t["format"] == "txt"
    assert out_t["data"] == ["line1", "line2"]

    u = tmp_path / "a.bin"
    u.write_text("raw", encoding="utf-8")
    out_u = annotation.parse_annotation_file(str(u))
    assert out_u["format"] == "unknown"
    assert out_u["data"] == "raw"


def test_get_annotation_explicit_and_heuristic(monkeypatch, tmp_path):
    j = tmp_path / "ann.json"
    j.write_text(json.dumps({"category": "pedestrian"}), encoding="utf-8")

    out_explicit = annotation.get_annotation("ignored", explicit_path=str(j))
    assert out_explicit is not None
    assert out_explicit["format"] == "json"

    monkeypatch.setattr(annotation.parser, "find_annotation_for_record", lambda _: str(j))
    out_heur = annotation.get_annotation("record-1")
    assert out_heur is not None
    assert out_heur["data"]["category"] == "pedestrian"

    monkeypatch.setattr(annotation.parser, "find_annotation_for_record", lambda _: None)
    assert annotation.get_annotation("record-2") is None


def test_extract_labels_dict_and_list_and_other():
    d = {"data": {"label": "car", "category": "vehicle"}}
    assert annotation.extract_labels(d) == ["car", "vehicle"]

    l = {"data": [{"label": "a"}, {"class": "b"}, {"x": 1}]}
    assert annotation.extract_labels(l) == ["a", "b"]

    assert annotation.extract_labels({"data": "not-structured"}) == []
