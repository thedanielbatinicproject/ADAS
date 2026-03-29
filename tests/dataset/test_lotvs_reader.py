import os
import tempfile
import shutil
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from adas.dataset import lotvs_reader

def make_fake_csv(path):
    content = (
        "video;weather;light;scenes;linear;type;accident_flag;abnormal_start_frame;accident_frame;abnormal_end_frame;total_frames;int1;int2;int3;int4;int5;texts;causes;measures\n"
        "001;1;1;4;1;1;1;30;63;115;440;30;33;85;52;325;desc;cause;measure\n"
        "002;2;2;5;2;2;0;110;-1;200;344;110;58;90;32;144;desc2;cause2;measure2\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def test_load_annotations_and_ids():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "DADA2000_video_annotations.csv")
        make_fake_csv(csv_path)
        annotations = lotvs_reader.load_annotations(csv_path)
        assert isinstance(annotations, dict)
        assert "001" in annotations and "002" in annotations
        ids = lotvs_reader.get_all_video_ids(annotations)
        assert set(ids) == {"001", "002"}

def test_get_annotation_for_video():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "DADA2000_video_annotations.csv")
        make_fake_csv(csv_path)
        annotations = lotvs_reader.load_annotations(csv_path)
        ann = lotvs_reader.get_annotation_for_video(annotations, "001")
        assert ann["video_id"] == "001"
        assert ann["weather"] == "1"
        assert ann["accident_flag"] == 1

def test_get_annotation_for_frame():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "DADA2000_video_annotations.csv")
        make_fake_csv(csv_path)
        annotations = lotvs_reader.load_annotations(csv_path)
        # accident frame
        ann = lotvs_reader.get_annotation_for_frame(annotations, "001", 63)
        assert ann["label"] == "accident_frame"
        # abnormal
        ann = lotvs_reader.get_annotation_for_frame(annotations, "001", 50)
        assert ann["label"] == "abnormal"
        # normal
        ann = lotvs_reader.get_annotation_for_frame(annotations, "001", 10)
        assert ann["label"] == "normal"

def test_get_video_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create folder structure: tmpdir/1/001
        d1 = os.path.join(tmpdir, "1", "001")
        os.makedirs(d1)
        path = lotvs_reader.get_video_path(tmpdir, "001")
        assert path is not None
        assert os.path.basename(path) == "001"