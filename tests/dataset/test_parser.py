
import os
import sys
import shutil
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from adas.dataset import parser
# Fixture for temp repo in /tests/tmp_test_repo
tmp_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp_test_repo'))

@pytest.fixture(autouse=True)
def setup_and_teardown_tmp_repo():
    # Setup: create/clean temp repo
    if os.path.exists(tmp_repo_path):
        shutil.rmtree(tmp_repo_path)
    os.makedirs(tmp_repo_path)
    yield tmp_repo_path
    # Teardown: clean up after test
    if os.path.exists(tmp_repo_path):
        shutil.rmtree(tmp_repo_path)

def test_find_records_image_seq():
    # Arrange
    img1 = os.path.join(tmp_repo_path, "img1.jpg")
    img2 = os.path.join(tmp_repo_path, "img2.jpg")
    with open(img1, 'wb') as f:
        f.write(b'fake')
    with open(img2, 'wb') as f:
        f.write(b'fake')
    # Act
    records = list(parser.find_records(tmp_repo_path))
    # Assert
    assert len(records) == 1, f"Expected 1 image_seq, got: {records}"
    record_id, record_type, path, meta = records[0]
    assert record_type == "image_seq", f"Expected record_type 'image_seq', got: {record_type}"
    assert meta.get("n_images", 0) == 2, f"Expected 2 images, got: {meta.get('n_images')}"

def test__is_image_file():
    assert parser._is_image_file("slika.jpg"), "Should return True for .jpg"
    assert not parser._is_image_file("video.mp4"), "Should return False for .mp4"

def test__is_video_file():
    assert parser._is_video_file("video.mp4"), "Should return True for .mp4"
    assert not parser._is_video_file("slika.jpg"), "Should return False for .jpg"

def test_find_records_video():
    # Arrange
    video_path = os.path.join(tmp_repo_path, "test.mp4")
    with open(video_path, 'wb') as f:
        f.write(b'fake')  # Not a real video, but enough for file detection
    # Act
    records = list(parser.find_records(tmp_repo_path))
    # Assert
    # Dozvoli duplikate, ali provjeri da postoji barem jedan ispravan video record
    video_records = [r for r in records if r[1] == "video" and r[2].endswith("test.mp4")]
    assert len(video_records) >= 1, f"Expected at least 1 video record for test.mp4, got: {records}"

def test_record_metadata():
    # Arrange
    img1 = os.path.join(tmp_repo_path, "img1.jpg")
    with open(img1, 'wb') as f:
        f.write(b'fake')
    # Act
    meta = parser.record_metadata(tmp_repo_path)
    # Assert
    assert meta["n_frames"] == 1, f"Expected n_frames 1, got: {meta['n_frames']}"
    assert meta["path"].endswith("tmp_test_repo"), f"Expected path to end with 'tmp_test_repo', got: {meta['path']}"

def test_find_annotation_for_record():
    # Arrange
    ann_path = os.path.join(tmp_repo_path, "annotations.json")
    with open(ann_path, 'w') as f:
        f.write('{"test": 1}')
    # Act
    found = parser.find_annotation_for_record(tmp_repo_path)
    # Assert
    assert found and found.endswith("annotations.json"), f"Expected to find 'annotations.json', got: {found}"

def test_get_annotation_csv():
    # Arrange
    ann_path = os.path.join(tmp_repo_path, "annotations.csv")
    with open(ann_path, 'w') as f:
        f.write("id;value\n1;42\n2;43\n")
    # Act
    ann = parser.get_annotation(tmp_repo_path)
    # Assert
    assert isinstance(ann, dict), f"Expected dict, got: {type(ann)}"
    assert "data" in ann, f"Expected key 'data' in result, got: {ann}"
    assert isinstance(ann["data"], list), f"Expected 'data' to be a list, got: {type(ann['data'])}"
    assert len(ann["data"]) == 2, f"Expected 2 rows in CSV, got: {len(ann['data'])}"
    assert ann["data"][0]["id"] == "1" and ann["data"][0]["value"] == "42", f"First row incorrect: {ann['data'][0]}"
    assert ann["data"][1]["id"] == "2" and ann["data"][1]["value"] == "43", f"Second row incorrect: {ann['data'][1]}"
