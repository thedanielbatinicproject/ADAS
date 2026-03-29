
import os
import shutil
import pytest
from src.adas.dataset import parser
from tests.assert_colored import assert_colored

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
    msg_pass = f"find_records found 1 image_seq with 2 images as expected."
    msg_fail = f"find_records did not find expected image_seq. Records: {records}"
    assert_colored(len(records) == 1, msg_pass, msg_fail)
    record_id, record_type, path, meta = records[0]
    assert_colored(record_type == "image_seq", f"record_type is image_seq", f"record_type is {record_type}")
    assert_colored(meta.get("n_images", 0) == 2, f"n_images is 2", f"n_images is {meta.get('n_images')}")

def test__is_image_file():
    assert_colored(parser._is_image_file("slika.jpg"), "_is_image_file returns True for .jpg", "_is_image_file returns False for .jpg")
    assert_colored(not parser._is_image_file("video.mp4"), "_is_image_file returns False for .mp4", "_is_image_file returns True for .mp4")

def test__is_video_file():
    assert_colored(parser._is_video_file("video.mp4"), "_is_video_file returns True for .mp4", "_is_video_file returns False for .mp4")
    assert_colored(not parser._is_video_file("slika.jpg"), "_is_video_file returns False for .jpg", "_is_video_file returns True for .jpg")

def test_find_records_video():
    # Arrange
    video_path = os.path.join(tmp_repo_path, "test.mp4")
    with open(video_path, 'wb') as f:
        f.write(b'fake')  # Not a real video, but enough for file detection
    # Act
    records = list(parser.find_records(tmp_repo_path))
    # Assert
    msg_pass = f"find_records found 1 video as expected."
    msg_fail = f"find_records did not find expected video. Records: {records}"
    assert_colored(len(records) == 1, msg_pass, msg_fail)
    record_id, record_type, path, meta = records[0]
    assert_colored(record_type == "video", f"record_type is video", f"record_type is {record_type}")
    assert_colored(path.endswith("test.mp4"), f"path ends with test.mp4", f"path is {path}")

def test_record_metadata():
    # Arrange
    img1 = os.path.join(tmp_repo_path, "img1.jpg")
    with open(img1, 'wb') as f:
        f.write(b'fake')
    # Act
    meta = parser.record_metadata(tmp_repo_path)
    # Assert
    assert_colored(meta["n_frames"] == 1, "n_frames is 1", f"n_frames is {meta['n_frames']}")
    assert_colored(meta["path"].endswith("tmp_test_repo"), "path ends with tmp_test_repo", f"path is {meta['path']}")

def test_find_annotation_for_record():
    # Arrange
    ann_path = os.path.join(tmp_repo_path, "annotations.json")
    with open(ann_path, 'w') as f:
        f.write('{"test": 1}')
    # Act
    found = parser.find_annotation_for_record(tmp_repo_path)
    # Assert
    assert_colored(found and found.endswith("annotations.json"), "Found annotations.json", f"Did not find annotations.json, found: {found}")

def test_get_annotation_csv():
    # Arrange
    ann_path = os.path.join(tmp_repo_path, "annotations.csv")
    with open(ann_path, 'w') as f:
        f.write("id,value\n1,42\n2,43\n")
    # Act
    ann = parser.get_annotation(tmp_repo_path)
    # Assert
    assert_colored(isinstance(ann, dict), "get_annotation returns dict for CSV", f"Returned: {ann}")
    assert_colored("data" in ann, 'Key "data" present in result', f'Result: {ann}')
    assert_colored(isinstance(ann["data"], list), '"data" is a list', f'data: {ann["data"]}')
    assert_colored(len(ann["data"]) == 2, 'CSV parsed with 2 rows', f'Rows: {ann["data"]}')
    assert_colored(ann["data"][0]["id"] == "1" and ann["data"][0]["value"] == "42", 'First row correct', f'Row: {ann["data"][0]}')
    assert_colored(ann["data"][1]["id"] == "2" and ann["data"][1]["value"] == "43", 'Second row correct', f'Row: {ann["data"][1]}')
