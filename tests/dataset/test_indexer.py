
# Ensure src is in sys.path for imports
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..', 'src')))

import tempfile
import shutil
import sqlite3
import pytest
from adas.dataset import indexer
from adas.dataset.parser import find_records, record_metadata

# Helper: create a minimal fake dataset structure
@pytest.fixture
def fake_dataset(tmp_path):
    root = tmp_path / "dada"
    os.makedirs(root, exist_ok=True)
    # Create a fake record directory and a dummy file
    rec_dir = root / "cat1" / "001"
    rec_dir.mkdir(parents=True)
    dummy_file = rec_dir / "dummy.jpg"
    dummy_file.write_text("test")
    return str(root)

def test_build_and_query_index(fake_dataset, tmp_path):
    # Use a temp index path, never touch the real one
    index_path = tmp_path / "index.db"
    # Build index
    indexer.build_index(fake_dataset, str(index_path), overwrite=True)
    # Check index file exists
    assert os.path.exists(index_path)
    # Query all records
    records = indexer.list_records(str(index_path))
    assert isinstance(records, list)
    assert len(records) == 1
    rec = records[0]
    assert rec["record_id"] == "cat1/001"
    assert rec["category"] == "cat1"
    # Query by record_id
    rec2 = indexer.get_record(str(index_path), "cat1/001")
    assert rec2 is not None
    assert rec2["record_id"] == "cat1/001"
    # Index is fresh
    assert indexer.is_index_fresh(fake_dataset, str(index_path))

def test_index_not_fresh_on_new_file(fake_dataset, tmp_path):
    index_path = tmp_path / "index.db"
    indexer.build_index(fake_dataset, str(index_path), overwrite=True)
    # Add a new file to dataset
    new_file = os.path.join(fake_dataset, "cat1", "001", "new.txt")
    with open(new_file, "w") as f:
        f.write("new")
    import time
    time.sleep(0.1)  # Ensure mtime is updated
    assert not indexer.is_index_fresh(fake_dataset, str(index_path))
