
"""
Indexer for dataset records metadata (robust, production-ready).

Scans the dataset, builds a SQLite index with metadata (record_id, category, path, n_frames, maps_path),
and enables fast queries and resume without rescanning the full dataset.

Index is stored by default in /data/processed/index.db (can be overridden).
Atomic: writes to index.db.tmp, then renames to index.db.
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any
from .parser import find_records, record_metadata

DEFAULT_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/processed/index.db'))
INDEX_VERSION = 1

def build_index(dataset_root: str, index_path: Optional[str] = None, overwrite: bool = False) -> None:
	"""
	Scan dataset_root, build SQLite index with metadata for all records.
	Atomic: writes to index.db.tmp, then renames to index.db.
	If overwrite=True, existing index will be replaced.
	"""
	if index_path is None:
		index_path = DEFAULT_INDEX_PATH
	os.makedirs(os.path.dirname(index_path), exist_ok=True)
	tmp_index_path = index_path + ".tmp"
	if overwrite and os.path.exists(index_path):
		os.remove(index_path)
	if os.path.exists(tmp_index_path):
		os.remove(tmp_index_path)
	conn = sqlite3.connect(tmp_index_path)
	try:
		c = conn.cursor()
		c.execute("""
			CREATE TABLE IF NOT EXISTS records (
				record_id TEXT PRIMARY KEY,
				category TEXT,
				path TEXT,
				n_frames INTEGER,
				maps_path TEXT
			)
		""")
		c.execute("DELETE FROM records")
		for record_id, record_type, path, meta in find_records(dataset_root):
			meta = record_metadata(path)
			c.execute(
				"INSERT OR REPLACE INTO records (record_id, category, path, n_frames, maps_path) VALUES (?, ?, ?, ?, ?)",
				(
					record_id,
					meta.get("category"),
					meta.get("path"),
					meta.get("n_frames"),
					meta.get("maps_path"),
				),
			)
		conn.commit()
	finally:
		conn.close()
	os.replace(tmp_index_path, index_path)

def is_index_fresh(dataset_root: str, index_path: Optional[str] = None) -> bool:
	"""
	Check if index exists and is newer than the latest file in dataset_root.
	"""
	if index_path is None:
		index_path = DEFAULT_INDEX_PATH
	if not os.path.exists(index_path):
		return False
	index_mtime = os.path.getmtime(index_path)
	for root, dirs, files in os.walk(dataset_root):
		for name in files + dirs:
			fpath = os.path.join(root, name)
			if os.path.getmtime(fpath) > index_mtime:
				return False
	return True

def get_record(index_path: Optional[str], record_id: str) -> Optional[Dict[str, Any]]:
	"""
	Return metadata for a given record_id from the index.
	"""
	if index_path is None:
		index_path = DEFAULT_INDEX_PATH
	conn = sqlite3.connect(index_path)
	try:
		c = conn.cursor()
		c.execute("SELECT record_id, category, path, n_frames, maps_path FROM records WHERE record_id = ?", (record_id,))
		row = c.fetchone()
		if row:
			return dict(zip(["record_id", "category", "path", "n_frames", "maps_path"], row))
		return None
	finally:
		conn.close()

def list_records(index_path: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
	"""
	List all (or filtered) records from the index.
	"""
	if index_path is None:
		index_path = DEFAULT_INDEX_PATH
	conn = sqlite3.connect(index_path)
	try:
		c = conn.cursor()
		if category:
			c.execute("SELECT record_id, category, path, n_frames, maps_path FROM records WHERE category = ?", (category,))
		else:
			c.execute("SELECT record_id, category, path, n_frames, maps_path FROM records")
		rows = c.fetchall()
		return [dict(zip(["record_id", "category", "path", "n_frames", "maps_path"], row)) for row in rows]
	finally:
		conn.close()
