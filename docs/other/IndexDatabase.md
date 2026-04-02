# Index Database (Local SQLite)

Database file: `data/processed/index.db`

This local SQLite index contains two linked tables:

1. `records` (discovered dataset records from filesystem scan)
2. `annotations` (parsed rows from `DADA2000_video_annotations.csv`)

Join key between tables: `(category_id, video_id)`

## Table: records

| Column | Type | Description |
|---|---|---|
| record_id | TEXT (PK) | Relative record id from scan (example: `1/001`) |
| category | TEXT | Parsed category string from path metadata |
| path | TEXT | Absolute path to record folder or video |
| n_frames | INTEGER | Number of frames (best-effort) |
| maps_path | TEXT | Reserved field (nullable) |
| category_id | INTEGER | Normalized numeric category id (example: `1`) |
| video_id | INTEGER | Normalized numeric video id (example: `1`) |
| annotation_status | TEXT | `matched`, `missing_annotation`, `ambiguous_record_key`, or `NULL` |

## Table: annotations

| Column | Type | Description |
|---|---|---|
| category_id | INTEGER | Numeric category id (from CSV `type`) |
| video_id | INTEGER | Numeric video id (from CSV `video`) |
| video_raw | TEXT | Original raw CSV video value |
| weather | INTEGER | Weather code |
| light | INTEGER | Light code |
| scenes | INTEGER | Scene code |
| linear | INTEGER | Linear code |
| type | INTEGER | Category/type code |
| accident_occurred | INTEGER | 1/0 flag |
| abnormal_start_frame | INTEGER | Abnormal start frame |
| accident_frame | INTEGER | Accident frame |
| abnormal_end_frame | INTEGER | Abnormal end frame |
| total_frames | INTEGER | Total frame count from CSV |
| interval_0_tai | INTEGER | Interval [0, tai] |
| interval_tai_tco | INTEGER | Interval [tai, tco] |
| interval_tai_tae | INTEGER | Interval [tai, tae] |
| interval_tco_tae | INTEGER | Interval [tco, tae] |
| interval_tae_end | INTEGER | Interval [tae, end] |
| texts | TEXT | Text description |
| causes | TEXT | Cause description |
| measures | TEXT | Suggested measures |

Primary key: `(category_id, video_id)`

## Build index (with annotations)

Example command:

```bash
python scripts/dataset/build_index.py \
  --dataset-root data/raw/DADA2000 \
  --index-path data/processed/index.db
```

Notes:
- If `--annotations-csv` is omitted, script auto-detects `DADA2000_video_annotations.csv` next to `--dataset-root` parent.
- Progress is logged every 10 seconds by default.
- Missing links are reported as one-line warnings:
  - missing annotation for a discovered record
  - missing record for a CSV annotation row

## Python call examples and returns

### 1) Build index

```python
from adas.dataset import indexer

indexer.build_index(
    dataset_root="data/raw/DADA2000",
    index_path="data/processed/index.db",
    overwrite=True,
    annotations_csv_path="data/raw/DADA2000_video_annotations.csv",
    progress_interval_sec=10,
)
```

Return: `None` (writes SQLite DB and prints progress/warnings)

### 2) List records

```python
from adas.dataset import indexer

rows = indexer.list_records("data/processed/index.db")
print(rows[0])
```

Example return item:

```python
{
  "record_id": "1/001",
  "category": "1",
  "path": "/app/data/raw/DADA2000/1/001",
  "n_frames": 440,
  "maps_path": None,
  "category_id": 1,
  "video_id": 1,
  "annotation_status": "matched"
}
```

### 3) Get single record

```python
row = indexer.get_record("data/processed/index.db", "1/001")
```

Return: single dict or `None`

### 4) Get joined record + annotation

```python
joined = indexer.get_record_with_annotation("data/processed/index.db", "1/001")
```

Return: dict containing `records.*` and `annotations.*` columns (annotation columns are `None` if no match).
