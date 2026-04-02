Popis templateova za skripte koje se mogu pokrenuti:

1) Build index (records + annotations)

```bash
python scripts/dataset/build_index.py \
	--dataset-root data/raw/DADA2000 \
	--index-path data/processed/index.db
```

2) Pusti točno određeni video po category/video ID

```bash
python scripts/dataset/play_index_video.py \
	--index-path data/processed/index.db \
	--category-id 1 \
	--video-id 1 \
	--delay-ms 33
```
Napomena: 33ms ~ 30 FPS simulacija.