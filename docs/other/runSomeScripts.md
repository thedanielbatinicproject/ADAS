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

3) Dohvati nekoliko nasumičnih objekata iz dataseta za svaku kategoriju
```bash
python scripts/dataset/sample_conditions.py \
	--n N \
	--seed SEED \
	--index-path INDEX_PATH \
	--dataset-root DATASET_ROOT
```
sa zadanim vrijednostima:
n default: 10
INDEX_PATH default: /app/data/processed/index.db
DATASET_ROOT default: /app/data/raw/DADA2000

4) Pokreni ADAS scenario pipeline na jednom videu

Pokreće cijeli pipeline: učitavanje frameova -> detekcija traka -> detekcija prepreka -> procjena rizika sudara -> kontekst -> UI dashboard -> audio upozorenja.

Preduvjeti: izgrađen index (`build_index.py`), X server za GUI (VcXsrv), opcionalno PulseAudio za zvuk.

Za audio: na Windows hostu pokrenite `scripts\start_pulseaudio.bat` (PulseAudio je uključen u `dep\pulseaudio-1.1`, nije potrebna zasebna instalacija).

```bash
python scripts/run_scenario.py \
	--category-id 1 \
	--video-id 1
```
Za pokretanje s eksplicitnim audio checkom:
```bash
python scripts/run_scenario.py \
	--category-id 1 \
	--video-id 1 \
	--audio
```
Za headless mode (bez GUI prozora):
```bash
python scripts/run_scenario.py \
	--category-id 1 \
	--video-id 1 \
	--ui-backend none \
	--max-frames 100
```
Dodatni parametri: `--no-audio`, `--audio`, `--no-dashboard`, `--no-lanes`, `--no-obstacles`, `--no-risk`, `--ui-backend {dpg,cv2,none}`, `--target-fps FPS`, `--log-file PATH`.

5) Debug lane detekcije

Prikazuje Canny rubove, fitane polinomne granice i masku traka na videostreamu. Ne pokreće detekciju prepreka ni procjenu rizika.

Preduvjeti: izgrađen index, X server za GUI.

```bash
python scripts/debug_lanes.py \
	--category-id 1 \
	--video-id 1
```
Za prikaz Canny rubova:
```bash
python scripts/debug_lanes.py \
	--category-id 1 \
	--video-id 1 \
	--show-edges
```
Dodatni parametri: `--max-frames N`, `--delay-ms MS`, `--max-width PX`, `--ui-backend {dpg,cv2}`.

6) Debug detekcije prepreka

Prikazuje detektirane i praćene prepreke na videostreamu. Opcionalno prikazuje i trake (`--with-lanes`).

Preduvjeti: izgrađen index, X server za GUI.

```bash
python scripts/debug_obstacles.py \
	--category-id 1 \
	--video-id 1
```
S prikazom traka:
```bash
python scripts/debug_obstacles.py \
	--category-id 1 \
	--video-id 1 \
	--with-lanes
```
Dodatni parametri: `--max-frames N`, `--delay-ms MS`, `--max-width PX`, `--ui-backend {dpg,cv2}`.

7) Konteksna analiza videa

Pokreće konteksnu analizu (vidljivost, vremenski uvjeti, stanje traka, površina ceste) na svakom N-tom frameu. U terminal modu ispisuje tablicu, u GUI modu prikazuje player s overlay panelom.

Preduvjeti: izgrađen index, za GUI mod: X server.

Terminal mod (ispis tablice):
```bash
python scripts/context/analyze_video.py \
	--category-id 1 \
	--video-id 1 \
	--every-n 10
```
GUI mod s overlay panelom:
```bash
python scripts/context/analyze_video.py \
	--category-id 1 \
	--video-id 1 \
	--every-n 10 \
	--gui
```
Dodatni parametri: `--max-frames N`, `--delay-ms MS`, `--max-width PX`, `--ui-backend {dpg,cv2}`, `--record-type TYPE`.