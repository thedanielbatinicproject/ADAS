# ADAS — Primjena računalnog vida u sustavima za izbjegavanje sudara

Primjena metoda klasične obrade slike za prepoznavanje prometnih traka i prepreka u sustavima asistencije vozaču (ADAS).

## Pokretanje projekta

### Preduvjeti

- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/) + ekstenzija [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup

```bash
git clone https://github.com/thedanielbatinicproject/ADAS.git
cd ADAS
```

Korisnici otvaraju VS Code:

```bash
code .
```

Kada se VS Code otvori, korisnici kliknu **"Reopen in Container"** ili:

`Ctrl+Shift+P` → **Dev Containers: Reopen in Container**

> Prvi build traje ~10-15 minuta (instalacija LaTeX-a i Python dependencija).

### Što se instalira u containeru

- Python 3.12
- Poetry + sve dependencije iz `pyproject.toml`
- LaTeX (`texlive-full`)

#### Dodatna dokumentacija
Detaljnije o projektnim specifikacijama korisnici mogu pročitati [ovdje](/docs/other/ProjectSpecifications.md).

Opis lokalne indeksirane baze (records + annotations), svih stupaca i primjera poziva dostupan je u [docs/other/IndexDatabase.md](docs/other/IndexDatabase.md).

Popis skripti za pokretanje testova i ostalih modula nalazi se u [docs/other/runSomeScripts.md](docs/other/runSomeScripts.md). Za svaku skriptu podrazumijeva se da su korisnici pozicionirani u root direktoriju projekta, što je u docker kontenjeru "root@cont-id:/app".

## Glavni dashboard (preporučeni način)

Za najjednostavniji rad koristi se jedan glavni UI panel koji pokreće sve ostalo.

1. Korisnici povuku repozitorij i pripreme dataset.
2. Dashboard se pokreće naredbom:

```bash
python main.py
```

3. U startup overlayu korisnici kliknu `RUN DOCKER`.
	- dashboard pokreće bundled PulseAudio (`scripts\\start_pulseaudio.bat`)
	- provjerava X server (VcXsrv)
	- pokreće `docker compose up -d adas`
4. Nakon toga korisnici u glavnom panelu biraju video iz tablice i pokreću module preko kartica.

Napomena: tablica će biti prazna dok se ne izgradi index (Build Index kartica ili `scripts/dataset/build_index.py`).


## Opcionalno: GUI Playback u Dockeru (Windows Host)

Ovaj korak je opcionalan i potreban je samo ako korisnici žele otvoriti player prozor iz dev containera (OpenCV `imshow`).

1. Korisnici instaliraju i pokreću **VcXsrv** na Windows hostu (XLaunch):
	- `Multiple windows`
	- `Start no client`
	- `Disable access control` (za početni test)

2. U `docker-compose.yml` koriste se ove varijable za servis:
	- `DISPLAY=host.docker.internal:0.0`
	- `QT_QPA_PLATFORM=xcb`
	- `QT_X11_NO_MITSHM=1`

3. Potrebno je rebuildati dev container:
	- VS Code: `Dev Containers: Rebuild and Reopen in Container`

4. Player se pokreće unutar containera:

```bash
python scripts/dataset/play_index_video.py \
  --index-path data/processed/index.db \
  --category-id 1 \
  --video-id 1 \
  --delay-ms 33
```

Ako se GUI ne otvori, korisnici provjeravaju je li VcXsrv pokrenut i je li firewall dopustio pristup.


## Opcionalno: Audio u Dockeru (Windows Host)

Docker kontejner nema pristup zvučnoj kartici hosta. Za audio upozorenja (kočenje, upozorenje) PulseAudio server mora biti pokrenut na Windows hostu.

PulseAudio je uključen u repozitorij (`dep/pulseaudio-1.1`) — **nije potrebna posebna instalacija**.

1. Na Windows hostu korisnici pokreću batch skriptu iz korijena projekta:

```
scripts\start_pulseaudio.bat
```

2. Taj prozor ostaje otvoren dok korisnici koriste Docker kontejner.

3. U containeru se pokreće ADAS scenario s audiom:

```bash
python scripts/run_scenario.py --category-id 1 --video-id 1
```

Audio je uključen po defaultu. Za pokretanje bez zvuka dodaje se `--no-audio`.

> Bez pokrenutog PulseAudio servera zvuk neće raditi, ali sve ostale funkcionalnosti (video, detekcija, dashboard) rade normalno.


