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

Otvorite VS Code:

```bash
code .
```

Kada se VS Code otvori, kliknite **"Reopen in Container"** ili:

`Ctrl+Shift+P` → **Dev Containers: Reopen in Container**

> Prvi build traje ~10-15 minuta (instalacija LaTeX-a i Python dependencija).

### Što se instalira u containeru

- Python 3.12
- Poetry + sve dependencije iz `pyproject.toml`
- LaTeX (`texlive-full`)

#### Dodatna dokumentacija
Detaljnije o projektnim specifikacijama pročitajte [ovdje](/docs/other/ProjectSpecifications.md).

Opis lokalne indeksirane baze (records + annotations), svih stupaca i primjera poziva dostupan je u [docs/other/IndexDatabase.md](docs/other/IndexDatabase.md).

Popis skripti za pokretanje testova i ostalih modula nalazi se u [docs/other/runSomeScripts.md](docs/other/runSomeScripts.md). Za svaku skriptu se podrazumijeva da ste pozicionirani u root direktoriju projekta, što je u docker kontenjeru "root@cont-id:/app".


## Opcionalno: GUI Playback u Dockeru (Windows Host)

Ovaj korak je opcionalan i potreban je samo ako želite otvoriti player prozor iz dev containera (OpenCV `imshow`).

1. Instalirajte i pokrenite **VcXsrv** na Windows hostu (XLaunch):
	- `Multiple windows`
	- `Start no client`
	- `Disable access control` (za početni test)

2. U `docker-compose.yml` koristite ove varijable za servis:
	- `DISPLAY=host.docker.internal:0.0`
	- `QT_QPA_PLATFORM=xcb`
	- `QT_X11_NO_MITSHM=1`

3. Rebuildajte dev container:
	- VS Code: `Dev Containers: Rebuild and Reopen in Container`

4. Pokrenite player unutar containera:

```bash
python scripts/dataset/play_index_video.py \
  --index-path data/processed/index.db \
  --category-id 1 \
  --video-id 1 \
  --delay-ms 33
```

Ako se GUI ne otvori, provjerite da je VcXsrv pokrenut i da je firewall dopustio pristup.


## Opcionalno: Audio u Dockeru (Windows Host)

Docker kontejner nema pristup zvučnoj kartici hosta. Za audio upozorenja (kočenje, upozorenje) PulseAudio server mora biti pokrenut na Windows hostu.

PulseAudio je uključen u repozitorij (`dep/pulseaudio-1.1`) — **nije potrebna posebna instalacija**.

1. Na Windows hostu pokrenite batch skriptu iz korijena projekta:

```
scripts\start_pulseaudio.bat
```

2. Ostavite taj prozor otvoren dok koristite Docker kontejner.

3. U containeru pokrenite ADAS scenario s audiom:

```bash
python scripts/run_scenario.py --category-id 1 --video-id 1
```

Audio je uključen po defaultu. Za pokretanje bez zvuka dodajte `--no-audio`.

> Bez pokrenutog PulseAudio servera zvuk neće raditi, ali sve ostale funkcionalnosti (video, detekcija, dashboard) rade normalno.


