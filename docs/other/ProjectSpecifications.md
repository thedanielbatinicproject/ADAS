# Specifikacije Projekta

## Pregled Projekta

**Naslov:** Primjena računalnog vida u sustavima za izbjegavanje sudara kod autonomnih vozila i pametnih sustava asistencije vozaču (ADAS)

**Autor:** Daniel Batinić

**Cilj:** Implementirati deterministički pipeline klasične obrade slike za detekciju prometnih traka i prepreka, uz kinematički algoritam procjene rizika od sudara koji aktivira upozorenje vozaču ili signal automatskog kočenja.

---

## Opseg

| U opsegu | Izvan opsega |
|---|---|
| Detekcija prometnih traka (Canny, transformacija perspektive) | Neuralne mreže / duboko učenje |
| Detekcija prepreka (klasične CV metode) | Lateralno upravljanje / promjena trake |
| Izračun vremena do sudara (TTC) | Višekamerainski input |
| Aktivacija upozorenja vozaču | Integracija CARLA simulatora |
| Aktivacija automatskog kočenja | |
| Obrada videa u stvarnom vremenu | |
| Kvantitativna evaluacija | |

---

## Dataset

| Svojstvo | Vrijednost |
|---|---|
| Dataset | DADA-2000 |
| Scenariji | Bliske nesreće, prometne nesreće, kritične situacije |
| Uvjeti | Dnevno svjetlo, noć, kiša, magla |
| Okruženje | Gradska vožnja i autocesta |
| Kamera | Jedna prednja dashcam perspektiva |
| Rezolucija | 1280x720 (720p) |

---

## Sistemski Zahtjevi

### Procesorska Moć

| Svojstvo | Vrijednost |
|---|---|
| Procesorska jedinica | Samo CPU (bez GPU) |
| Broj jezgri | 4 |
| Brzina procesora | 2.0 GHz |
| RAM | 8 GB |
| Ograničenje resursa | Docker resource limits |

### Ciljevi Performansi

| Metrika | Cilj |
|---|---|
| Minimalni FPS | 20 |
| Ciljani FPS | 25-30 |
| Ulazna rezolucija | 1280x720 |

---

## Tehnološki Stack

| Komponenta | Tehnologija |
|---|---|
| Programski jezik | Python 3.12 |
| Obrada slike | scikit-image |
| Numerička računanja | torch, torchvision (samo CPU) |
| Verzioniranje podataka | DVC |
| Notebook okruženje | JupyterLab |
| Upravljanje ovisnostima | Poetry |
| Kontejnerizacija | Docker + docker-compose |
| Kvaliteta koda | Ruff, Black, pre-commit |
| Testiranje | pytest |
| Dokumentacija | LaTeX |

---

## Reakcija Sustava

| Uvjet | Reakcija |
|---|---|
| TTC iznad sigurnog praga | Nema akcije |
| TTC se približava kritičnom pragu | Upozorenje vozaču |
| TTC ispod kritičnog praga | Signal automatskog kočenja |

---

## Budući Razvoj

- Višekamerainski input (lijevo, desno, stražnja kamera)
- Integracija CARLA simulatora kao ulaznog modula
- Evaluacija u dodatnim ekstremnim vremenskim uvjetima