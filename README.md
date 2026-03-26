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

Otvori VS Code:

```bash
code .
```

Kada se VS Code otvori, klikni **"Reopen in Container"** ili:

`Ctrl+Shift+P` → **Dev Containers: Reopen in Container**

> Prvi build traje ~10-15 minuta (instalacija LaTeX-a i Python dependencija).

### Što se instalira u containeru

- Python 3.12
- Poetry + sve dependencije iz `pyproject.toml`
- LaTeX (`texlive-full`)
