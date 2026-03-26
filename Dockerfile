FROM python:3.12-slim

WORKDIR /app

# System dependencies + LaTeX
RUN apt-get update && apt-get install -y \
    git \
    curl \
    texlive-full \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install poetry==2.3.2

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-root --with dev

COPY . .