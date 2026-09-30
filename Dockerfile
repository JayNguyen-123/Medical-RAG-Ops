# syntax=docker/dockerfile:1.7
# Single image for worker, beat, UI and ingest (commands set in docker-compose.yml).

# ---------- Stage 1: build dependencies into an isolated venv ----------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
# Prefer a hashed lock file in production: pip install --require-hashes -r requirements.lock
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- Stage 2: minimal runtime ----------
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="medical-rag-ops" \
      compliance.scope="HIPAA-Technical-Safeguards"

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/app

# Unprivileged runtime user
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app *.py ./

USER app

EXPOSE 8501
CMD ["celery", "-A", "tasks", "worker", "--loglevel=INFO"]
