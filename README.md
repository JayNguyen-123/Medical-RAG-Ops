# Secure Medical RAG Pipeline

Retrieval-augmented clinical reference assistant built with LangChain, Pinecone, Redis Stack and Arize Phoenix. It includes application-level AES-256-GCM encryption, tenant (clearance-group) isolation, grounded answers with citations, continuous PHI-leak monitoring and a Ragas quality gate in CI.

> This is a reference implementation. HIPAA compliance also depends on BAAs, hosting, access policies and operational controls. See `REVIEW.md`.

## Architecture

- **Query path** (`main_pipeline.py`): authorise → validate → embed once → tenant-scoped encrypted semantic cache → Pinecone retrieval filtered by `clearance_group` with a relevance floor → grounded GPT-4o answer with `[S#]` citations → cache (successful answers only) → audit log.
- **Ingest** (`ingest.py`): PDF → page-aware chunks → deterministic vector IDs → stale chunk cleanup → cache invalidation.
- **Monitoring** (`tasks.py`): Celery beat every 5 min → new LLM spans from Phoenix → regex + LLM-judge PHI checks → annotations in Phoenix → identifier-only Slack alert.
- **UI** (`app.py`): Streamlit behind an SSO reverse proxy; groups come from `X-Forwarded-Groups`.

## Layout

```
medical-rag-ops/
├── .github/workflows/rag-eval.yml   # lint + unit (offline) → docker build → staging Ragas gate
├── .env.example                     # configuration template (copy to .env; never commit .env)
├── .gitignore / .dockerignore
├── docker-compose.yml               # postgres, redis-stack, phoenix, worker, beat, frontend, ingest (tools profile)
├── Dockerfile                       # single non-root image for all app services
├── requirements.txt / requirements-dev.txt
├── pyproject.toml / pytest.ini
├── config.py                        # validated settings (pydantic-settings)
├── crypto.py                        # AES-256-GCM with AAD + key rotation; HMAC keyed hashing
├── database.py                      # lazy clients; idempotent Redis/Pinecone index bootstrap
├── cache.py                         # encrypted, tenant-scoped semantic cache
├── observability.py                 # OpenTelemetry → Phoenix
├── ingest.py                        # CLI ingestion
├── main_pipeline.py                 # query engine
├── tasks.py                         # Celery PHI compliance monitor
├── app.py                           # Streamlit UI
└── tests/                           # unit, pipeline (mocked) and eval (live, -m eval)
```

## Quick start

```bash
cp .env.example .env        # fill in every blank value
docker compose up -d --build
docker compose ps           # all services should be healthy

# First boot: log into Phoenix (http://127.0.0.1:6006), create a system API key,
# set PHOENIX_API_KEY in .env, then: docker compose up -d

# Ingest a guideline into a clearance group
docker compose run --rm rag_ingest --file /data/clinical_guidelines.pdf --group cardiology
```

Generate keys:

```bash
python -c "import os,base64;print(base64.urlsafe_b64encode(os.urandom(32)).decode())"   # MEDICAL_ENCRYPTION_KEY
python -c "import secrets;print(secrets.token_urlsafe(32))"                              # REDIS_PASSWORD / PHOENIX_SECRET
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not eval"        # offline, no credentials needed
pytest -m eval -s           # live Ragas gate against the STAGING index
```

## Key rotation

1. Move the current `MEDICAL_ENCRYPTION_KEY` into `MEDICAL_ENCRYPTION_PREVIOUS_KEYS`.
2. Set a new `MEDICAL_ENCRYPTION_KEY` and restart. New writes use the new key, and old entries still decrypt.
3. After `CACHE_TTL_SECONDS` has passed, remove the old key.

## Operational notes

- Run exactly **one** `celery_beat` instance.
- Streamlit and Phoenix bind to loopback; expose them only through a TLS, authenticating reverse proxy.
- `CACHE_SIMILARITY_THRESHOLD` is a cosine *distance*. Keep it strict (≤ 0.02) for clinical content, or set `SEMANTIC_CACHE_ENABLED=false`.
