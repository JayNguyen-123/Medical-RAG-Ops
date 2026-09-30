# database.py
"""
Lazily-constructed, process-wide clients for OpenAI embeddings, Pinecone and Redis.

Nothing here opens a network connection at import time, so modules can be imported by
tests, linters and CLI tools without live infrastructure.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

import redis
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from redis.commands.search.field import TagField, VectorField

try:  # redis-py >= 6
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:  # redis-py 5.x
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType

from config import get_settings

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "v1"
CACHE_INDEX_NAME = f"idx:prompt_cache:{CACHE_SCHEMA_VERSION}"
CACHE_KEY_PREFIX = f"cache:{CACHE_SCHEMA_VERSION}:"


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(
        model=s.embedding_model,
        dimensions=s.embedding_dimension,
        api_key=s.openai_api_key.get_secret_value(),
        timeout=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )


@lru_cache(maxsize=1)
def get_pinecone() -> Pinecone:
    return Pinecone(api_key=get_settings().pinecone_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_vector_store() -> PineconeVectorStore:
    s = get_settings()
    return PineconeVectorStore(
        index=get_pinecone().Index(s.pinecone_index_name),
        embedding=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    s = get_settings()
    return redis.Redis(
        host=s.redis_host,
        port=s.redis_port,
        password=s.redis_password.get_secret_value() if s.redis_password else None,
        ssl=s.redis_ssl,
        decode_responses=False,  # vectors are raw bytes
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
        health_check_interval=30,
        retry_on_timeout=True,
    )


# ---------------------------------------------------------------------------
# Idempotent schema bootstrap (call from start-up / CLI, not on import)
# ---------------------------------------------------------------------------
def ensure_redis_cache_index() -> None:
    """Create the RediSearch HNSW index if missing. Requires Redis Stack / Redis 8 (query engine)."""
    client = get_redis()
    try:
        client.ft(CACHE_INDEX_NAME).info()
        return
    except redis.exceptions.ResponseError as exc:
        if "unknown command" in str(exc).lower():
            raise RuntimeError(
                "Redis server has no search module. Use redis/redis-stack-server or Redis 8+."
            ) from exc
    logger.info("creating redis index %s", CACHE_INDEX_NAME)
    # Only the tenant tag and the vector are indexed; encrypted blobs are stored but NOT indexed.
    schema = (
        TagField("clearance_group"),
        VectorField(
            "query_embedding",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": get_settings().embedding_dimension, "DISTANCE_METRIC": "COSINE"},
        ),
    )
    try:
        client.ft(CACHE_INDEX_NAME).create_index(
            fields=schema,
            definition=IndexDefinition(prefix=[CACHE_KEY_PREFIX], index_type=IndexType.HASH),
        )
    except redis.exceptions.ResponseError as exc:
        if "index already exists" not in str(exc).lower():  # lost a race with another worker
            raise


def ensure_pinecone_index(timeout_seconds: int = 300) -> None:
    s = get_settings()
    pc = get_pinecone()
    if s.pinecone_index_name in pc.list_indexes().names():
        desc = pc.describe_index(s.pinecone_index_name)
        if desc.dimension != s.embedding_dimension:
            raise RuntimeError(
                f"Pinecone index dimension {desc.dimension} != embedding dimension {s.embedding_dimension}"
            )
        return
    logger.info("creating pinecone index %s", s.pinecone_index_name)
    pc.create_index(
        name=s.pinecone_index_name,
        dimension=s.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        deletion_protection="enabled" if s.environment == "production" else "disabled",
    )
    deadline = time.monotonic() + timeout_seconds
    while not pc.describe_index(s.pinecone_index_name).status["ready"]:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Pinecone index not ready after {timeout_seconds}s")
        time.sleep(2)
