# cache.py
"""
Tenant-isolated, encrypted semantic cache on Redis (RediSearch HNSW).

Safety properties:
  * Lookups are pre-filtered by the caller's clearance_group TAG -> no cross-tenant hits.
  * Query/response text is AES-256-GCM encrypted with AAD bound to (key, field, group).
  * Keys use an HMAC of the normalised question (stable across processes, not reversible).
  * Every entry has a TTL; ingest invalidates a group's entries when its corpus changes.
  * Any cache failure degrades to a miss -- the cache is never on the critical path.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import numpy as np
from redis.commands.search.query import Query

from config import CLEARANCE_GROUP_PATTERN, get_settings
from crypto import DecryptionError, get_crypto_engine
from database import CACHE_INDEX_NAME, CACHE_KEY_PREFIX, ensure_redis_cache_index, get_redis

logger = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)
    distance: float = 0.0


def normalise_question(question: str) -> str:
    return " ".join(question.lower().split())


def validate_clearance_group(group: str) -> str:
    if not isinstance(group, str) or not CLEARANCE_GROUP_PATTERN.match(group):
        raise ValueError("invalid clearance group identifier")
    return group


def make_cache_key(question: str, clearance_group: str) -> str:
    validate_clearance_group(clearance_group)
    s = get_settings()
    # Model + corpus-index are part of the identity so a model/index change never serves stale answers.
    material = f"{s.llm_model}|{s.pinecone_index_name}|{clearance_group}|{normalise_question(question)}"
    return f"{CACHE_KEY_PREFIX}{clearance_group}:{get_crypto_engine().keyed_hash(material)}"


def build_knn_query_string(clearance_group: str) -> str:
    # clearance_group is validated against [a-z0-9_] so no TAG escaping is required.
    validate_clearance_group(clearance_group)
    return f"(@clearance_group:{{{clearance_group}}})=>[KNN 1 @query_embedding $vec AS vector_score]"


def _aad(cache_key: str, field_name: str) -> str:
    return f"{cache_key}|{field_name}"


_index_ready = False


def _ensure_index() -> None:
    global _index_ready
    if not _index_ready:
        ensure_redis_cache_index()
        _index_ready = True


def to_vector_bytes(embedding: list[float]) -> bytes:
    return np.asarray(embedding, dtype=np.float32).tobytes()


def lookup(query_embedding: list[float], clearance_group: str) -> CachedAnswer | None:
    s = get_settings()
    if not s.semantic_cache_enabled:
        return None
    try:
        _ensure_index()
        q = (
            Query(build_knn_query_string(clearance_group))
            .return_fields("response", "sources", "vector_score")
            .sort_by("vector_score", asc=True)
            .paging(0, 1)
            .dialect(2)
        )
        res = get_redis().ft(CACHE_INDEX_NAME).search(q, query_params={"vec": to_vector_bytes(query_embedding)})
        if not res.docs:
            return None
        doc = res.docs[0]
        distance = float(doc.vector_score)
        if distance > s.cache_similarity_threshold:
            return None
        key = doc.id if isinstance(doc.id, str) else doc.id.decode()
        engine = get_crypto_engine()
        answer = engine.decrypt(doc.response, _aad(key, "response"))
        sources = json.loads(engine.decrypt(doc.sources, _aad(key, "sources"))) if getattr(doc, "sources", None) else []
        return CachedAnswer(answer=answer, sources=sources, distance=distance)
    except DecryptionError:
        logger.warning("semantic cache entry failed authentication; treating as miss")
    except Exception:  # noqa: BLE001 - cache must never break the request path
        logger.exception("semantic cache lookup failed; treating as miss")
    return None


def store(question: str, query_embedding: list[float], clearance_group: str, answer: str, sources: list[dict]) -> None:
    s = get_settings()
    if not s.semantic_cache_enabled:
        return
    try:
        _ensure_index()
        key = make_cache_key(question, clearance_group)
        engine = get_crypto_engine()
        pipe = get_redis().pipeline(transaction=True)
        pipe.hset(key, mapping={
            "query": engine.encrypt(question, _aad(key, "query")),
            "response": engine.encrypt(answer, _aad(key, "response")),
            "sources": engine.encrypt(json.dumps(sources), _aad(key, "sources")),
            "clearance_group": clearance_group,
            "query_embedding": to_vector_bytes(query_embedding),
        })
        pipe.expire(key, s.cache_ttl_seconds)
        pipe.execute()
    except Exception:  # noqa: BLE001
        logger.exception("semantic cache write failed")


def invalidate_group(clearance_group: str) -> int:
    """Delete every cached answer for a clearance group (call after its corpus changes)."""
    validate_clearance_group(clearance_group)
    client = get_redis()
    deleted = 0
    batch: list[bytes] = []
    for key in client.scan_iter(match=f"{CACHE_KEY_PREFIX}{clearance_group}:*", count=500):
        batch.append(key)
        if len(batch) >= 500:
            deleted += client.unlink(*batch)
            batch.clear()
    if batch:
        deleted += client.unlink(*batch)
    logger.info("invalidated %d cache entries for group=%s", deleted, clearance_group)
    return deleted
