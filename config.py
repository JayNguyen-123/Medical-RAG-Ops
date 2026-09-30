# config.py
"""
Centralised, validated runtime configuration.

Fails fast at process start-up if a required secret is missing or malformed.
Secrets are held as SecretStr so they never render in reprs, tracebacks or logs.
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CLEARANCE_GROUP_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


def decode_aes_key(raw: str) -> bytes:
    """Decode a URL-safe base64 string and require exactly 32 bytes (AES-256)."""
    try:
        key = base64.urlsafe_b64decode(raw.strip().encode("ascii"))
    except (binascii.Error, ValueError, UnicodeEncodeError) as exc:
        raise ValueError("encryption key is not valid URL-safe base64") from exc
    if len(key) != 32:
        raise ValueError(f"encryption key must decode to 32 bytes for AES-256 (got {len(key)})")
    return key


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ---- Runtime --------------------------------------------------------
    environment: Literal["development", "staging", "production"] = Field("production", alias="ENVIRONMENT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ---- Provider credentials -------------------------------------------
    openai_api_key: SecretStr = Field(..., alias="OPENAI_API_KEY")
    pinecone_api_key: SecretStr = Field(..., alias="PINECONE_API_KEY")

    # ---- Cryptography (HIPAA technical safeguard) -----------------------
    # Current key encrypts; previous keys (comma separated) are decrypt-only for rotation.
    medical_encryption_key: SecretStr = Field(..., alias="MEDICAL_ENCRYPTION_KEY")
    medical_encryption_previous_keys: SecretStr = Field(SecretStr(""), alias="MEDICAL_ENCRYPTION_PREVIOUS_KEYS")

    # ---- Models -----------------------------------------------------------
    llm_model: str = Field("gpt-4o", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(60.0, alias="LLM_TIMEOUT_SECONDS", gt=0)
    llm_max_retries: int = Field(3, alias="LLM_MAX_RETRIES", ge=0, le=10)
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, alias="EMBEDDING_DIMENSION", gt=0)

    # ---- Pinecone ---------------------------------------------------------
    pinecone_index_name: str = Field("medical-knowledge-index", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field("aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field("us-east-1", alias="PINECONE_REGION")
    retrieval_k: int = Field(4, alias="RETRIEVAL_K", ge=1, le=20)
    # Cosine similarity floor; chunks below this are treated as "not relevant".
    retrieval_min_score: float = Field(0.30, alias="RETRIEVAL_MIN_SCORE", ge=0.0, le=1.0)

    # ---- Access control ---------------------------------------------------
    allowed_clearance_groups: str = Field("cardiology", alias="ALLOWED_CLEARANCE_GROUPS")
    max_question_chars: int = Field(2000, alias="MAX_QUESTION_CHARS", gt=0)

    # ---- Redis / semantic cache ------------------------------------------
    redis_host: str = Field("redis", alias="REDIS_HOST")
    redis_port: int = Field(6379, alias="REDIS_PORT")
    redis_password: Optional[SecretStr] = Field(None, alias="REDIS_PASSWORD")
    redis_ssl: bool = Field(False, alias="REDIS_SSL")
    semantic_cache_enabled: bool = Field(True, alias="SEMANTIC_CACHE_ENABLED")
    # Cosine DISTANCE (0 = identical). Kept deliberately strict: clinically different questions
    # ("adult" vs "pediatric" dose) can sit very close in embedding space.
    cache_similarity_threshold: float = Field(0.02, alias="CACHE_SIMILARITY_THRESHOLD", ge=0.0, le=0.2)
    cache_ttl_seconds: int = Field(86_400, alias="CACHE_TTL_SECONDS", gt=0)

    # ---- Celery -----------------------------------------------------------
    celery_broker_url: Optional[SecretStr] = Field(None, alias="CELERY_BROKER_URL")
    celery_result_backend: Optional[SecretStr] = Field(None, alias="CELERY_RESULT_BACKEND")

    # ---- Observability ----------------------------------------------------
    tracing_enabled: bool = Field(True, alias="TRACING_ENABLED")
    phoenix_collector_endpoint: str = Field("http://phoenix:6006", alias="PHOENIX_COLLECTOR_ENDPOINT")
    phoenix_api_key: Optional[SecretStr] = Field(None, alias="PHOENIX_API_KEY")
    phoenix_project_name: str = Field("medical-rag-enterprise", alias="PHOENIX_PROJECT_NAME")
    phi_eval_lookback_minutes: int = Field(15, alias="PHI_EVAL_LOOKBACK_MINUTES", gt=0)
    phi_eval_max_spans: int = Field(500, alias="PHI_EVAL_MAX_SPANS", gt=0)
    slack_webhook_url: Optional[SecretStr] = Field(None, alias="SLACK_WEBHOOK_URL")

    # ---- Validators -----------------------------------------------------
    @field_validator("medical_encryption_key")
    @classmethod
    def _validate_key(cls, value: SecretStr) -> SecretStr:
        decode_aes_key(value.get_secret_value())
        return value

    @field_validator("medical_encryption_previous_keys")
    @classmethod
    def _validate_previous_keys(cls, value: SecretStr) -> SecretStr:
        for k in filter(None, (p.strip() for p in value.get_secret_value().split(","))):
            decode_aes_key(k)
        return value

    @field_validator("slack_webhook_url", mode="before")
    @classmethod
    def _validate_slack(cls, value):
        if value in (None, ""):
            return None
        raw = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        if not raw.startswith("https://hooks.slack.com/"):
            raise ValueError("SLACK_WEBHOOK_URL must be an https://hooks.slack.com/... incoming webhook")
        return raw

    @field_validator("allowed_clearance_groups")
    @classmethod
    def _validate_groups(cls, value: str) -> str:
        groups = [g.strip() for g in value.split(",") if g.strip()]
        if not groups:
            raise ValueError("ALLOWED_CLEARANCE_GROUPS must list at least one group")
        bad = [g for g in groups if not CLEARANCE_GROUP_PATTERN.match(g)]
        if bad:
            raise ValueError(f"invalid clearance group names: {bad} (use lowercase a-z, 0-9, _)")
        return ",".join(groups)

    @model_validator(mode="after")
    def _production_guards(self) -> "AppSettings":
        if self.environment == "production" and self.redis_password is None:
            raise ValueError("REDIS_PASSWORD is required when ENVIRONMENT=production")
        return self

    # ---- Derived values -------------------------------------------------
    @property
    def allowed_groups(self) -> frozenset[str]:
        return frozenset(self.allowed_clearance_groups.split(","))

    @property
    def encryption_keys(self) -> list[bytes]:
        """Current key first, then decrypt-only previous keys."""
        prev = [p.strip() for p in self.medical_encryption_previous_keys.get_secret_value().split(",") if p.strip()]
        return [decode_aes_key(self.medical_encryption_key.get_secret_value())] + [decode_aes_key(p) for p in prev]

    def redis_url(self, db: int = 0) -> str:
        scheme = "rediss" if self.redis_ssl else "redis"
        auth = f":{quote(self.redis_password.get_secret_value(), safe='')}@" if self.redis_password else ""
        return f"{scheme}://{auth}{self.redis_host}:{self.redis_port}/{db}"

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url.get_secret_value() if self.celery_broker_url else self.redis_url(1)

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend.get_secret_value() if self.celery_result_backend else self.redis_url(2)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def configure_logging() -> None:
    """Idempotent structured-ish logging. Never log query text, answers or user names (PHI)."""
    root = logging.getLogger()
    if getattr(root, "_medical_rag_configured", False):
        return
    logging.basicConfig(
        level=get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Third-party HTTP clients can log full request bodies at DEBUG.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._medical_rag_configured = True  # type: ignore[attr-defined]


# Backwards-compatible module-level handle (resolved lazily on first attribute access).
class _LazySettings:
    def __getattr__(self, item):
        return getattr(get_settings(), item)


settings = _LazySettings()
