# observability.py
"""
OpenTelemetry -> Arize Phoenix tracing for the LangChain pipeline.

NOTE (HIPAA): traces contain prompts, retrieved context and answers in plaintext.
Treat the Phoenix server + its Postgres volume as a PHI system of record: auth enabled,
encrypted storage, network-restricted, retention policy set. Set
OPENINFERENCE_HIDE_INPUTS / OPENINFERENCE_HIDE_OUTPUTS=true if that is not acceptable
(the PHI compliance task then has nothing to inspect).
"""
from __future__ import annotations

import logging
import threading

from config import get_settings

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_initialised = False


def setup_tracing() -> None:
    global _initialised
    s = get_settings()
    if not s.tracing_enabled or _initialised:
        return
    with _lock:
        if _initialised:
            return
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from phoenix.otel import register

            headers = (
                {"authorization": f"Bearer {s.phoenix_api_key.get_secret_value()}"} if s.phoenix_api_key else None
            )
            tracer_provider = register(
                project_name=s.phoenix_project_name,
                endpoint=f"{s.phoenix_collector_endpoint.rstrip('/')}/v1/traces",
                headers=headers,
                batch=True,  # never block requests on the exporter
                set_global_tracer_provider=False,
            )
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
            logger.info("tracing enabled -> project=%s", s.phoenix_project_name)
        except Exception:  # noqa: BLE001 - observability must not take the service down
            logger.exception("failed to initialise tracing; continuing without it")
        _initialised = True
