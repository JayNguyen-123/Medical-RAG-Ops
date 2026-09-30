# ingest.py
"""
Idempotent ingestion of clinical PDFs into Pinecone with tenant metadata.

Usage:
    python ingest.py --file data/clinical_guidelines.pdf --group cardiology

Re-running on the same document overwrites its vectors (deterministic IDs) and deletes
chunks that no longer exist, so there are no duplicates and no orphaned stale guidance.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import cache
from config import configure_logging, get_settings
from database import ensure_pinecone_index, get_pinecone, get_vector_store

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
UPSERT_BATCH = 100
MAX_PDF_BYTES = 100 * 1024 * 1024


def _doc_slug(path: Path) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", path.stem.lower()).strip("-")
    if not slug:
        raise ValueError(f"cannot derive a document id from filename {path.name!r}")
    return slug[:80]


def _id_prefix(clearance_group: str, doc_slug: str) -> str:
    return f"{clearance_group}#{doc_slug}#"


def build_chunks(path: Path, clearance_group: str) -> tuple[list[Document], list[str]]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise ValueError(f"{path.name} is password protected; decrypt it before ingestion")

    file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    slug = _doc_slug(path)
    prefix = _id_prefix(clearance_group, slug)
    ingested_at = datetime.now(timezone.utc).isoformat()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    ids: list[str] = []
    for page_idx, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            logger.warning("page %d of %s has no extractable text (scanned image?)", page_idx + 1, path.name)
            continue
        for chunk_idx, chunk in enumerate(splitter.split_text(text)):
            chunk_id = f"{prefix}p{page_idx + 1}#c{chunk_idx}"
            ids.append(chunk_id)
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "chunk_id": chunk_id,
                    "source_document": path.name,
                    "page_number": page_idx + 1,
                    "clearance_group": clearance_group,
                    "classification": "Internal Medical Guideline",
                    "document_sha256": file_sha256,
                    "ingested_at": ingested_at,
                },
            ))
    return docs, ids


def _delete_stale_vectors(prefix: str, keep_ids: set[str]) -> int:
    """Serverless-compatible cleanup: list IDs by prefix and delete those no longer produced."""
    index = get_pinecone().Index(get_settings().pinecone_index_name)
    stale: list[str] = []
    for id_batch in index.list(prefix=prefix):
        stale.extend(i for i in id_batch if i not in keep_ids)
    for i in range(0, len(stale), 1000):
        index.delete(ids=stale[i:i + 1000])
    return len(stale)


def process_and_ingest_pdf(file_path: str, clearance_group: str) -> int:
    s = get_settings()
    cache.validate_clearance_group(clearance_group)
    if clearance_group not in s.allowed_groups:
        raise PermissionError(f"clearance group {clearance_group!r} is not in ALLOWED_CLEARANCE_GROUPS")

    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source document not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("only .pdf files are supported")
    if path.stat().st_size > MAX_PDF_BYTES:
        raise ValueError(f"{path.name} exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB limit")

    ensure_pinecone_index()

    docs, ids = build_chunks(path, clearance_group)
    if not docs:
        raise ValueError(f"no extractable text in {path.name}; OCR it before ingestion")
    logger.info("doc=%s group=%s chunks=%d", path.name, clearance_group, len(docs))

    get_vector_store().add_documents(docs, ids=ids, batch_size=UPSERT_BATCH)
    removed = _delete_stale_vectors(_id_prefix(clearance_group, _doc_slug(path)), set(ids))
    if removed:
        logger.info("removed %d stale chunks for doc=%s", removed, path.name)

    # Corpus changed -> previously cached answers for this group may be outdated.
    try:
        cache.invalidate_group(clearance_group)
    except Exception:  # noqa: BLE001
        logger.exception("cache invalidation failed; stale answers may persist until TTL expiry")

    return len(docs)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Ingest a clinical PDF into the RAG index.")
    parser.add_argument("--file", required=True, help="path to the PDF")
    parser.add_argument("--group", required=True, help="clearance group that may read this document")
    args = parser.parse_args(argv)
    try:
        n = process_and_ingest_pdf(args.file, args.group)
    except Exception:
        logger.exception("ingestion failed")
        return 1
    logger.info("ingestion complete: %d chunks", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
