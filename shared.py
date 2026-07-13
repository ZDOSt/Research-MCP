import hashlib
import ipaddress
import json
import logging
import os
import re
import time
import asyncio
import threading
import uuid
from collections import Counter
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastembed import TextEmbedding
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchAny,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    SetPayload,
    SetPayloadOperation,
    VectorParams,
    WriteOrdering,
)
from redaction import redact_sensitive_text
from url_identity import canonicalize_source_identity

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def safe_error_detail(value: object, limit: int = 2000) -> str:
    redacted, _ = redact_sensitive_text(str(value or ""))
    return redacted[:limit]


SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")
CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235")
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
RESEARCH_API_URL = os.getenv("RESEARCH_API_URL", "")
RESEARCH_API_TOKEN = os.getenv("RESEARCH_API_TOKEN", "")
RESEARCH_API_MAX_RESPONSE_BYTES = max(
    1024,
    int(os.getenv("RESEARCH_API_MAX_RESPONSE_BYTES", "8388608")),
)
RESEARCH_API_TOTAL_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("RESEARCH_API_TOTAL_TIMEOUT_SECONDS", "180")),
)
RESEARCH_API_ALLOW_INSECURE_HTTP = os.getenv(
    "RESEARCH_API_ALLOW_INSECURE_HTTP", "false"
).lower() in {"1", "true", "yes", "on"}
RAG_MAX_INGEST_CHARS = max(
    1_000,
    int(os.getenv("RAG_MAX_INGEST_CHARS", "1000000")),
)
RAG_MAX_METADATA_BYTES = max(
    1_024,
    int(os.getenv("RAG_MAX_METADATA_BYTES", "262144")),
)
RERANKER_MAX_RESPONSE_BYTES = max(
    1024,
    int(os.getenv("RERANKER_MAX_RESPONSE_BYTES", "2097152")),
)
RERANKER_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("RERANKER_TIMEOUT_SECONDS", "20")),
)
RERANKER_TOTAL_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("RERANKER_TOTAL_TIMEOUT_SECONDS", "30")),
)
USE_RESEARCH_API_RAG = os.getenv("USE_RESEARCH_API_RAG", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "librechat_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")
DEFAULT_NAMESPACE = os.getenv("DEFAULT_RESEARCH_NAMESPACE", "default")


def _load_research_timezone(value: str) -> ZoneInfo:
    timezone_name = value.strip() or "UTC"
    try:
        return ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(
            "RESEARCH_TIMEZONE must be a valid IANA timezone name, such as "
            "UTC or America/New_York"
        ) from exc


RESEARCH_TIMEZONE_NAME = os.getenv("RESEARCH_TIMEZONE", "UTC").strip() or "UTC"
RESEARCH_TIMEZONE = _load_research_timezone(RESEARCH_TIMEZONE_NAME)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
if not 100 <= CHUNK_SIZE <= 100_000:
    raise ValueError("CHUNK_SIZE must be between 100 and 100000")
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE")

VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "384"))
QDRANT_INIT_RETRIES = int(os.getenv("QDRANT_INIT_RETRIES", "30"))
QDRANT_INIT_DELAY_SECONDS = float(os.getenv("QDRANT_INIT_DELAY_SECONDS", "2"))
QDRANT_HISTORY_RETENTION_SECONDS = int(
    os.getenv("QDRANT_HISTORY_RETENTION_SECONDS", "2592000")
)
if VECTOR_SIZE <= 0:
    raise ValueError("VECTOR_SIZE must be positive")
if QDRANT_INIT_RETRIES <= 0 or QDRANT_INIT_DELAY_SECONDS < 0:
    raise ValueError(
        "QDRANT initialization retries must be positive and delay non-negative"
    )
if QDRANT_HISTORY_RETENTION_SECONDS < 0:
    raise ValueError("QDRANT_HISTORY_RETENTION_SECONDS must be non-negative")

_qdrant_client: Optional[QdrantClient] = None
_embedder: Optional[TextEmbedding] = None
_client_lock = threading.Lock()
_qdrant_init_lock = threading.Lock()
_ingestion_order_lock = threading.Lock()
_qdrant_ready = False
_last_ingestion_order_ns = 0
_INGESTION_TOMBSTONE_RECORD_TYPE = "ingestion_attempt_tombstone"
IngestionCommitGuard = Callable[[], Awaitable[bool]]
_ingestion_commit_guard: ContextVar[Optional[IngestionCommitGuard]] = ContextVar(
    "ingestion_commit_guard",
    default=None,
)

PAYLOAD_INDEXES = {
    "namespace": PayloadSchemaType.KEYWORD,
    "source": PayloadSchemaType.KEYWORD,
    "source_identity": PayloadSchemaType.KEYWORD,
    "url": PayloadSchemaType.KEYWORD,
    "research_run_id": PayloadSchemaType.KEYWORD,
    "source_version": PayloadSchemaType.KEYWORD,
    "ingestion_id": PayloadSchemaType.KEYWORD,
    "ingestion_attempt_id": PayloadSchemaType.KEYWORD,
    "ingestion_order_ns": PayloadSchemaType.INTEGER,
    "ingestion_committed": PayloadSchemaType.BOOL,
    "is_latest_version": PayloadSchemaType.BOOL,
    "lifecycle_status": PayloadSchemaType.KEYWORD,
}


class QdrantSchemaError(RuntimeError):
    """Raised when an existing collection cannot store this application's vectors."""


class StrictRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestRequest(StrictRequestModel):
    text: str = Field(min_length=1, max_length=RAG_MAX_INGEST_CHARS)
    metadata: Optional[Dict[str, Any]] = Field(default=None, max_length=100)

    @field_validator("metadata")
    @classmethod
    def validate_metadata_size(
        cls,
        value: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        if len(encoded) > RAG_MAX_METADATA_BYTES:
            raise ValueError(
                f"metadata exceeds the {RAG_MAX_METADATA_BYTES}-byte limit"
            )
        return value


class QueryRequest(StrictRequestModel):
    query: str = Field(min_length=1, max_length=8_000)
    top_k: int = Field(default=5, ge=1, le=30)
    namespace: str = Field(default=DEFAULT_NAMESPACE, min_length=1, max_length=200)
    research_run_id: Optional[str] = Field(default=None, max_length=200)
    ingestion_attempt_id: Optional[str] = Field(default=None, max_length=128)


class SourceDeleteRequest(StrictRequestModel):
    source: str = Field(min_length=1, max_length=4_096)
    namespace: str = Field(default=DEFAULT_NAMESPACE, min_length=1, max_length=200)


class SourceListRequest(StrictRequestModel):
    limit: int = Field(default=50, ge=1, le=500)
    namespace: str = Field(default=DEFAULT_NAMESPACE, min_length=1, max_length=200)


class IngestionAttemptInvalidationRequest(StrictRequestModel):
    ingestion_attempt_id: str = Field(min_length=1, max_length=128)
    reason: str = Field(default="remote_worker_abandoned", min_length=1, max_length=128)


def normalize_namespace(value: Optional[str]) -> str:
    value = (value or DEFAULT_NAMESPACE).strip()
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return value[:100] or DEFAULT_NAMESPACE


def _validated_ingestion_attempt_id(metadata: Dict[str, Any]) -> Optional[str]:
    supplied_attempt_id = metadata.get("ingestion_attempt_id")
    if supplied_attempt_id is not None and not isinstance(supplied_attempt_id, str):
        raise ValueError("ingestion_attempt_id must be a string")
    ingestion_attempt_id = (supplied_attempt_id or "").strip() or None
    if ingestion_attempt_id and len(ingestion_attempt_id) > 128:
        raise ValueError("ingestion_attempt_id must contain at most 128 characters")
    return ingestion_attempt_id


def set_ingestion_commit_guard(guard: IngestionCommitGuard) -> Token:
    if not callable(guard):
        raise TypeError("ingestion commit guard must be callable")
    return _ingestion_commit_guard.set(guard)


def reset_ingestion_commit_guard(token: Token) -> None:
    _ingestion_commit_guard.reset(token)


def _ingestion_attempt_tombstone_id(ingestion_attempt_id: str) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"research-mcp:{COLLECTION_NAME}:invalid-attempt:{ingestion_attempt_id}",
        )
    )


async def _ingestion_attempt_is_invalidated(ingestion_attempt_id: str) -> bool:
    client = get_qdrant_client()

    def retrieve() -> bool:
        records = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[_ingestion_attempt_tombstone_id(ingestion_attempt_id)],
            with_payload=False,
            with_vectors=False,
        )
        return bool(records)

    return await asyncio.to_thread(retrieve)


async def _assert_ingestion_commit_allowed(
    ingestion_attempt_id: Optional[str] = None,
) -> None:
    guard = _ingestion_commit_guard.get()
    if guard is not None:
        try:
            allowed = await guard()
        except Exception as exc:
            raise RuntimeError("ingestion lease validation failed") from exc
        if allowed is not True:
            raise RuntimeError("ingestion lease is no longer valid")
    if ingestion_attempt_id and await _ingestion_attempt_is_invalidated(
        ingestion_attempt_id
    ):
        raise RuntimeError("ingestion attempt has been invalidated")


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _client_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        with _client_lock:
            if _embedder is None:
                logger.info("Loading embedding model '%s'.", EMBEDDING_MODEL)
                _embedder = TextEmbedding(
                    model_name=EMBEDDING_MODEL, cache_dir=MODEL_CACHE_DIR
                )
    return _embedder


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def next_ingestion_order_ns() -> int:
    """Return a process-monotonic, Qdrant-compatible signed 64-bit order value."""
    global _last_ingestion_order_ns
    with _ingestion_order_lock:
        value = max(time.time_ns(), _last_ingestion_order_ns + 1)
        if value > 2**63 - 1:
            raise RuntimeError(
                "System time exceeds Qdrant's signed 64-bit integer range"
            )
        _last_ingestion_order_ns = value
        return value


def runtime_retrieval_context() -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    local_now = now.astimezone(RESEARCH_TIMEZONE)
    return {
        "retrieved_at_utc": now.isoformat(),
        "current_date_utc": now.date().isoformat(),
        "timezone": RESEARCH_TIMEZONE_NAME,
        "retrieved_at_local": local_now.isoformat(),
        "current_date_local": local_now.date().isoformat(),
        "freshness": "runtime_retrieved",
        "guidance": (
            "This MCP result was retrieved or queried at server runtime. "
            "Interpret relative dates using current_date_local and timezone. "
            "Information dated after the answering model's training cutoff can be valid; "
            "do not discard it solely because it is newer than the model cutoff."
        ),
    }


def normalize_url(url: str) -> str:
    return canonicalize_source_identity(url)


def get_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain[4:] if domain.startswith("www.") else domain
    except Exception:
        return ""


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _looks_not_found(exc: Exception) -> bool:
    message = str(exc).lower()
    return "not found" in message or "404" in message


def _looks_already_exists(exc: Exception) -> bool:
    message = str(exc).lower()
    return "already exists" in message or "409" in message


def _research_api_is_internal(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.username is not None or parsed.password is not None:
            return False
        host = (parsed.hostname or "").rstrip(".").lower()
        if not host:
            return False
        if host in {"localhost", "localhost.localdomain"} or host.endswith(
            (".local", ".internal")
        ):
            return True
        if "." not in host:
            return True
        try:
            return not ipaddress.ip_address(host).is_global
        except ValueError:
            return False
    except (TypeError, ValueError):
        return False


def _remote_rag_headers() -> Dict[str, str]:
    try:
        parsed = urlparse(RESEARCH_API_URL)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Invalid RESEARCH_API_URL: {safe_error_detail(exc)}",
        ) from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(
            status_code=503,
            detail="RESEARCH_API_URL must be an http(s) URL with a hostname",
        )
    if parsed.username is not None or parsed.password is not None:
        raise HTTPException(
            status_code=503, detail="RESEARCH_API_URL must not contain URL credentials"
        )
    if (
        parsed.scheme == "http"
        and not _research_api_is_internal(RESEARCH_API_URL)
        and not RESEARCH_API_ALLOW_INSECURE_HTTP
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Public RESEARCH_API_URL endpoints must use HTTPS; set "
                "RESEARCH_API_ALLOW_INSECURE_HTTP=true only for a trusted endpoint"
            ),
        )
    if RESEARCH_API_TOKEN:
        return {"Authorization": f"Bearer {RESEARCH_API_TOKEN}"}
    if not _research_api_is_internal(RESEARCH_API_URL):
        raise HTTPException(
            status_code=503,
            detail=(
                "Remote RAG is configured without RESEARCH_API_TOKEN. "
                "Unauthenticated remote RAG is allowed only for loopback or internal service URLs."
            ),
        )
    return {}


async def _remote_rag_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not RESEARCH_API_URL:
        raise HTTPException(
            status_code=503,
            detail="Remote RAG is enabled but RESEARCH_API_URL is empty",
        )

    try:
        async with asyncio.timeout(RESEARCH_API_TOTAL_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                timeout=180.0, headers=_remote_rag_headers()
            ) as client:
                async with client.stream(
                    method,
                    f"{RESEARCH_API_URL.rstrip('/')}{path}",
                    json=json_body,
                    params=params,
                ) as response:
                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            declared_length = int(content_length)
                        except ValueError as exc:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Remote RAG returned an invalid Content-Length for {path}",
                            ) from exc
                        if declared_length > RESEARCH_API_MAX_RESPONSE_BYTES:
                            raise HTTPException(
                                status_code=502,
                                detail=(
                                    f"Remote RAG response exceeds RESEARCH_API_MAX_RESPONSE_BYTES="
                                    f"{RESEARCH_API_MAX_RESPONSE_BYTES} for {path}"
                                ),
                            )

                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > RESEARCH_API_MAX_RESPONSE_BYTES:
                            raise HTTPException(
                                status_code=502,
                                detail=(
                                    f"Remote RAG response exceeds RESEARCH_API_MAX_RESPONSE_BYTES="
                                    f"{RESEARCH_API_MAX_RESPONSE_BYTES} for {path}"
                                ),
                            )
                        body.extend(chunk)

        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Remote RAG returned invalid JSON for {path}",
            ) from exc
    except HTTPException:
        raise
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Remote RAG request exceeded its total deadline for {path}",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Remote RAG request failed with HTTP {exc.response.status_code}: {path}",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Remote RAG transport failed for {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=f"Remote RAG returned a non-object response for {path}",
        )
    return payload


async def _compensate_remote_ingestion_attempt(
    ingestion_attempt_id: Optional[str],
    *,
    reason: str,
) -> None:
    if not ingestion_attempt_id:
        return
    try:
        await _remote_rag_request(
            "POST",
            "/rag/invalidate-attempt",
            json_body={
                "ingestion_attempt_id": ingestion_attempt_id,
                "reason": reason,
            },
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.error(
            "Remote RAG attempt compensation failed for %s: %s",
            ingestion_attempt_id[:16],
            type(exc).__name__,
        )


def _schema_value(value: Any) -> str:
    value = getattr(value, "value", value)
    return str(value or "").strip().lower()


def _validate_collection_schema(collection: Any) -> None:
    config = getattr(collection, "config", None)
    params = getattr(config, "params", None)
    vectors = getattr(params, "vectors", None)
    if vectors is None and isinstance(collection, dict):
        vectors = collection.get("config", {}).get("params", {}).get("vectors")

    if isinstance(vectors, dict):
        if "size" in vectors and "distance" in vectors:
            vector_size = vectors.get("size")
            vector_distance = vectors.get("distance")
        else:
            raise QdrantSchemaError(
                f"Collection '{COLLECTION_NAME}' uses named vectors, but Research MCP requires one unnamed vector."
            )
    else:
        vector_size = getattr(vectors, "size", None)
        vector_distance = getattr(vectors, "distance", None)

    if vector_size != VECTOR_SIZE or _schema_value(vector_distance) != _schema_value(
        Distance.COSINE
    ):
        raise QdrantSchemaError(
            f"Collection '{COLLECTION_NAME}' vector schema is size={vector_size}, distance={vector_distance}; "
            f"expected size={VECTOR_SIZE}, distance={Distance.COSINE.value}. Recreate the collection or fix configuration."
        )


def _payload_index_type(value: Any) -> str:
    return _schema_value(getattr(value, "data_type", value))


def _ensure_payload_indexes(client: QdrantClient, collection: Any) -> None:
    payload_schema = getattr(collection, "payload_schema", None) or {}
    for field_name, expected_type in PAYLOAD_INDEXES.items():
        existing = payload_schema.get(field_name)
        if existing is not None:
            if _payload_index_type(existing) != _schema_value(expected_type):
                raise QdrantSchemaError(
                    f"Collection '{COLLECTION_NAME}' payload index '{field_name}' has type "
                    f"{_payload_index_type(existing)!r}; expected {_schema_value(expected_type)!r}."
                )
            continue
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=expected_type,
                wait=True,
            )
        except Exception as exc:
            if not _looks_already_exists(exc):
                raise


def init_qdrant() -> None:
    global _qdrant_ready
    if _qdrant_ready:
        return

    last_error = None
    client = get_qdrant_client()

    with _qdrant_init_lock:
        if _qdrant_ready:
            return

        for attempt in range(1, QDRANT_INIT_RETRIES + 1):
            try:
                collection = client.get_collection(COLLECTION_NAME)
                _validate_collection_schema(collection)
                _ensure_payload_indexes(client, collection)
                logger.info("Collection '%s' exists.", COLLECTION_NAME)
                _qdrant_ready = True
                return
            except QdrantSchemaError:
                raise
            except Exception as exc:
                last_error = exc

                if _looks_not_found(exc):
                    try:
                        client.create_collection(
                            collection_name=COLLECTION_NAME,
                            vectors_config=VectorParams(
                                size=VECTOR_SIZE, distance=Distance.COSINE
                            ),
                        )
                        collection = client.get_collection(COLLECTION_NAME)
                        _validate_collection_schema(collection)
                        _ensure_payload_indexes(client, collection)
                        logger.info("Created collection '%s'.", COLLECTION_NAME)
                        _qdrant_ready = True
                        return
                    except QdrantSchemaError:
                        raise
                    except Exception as create_exc:
                        if _looks_already_exists(create_exc):
                            collection = client.get_collection(COLLECTION_NAME)
                            _validate_collection_schema(collection)
                            _ensure_payload_indexes(client, collection)
                            logger.info(
                                "Collection '%s' was created by another process.",
                                COLLECTION_NAME,
                            )
                            _qdrant_ready = True
                            return
                        last_error = create_exc

                logger.warning(
                    "Qdrant init attempt %s/%s failed: %s",
                    attempt,
                    QDRANT_INIT_RETRIES,
                    last_error,
                )
                if attempt < QDRANT_INIT_RETRIES:
                    time.sleep(QDRANT_INIT_DELAY_SECONDS)

        raise RuntimeError(
            f"Qdrant initialization failed after {QDRANT_INIT_RETRIES} attempts: {last_error}"
        )


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def split_markdown_sections(text: str) -> List[Dict[str, str]]:
    text = clean_text(text)
    if not text:
        return []

    lines = text.splitlines()
    sections = []
    current_heading = "Document"
    current_lines = []

    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    for line in lines:
        match = heading_re.match(line)
        if match and current_lines:
            sections.append(
                {
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                }
            )
            current_heading = match.group(2).strip()
            current_lines = [line]
        elif match:
            current_heading = match.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading": current_heading,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]


def split_long_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    text = clean_text(text)
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)

        if end < len(text):
            sentence_boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind("\n\n", start, end),
            )

            if sentence_boundary > start + int(chunk_size * 0.55):
                end = sentence_boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    sections = split_markdown_sections(text)
    chunks = []

    for section_index, section in enumerate(sections):
        section_chunks = split_long_text(section["text"], chunk_size, overlap)

        for section_chunk_index, chunk in enumerate(section_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "section": section["heading"],
                    "section_index": section_index,
                    "section_chunk_index": section_chunk_index,
                }
            )

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    return [list(vec) for vec in get_embedder().embed(texts)]


async def embed_texts_async(texts: List[str]) -> List[List[float]]:
    return await asyncio.to_thread(embed_texts, texts)


def point_id_for(
    source: str,
    chunk_index: int,
    namespace: str,
    source_version: str,
    research_run_id: Optional[str] = None,
    ingestion_id: Optional[str] = None,
) -> str:
    research_run_id = (research_run_id or "").strip()
    run_scope = f"run:{research_run_id}" if research_run_id else "memory"
    attempt_scope = f":ingestion:{ingestion_id.strip()}" if ingestion_id else ""
    identity = (
        f"{normalize_namespace(namespace)}:{source}:{source_version}:"
        f"{run_scope}{attempt_scope}:{chunk_index}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def qdrant_query_points(
    query_vec: List[float], limit: int, query_filter: Optional[Filter] = None
):
    client = get_qdrant_client()
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return getattr(response, "points", response)

    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        query_filter=query_filter,
        limit=limit,
    )


async def qdrant_query_points_async(
    query_vec: List[float],
    limit: int,
    query_filter: Optional[Filter] = None,
):
    return await asyncio.to_thread(qdrant_query_points, query_vec, limit, query_filter)


async def qdrant_upsert_async(points: List[PointStruct]) -> None:
    client = get_qdrant_client()
    await asyncio.to_thread(
        client.upsert,
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
        ordering=WriteOrdering.STRONG,
    )


def _source_filter(namespace: str, source: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="namespace", match=MatchValue(value=namespace)),
            FieldCondition(key="source", match=MatchValue(value=source)),
        ]
    )


def _scroll_all_matching(
    client: QdrantClient,
    query_filter: Optional[Filter],
    *,
    max_points: Optional[int] = None,
) -> tuple[List[Any], bool]:
    records: List[Any] = []
    offset = None
    page_size = 1000

    while max_points is None or len(records) < max_points:
        limit = (
            page_size
            if max_points is None
            else min(page_size, max_points - len(records))
        )
        if limit <= 0:
            break
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            offset=offset,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(points)
        if offset is None or not points:
            return records, False

    return records, offset is not None


def _scroll_matching_batch(
    client: QdrantClient,
    query_filter: Optional[Filter],
    *,
    max_points: int,
    cursor: Any = None,
) -> tuple[List[Any], Any]:
    """Read one bounded maintenance batch and return its resume cursor."""
    records: List[Any] = []
    offset = cursor
    page_size = 1000

    while len(records) < max_points:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=min(page_size, max_points - len(records)),
            offset=offset,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(points)
        if offset is None or not points:
            break

    return records, offset


def _valid_ingestion_key(payload: Dict[str, Any]) -> Optional[tuple[int, str]]:
    ingestion_id = payload.get("ingestion_id")
    ingestion_order_ns = payload.get("ingestion_order_ns")
    if (
        not isinstance(ingestion_id, str)
        or not ingestion_id
        or isinstance(ingestion_order_ns, bool)
        or not isinstance(ingestion_order_ns, int)
        or not 0 < ingestion_order_ns <= 2**63 - 1
    ):
        return None
    return ingestion_order_ns, ingestion_id


def _complete_committed_ingestions(
    records: List[Any],
) -> tuple[Dict[tuple[int, str], List[Any]], List[Any]]:
    grouped: Dict[tuple[int, str], List[Any]] = {}
    invalid: List[Any] = []

    for record in records:
        payload = record.payload or {}
        if payload.get("ingestion_committed") is not True:
            continue
        key = _valid_ingestion_key(payload)
        if key is None:
            invalid.append(record)
            continue
        grouped.setdefault(key, []).append(record)

    complete: Dict[tuple[int, str], List[Any]] = {}
    for key, ingestion_records in grouped.items():
        payloads = [record.payload or {} for record in ingestion_records]
        expected_values = {payload.get("ingestion_chunk_count") for payload in payloads}
        chunk_indexes = {payload.get("chunk_index") for payload in payloads}
        source_versions = {payload.get("source_version") for payload in payloads}
        expected = next(iter(expected_values)) if len(expected_values) == 1 else None
        is_complete = (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected > 0
            and len(ingestion_records) == expected
            and chunk_indexes == set(range(expected))
            and len(source_versions) == 1
            and None not in source_versions
        )
        if is_complete:
            complete[key] = ingestion_records
        else:
            invalid.extend(ingestion_records)

    return complete, invalid


def _payload_is_active(payload: Dict[str, Any]) -> bool:
    committed = payload.get("ingestion_committed")
    if committed is False:
        return False
    if committed is True:
        return (
            payload.get("is_latest_version") is True
            and payload.get("lifecycle_status") == "active"
        )
    return (
        not payload.get("ingestion_id")
        and payload.get("is_latest_version") is not False
        and payload.get("lifecycle_status")
        not in {"pending", "committed_pending_reconciliation", "superseded", "invalid"}
    )


def _retrievable_lifecycle_filter() -> Filter:
    """Match modern active points plus compatible pre-lifecycle records."""
    return Filter(
        should=[
            Filter(
                must=[
                    FieldCondition(
                        key="ingestion_committed", match=MatchValue(value=True)
                    ),
                    FieldCondition(
                        key="is_latest_version", match=MatchValue(value=True)
                    ),
                    FieldCondition(
                        key="lifecycle_status", match=MatchValue(value="active")
                    ),
                ]
            ),
            Filter(
                must=[
                    IsEmptyCondition(is_empty=PayloadField(key="ingestion_committed")),
                    IsEmptyCondition(is_empty=PayloadField(key="ingestion_id")),
                    Filter(
                        should=[
                            FieldCondition(
                                key="is_latest_version", match=MatchValue(value=True)
                            ),
                            IsEmptyCondition(
                                is_empty=PayloadField(key="is_latest_version")
                            ),
                        ]
                    ),
                ],
                must_not=[
                    FieldCondition(
                        key="lifecycle_status",
                        match=MatchAny(
                            any=[
                                "pending",
                                "committed_pending_reconciliation",
                                "superseded",
                                "invalid",
                            ]
                        ),
                    )
                ],
            ),
        ]
    )


async def reconcile_source_versions_async(
    *,
    namespace: str,
    source: str,
    reconciled_at: Optional[str] = None,
    max_rounds: int = 4,
) -> Dict[str, Any]:
    """Converge one source to exactly one complete committed ingestion."""
    namespace = normalize_namespace(namespace)
    source = normalize_url(source)
    if not source:
        raise ValueError("source is required")
    reconciled_at = reconciled_at or utc_now_iso()
    reconciled_at_unix = int(time.time())
    client = get_qdrant_client()

    def reconcile() -> Dict[str, Any]:
        for _round in range(max(1, max_rounds)):
            records, _ = _scroll_all_matching(client, _source_filter(namespace, source))
            complete, invalid_records = _complete_committed_ingestions(records)
            if not complete:
                invalid_ids = [
                    record.id
                    for record in invalid_records
                    if (record.payload or {}).get("lifecycle_status") != "invalid"
                    or not (record.payload or {}).get("superseded_at_unix")
                ]
                if invalid_ids:
                    client.set_payload(
                        collection_name=COLLECTION_NAME,
                        payload={
                            "is_latest_version": False,
                            "lifecycle_status": "invalid",
                            "superseded_at": reconciled_at,
                            "superseded_at_unix": reconciled_at_unix,
                            "superseded_reason": "incomplete_or_invalid_committed_ingestion",
                        },
                        points=invalid_ids,
                        wait=True,
                        ordering=WriteOrdering.STRONG,
                    )
                return {
                    "is_latest_version": False,
                    "lifecycle_status": "no_complete_committed_ingestion",
                    "winner_ingestion_id": None,
                    "winner_ingestion_order_ns": None,
                }

            winner_key = max(complete)
            winner_order_ns, winner_id = winner_key
            winner_records = complete[winner_key]
            winner_point_ids = {record.id for record in winner_records}
            demote_ids = []
            invalid_ids = []

            for record in records:
                payload = record.payload or {}
                if record.id in winner_point_ids:
                    continue
                if payload.get("ingestion_committed") is False:
                    continue
                if payload.get("ingestion_committed") is True:
                    if record in invalid_records:
                        if payload.get(
                            "lifecycle_status"
                        ) != "invalid" or not payload.get("superseded_at_unix"):
                            invalid_ids.append(record.id)
                    elif (
                        payload.get("lifecycle_status") != "superseded"
                        or _payload_is_active(payload)
                        or not payload.get("superseded_at_unix")
                    ):
                        demote_ids.append(record.id)
                elif _payload_is_active(payload) or not payload.get(
                    "superseded_at_unix"
                ):
                    demote_ids.append(record.id)

            winner_needs_promotion = any(
                not _payload_is_active(record.payload or {})
                or (record.payload or {}).get("lifecycle_status") != "active"
                for record in winner_records
            )
            operations = []
            superseded_payload = {
                "is_latest_version": False,
                "lifecycle_status": "superseded",
                "superseded_at": reconciled_at,
                "superseded_at_unix": reconciled_at_unix,
                "superseded_by_ingestion_id": winner_id,
                "superseded_by_ingestion_order_ns": winner_order_ns,
            }
            if winner_needs_promotion:
                # Filter evaluation happens when Qdrant applies the batch. It
                # therefore also demotes an active ingestion promoted by a
                # concurrent reconciler after our scan but before this write.
                operations.append(
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload=superseded_payload,
                            filter=Filter(
                                must=_source_filter(namespace, source).must
                                + [_retrievable_lifecycle_filter()]
                            ),
                        )
                    )
                )
            if demote_ids:
                operations.append(
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload=superseded_payload, points=demote_ids
                        )
                    )
                )
            if invalid_ids:
                operations.append(
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload={
                                "is_latest_version": False,
                                "lifecycle_status": "invalid",
                                "superseded_at": reconciled_at,
                                "superseded_at_unix": reconciled_at_unix,
                                "superseded_reason": (
                                    "incomplete_or_invalid_committed_ingestion"
                                ),
                            },
                            points=invalid_ids,
                        )
                    )
                )
            if winner_needs_promotion:
                operations.append(
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload={
                                "is_latest_version": True,
                                "lifecycle_status": "active",
                                "activated_at": reconciled_at,
                                "superseded_at": None,
                                "superseded_at_unix": None,
                                "superseded_reason": None,
                                "superseded_by_ingestion_id": None,
                                "superseded_by_ingestion_order_ns": None,
                            },
                            points=list(winner_point_ids),
                        )
                    )
                )

            if operations:
                # Qdrant applies this as one ordered request. Every promotion is
                # preceded by a source-wide active demotion, so stale concurrent
                # reconcilers cannot leave two active committed versions.
                client.batch_update_points(
                    collection_name=COLLECTION_NAME,
                    update_operations=operations,
                    wait=True,
                    ordering=WriteOrdering.STRONG,
                )

            verified, _ = _scroll_all_matching(
                client, _source_filter(namespace, source)
            )
            verified_complete, _ = _complete_committed_ingestions(verified)
            if not verified_complete:
                continue
            verified_winner = max(verified_complete)
            active_ids = {
                record.id
                for record in verified
                if _payload_is_active(record.payload or {})
            }
            expected_active_ids = {
                record.id for record in verified_complete[verified_winner]
            }
            if verified_winner == winner_key and active_ids == expected_active_ids:
                return {
                    "is_latest_version": True,
                    "lifecycle_status": "active",
                    "winner_ingestion_id": winner_id,
                    "winner_ingestion_order_ns": winner_order_ns,
                }

        raise RuntimeError("Qdrant lifecycle reconciliation did not converge")

    return await asyncio.to_thread(reconcile)


async def supersede_source_versions_async(
    *,
    namespace: str,
    source: str,
    active_ingestion_id: str,
    active_source_version: str,
    active_ingestion_order_ns: int,
    superseded_at: str,
    active_ingestion_attempt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit a complete ingestion, then deterministically reconcile its source."""
    namespace = normalize_namespace(namespace)
    source = normalize_url(source)
    if not source:
        raise ValueError("source is required")
    client = get_qdrant_client()

    def commit() -> None:
        ingestion_filter = Filter(
            must=_source_filter(namespace, source).must
            + [
                FieldCondition(
                    key="ingestion_id", match=MatchValue(value=active_ingestion_id)
                )
            ]
        )
        records, _ = _scroll_all_matching(client, ingestion_filter)
        if not records:
            raise RuntimeError("Cannot commit an ingestion with no stored chunks")
        payloads = [record.payload or {} for record in records]
        expected_values = {payload.get("ingestion_chunk_count") for payload in payloads}
        expected = next(iter(expected_values)) if len(expected_values) == 1 else None
        chunk_indexes = {payload.get("chunk_index") for payload in payloads}
        valid = (
            isinstance(expected, int)
            and not isinstance(expected, bool)
            and expected > 0
            and len(records) == expected
            and chunk_indexes == set(range(expected))
            and all(
                payload.get("ingestion_order_ns") == active_ingestion_order_ns
                and payload.get("source_version") == active_source_version
                for payload in payloads
            )
        )
        if not valid:
            raise RuntimeError("Cannot commit an incomplete or inconsistent ingestion")
        if all(payload.get("ingestion_committed") is True for payload in payloads):
            return
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "ingestion_committed": True,
                "lifecycle_status": "committed_pending_reconciliation",
                "committed_at": utc_now_iso(),
                "committed_at_unix": int(time.time()),
            },
            points=[record.id for record in records],
            wait=True,
            ordering=WriteOrdering.STRONG,
        )

    def abandon() -> None:
        ingestion_filter = Filter(
            must=_source_filter(namespace, source).must
            + [
                FieldCondition(
                    key="ingestion_id", match=MatchValue(value=active_ingestion_id)
                ),
                FieldCondition(
                    key="ingestion_order_ns",
                    match=MatchValue(value=active_ingestion_order_ns),
                ),
            ]
        )
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "ingestion_committed": False,
                "is_latest_version": False,
                "lifecycle_status": "invalid",
                "superseded_at": utc_now_iso(),
                "superseded_at_unix": int(time.time()),
                "superseded_reason": "worker_lease_lost",
            },
            points=ingestion_filter,
            wait=True,
            ordering=WriteOrdering.STRONG,
        )

    async def abandon_and_reconcile() -> None:
        await asyncio.to_thread(abandon)
        await reconcile_source_versions_async(
            namespace=namespace,
            source=source,
        )

    await _assert_ingestion_commit_allowed(active_ingestion_attempt_id)
    commit_task = asyncio.create_task(asyncio.to_thread(commit))
    try:
        await asyncio.shield(commit_task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(commit_task)
        finally:
            await asyncio.shield(asyncio.to_thread(abandon))
        raise
    except Exception:
        await asyncio.shield(asyncio.to_thread(abandon))
        raise

    try:
        await _assert_ingestion_commit_allowed(active_ingestion_attempt_id)
    except asyncio.CancelledError:
        await asyncio.shield(asyncio.to_thread(abandon))
        raise
    except Exception:
        await asyncio.shield(asyncio.to_thread(abandon))
        raise

    reconcile_task = asyncio.create_task(
        reconcile_source_versions_async(
            namespace=namespace,
            source=source,
            reconciled_at=superseded_at,
        )
    )
    try:
        reconciliation = await asyncio.shield(reconcile_task)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(reconcile_task)
        except Exception:
            pass
        await asyncio.shield(abandon_and_reconcile())
        raise
    except Exception:
        await asyncio.shield(abandon_and_reconcile())
        raise

    try:
        await _assert_ingestion_commit_allowed(active_ingestion_attempt_id)
    except asyncio.CancelledError:
        await asyncio.shield(abandon_and_reconcile())
        raise
    except Exception:
        await asyncio.shield(abandon_and_reconcile())
        raise

    is_winner = reconciliation.get("winner_ingestion_id") == active_ingestion_id
    return {
        **reconciliation,
        "is_latest_version": is_winner,
        "lifecycle_status": "active" if is_winner else "superseded",
    }


async def invalidate_ingestion_attempt_async(
    ingestion_attempt_id: str,
    *,
    reason: str,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """Make every ingestion from a failed worker attempt non-retrievable."""
    if not isinstance(ingestion_attempt_id, str):
        raise TypeError("ingestion_attempt_id must be a string")
    ingestion_attempt_id = ingestion_attempt_id.strip()
    if not ingestion_attempt_id or len(ingestion_attempt_id) > 128:
        raise ValueError("ingestion_attempt_id must contain 1 to 128 characters")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    reason = reason.strip()[:128]
    normalized_namespace = (
        normalize_namespace(namespace) if namespace is not None else None
    )
    await asyncio.to_thread(init_qdrant)
    client = get_qdrant_client()
    must = [
        FieldCondition(
            key="ingestion_attempt_id",
            match=MatchValue(value=ingestion_attempt_id),
        )
    ]
    if normalized_namespace is not None:
        must.append(
            FieldCondition(
                key="namespace",
                match=MatchValue(value=normalized_namespace),
            )
        )
    attempt_filter = Filter(must=must)

    def invalidate() -> tuple[int, set[tuple[str, str]]]:
        invalidated_at = utc_now_iso()
        invalidated_at_unix = int(time.time())
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=_ingestion_attempt_tombstone_id(ingestion_attempt_id),
                    vector=[0.0] * VECTOR_SIZE,
                    payload={
                        "record_type": _INGESTION_TOMBSTONE_RECORD_TYPE,
                        "namespace": "__research_mcp_internal__",
                        "ingestion_attempt_id": ingestion_attempt_id,
                        "ingestion_committed": False,
                        "is_latest_version": False,
                        "lifecycle_status": "invalid",
                        "superseded_at": invalidated_at,
                        "superseded_at_unix": invalidated_at_unix,
                        "superseded_reason": reason,
                    },
                )
            ],
            wait=True,
            ordering=WriteOrdering.STRONG,
        )
        records, _ = _scroll_all_matching(client, attempt_filter)
        ingestion_records = [
            record
            for record in records
            if (record.payload or {}).get("record_type")
            != _INGESTION_TOMBSTONE_RECORD_TYPE
        ]
        sources = {
            (normalize_namespace((record.payload or {}).get("namespace")), source)
            for record in ingestion_records
            if (
                source := normalize_url(
                    (record.payload or {}).get("source")
                    or (record.payload or {}).get("url")
                )
            )
        }
        if ingestion_records:
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={
                    "ingestion_committed": False,
                    "is_latest_version": False,
                    "lifecycle_status": "invalid",
                    "superseded_at": invalidated_at,
                    "superseded_at_unix": invalidated_at_unix,
                    "superseded_reason": reason,
                },
                points=attempt_filter,
                wait=True,
                ordering=WriteOrdering.STRONG,
            )
        return len(ingestion_records), sources

    invalidated, sources = await asyncio.to_thread(invalidate)
    reconciled = 0
    for source_namespace, source in sorted(sources):
        await reconcile_source_versions_async(
            namespace=source_namespace,
            source=source,
        )
        reconciled += 1
    return {
        "ingestion_attempt_id": ingestion_attempt_id,
        "invalidated": invalidated,
        "sources_reconciled": reconciled,
    }


async def invalidate_ingestion_attempt_impl(
    ingestion_attempt_id: str,
    *,
    reason: str,
) -> Dict[str, Any]:
    if USE_RESEARCH_API_RAG:
        return await _remote_rag_request(
            "POST",
            "/rag/invalidate-attempt",
            json_body={
                "ingestion_attempt_id": ingestion_attempt_id,
                "reason": reason,
            },
        )
    return await invalidate_ingestion_attempt_async(
        ingestion_attempt_id,
        reason=reason,
    )


async def rerank_docs(
    query: str, docs: List[Dict[str, Any]], top_k: int
) -> List[Dict[str, Any]]:
    if not docs:
        return []

    texts = [doc["text"] for doc in docs]

    try:
        async with asyncio.timeout(RERANKER_TOTAL_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(timeout=RERANKER_TIMEOUT_SECONDS) as client:
                async with client.stream(
                    "POST",
                    f"{RERANKER_URL.rstrip('/')}/rerank",
                    json={"query": query, "texts": texts},
                ) as response:
                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length:
                        declared_length = int(content_length)
                        if declared_length > RERANKER_MAX_RESPONSE_BYTES:
                            raise ValueError(
                                "Reranker response exceeded RERANKER_MAX_RESPONSE_BYTES"
                            )
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > RERANKER_MAX_RESPONSE_BYTES:
                            raise ValueError(
                                "Reranker response exceeded RERANKER_MAX_RESPONSE_BYTES"
                            )
                        body.extend(chunk)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Reranker returned invalid JSON") from exc

        scored_by_index: Dict[int, Dict[str, Any]] = {}

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue

                index = item.get("index")
                score = item.get("score", 0)

                if isinstance(index, int) and 0 <= index < len(docs):
                    doc = dict(docs[index])
                    doc["rerank_score"] = score
                    scored_by_index[index] = doc

        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            for item in payload["results"]:
                if not isinstance(item, dict):
                    continue

                index = item.get("index")
                score = item.get("score", 0)
                text = item.get("text")

                if isinstance(index, int) and 0 <= index < len(docs):
                    doc = dict(docs[index])
                    doc["rerank_score"] = score
                    scored_by_index[index] = doc
                elif text:
                    for doc_index, doc in enumerate(docs):
                        if doc["text"] == text:
                            ranked_doc = dict(doc)
                            ranked_doc["rerank_score"] = score
                            scored_by_index[doc_index] = ranked_doc
                            break

        if len(scored_by_index) == len(docs):
            scored_docs = list(scored_by_index.values())
            scored_docs.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
            return scored_docs[:top_k]
        if scored_by_index:
            logger.warning(
                "Reranker returned %d/%d unique documents; using vector order.",
                len(scored_by_index),
                len(docs),
            )

    except Exception as exc:
        logger.warning(
            "Reranker failed, using vector order: %s", safe_error_detail(exc)
        )

    return docs[:top_k]


async def rag_ingest_impl(req: IngestRequest) -> Dict[str, Any]:
    if USE_RESEARCH_API_RAG:
        ingestion_attempt_id = _validated_ingestion_attempt_id(req.metadata or {})
        guarded_worker_ingestion = _ingestion_commit_guard.get() is not None
        if guarded_worker_ingestion and not ingestion_attempt_id:
            raise RuntimeError(
                "Leased remote worker ingestion requires an ingestion_attempt_id"
            )
        await _assert_ingestion_commit_allowed()
        try:
            result = await _remote_rag_request(
                "POST", "/rag/ingest", json_body=req.model_dump()
            )
        except asyncio.CancelledError:
            if guarded_worker_ingestion:
                await _compensate_remote_ingestion_attempt(
                    ingestion_attempt_id,
                    reason="remote_request_cancelled",
                )
            raise
        except Exception:
            if guarded_worker_ingestion:
                await _compensate_remote_ingestion_attempt(
                    ingestion_attempt_id,
                    reason="remote_request_outcome_unknown",
                )
            raise
        try:
            await _assert_ingestion_commit_allowed()
        except Exception:
            await _compensate_remote_ingestion_attempt(
                ingestion_attempt_id,
                reason="worker_lease_lost",
            )
            raise
        return result

    try:
        await asyncio.to_thread(init_qdrant)
        text = clean_text(req.text)
        if not text:
            return {"stored": 0}

        metadata = req.metadata or {}

        source = normalize_url(
            metadata.get("source") or metadata.get("url") or "unknown"
        )
        if not source:
            raise ValueError("source must be a valid URL or non-URL identifier")
        url = normalize_url(metadata.get("url") or source) or source
        requested_url = normalize_url(metadata.get("requested_url") or url) or url
        title = metadata.get("title")
        domain = metadata.get("domain") or get_domain(url)
        query = metadata.get("query")
        content_type = metadata.get("content_type", "webpage")
        source_score = metadata.get("source_score")
        source_reason = metadata.get("source_reason")
        retrieved_at_utc = metadata.get("retrieved_at_utc")
        retrieval_current_date_utc = metadata.get("retrieval_current_date_utc")
        namespace = normalize_namespace(metadata.get("namespace"))
        supplied_run_id = metadata.get("research_run_id")
        if supplied_run_id is not None and not isinstance(supplied_run_id, str):
            raise ValueError("research_run_id must be a string")
        research_run_id = (supplied_run_id or "").strip() or None
        ingestion_attempt_id = _validated_ingestion_attempt_id(metadata)
        if ingestion_attempt_id and await _ingestion_attempt_is_invalidated(
            ingestion_attempt_id
        ):
            raise RuntimeError("ingestion attempt has been invalidated")
        supplied_source_version = metadata.get("source_version")
        if supplied_source_version is None or supplied_source_version == "":
            source_version = hash_text(text)
        elif not isinstance(supplied_source_version, str):
            raise ValueError("source_version must be a string")
        else:
            source_version = supplied_source_version.strip()
            if not source_version or len(source_version) > 256:
                raise ValueError("source_version must contain 1 to 256 characters")
        supplied_ingestion_id = metadata.get("ingestion_id")
        if supplied_ingestion_id is None or supplied_ingestion_id == "":
            ingestion_id = uuid.uuid4().hex
        elif not isinstance(supplied_ingestion_id, str):
            raise ValueError("ingestion_id must be a string")
        else:
            ingestion_id = supplied_ingestion_id.strip()
            if not ingestion_id or len(ingestion_id) > 128:
                raise ValueError("ingestion_id must contain 1 to 128 characters")
        ingestion_order_ns = metadata.get("ingestion_order_ns")
        if ingestion_order_ns is None:
            ingestion_order_ns = next_ingestion_order_ns()
        if (
            isinstance(ingestion_order_ns, bool)
            or not isinstance(ingestion_order_ns, int)
            or not 0 < ingestion_order_ns <= 2**63 - 1
        ):
            raise ValueError(
                "ingestion_order_ns must be a positive signed 64-bit integer"
            )
        snapshot_id = metadata.get("snapshot_id") or str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{source}:{source_version}")
        )
        artifact_id = metadata.get("artifact_id")
        artifact_path = metadata.get("artifact_path")
        ingested_at = utc_now_iso()
        ingested_at_unix = int(time.time())

        chunks = chunk_text_with_metadata(text)

        if not chunks:
            return {"stored": 0, "source": source}

        vectors = await embed_texts_async([chunk["text"] for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"Embedding model returned {len(vectors)} vectors for {len(chunks)} chunks."
            )
        invalid_dimensions = [
            index for index, vector in enumerate(vectors) if len(vector) != VECTOR_SIZE
        ]
        if invalid_dimensions:
            raise ValueError(
                f"Embedding model returned an unexpected dimension for vector indexes "
                f"{invalid_dimensions[:10]}; VECTOR_SIZE={VECTOR_SIZE}. "
                "Set VECTOR_SIZE to match EMBEDDING_MODEL and recreate the collection."
            )

        points = []

        for index, (chunk, vec) in enumerate(zip(chunks, vectors)):
            chunk_text = chunk["text"]
            chunk_hash = hash_text(chunk_text)

            payload = {
                "text": chunk_text,
                "source": source,
                "source_identity": source,
                "url": url,
                "requested_url": requested_url,
                "domain": domain,
                "hash": chunk_hash,
                "chunk_index": index,
                "section": chunk.get("section"),
                "section_index": chunk.get("section_index"),
                "section_chunk_index": chunk.get("section_chunk_index"),
                "content_type": content_type,
                "namespace": namespace,
                "research_run_id": research_run_id,
                "ingestion_attempt_id": ingestion_attempt_id,
                "source_version": source_version,
                "ingestion_id": ingestion_id,
                "ingestion_order_ns": ingestion_order_ns,
                "ingestion_chunk_count": len(chunks),
                "ingestion_committed": False,
                "is_latest_version": False,
                "lifecycle_status": "pending",
                "snapshot_id": snapshot_id,
                "artifact_id": artifact_id,
                "artifact_path": artifact_path,
                "artifact_lifecycle": (
                    "retention_managed_independently" if artifact_path else None
                ),
                "ingested_at": ingested_at,
                "ingested_at_unix": ingested_at_unix,
                "retrieved_at_utc": retrieved_at_utc,
                "retrieval_current_date_utc": retrieval_current_date_utc,
            }

            if title:
                payload["title"] = title
            if query:
                payload["query"] = query
            if source_score is not None:
                payload["source_score"] = source_score
            if source_reason:
                payload["source_reason"] = source_reason

            points.append(
                PointStruct(
                    id=point_id_for(
                        source,
                        index,
                        namespace,
                        source_version,
                        research_run_id=research_run_id,
                        ingestion_id=ingestion_id,
                    ),
                    vector=vec,
                    payload=payload,
                )
            )

        await qdrant_upsert_async(points)
        reconciliation = await supersede_source_versions_async(
            namespace=namespace,
            source=source,
            active_ingestion_id=ingestion_id,
            active_source_version=source_version,
            active_ingestion_order_ns=ingestion_order_ns,
            superseded_at=ingested_at,
            active_ingestion_attempt_id=ingestion_attempt_id,
        )
        logger.info("Ingested %d chunks from %s.", len(points), source)

        return {
            "stored": len(points),
            "source": source,
            "url": url,
            "requested_url": requested_url,
            "title": title,
            "domain": domain,
            "namespace": namespace,
            "research_run_id": research_run_id,
            "ingestion_attempt_id": ingestion_attempt_id,
            "source_version": source_version,
            "ingestion_id": ingestion_id,
            "ingestion_order_ns": ingestion_order_ns,
            "is_latest_version": reconciliation["is_latest_version"],
            "lifecycle_status": reconciliation["lifecycle_status"],
            "snapshot_id": snapshot_id,
            "artifact_id": artifact_id,
            "artifact_path": artifact_path,
            "ingested_at": ingested_at,
        }

    except Exception as exc:
        detail = safe_error_detail(exc)
        logger.error("Ingest failed: %s", detail)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {detail}")


async def rag_query_impl(req: QueryRequest) -> Dict[str, Any]:
    if USE_RESEARCH_API_RAG:
        return await _remote_rag_request(
            "POST", "/rag/query", json_body=req.model_dump()
        )

    try:
        await asyncio.to_thread(init_qdrant)
        top_k = max(1, min(req.top_k, 30))
        query_vectors = await embed_texts_async([req.query])
        if len(query_vectors) != 1:
            raise ValueError(
                f"Embedding model returned {len(query_vectors)} vectors for one query."
            )
        query_vec = query_vectors[0]
        if len(query_vec) != VECTOR_SIZE:
            raise ValueError(
                f"Embedding model returned {len(query_vec)} dimensions but VECTOR_SIZE={VECTOR_SIZE}."
            )

        namespace = normalize_namespace(req.namespace)
        must_conditions = [
            FieldCondition(key="namespace", match=MatchValue(value=namespace)),
            _retrievable_lifecycle_filter(),
        ]
        if req.research_run_id:
            must_conditions.append(
                FieldCondition(
                    key="research_run_id", match=MatchValue(value=req.research_run_id)
                )
            )
        if req.ingestion_attempt_id:
            must_conditions.append(
                FieldCondition(
                    key="ingestion_attempt_id",
                    match=MatchValue(value=req.ingestion_attempt_id),
                )
            )
        query_filter = Filter(must=must_conditions)

        hits = await qdrant_query_points_async(
            query_vec=query_vec,
            limit=top_k * 5,
            query_filter=query_filter,
        )

        unique_docs = []
        seen_text = set()

        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")

            if not text or text in seen_text:
                continue

            doc = {
                "text": text,
                "source": payload.get("source", "unknown"),
                "url": payload.get("url", payload.get("source", "unknown")),
                "requested_url": payload.get("requested_url"),
                "title": payload.get("title"),
                "domain": payload.get("domain"),
                "section": payload.get("section"),
                "chunk_index": payload.get("chunk_index"),
                "content_type": payload.get("content_type"),
                "ingested_at": payload.get("ingested_at"),
                "retrieved_at_utc": payload.get("retrieved_at_utc"),
                "retrieval_current_date_utc": payload.get("retrieval_current_date_utc"),
                "source_score": payload.get("source_score"),
                "source_reason": payload.get("source_reason"),
                "namespace": payload.get("namespace", namespace),
                "research_run_id": payload.get("research_run_id"),
                "ingestion_attempt_id": payload.get("ingestion_attempt_id"),
                "source_version": payload.get("source_version"),
                "ingestion_id": payload.get("ingestion_id"),
                "ingestion_order_ns": payload.get("ingestion_order_ns"),
                "ingestion_committed": payload.get("ingestion_committed"),
                "is_latest_version": payload.get("is_latest_version"),
                "lifecycle_status": payload.get("lifecycle_status"),
                "snapshot_id": payload.get("snapshot_id"),
                "artifact_id": payload.get("artifact_id"),
                "artifact_path": payload.get("artifact_path"),
                "artifact_lifecycle": payload.get("artifact_lifecycle"),
                "vector_score": getattr(hit, "score", None),
            }

            unique_docs.append(doc)
            seen_text.add(text)

        if not unique_docs:
            return {
                "query": req.query,
                "namespace": namespace,
                "research_run_id": req.research_run_id,
                "ingestion_attempt_id": req.ingestion_attempt_id,
                "results": [],
            }

        final = await rerank_docs(req.query, unique_docs, top_k)

        return {
            "query": req.query,
            "namespace": namespace,
            "research_run_id": req.research_run_id,
            "ingestion_attempt_id": req.ingestion_attempt_id,
            "results": final,
        }

    except Exception as exc:
        detail = safe_error_detail(exc)
        logger.error("Query failed: %s", detail)
        raise HTTPException(status_code=500, detail=f"Query failed: {detail}")


def collect_points(
    limit_per_page: int = 10000,
    max_points: int = 100000,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[List[Any], bool]:
    init_qdrant()
    offset = None
    total = 0
    records: List[Any] = []
    namespace = normalize_namespace(namespace)
    scroll_filter = Filter(
        must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
    )
    client = get_qdrant_client()

    while total < max_points:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=min(limit_per_page, max_points - total),
            offset=offset,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        total += len(points)
        records.extend(points)

        if offset is None:
            break

    return records, bool(offset is not None and total >= max_points)


def scroll_points(
    limit_per_page: int = 10000,
    max_points: int = 100000,
    namespace: str = DEFAULT_NAMESPACE,
):
    records, _ = collect_points(
        limit_per_page=limit_per_page,
        max_points=max_points,
        namespace=namespace,
    )
    yield from records


def _payload_timestamp(payload: Dict[str, Any], prefix: str) -> Optional[int]:
    unix_value = payload.get(f"{prefix}_unix")
    if (
        isinstance(unix_value, int)
        and not isinstance(unix_value, bool)
        and unix_value > 0
    ):
        return unix_value
    iso_value = payload.get(prefix)
    if not isinstance(iso_value, str) or not iso_value:
        return None
    try:
        parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    except ValueError:
        return None


async def prune_qdrant_history_async(
    *,
    namespace: Optional[str] = None,
    retention_seconds: Optional[int] = None,
    max_points: int = 100000,
    cursor: Any = None,
) -> Dict[str, Any]:
    """Delete bounded-age superseded, invalid, and abandoned pending chunks."""
    retention = (
        QDRANT_HISTORY_RETENTION_SECONDS
        if retention_seconds is None
        else retention_seconds
    )
    if isinstance(retention, bool) or not isinstance(retention, int) or retention < 0:
        raise ValueError("retention_seconds must be a non-negative integer")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if retention == 0:
        return {
            "enabled": False,
            "scanned": 0,
            "deleted": 0,
            "truncated": False,
            "next_cursor": None,
        }

    await asyncio.to_thread(init_qdrant)
    normalized_namespace = (
        normalize_namespace(namespace) if namespace is not None else None
    )
    query_filter = (
        Filter(
            must=[
                FieldCondition(
                    key="namespace", match=MatchValue(value=normalized_namespace)
                )
            ]
        )
        if normalized_namespace is not None
        else None
    )
    client = get_qdrant_client()

    def prune() -> Dict[str, Any]:
        records, next_cursor = _scroll_matching_batch(
            client,
            query_filter,
            max_points=max_points,
            cursor=cursor,
        )
        cutoff = int(time.time()) - retention
        delete_ids = []
        for record in records:
            payload = record.payload or {}
            if payload.get("record_type") == _INGESTION_TOMBSTONE_RECORD_TYPE:
                continue
            if _payload_is_active(payload):
                continue
            lifecycle_status = payload.get("lifecycle_status")
            if lifecycle_status in {"superseded", "invalid"}:
                timestamp = _payload_timestamp(payload, "superseded_at")
            elif payload.get("ingestion_committed") is False:
                timestamp = _payload_timestamp(payload, "ingested_at")
            else:
                continue
            if timestamp is not None and timestamp < cutoff:
                delete_ids.append(record.id)

        for index in range(0, len(delete_ids), 1000):
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=delete_ids[index : index + 1000],
                wait=True,
                ordering=WriteOrdering.STRONG,
            )
        return {
            "enabled": True,
            "retention_seconds": retention,
            "scanned": len(records),
            "deleted": len(delete_ids),
            "truncated": next_cursor is not None,
            "next_cursor": next_cursor,
        }

    return await asyncio.to_thread(prune)


async def repair_qdrant_lifecycle_async(
    *,
    namespace: Optional[str] = None,
    max_points: int = 100000,
    prune_history: bool = True,
    cursor: Any = None,
    history_cursor: Any = None,
) -> Dict[str, Any]:
    """Repair committed/pending lifecycle state and canonical source identities."""
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    await asyncio.to_thread(init_qdrant)
    normalized_namespace = (
        normalize_namespace(namespace) if namespace is not None else None
    )
    query_filter = (
        Filter(
            must=[
                FieldCondition(
                    key="namespace", match=MatchValue(value=normalized_namespace)
                )
            ]
        )
        if normalized_namespace is not None
        else None
    )
    client = get_qdrant_client()

    def discover() -> tuple[set[tuple[str, str]], int, Any]:
        records, next_cursor = _scroll_matching_batch(
            client,
            query_filter,
            max_points=max_points,
            cursor=cursor,
        )
        sources: set[tuple[str, str]] = set()
        identity_updates: Dict[str, List[Any]] = {}
        for record in records:
            payload = record.payload or {}
            source_namespace = normalize_namespace(payload.get("namespace"))
            raw_source = payload.get("source") or payload.get("url")
            canonical_source = normalize_url(raw_source)
            if not canonical_source:
                continue
            sources.add((source_namespace, canonical_source))
            if (
                payload.get("source") != canonical_source
                or payload.get("source_identity") != canonical_source
            ):
                identity_updates.setdefault(canonical_source, []).append(record.id)

        for canonical_source, point_ids in identity_updates.items():
            for index in range(0, len(point_ids), 1000):
                client.set_payload(
                    collection_name=COLLECTION_NAME,
                    payload={
                        "source": canonical_source,
                        "source_identity": canonical_source,
                    },
                    points=point_ids[index : index + 1000],
                    wait=True,
                    ordering=WriteOrdering.STRONG,
                )
        return sources, sum(map(len, identity_updates.values())), next_cursor

    sources, identities_repaired, next_cursor = await asyncio.to_thread(discover)
    repaired = 0
    for source_namespace, source in sorted(sources):
        result = await reconcile_source_versions_async(
            namespace=source_namespace,
            source=source,
        )
        if result.get("winner_ingestion_id"):
            repaired += 1

    cleanup = (
        await prune_qdrant_history_async(
            namespace=normalized_namespace,
            max_points=max_points,
            cursor=history_cursor,
        )
        if prune_history
        else {
            "enabled": False,
            "scanned": 0,
            "deleted": 0,
            "truncated": False,
            "next_cursor": None,
        }
    )
    return {
        "sources_scanned": len(sources),
        "sources_reconciled": repaired,
        "source_identities_repaired": identities_repaired,
        "scan_truncated": next_cursor is not None,
        "next_cursor": next_cursor,
        "history_cleanup": cleanup,
    }


async def list_sources_impl(
    limit: int = 50, namespace: str = DEFAULT_NAMESPACE
) -> Dict[str, Any]:
    limit = max(1, min(limit, 500))
    namespace = normalize_namespace(namespace)

    if USE_RESEARCH_API_RAG:
        return await _remote_rag_request(
            "POST",
            "/rag/sources",
            json_body=SourceListRequest(limit=limit, namespace=namespace).model_dump(),
        )

    try:
        sources = {}
        points, truncated = await asyncio.to_thread(collect_points, namespace=namespace)

        for point in points:
            payload = point.payload or {}
            source = payload.get("source") or payload.get("url") or "unknown"

            if source not in sources:
                sources[source] = {
                    "source": source,
                    "url": payload.get("url", source),
                    "title": payload.get("title"),
                    "domain": payload.get("domain"),
                    "content_type": payload.get("content_type"),
                    "ingested_at": payload.get("ingested_at"),
                    "chunks": 0,
                    "total_chunks": 0,
                    "superseded_chunks": 0,
                    "pending_chunks": 0,
                    "invalid_chunks": 0,
                    "source_versions": set(),
                }

            sources[source]["total_chunks"] += 1
            if payload.get("source_version"):
                sources[source]["source_versions"].add(payload["source_version"])
            if payload.get("ingestion_committed") is False:
                sources[source]["pending_chunks"] += 1
            elif payload.get("lifecycle_status") == "invalid":
                sources[source]["invalid_chunks"] += 1
            elif not _payload_is_active(payload):
                sources[source]["superseded_chunks"] += 1
            else:
                sources[source]["chunks"] += 1

            current_ingested = sources[source].get("ingested_at")
            new_ingested = payload.get("ingested_at")
            if new_ingested and (
                not current_ingested or new_ingested > current_ingested
            ):
                sources[source].update(
                    {
                        "url": payload.get("url", source),
                        "title": payload.get("title"),
                        "domain": payload.get("domain"),
                        "content_type": payload.get("content_type"),
                        "ingested_at": new_ingested,
                    }
                )

        for item in sources.values():
            versions = item.pop("source_versions")
            item["version_count"] = len(versions)

        sorted_sources = sorted(
            sources.values(),
            key=lambda item: item.get("ingested_at") or "",
            reverse=True,
        )

        return {
            "namespace": namespace,
            "count": len(sorted_sources),
            "sources": sorted_sources[:limit],
            "scanned_chunks": len(points),
            "scan_limit": 100000,
            "truncated": truncated,
        }

    except Exception as exc:
        detail = safe_error_detail(exc)
        logger.error("List sources failed: %s", detail)
        raise HTTPException(status_code=500, detail=f"List sources failed: {detail}")


async def source_stats_impl(namespace: str = DEFAULT_NAMESPACE) -> Dict[str, Any]:
    namespace = normalize_namespace(namespace)
    if USE_RESEARCH_API_RAG:
        return await _remote_rag_request(
            "GET",
            "/rag/source-stats",
            params={"namespace": namespace},
        )

    try:
        source_counter = Counter()
        domain_counter = Counter()
        content_type_counter = Counter()
        total = 0
        active = 0
        superseded = 0
        pending = 0
        invalid = 0
        points, truncated = await asyncio.to_thread(collect_points, namespace=namespace)

        for point in points:
            total += 1
            payload = point.payload or {}
            if payload.get("ingestion_committed") is False:
                pending += 1
                continue
            if payload.get("lifecycle_status") == "invalid":
                invalid += 1
                continue
            if not _payload_is_active(payload):
                superseded += 1
                continue
            active += 1
            source_counter[payload.get("source", "unknown")] += 1
            domain_counter[payload.get("domain", "unknown")] += 1
            content_type_counter[payload.get("content_type", "unknown")] += 1

        return {
            "collection": COLLECTION_NAME,
            "namespace": namespace,
            "total_chunks_sampled": total,
            "active_chunks_sampled": active,
            "superseded_chunks_sampled": superseded,
            "pending_chunks_sampled": pending,
            "invalid_chunks_sampled": invalid,
            "unique_sources": len(source_counter),
            "top_domains": [
                {"domain": domain, "chunks": count}
                for domain, count in domain_counter.most_common(25)
            ],
            "top_sources": [
                {"source": source, "chunks": count}
                for source, count in source_counter.most_common(25)
            ],
            "content_types": [
                {"content_type": content_type, "chunks": count}
                for content_type, count in content_type_counter.most_common()
            ],
            "scan_limit": 100000,
            "truncated": truncated,
        }

    except Exception as exc:
        detail = safe_error_detail(exc)
        logger.error("Source stats failed: %s", detail)
        raise HTTPException(status_code=500, detail=f"Source stats failed: {detail}")


async def delete_source_impl(
    source: str, namespace: str = DEFAULT_NAMESPACE
) -> Dict[str, Any]:
    raw_source = (source or "").strip()
    source = normalize_url(raw_source)
    namespace = normalize_namespace(namespace)

    if not source:
        raise HTTPException(status_code=400, detail="source is required")

    if USE_RESEARCH_API_RAG:
        return await _remote_rag_request(
            "POST",
            "/rag/delete-source",
            json_body=SourceDeleteRequest(
                source=source, namespace=namespace
            ).model_dump(),
        )

    try:
        await asyncio.to_thread(init_qdrant)
        delete_filter = Filter(
            must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))],
            should=[
                FieldCondition(key="source", match=MatchValue(value=source)),
                FieldCondition(key="url", match=MatchValue(value=source)),
                FieldCondition(key="source_identity", match=MatchValue(value=source)),
                *(
                    [
                        FieldCondition(
                            key="source", match=MatchValue(value=raw_source)
                        ),
                        FieldCondition(key="url", match=MatchValue(value=raw_source)),
                    ]
                    if raw_source != source
                    else []
                ),
            ],
        )

        client = get_qdrant_client()
        await asyncio.to_thread(
            client.delete,
            collection_name=COLLECTION_NAME,
            points_selector=delete_filter,
            wait=True,
        )

        return {
            "deleted": True,
            "source": source,
            "namespace": namespace,
            "deleted_scope": "all_vector_memory_versions_for_source",
            "artifact_files_deleted": False,
            "artifact_lifecycle": "managed_independently_from_vector_memory",
        }

    except Exception as exc:
        detail = safe_error_detail(exc)
        logger.error("Delete source failed: %s", detail)
        raise HTTPException(status_code=500, detail=f"Delete source failed: {detail}")
