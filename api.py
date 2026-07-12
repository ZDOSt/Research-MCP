import os
import secrets
from typing import Awaitable, Callable

from fastapi import FastAPI
from fastapi import Depends, Header, HTTPException, status
from starlette.responses import JSONResponse

from shared import (
    COLLECTION_NAME,
    IngestRequest,
    IngestionAttemptInvalidationRequest,
    QueryRequest,
    SourceDeleteRequest,
    SourceListRequest,
    delete_source_impl,
    get_qdrant_client,
    invalidate_ingestion_attempt_async,
    list_sources_impl,
    rag_ingest_impl,
    rag_query_impl,
    source_stats_impl,
)

RESEARCH_API_MAX_REQUEST_BYTES = max(
    1_024,
    int(os.getenv("RESEARCH_API_MAX_REQUEST_BYTES", "4194304")),
)


class RequestBodyLimitMiddleware:
    """Bound request bodies while they are read, including chunked requests."""

    def __init__(self, app: Callable[..., Awaitable[None]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                declared_length = int(raw_length)
            except (TypeError, ValueError):
                response = JSONResponse(
                    {"detail": "Invalid Content-Length"},
                    status_code=status.HTTP_400_BAD_REQUEST,
                )
                await response(scope, receive, send)
                return
            if declared_length < 0 or declared_length > self.max_bytes:
                response = JSONResponse(
                    {"detail": "Request body exceeds the configured byte limit"},
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )
                await response(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body") or b""
            if len(body) + len(chunk) > self.max_bytes:
                response = JSONResponse(
                    {"detail": "Request body exceeds the configured byte limit"},
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                )
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def bounded_receive():
            nonlocal replayed
            if replayed:
                return {"type": "http.disconnect"}
            replayed = True
            return {
                "type": "http.request",
                "body": bytes(body),
                "more_body": False,
            }

        await self.app(scope, bounded_receive, send)


app = FastAPI(title="Research RAG API")
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_bytes=RESEARCH_API_MAX_REQUEST_BYTES,
)
RESEARCH_API_TOKEN = os.getenv("RESEARCH_API_TOKEN", "").strip()
RESEARCH_API_ALLOW_UNAUTHENTICATED = os.getenv(
    "RESEARCH_API_ALLOW_UNAUTHENTICATED", "false"
).lower() in {"1", "true", "yes", "on"}


async def require_api_authorization(authorization: str = Header(default="")) -> None:
    if not RESEARCH_API_TOKEN:
        if RESEARCH_API_ALLOW_UNAUTHENTICATED:
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RESEARCH_API_TOKEN is required before enabling RAG management routes",
        )
    scheme, _, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        supplied.strip(), RESEARCH_API_TOKEN
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/health")
async def health():
    detail = None
    try:
        get_qdrant_client().get_collection(COLLECTION_NAME)
        qdrant_status = "ok"
    except Exception:
        qdrant_status = "error"
        detail = "qdrant unavailable"

    return {
        "status": "ok" if qdrant_status == "ok" else "degraded",
        "qdrant": qdrant_status,
        "detail": detail,
    }


@app.get("/rag/health")
async def rag_health():
    return await health()


@app.post("/rag/ingest", dependencies=[Depends(require_api_authorization)])
async def rag_ingest_route(body: IngestRequest):
    return await rag_ingest_impl(body)


@app.post("/rag/invalidate-attempt", dependencies=[Depends(require_api_authorization)])
async def invalidate_ingestion_attempt_route(body: IngestionAttemptInvalidationRequest):
    return await invalidate_ingestion_attempt_async(
        body.ingestion_attempt_id,
        reason=body.reason,
    )


@app.post("/rag/query", dependencies=[Depends(require_api_authorization)])
async def rag_query_route(body: QueryRequest):
    return await rag_query_impl(body)


@app.post("/rag/sources", dependencies=[Depends(require_api_authorization)])
async def list_sources_route(body: SourceListRequest):
    return await list_sources_impl(limit=body.limit, namespace=body.namespace)


@app.get("/rag/source-stats", dependencies=[Depends(require_api_authorization)])
async def source_stats_route(namespace: str = "default"):
    return await source_stats_impl(namespace=namespace)


@app.post("/rag/delete-source", dependencies=[Depends(require_api_authorization)])
async def delete_source_route(body: SourceDeleteRequest):
    return await delete_source_impl(body.source, namespace=body.namespace)
