import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import ValidationError

import api
import shared


def test_ingest_request_bounds_text_and_metadata(monkeypatch):
    with pytest.raises(ValidationError, match="String should have at most"):
        shared.IngestRequest(text="x" * (shared.RAG_MAX_INGEST_CHARS + 1))

    with pytest.raises(ValidationError, match="Dictionary should have at most"):
        shared.IngestRequest(
            text="content",
            metadata={f"key-{index}": index for index in range(101)},
        )

    monkeypatch.setattr(shared, "RAG_MAX_METADATA_BYTES", 8)
    with pytest.raises(ValidationError, match="metadata exceeds"):
        shared.IngestRequest(text="content", metadata={"key": "long value"})

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        shared.IngestRequest(text="content", ignored="x" * 10_000)


@pytest.mark.asyncio
async def test_request_body_limit_rejects_chunked_body_before_endpoint():
    inner = FastAPI()
    called = False

    @inner.post("/ingest")
    async def ingest(request: Request):
        nonlocal called
        body = await request.body()
        called = True
        return {"size": len(body)}

    bounded_app = api.RequestBodyLimitMiddleware(inner, max_bytes=4)
    transport = httpx.ASGITransport(app=bounded_app)

    async def chunks():
        yield b"123"
        yield b"45"

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post("/ingest", content=chunks())

    assert response.status_code == 413
    assert called is False


@pytest.mark.asyncio
async def test_request_body_limit_ignores_lying_content_length():
    inner = FastAPI()
    called = False

    @inner.post("/ingest")
    async def ingest(request: Request):
        nonlocal called
        await request.body()
        called = True
        return {"ok": True}

    bounded_app = api.RequestBodyLimitMiddleware(inner, max_bytes=4)
    transport = httpx.ASGITransport(app=bounded_app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/ingest",
            content=b"12345",
            headers={"content-length": "1"},
        )

    assert response.status_code == 413
    assert called is False
