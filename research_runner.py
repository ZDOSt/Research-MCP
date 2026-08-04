"""Private synchronous research runner exposed only over a Unix socket.

The MCP gateway uses this service for ordinary (non-deep) assistant requests.
Keeping the execution path separate from the durable Redis worker prevents a
long-running background job from consuming the interactive response window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator


LOGGER = logging.getLogger("research-runner")
RESEARCH_RUNNER_SOCKET = os.getenv(
    "RESEARCH_RUNNER_SOCKET", "/run/research-interactive/runner.sock"
)
RESEARCH_RUNNER_MAX_REQUEST_BYTES = max(
    1024,
    int(os.getenv("RESEARCH_RUNNER_MAX_REQUEST_BYTES", str(512 * 1024))),
)
RESEARCH_RUNNER_MAX_RESPONSE_BYTES = max(
    4096,
    int(os.getenv("RESEARCH_RUNNER_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))),
)
RESEARCH_RUNNER_MAX_CONCURRENT = max(
    1, int(os.getenv("RESEARCH_RUNNER_MAX_CONCURRENT", "2"))
)
_semaphore = asyncio.Semaphore(RESEARCH_RUNNER_MAX_CONCURRENT)


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1, max_length=8000)
    mode: str = Field(default="auto", pattern="^(auto|quick|balanced|technical|academic)$")
    namespace: str = Field(default="default", min_length=1, max_length=128)
    search_cache_scope: str | None = Field(default=None, max_length=128)
    github_access_policy: dict[str, Any] | None = None
    time_budget_seconds: float = Field(default=36.0, ge=1.0, le=120.0)

    @field_validator("github_access_policy")
    @classmethod
    def validate_github_access_policy(cls, value):
        if value is None:
            return None
        if set(value) - {"allowed", "repositories"}:
            raise ValueError("unsupported GitHub access policy field")
        if not isinstance(value.get("allowed"), bool):
            raise ValueError("GitHub access policy allowed must be boolean")
        repositories = value.get("repositories", [])
        if not isinstance(repositories, list) or len(repositories) > 256:
            raise ValueError("GitHub access policy repositories are invalid")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 300 for item in repositories):
            raise ValueError("GitHub access policy repository is invalid")
        return {
            "allowed": value["allowed"],
            "repositories": list(dict.fromkeys(item.strip() for item in repositories)),
        }


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Restrict the private runner socket to the app group after Uvicorn binds it."""
    socket_path = Path(RESEARCH_RUNNER_SOCKET)
    if socket_path.exists():
        socket_path.chmod(0o660)
    yield


app = FastAPI(
    title="Research MCP interactive runner",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next):
    declared = request.headers.get("content-length")
    if declared:
        try:
            size = int(declared)
        except ValueError:
            return Response(status_code=400)
        if size < 0 or size > RESEARCH_RUNNER_MAX_REQUEST_BYTES:
            return Response(status_code=413)
    # Cache the bounded body so FastAPI can validate it after this middleware
    # has inspected the actual payload (not just an optional Content-Length).
    body = await request.body()
    if len(body) > RESEARCH_RUNNER_MAX_REQUEST_BYTES:
        return Response(status_code=413)
    request._body = body
    return await call_next(request)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/research_assistant")
async def research_assistant(payload: ResearchRequest) -> dict[str, Any]:
    # Import lazily so the tiny health endpoint remains available while model
    # and embedding dependencies are loading.
    from research_agent import run_research_assistant
    from shared import normalize_namespace

    encoded = json.dumps(payload.model_dump(), separators=(",", ":")).encode("utf-8")
    if len(encoded) > RESEARCH_RUNNER_MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="research request exceeds the byte limit")

    async with _semaphore:
        try:
            result = await run_research_assistant(
                request=payload.request,
                mode=payload.mode,
                namespace=normalize_namespace(payload.namespace),
                research_run_id=uuid.uuid4().hex,
                # Interactive calls must not wait for archival or vector
                # indexing; the durable deep path owns persistence. Setting
                # defer_persistence keeps the pipeline from entering Qdrant
                # on this latency-critical path.
                persist_source_artifacts=False,
                defer_persistence=True,
                search_cache_scope=payload.search_cache_scope,
                github_access_policy=payload.github_access_policy,
                time_budget_seconds=payload.time_budget_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Interactive research failed: %s", type(exc).__name__)
            raise HTTPException(
                status_code=502,
                detail="interactive research execution failed",
            ) from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="interactive research returned invalid data")
    if result.get("status") in {"queued", "running"}:
        LOGGER.error("Interactive runner returned a non-terminal result")
        raise HTTPException(status_code=502, detail="interactive research did not complete")
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > RESEARCH_RUNNER_MAX_RESPONSE_BYTES:
        raise HTTPException(status_code=502, detail="interactive research response exceeds the byte limit")
    return result


async def healthcheck(socket_path: str = RESEARCH_RUNNER_SOCKET) -> bool:
    transport = httpx.AsyncHTTPTransport(uds=socket_path)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://research-runner",
            timeout=2.0,
            trust_env=False,
        ) as client:
            response = await client.get("/healthz")
            return response.status_code == 200 and response.json() == {"ok": True}
    except Exception:
        return False


def _prepare_socket_path() -> None:
    path = Path(RESEARCH_RUNNER_SOCKET)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_socket():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Research MCP interactive runner")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.healthcheck:
        return 0 if asyncio.run(healthcheck()) else 1
    _prepare_socket_path()
    try:
        uvicorn.run(
            app,
            uds=RESEARCH_RUNNER_SOCKET,
            log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
            access_log=False,
            proxy_headers=False,
        )
    finally:
        with suppress(FileNotFoundError):
            Path(RESEARCH_RUNNER_SOCKET).unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
