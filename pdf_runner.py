"""No-network PDF parser service exposed only through a Unix socket."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import suppress
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from crawler import PDF_MAX_RESPONSE_BYTES, _extract_pdf_text_subprocess


LOGGER = logging.getLogger("pdf-runner")
PDF_RUNNER_SOCKET = os.getenv("PDF_RUNNER_SOCKET", "/run/research-pdf/runner.sock")
_parser_semaphore = asyncio.Semaphore(1)


def require_isolated_runtime() -> None:
    marker = os.getenv("RESEARCH_PDF_NETWORK_ISOLATED", "").strip().lower()
    if marker not in {"1", "true", "yes", "on"}:
        raise RuntimeError("pdf-runner requires an isolated network namespace")


app = FastAPI(
    title="Research MCP isolated PDF runner",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/extract")
async def extract(request: Request) -> dict[str, object]:
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) < 0 or int(declared) > PDF_MAX_RESPONSE_BYTES:
                raise HTTPException(status_code=413, detail="PDF exceeds the byte limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc

    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > PDF_MAX_RESPONSE_BYTES:
            raise HTTPException(status_code=413, detail="PDF exceeds the byte limit")

    async with _parser_semaphore:
        content, title, error = await _extract_pdf_text_subprocess(bytes(body))
    return {"content": content, "title": title, "error": error}


async def healthcheck(socket_path: str = PDF_RUNNER_SOCKET) -> bool:
    transport = httpx.AsyncHTTPTransport(uds=socket_path)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://pdf-runner",
            timeout=2.0,
            trust_env=False,
        ) as client:
            response = await client.get("/healthz")
            return response.status_code == 200 and response.json() == {"ok": True}
    except Exception:
        return False


def _prepare_socket_path() -> None:
    path = Path(PDF_RUNNER_SOCKET)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_socket():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Research MCP isolated PDF runner")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.healthcheck:
        return 0 if asyncio.run(healthcheck()) else 1
    require_isolated_runtime()
    _prepare_socket_path()
    try:
        uvicorn.run(
            app,
            uds=PDF_RUNNER_SOCKET,
            log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
            access_log=False,
            proxy_headers=False,
        )
    finally:
        with suppress(FileNotFoundError):
            Path(PDF_RUNNER_SOCKET).unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
