"""No-network PDF parser exposed only through a Unix socket."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request


PDF_RUNNER_SOCKET = os.getenv(
    "PDF_RUNNER_SOCKET", "/run/search-gateway-pdf/runner.sock"
)
PDF_MAX_RESPONSE_BYTES = max(
    65_536, int(os.getenv("GATEWAY_DIRECT_MAX_BYTES", str(8 * 1024 * 1024)))
)
PDF_OUTPUT_BYTES = max(
    65_536, int(os.getenv("PDF_SANDBOX_OUTPUT_BYTES", str(6 * 1024 * 1024)))
)
PDF_TIMEOUT_SECONDS = max(2.0, float(os.getenv("PDF_TIMEOUT_SECONDS", "20")))
_semaphore = asyncio.Semaphore(1)


def require_isolated_runtime() -> None:
    if os.getenv("RESEARCH_PDF_NETWORK_ISOLATED", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("PDF runner requires an internal Docker network")


async def parse_pdf(body: bytes) -> dict[str, object]:
    env = {
        key: os.environ[key]
        for key in ("LANG", "LC_ALL", "PATH")
        if os.environ.get(key)
    }
    env.update(
        {
            "PDF_MAX_RESPONSE_BYTES": str(PDF_MAX_RESPONSE_BYTES),
            "PDF_MAX_PAGES": os.getenv("GATEWAY_PDF_MAX_PAGES", "100"),
            "PDF_MAX_EXTRACTED_CHARS": os.getenv("GATEWAY_PAGE_MAX_CHARS", "300000"),
            "PDF_SANDBOX_MEMORY_BYTES": os.getenv(
                "PDF_SANDBOX_MEMORY_BYTES", str(512 * 1024 * 1024)
            ),
            "PDF_SANDBOX_CPU_SECONDS": os.getenv("PDF_SANDBOX_CPU_SECONDS", "15"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name("pdf_sandbox.py")),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    try:
        async with asyncio.timeout(PDF_TIMEOUT_SECONDS):
            stdout, _ = await process.communicate(body)
    except (TimeoutError, asyncio.CancelledError) as exc:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        if isinstance(exc, asyncio.CancelledError):
            raise
        return {"content": "", "title": None, "error": "PDF extraction timed out"}
    if process.returncode != 0 or len(stdout) > PDF_OUTPUT_BYTES:
        return {"content": "", "title": None, "error": "PDF extraction failed safely"}
    try:
        payload = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"content": "", "title": None, "error": "PDF parser returned invalid output"}
    return payload if isinstance(payload, dict) else {
        "content": "",
        "title": None,
        "error": "PDF parser returned invalid output",
    }


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/extract")
async def extract(request: Request) -> dict[str, object]:
    declared = request.headers.get("content-length")
    if declared:
        try:
            if not 0 <= int(declared) <= PDF_MAX_RESPONSE_BYTES:
                raise HTTPException(status_code=413, detail="PDF exceeds the byte limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > PDF_MAX_RESPONSE_BYTES:
            raise HTTPException(status_code=413, detail="PDF exceeds the byte limit")
    async with _semaphore:
        return await parse_pdf(bytes(body))


async def healthcheck() -> bool:
    transport = httpx.AsyncHTTPTransport(uds=PDF_RUNNER_SOCKET)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://pdf-runner",
            timeout=2.0,
            trust_env=False,
        ) as client:
            response = await client.get("/healthz")
            return response.status_code == 200
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    if args.healthcheck:
        return 0 if asyncio.run(healthcheck()) else 1
    require_isolated_runtime()
    path = Path(PDF_RUNNER_SOCKET)
    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(FileNotFoundError):
        path.unlink()
    try:
        uvicorn.run(app, uds=PDF_RUNNER_SOCKET, access_log=False, proxy_headers=False)
    finally:
        with suppress(FileNotFoundError):
            path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
