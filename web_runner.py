"""Isolated browser and Crawl4AI control service exposed only over a Unix socket."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field

from browser import (
    ABSOLUTE_MAX_CHARS,
    DEFAULT_MAX_CHARS,
    chromium_launch_options,
    chromium_sandbox_mode,
    playwright_explore_page_local,
    set_resolved_chromium_sandbox,
)
from crawler import (
    CRAWL4AI_MAX_RESPONSE_BYTES,
    CRAWL4AI_TOTAL_TIMEOUT_SECONDS,
    _read_limited_response,
    validate_url_safety,
)


LOGGER = logging.getLogger("web-runner")
WEB_RUNNER_SOCKET = os.getenv("WEB_RUNNER_SOCKET", "/run/research-web/runner.sock")
WEB_RUNNER_MAX_REQUEST_BYTES = max(
    1024, int(os.getenv("WEB_RUNNER_MAX_REQUEST_BYTES", str(2 * 1024 * 1024)))
)
WEB_RUNNER_PROXY_URL = os.getenv("RESEARCH_BROWSER_PROXY", "").strip()
WEB_RUNNER_CRAWL4AI_URL = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235").rstrip("/")
WEB_RUNNER_CRAWL4AI_API_TOKEN = os.getenv("CRAWL4AI_API_TOKEN", "").strip()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def crawl4ai_auth_headers() -> dict[str, str]:
    if not WEB_RUNNER_CRAWL4AI_API_TOKEN:
        return {}
    return {"Authorization": f"Bearer {WEB_RUNNER_CRAWL4AI_API_TOKEN}"}


def require_isolated_runtime() -> None:
    if not _env_flag("RESEARCH_REQUIRE_WEB_ISOLATION"):
        raise RuntimeError("web-runner requires RESEARCH_REQUIRE_WEB_ISOLATION=true")
    if not _env_flag("RESEARCH_WEB_NETWORK_ISOLATED"):
        raise RuntimeError("web-runner requires an internal Docker network")
    if not WEB_RUNNER_PROXY_URL.lower().startswith("socks5://"):
        raise RuntimeError("web-runner requires a SOCKS5 RESEARCH_BROWSER_PROXY")


def chromium_sandbox_denied_by_host(exc: Exception) -> bool:
    detail = str(exc).lower()
    return (
        "sandbox/linux/services/credentials.cc" in detail
        and "permission denied (13)" in detail
    )


async def _verify_chromium_launch(sandbox_enabled: bool) -> None:
    async with asyncio.timeout(45.0):
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                **chromium_launch_options(sandbox_enabled=sandbox_enabled)
            )
            await browser.close()


async def verify_chromium_runtime() -> None:
    """Resolve and verify the Chromium sandbox mode before serving requests."""
    mode = chromium_sandbox_mode()
    sandbox_enabled = mode != "disabled"
    try:
        await _verify_chromium_launch(sandbox_enabled)
    except Exception as exc:
        if mode != "auto" or not chromium_sandbox_denied_by_host(exc):
            raise RuntimeError("Chromium sandbox preflight failed") from exc
        LOGGER.warning(
            "Chromium native sandbox was denied by host policy; using the "
            "hardened web-runner container as the compatibility sandbox"
        )
        try:
            await _verify_chromium_launch(False)
        except Exception as fallback_exc:
            raise RuntimeError("Chromium compatibility preflight failed") from fallback_exc
        sandbox_enabled = False
    set_resolved_chromium_sandbox(sandbox_enabled)


@asynccontextmanager
async def lifespan(_: FastAPI):
    require_isolated_runtime()
    await verify_chromium_runtime()
    socket_path = Path(WEB_RUNNER_SOCKET)
    if socket_path.exists():
        socket_path.chmod(0o660)
    yield


app = FastAPI(
    title="Research MCP isolated web runner",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


class ExploreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=8192)
    labels: list[str] = Field(default_factory=list, max_length=50)
    task: str = Field(default="", max_length=4000)
    max_chars: int = Field(default=DEFAULT_MAX_CHARS, ge=10_000, le=ABSOLUTE_MAX_CHARS)
    profile: Literal["targeted", "balanced", "exhaustive"] = "targeted"
    timeout_ms: int = Field(default=60_000, ge=5_000, le=60_000)


@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            return Response(status_code=400)
        if declared_size < 0 or declared_size > WEB_RUNNER_MAX_REQUEST_BYTES:
            return Response(status_code=413)
    return await call_next(request)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.post("/v1/explore")
async def explore(request: ExploreRequest) -> dict[str, Any]:
    return await playwright_explore_page_local(
        url=request.url,
        labels=request.labels,
        task=request.task,
        max_chars=request.max_chars,
        profile=request.profile,
        timeout_ms=request.timeout_ms,
    )


def _crawl_urls(payload: dict[str, Any]) -> list[str]:
    urls = payload.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list) or not urls or not all(isinstance(url, str) for url in urls):
        raise HTTPException(status_code=400, detail="Crawl4AI payload requires string URLs")
    if len(urls) > 20:
        raise HTTPException(status_code=400, detail="Crawl4AI URL count exceeds the limit")
    return urls


async def prepare_crawl_payload(payload: dict[str, Any]) -> dict[str, Any]:
    urls = _crawl_urls(payload)
    for url in urls:
        await validate_url_safety(url)

    forwarded_payload = dict(payload)
    browser_config = forwarded_payload.get("browser_config")
    browser_config = dict(browser_config) if isinstance(browser_config, dict) else {}
    for field in ("proxy", "proxy_config"):
        browser_config.pop(field, None)
    forwarded_payload["browser_config"] = browser_config

    crawler_config = forwarded_payload.get("crawler_config")
    crawler_config = dict(crawler_config) if isinstance(crawler_config, dict) else {}
    for field in (
        "proxy_config",
        "proxy_rotation_strategy",
        "proxy_session_id",
        "proxy_session_ttl",
        "proxy_session_auto_release",
    ):
        crawler_config.pop(field, None)
    forwarded_payload["crawler_config"] = crawler_config
    return forwarded_payload


@app.post("/v1/crawl4ai/{operation}")
async def relay_crawl4ai(
    operation: Literal["crawl", "md"],
    payload: dict[str, Any],
    timeout_seconds: float = 120.0,
) -> Response:
    if operation != "crawl":
        raise HTTPException(
            status_code=409,
            detail="Crawl4AI /md cannot be forced through the isolated proxy",
        )

    forwarded_payload = await prepare_crawl_payload(payload)

    total_timeout = min(max(1.0, float(timeout_seconds)), CRAWL4AI_TOTAL_TIMEOUT_SECONDS)
    try:
        async with asyncio.timeout(total_timeout):
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), trust_env=False) as client:
                async with client.stream(
                    "POST",
                    f"{WEB_RUNNER_CRAWL4AI_URL}/crawl",
                    headers=crawl4ai_auth_headers(),
                    json=forwarded_payload,
                ) as upstream:
                    upstream.raise_for_status()
                    body = await _read_limited_response(upstream, CRAWL4AI_MAX_RESPONSE_BYTES)
                    headers = {}
                    if upstream.headers.get("content-type"):
                        headers["content-type"] = upstream.headers["content-type"]
                    if upstream.encoding:
                        headers["x-upstream-encoding"] = upstream.encoding
                    return Response(content=body, status_code=upstream.status_code, headers=headers)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.warning("Isolated Crawl4AI request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Isolated Crawl4AI request failed") from exc


async def healthcheck(socket_path: str = WEB_RUNNER_SOCKET) -> bool:
    transport = httpx.AsyncHTTPTransport(uds=socket_path)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://web-runner",
            timeout=2.0,
            trust_env=False,
        ) as client:
            response = await client.get("/healthz")
            return response.status_code == 200 and response.json() == {"ok": True}
    except Exception:
        return False


def _prepare_socket_path() -> None:
    path = Path(WEB_RUNNER_SOCKET)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_socket():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Research MCP isolated web runner")
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
            uds=WEB_RUNNER_SOCKET,
            log_level=os.getenv("LOG_LEVEL", "INFO").lower(),
            access_log=False,
            proxy_headers=False,
        )
    finally:
        with suppress(FileNotFoundError):
            Path(WEB_RUNNER_SOCKET).unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
