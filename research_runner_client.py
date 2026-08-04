"""Gateway client for the private interactive research runner."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Mapping

import httpx


RESEARCH_RUNNER_SOCKET = os.getenv(
    "RESEARCH_RUNNER_SOCKET", "/run/research-interactive/runner.sock"
)
RESEARCH_RUNNER_MAX_RESPONSE_BYTES = max(
    4096,
    int(os.getenv("RESEARCH_RUNNER_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))),
)


async def _read_bounded_body(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > RESEARCH_RUNNER_MAX_RESPONSE_BYTES:
            raise ValueError("interactive research response exceeds the byte limit")
    return bytes(body)


async def run_interactive_research(
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute one ordinary research request without touching Redis queues."""
    transport = httpx.AsyncHTTPTransport(uds=RESEARCH_RUNNER_SOCKET)
    timeout = httpx.Timeout(
        connect=2.0,
        read=max(1.0, float(timeout_seconds) + 5.0),
        write=5.0,
        pool=2.0,
    )
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://research-runner",
            timeout=timeout,
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST", "/v1/research_assistant", json=dict(payload)
            ) as response:
                body = await _read_bounded_body(response)
    except asyncio.CancelledError:
        raise
    except ValueError:
        return {
            "error": "interactive_runner_response_too_large",
            "terminal": True,
            "answering_instructions": [
                "The interactive research response exceeded the configured size limit; do not invent an answer."
            ],
        }
    except Exception as exc:
        return {
            "error": "interactive_runner_unavailable",
            "terminal": True,
            "detail": type(exc).__name__,
            "answering_instructions": [
                "The private interactive research runner was unavailable. Do not claim that web research succeeded.",
                "Retry only after the server administrator restores the runner service.",
            ],
        }
    try:
        result = httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=body,
        ).json()
    except (TypeError, ValueError):
        result = None
    if response.status_code < 200 or response.status_code >= 300 or not isinstance(result, dict):
        return {
            "error": "interactive_runner_failed",
            "terminal": True,
            "detail": "runner returned an invalid response",
            "answering_instructions": [
                "Interactive research did not complete. Do not claim that web research succeeded."
            ],
        }
    if result.get("status") in {"queued", "running"}:
        return {
            "error": "interactive_runner_non_terminal",
            "terminal": True,
            "answering_instructions": [
                "The interactive runner returned a non-terminal response; do not present it as a completed answer."
            ],
        }
    return result
