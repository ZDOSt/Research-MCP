import asyncio
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

import gateway_pdf_runner


@pytest.mark.asyncio
async def test_pdf_runner_rejects_oversized_body_before_parsing(monkeypatch):
    parser = AsyncMock()
    monkeypatch.setattr(gateway_pdf_runner, "PDF_MAX_RESPONSE_BYTES", 4)
    monkeypatch.setattr(gateway_pdf_runner, "parse_pdf", parser)
    transport = httpx.ASGITransport(app=gateway_pdf_runner.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://pdf") as client:
        response = await client.post("/v1/extract", content=b"12345")
    assert response.status_code == 413
    parser.assert_not_awaited()


def test_pdf_runner_requires_network_isolation_marker():
    with patch.dict(
        os.environ,
        {"RESEARCH_PDF_NETWORK_ISOLATED": "false"},
        clear=False,
    ):
        with pytest.raises(RuntimeError, match="internal Docker network"):
            gateway_pdf_runner.require_isolated_runtime()


@pytest.mark.asyncio
async def test_pdf_subprocess_receives_no_application_secrets(monkeypatch):
    class Process:
        returncode = 0

        async def communicate(self, body):
            return b'{"content":"ok","title":null,"error":null}', b""

    create_process = AsyncMock(return_value=Process())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    with patch.dict(
        os.environ,
        {
            "CRAWL4AI_API_TOKEN": "do-not-copy",
            "SEARXNG_SECRET": "do-not-copy",
        },
        clear=False,
    ):
        result = await gateway_pdf_runner.parse_pdf(b"%PDF")
    assert result["content"] == "ok"
    child_environment = create_process.await_args.kwargs["env"]
    assert "CRAWL4AI_API_TOKEN" not in child_environment
    assert "SEARXNG_SECRET" not in child_environment


@pytest.mark.asyncio
async def test_pdf_subprocess_is_reaped_when_request_is_cancelled(monkeypatch):
    communicating = asyncio.Event()

    class Process:
        returncode = None
        killed = False
        waited = False

        async def communicate(self, body):
            communicating.set()
            await asyncio.Future()

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            self.waited = True
            return self.returncode

    process = Process()
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    task = asyncio.create_task(gateway_pdf_runner.parse_pdf(b"%PDF"))
    await communicating.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.killed is True
    assert process.waited is True
