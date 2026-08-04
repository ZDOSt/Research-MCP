import importlib
import os
import stat
import sys
from unittest.mock import AsyncMock, patch

import pytest
import httpx


def load_runner(monkeypatch):
    monkeypatch.setenv("RESEARCH_RUNNER_MAX_REQUEST_BYTES", "1024")
    monkeypatch.setenv("RESEARCH_RUNNER_MAX_RESPONSE_BYTES", "4096")
    sys.modules.pop("research_runner", None)
    return importlib.import_module("research_runner")


@pytest.mark.asyncio
async def test_runner_executes_without_persistence(monkeypatch):
    runner = load_runner(monkeypatch)
    result = {"status": "complete", "evidence": [{"url": "https://example.test"}]}
    call = AsyncMock(return_value=result)
    with patch("research_agent.run_research_assistant", call):
        response = await runner.research_assistant(
            runner.ResearchRequest(
                request="Find current documentation",
                mode="balanced",
                namespace="docs",
                time_budget_seconds=12,
            )
        )

    assert response == result
    kwargs = call.await_args.kwargs
    assert kwargs["persist_source_artifacts"] is False
    assert kwargs["defer_persistence"] is True
    assert kwargs["namespace"] == "docs"
    assert kwargs["time_budget_seconds"] == 12


@pytest.mark.asyncio
async def test_runner_rejects_non_terminal_result(monkeypatch):
    runner = load_runner(monkeypatch)
    with patch(
        "research_agent.run_research_assistant",
        AsyncMock(return_value={"status": "running", "terminal": False}),
    ):
        with pytest.raises(runner.HTTPException) as exc:
            await runner.research_assistant(
                runner.ResearchRequest(request="q")
            )
    assert exc.value.status_code == 502


@pytest.mark.asyncio
async def test_client_returns_terminal_error_when_socket_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCH_RUNNER_SOCKET", str(tmp_path / "missing.sock"))
    sys.modules.pop("research_runner_client", None)
    client = importlib.import_module("research_runner_client")
    result = await client.run_interactive_research(
        {"request": "q", "mode": "quick", "namespace": "default"},
        timeout_seconds=1,
    )
    assert result["error"] == "interactive_runner_unavailable"
    assert result["terminal"] is True
    assert "running" not in result


def test_runner_request_policy_rejects_unknown_fields(monkeypatch):
    runner = load_runner(monkeypatch)
    with pytest.raises(ValueError):
        runner.ResearchRequest(request="q", unknown="value")


def test_runner_request_policy_bounds_github_claims(monkeypatch):
    runner = load_runner(monkeypatch)
    with pytest.raises(ValueError):
        runner.ResearchRequest(
            request="q",
            github_access_policy={"allowed": True, "repositories": ["x"] * 257},
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX socket mode bits are not enforced on Windows")
@pytest.mark.asyncio
async def test_runner_lifespan_restricts_socket_permissions(monkeypatch, tmp_path):
    runner = load_runner(monkeypatch)
    socket_path = tmp_path / "runner.sock"
    socket_path.touch()
    monkeypatch.setattr(runner, "RESEARCH_RUNNER_SOCKET", str(socket_path))

    async with runner.lifespan(runner.app):
        mode = stat.S_IMODE(os.stat(socket_path).st_mode)
        assert mode & 0o007 == 0
        assert mode & 0o660 == 0o660


@pytest.mark.asyncio
async def test_client_stops_reading_oversized_response(monkeypatch):
    monkeypatch.setenv("RESEARCH_RUNNER_MAX_RESPONSE_BYTES", "4096")
    sys.modules.pop("research_runner_client", None)
    client = importlib.import_module("research_runner_client")

    response = httpx.Response(
        200,
        content=b"x" * 4097,
        request=httpx.Request("POST", "http://research-runner/v1/research_assistant"),
    )
    with pytest.raises(ValueError):
        await client._read_bounded_body(response)
