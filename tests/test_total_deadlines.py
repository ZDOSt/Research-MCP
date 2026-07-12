import asyncio

import pytest
from fastapi import HTTPException

import github_connector
import planner
import searching
import shared


class _ExpiredTimeout:
    async def __aenter__(self):
        raise TimeoutError

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_remote_rag_total_deadline_returns_gateway_timeout(monkeypatch):
    observed = []

    def expire_after(seconds):
        observed.append(seconds)
        return _ExpiredTimeout()

    monkeypatch.setattr(shared, "RESEARCH_API_URL", "https://rag.example.com")
    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "secret")
    monkeypatch.setattr(shared, "RESEARCH_API_TOTAL_TIMEOUT_SECONDS", 7.0)
    monkeypatch.setattr(shared.asyncio, "timeout", expire_after)

    with pytest.raises(HTTPException) as caught:
        await shared._remote_rag_request("GET", "/rag/source-stats")

    assert caught.value.status_code == 504
    assert "total deadline" in caught.value.detail
    assert observed == [7.0]


@pytest.mark.asyncio
async def test_reranker_total_deadline_falls_back_to_vector_order(monkeypatch):
    observed = []

    def expire_after(seconds):
        observed.append(seconds)
        return _ExpiredTimeout()

    docs = [{"text": "first"}, {"text": "second"}]
    monkeypatch.setattr(shared, "RERANKER_TOTAL_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(shared.asyncio, "timeout", expire_after)

    assert await shared.rerank_docs("query", docs, 2) == docs
    assert observed == [3.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "timeout_name", "timeout_seconds", "invoke"),
    [
        (
            searching,
            "SEARXNG_TIMEOUT_SECONDS",
            2.0,
            lambda: searching.searxng_search("query"),
        ),
        (
            planner,
            "PLANNER_TIMEOUT_SECONDS",
            3.0,
            lambda: planner._chat([{"role": "user", "content": "query"}]),
        ),
        (
            github_connector,
            "GITHUB_TIMEOUT_SECONDS",
            4.0,
            lambda: github_connector._github_get("/rate_limit"),
        ),
    ],
)
async def test_streamed_clients_enforce_total_deadline(
    monkeypatch,
    module,
    timeout_name,
    timeout_seconds,
    invoke,
):
    observed = []

    def expire_after(seconds):
        observed.append(seconds)
        return _ExpiredTimeout()

    if module is searching:
        monkeypatch.setattr(searching, "SEARXNG_URL", "http://searxng:8080")
    elif module is planner:
        monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://planner.example/v1")
        monkeypatch.setattr(planner, "PLANNER_MODEL", "private-model")

    monkeypatch.setattr(module, timeout_name, timeout_seconds)
    monkeypatch.setattr(module.asyncio, "timeout", expire_after)

    with pytest.raises(TimeoutError):
        await invoke()

    assert observed == [timeout_seconds]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "timeout_name", "configure", "invoke"),
    [
        (
            searching,
            "SEARXNG_TIMEOUT_SECONDS",
            lambda monkeypatch: monkeypatch.setattr(
                searching,
                "SEARXNG_URL",
                "http://searxng:8080",
            ),
            lambda: searching.searxng_search("query"),
        ),
        (
            planner,
            "PLANNER_TIMEOUT_SECONDS",
            lambda monkeypatch: (
                monkeypatch.setattr(
                    planner,
                    "PLANNER_BASE_URL",
                    "https://planner.example/v1",
                ),
                monkeypatch.setattr(planner, "PLANNER_MODEL", "private-model"),
            ),
            lambda: planner._chat([{"role": "user", "content": "query"}]),
        ),
        (
            github_connector,
            "GITHUB_TIMEOUT_SECONDS",
            lambda _monkeypatch: None,
            lambda: github_connector._github_get("/rate_limit"),
        ),
    ],
)
async def test_streamed_clients_stop_slow_drip_responses(
    monkeypatch,
    module,
    timeout_name,
    configure,
    invoke,
):
    class SlowResponse:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            while True:
                await asyncio.sleep(0.02)
                yield b" "

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return SlowResponse()

    configure(monkeypatch)
    monkeypatch.setattr(module, timeout_name, 0.01)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(TimeoutError):
        await invoke()
