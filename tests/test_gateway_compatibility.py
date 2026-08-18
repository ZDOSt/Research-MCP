import asyncio

import httpx
import pytest

import search_gateway


def _client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=search_gateway.app),
        base_url="http://gateway",
    )


@pytest.mark.asyncio
async def test_search_is_discovery_only_and_searx_compatible(monkeypatch):
    search_gateway._CACHE.clear()
    search_gateway._QUERY_LOCKS.clear()
    search_gateway._QUERY_LOCK_USERS.clear()

    async def no_cache(_):
        return None

    async def ignore_cache(*_):
        return None

    async def search(*args, **kwargs):
        return (
            [
                {
                    "title": "Guide",
                    "url": "https://docs.example/guide",
                    "snippet": "Useful guide",
                    "discovery_score": 2.0,
                    "engines": ["brave"],
                }
            ],
            [{"status": "ok", "result_count": 1}],
        )

    async def crawl(*args, **kwargs):
        raise AssertionError("discovery endpoint must not crawl")

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)

    async with _client() as client:
        response = await client.get("/search", params={"q": "install example"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["content"] == "Useful guide"
    assert payload["results"][0]["url"] == "https://docs.example/guide"
    assert payload["diagnostics"]["pipeline"] == "discovery"


@pytest.mark.asyncio
async def test_integrated_search_uses_bounded_research_budget(monkeypatch):
    captured = {}

    async def fake_research(request, *, budget_override, pipeline):
        captured.update(
            request=request, budget=budget_override, pipeline=pipeline
        )
        return {"query": request.query, "results": [], "number_of_results": 0}

    monkeypatch.setattr(search_gateway, "research", fake_research)
    async with _client() as client:
        response = await client.get(
            "/integrated/search", params={"q": "how to install example"}
        )

    assert response.status_code == 200
    assert captured["pipeline"] == "integrated"
    assert captured["budget"].crawl_pages == search_gateway.INTEGRATED_MAX_CRAWL_PAGES
    assert captured["budget"].follow_links == 0


@pytest.mark.asyncio
async def test_firecrawl_scrape_requires_bearer_and_returns_markdown(monkeypatch):
    monkeypatch.setattr(search_gateway, "FIRECRAWL_API_KEY", "test-key")
    called = []

    async def no_cache(_):
        return None

    async def ignore_cache(*_):
        return None

    async def fetch(url, query, timeout, **kwargs):
        called.append((url, query, timeout, kwargs))
        return {
            "url": "https://example.com/final",
            "title": "Example",
            "content": "# Extracted content",
            "status_code": 200,
            "links": [{"url": "https://example.com/next"}],
        }

    monkeypatch.setattr(search_gateway, "fetch_page", fetch)
    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    async with _client() as client:
        unauthorized = await client.post(
            "/v2/scrape", json={"url": "https://example.com"}
        )
        response = await client.post(
            "/v2/scrape",
            headers={"Authorization": "Bearer test-key"},
            json={"url": "https://example.com", "formats": ["markdown"]},
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["data"]["markdown"] == "# Extracted content"
    assert payload["data"]["metadata"]["sourceURL"] == "https://example.com/final"
    assert called and called[0][0] == "https://example.com/"


@pytest.mark.asyncio
async def test_firecrawl_scrape_rejects_non_markdown_formats(monkeypatch):
    monkeypatch.setattr(search_gateway, "FIRECRAWL_API_KEY", "test-key")
    async with _client() as client:
        response = await client.post(
            "/v2/scrape",
            headers={"Authorization": "Bearer test-key"},
            json={"url": "https://example.com", "formats": ["screenshot"]},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_firecrawl_scrape_preserves_ssrf_rejection(monkeypatch):
    monkeypatch.setattr(search_gateway, "FIRECRAWL_API_KEY", "test-key")

    async def no_cache(_):
        return None

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    async with _client() as client:
        response = await client.post(
            "/v2/scrape",
            headers={"Authorization": "Bearer test-key"},
            json={"url": "http://127.0.0.1/private"},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_cached_firecrawl_scrape_bypasses_saturated_admission(monkeypatch):
    monkeypatch.setattr(search_gateway, "FIRECRAWL_API_KEY", "test-key")
    cached = {
        "cached_at": search_gateway.time.time(),
        "result": {
            "success": True,
            "data": {"markdown": "cached", "metadata": {}},
        },
    }

    async def cache_get(_):
        return cached

    async def fetch(*args, **kwargs):
        raise AssertionError("fresh scrape cache must bypass retrieval")

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    monkeypatch.setattr(search_gateway, "_cache_get", cache_get)
    monkeypatch.setattr(search_gateway, "fetch_page", fetch)
    monkeypatch.setattr(search_gateway, "_ADMISSION", semaphore)
    async with _client() as client:
        response = await client.post(
            "/v2/scrape",
            headers={"Authorization": "Bearer test-key"},
            json={"url": "https://cached.example"},
        )
    semaphore.release()
    assert response.status_code == 200
    assert response.json()["data"]["markdown"] == "cached"


@pytest.mark.asyncio
async def test_uncached_firecrawl_scrape_obeys_admission_limit(monkeypatch):
    monkeypatch.setattr(search_gateway, "FIRECRAWL_API_KEY", "test-key")

    async def no_cache(_):
        return None

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_ADMISSION", semaphore)
    monkeypatch.setattr(search_gateway, "ADMISSION_TIMEOUT_SECONDS", 0.01)
    async with _client() as client:
        response = await client.post(
            "/v2/scrape",
            headers={"Authorization": "Bearer test-key"},
            json={"url": "https://uncached.example"},
        )
    semaphore.release()
    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


@pytest.mark.asyncio
async def test_firecrawl_search_returns_v2_web_results(monkeypatch):
    monkeypatch.setattr(search_gateway, "FIRECRAWL_API_KEY", "test-key")

    async def fake_discovery(request):
        assert request.query == "docker compose install"
        return {
            "results": [
                {
                    "title": "Install guide",
                    "url": "https://docs.example/install",
                    "snippet": "Official instructions",
                }
            ]
        }

    monkeypatch.setattr(search_gateway, "discovery_search", fake_discovery)
    async with _client() as client:
        response = await client.post(
            "/v2/search",
            headers={"Authorization": "Bearer test-key"},
            json={"query": "docker compose install", "limit": 3},
        )
    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {
            "web": [
                {
                    "url": "https://docs.example/install",
                    "title": "Install guide",
                    "description": "Official instructions",
                }
            ]
        },
    }
