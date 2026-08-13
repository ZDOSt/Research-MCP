import asyncio

import httpx
import pytest

import gateway_fetch
import search_gateway


def test_query_variants_are_bounded_and_intent_aware():
    variants = search_gateway._query_variants(
        "How do I install Example with Docker?", "balanced"
    )
    assert "install" in variants[0].lower()
    assert "example" in variants[0].lower()
    assert any("official documentation guide" in item for item in variants)
    assert len(variants) <= 3


def test_search_categories_are_inferred_from_intent():
    assert search_gateway._search_categories("Docker install error", []) == [
        "general",
        "it",
    ]
    assert search_gateway._search_categories("latest research study", []) == [
        "general",
        "news",
        "science",
    ]
    assert search_gateway._search_categories("anything", ["images"]) == ["images"]
    assert search_gateway._search_categories("find images of a nebula", []) == [
        "general",
        "images",
    ]


def test_game_query_uses_strategy_terms_not_product_measurements():
    variants = search_gateway._query_variants(
        "Best team composition in DragonSword Awakening", "balanced"
    )
    assert any("strategy guide wiki" in item for item in variants)
    assert all("measurements" not in item for item in variants)


def test_candidate_scoring_prefers_relevant_official_sources():
    official = {
        "title": "AW3426DW official support manual and settings",
        "url": "https://support.example.com/AW3426DW/settings",
        "content": "Recommended AW3426DW SDR HDR settings",
    }
    weak = {
        "title": "Unrelated monitor roundup",
        "url": "https://example.net/roundup",
        "content": "Many displays are available.",
    }
    query = "What are the best settings for my AW3426DW?"
    assert search_gateway._candidate_score(query, official, 2) > search_gateway._candidate_score(
        query, weak, 1
    )


@pytest.mark.asyncio
async def test_reranker_falls_back_to_lexical_order(monkeypatch):
    class FailingClient:
        async def __aenter__(self):
            raise httpx.ConnectError("offline")

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: FailingClient())
    docs = [
        {"text": "unrelated text", "candidate_index": 0},
        {"text": "AW3426DW recommended HDR settings", "candidate_index": 1},
    ]
    ranked, status = await search_gateway._rerank("AW3426DW HDR settings", docs, 2)
    assert status.startswith("fallback:")
    assert ranked[0]["candidate_index"] == 1


@pytest.mark.asyncio
async def test_reranker_preserves_unscored_documents(monkeypatch):
    response = httpx.Response(
        200,
        json=[{"index": 0, "score": 0.8}],
        request=httpx.Request("POST", "http://reranker/rerank"),
    )

    class Stream:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *args):
            return False

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return Stream()

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    docs = [
        {"text": "primary answer", "candidate_index": 0},
        {"text": "secondary answer", "candidate_index": 1},
    ]
    ranked, status = await search_gateway._rerank("answer", docs, 2)
    assert status == "partial"
    assert {item["candidate_index"] for item in ranked} == {0, 1}


def test_assemble_results_diversifies_domains_before_duplicates():
    pages = [
        {
            "url": "https://docs.example.com/a",
            "title": "A",
            "content": "A",
            "content_chars": 1,
            "extraction_method": "direct",
            "search": {"engines": ["x"], "snippet": ""},
        },
        {
            "url": "https://blog.example.com/b",
            "title": "B",
            "content": "B",
            "content_chars": 1,
            "extraction_method": "direct",
            "search": {"engines": ["x"], "snippet": ""},
        },
        {
            "url": "https://independent.test/c",
            "title": "C",
            "content": "C",
            "content_chars": 1,
            "extraction_method": "direct",
            "search": {"engines": ["x"], "snippet": ""},
        },
    ]
    passages = [
        {"page_index": 0, "text": "A evidence", "rerank_score": 0.99},
        {"page_index": 1, "text": "B evidence", "rerank_score": 0.98},
        {"page_index": 2, "text": "C evidence", "rerank_score": 0.90},
    ]
    results = search_gateway._assemble_results(pages, passages, 3)
    assert [item["title"] for item in results] == ["A", "C", "B"]


@pytest.mark.asyncio
async def test_fetch_page_uses_direct_result_without_expensive_fallback(monkeypatch):
    calls = []

    async def validate(url):
        return url

    async def direct(url, timeout):
        calls.append("direct")
        return {"content": "useful " * 500, "content_chars": 3500}

    async def crawl(*args):
        calls.append("crawl4ai")
        raise AssertionError("should not be called")

    monkeypatch.setattr(gateway_fetch, "validate_public_url", validate)
    monkeypatch.setattr(gateway_fetch, "direct_fetch", direct)
    monkeypatch.setattr(gateway_fetch, "crawl4ai_fetch", crawl)
    result = await gateway_fetch.fetch_page("https://example.com", "query", 10)
    assert result["content_chars"] == 3500
    assert calls == ["direct"]


@pytest.mark.asyncio
async def test_fetch_page_escalates_to_crawl4ai(monkeypatch):
    async def validate(url):
        return url

    async def direct(url, timeout):
        return {"content": "thin", "content_chars": 4}

    async def crawl(url, timeout):
        return {
            "content": "relevant content " * 100,
            "content_chars": 1700,
            "extraction_method": "crawl4ai",
        }

    monkeypatch.setattr(gateway_fetch, "validate_public_url", validate)
    monkeypatch.setattr(gateway_fetch, "direct_fetch", direct)
    monkeypatch.setattr(gateway_fetch, "crawl4ai_fetch", crawl)
    result = await gateway_fetch.fetch_page("https://example.com", "query", 10)
    assert result["extraction_method"] == "crawl4ai"


@pytest.mark.asyncio
async def test_research_returns_searx_compatible_enriched_results(monkeypatch):
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
                    "title": "Official guide",
                    "url": "https://docs.example.com/install",
                    "domain": "docs.example.com",
                    "snippet": "Install Example with Docker Compose",
                    "search_rank": 1,
                    "discovery_score": 4.0,
                    "published_at": None,
                    "engines": ["bing"],
                }
            ],
            [{"status": "ok", "result_count": 1}],
        )

    async def rerank(query, docs, top_k):
        output = []
        for index, doc in enumerate(docs[:top_k]):
            item = dict(doc)
            item["rerank_score"] = 1.0 - index * 0.01
            output.append(item)
        return output, "ok"

    async def crawl(candidates, query, deadline):
        return (
            [
                {
                    "url": candidates[0]["url"],
                    "title": candidates[0]["title"],
                    "content": "Docker Compose installation steps and configuration. " * 80,
                    "content_chars": 4400,
                    "links": [],
                    "extraction_method": "direct",
                    "search": candidates[0],
                }
            ],
            [],
        )

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_rerank", rerank)
    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)

    response = await search_gateway.research(
        search_gateway.SearchRequest(
            query="How do I install Example with Docker Compose?", max_results=3
        )
    )
    assert response["query"].startswith("How do I")
    assert response["number_of_results"] == 1
    assert response["results"][0]["url"] == "https://docs.example.com/install"
    assert "Docker Compose installation" in response["results"][0]["content"]


@pytest.mark.asyncio
async def test_research_uses_snippets_when_every_page_is_blocked(monkeypatch):
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
                    "title": "Useful result",
                    "url": "https://example.com/answer",
                    "domain": "example.com",
                    "snippet": "A useful search-engine excerpt",
                    "search_rank": 1,
                    "discovery_score": 2.0,
                    "published_at": None,
                    "engines": ["qwant"],
                }
            ],
            [],
        )

    async def rerank(query, docs, top_k):
        return ([dict(docs[0], rerank_score=1.0)] if docs else []), "ok"

    async def crawl(candidates, query, deadline):
        return [], [{"url": candidates[0]["url"], "error": "blocked"}]

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_rerank", rerank)
    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)

    response = await search_gateway.research(
        search_gateway.SearchRequest(query="specific fact", max_results=2)
    )
    assert response["results"][0]["content"] == "A useful search-engine excerpt"
    assert response["diagnostics"]["fallback"] == "search-snippets"


@pytest.mark.asyncio
async def test_research_cleans_up_query_lock(monkeypatch):
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
                    "title": "Result",
                    "url": "https://example.com/answer",
                    "domain": "example.com",
                    "snippet": "Useful answer",
                    "search_rank": 1,
                    "discovery_score": 1.0,
                    "published_at": None,
                    "engines": ["bing"],
                }
            ],
            [],
        )

    async def rerank(query, docs, top_k):
        return [dict(docs[0], rerank_score=1.0)], "ok"

    async def crawl(candidates, query, deadline):
        return [], []

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_rerank", rerank)
    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)

    request = search_gateway.SearchRequest(query="answer")
    key = search_gateway._cache_key(request)
    await search_gateway.research(request)
    assert key not in search_gateway._QUERY_LOCKS
    assert key not in search_gateway._QUERY_LOCK_USERS


@pytest.mark.asyncio
async def test_total_deadline_returns_discovered_snippet(monkeypatch):
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
                    "title": "Deadline result",
                    "url": "https://example.com/deadline",
                    "domain": "example.com",
                    "snippet": "Useful evidence discovered before the deadline",
                    "search_rank": 1,
                    "discovery_score": 1.0,
                    "published_at": None,
                    "engines": ["brave"],
                }
            ],
            [],
        )

    async def slow_rerank(query, docs, top_k):
        await asyncio.sleep(0.05)
        return [], "late"

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_rerank", slow_rerank)
    monkeypatch.setitem(
        search_gateway.MODE_BUDGETS,
        "quick",
        search_gateway.Budget(4, 1, 0, 2, 0.01),
    )

    result = await search_gateway.research(
        search_gateway.SearchRequest(query="deadline", mode="quick")
    )
    assert result["results"][0]["content"].startswith("Useful evidence")
    assert result["diagnostics"]["deadline_exceeded"] is True


@pytest.mark.asyncio
async def test_cancelled_lock_waiter_does_not_remove_active_lock():
    search_gateway._QUERY_LOCKS.clear()
    search_gateway._QUERY_LOCK_USERS.clear()
    request = search_gateway.SearchRequest(query="coalesced query")
    key = search_gateway._cache_key(request)
    lock = asyncio.Lock()
    await lock.acquire()
    search_gateway._QUERY_LOCKS[key] = lock
    search_gateway._QUERY_LOCK_USERS[key] = 1

    task = asyncio.create_task(search_gateway.research(request))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert search_gateway._QUERY_LOCKS[key] is lock
    lock.release()
    search_gateway._QUERY_LOCKS.clear()
    search_gateway._QUERY_LOCK_USERS.clear()
