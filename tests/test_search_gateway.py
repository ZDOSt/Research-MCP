import asyncio
import math

import httpx
import pytest

import gateway_fetch
import search_gateway


def test_query_variants_are_bounded_and_intent_aware():
    variants = search_gateway._query_variants(
        "How do I install Docker Compose on Ubuntu?", "balanced"
    )
    assert "install" in variants[0].lower()
    assert any("site:docs.docker.com" in item for item in variants)
    assert len(variants) <= 3


def test_generic_technical_query_uses_general_documentation_variant():
    variants = search_gateway._query_variants(
        "How do I install PostgreSQL on Ubuntu?", "balanced"
    )
    assert any("official documentation guide" in item for item in variants)
    assert all("site:docs.docker.com" not in item for item in variants)


def test_unrelated_compose_verb_does_not_trigger_docker_routing():
    variants = search_gateway._query_variants(
        "Compose an error report for this application", "balanced"
    )
    assert all("site:docs.docker.com" not in item for item in variants)


def test_installing_an_application_with_docker_does_not_target_docker_docs():
    variants = search_gateway._query_variants(
        "How do I install Example with Docker?", "balanced"
    )
    assert all("site:docs.docker.com" not in item for item in variants)
    assert any("official documentation guide" in item for item in variants)


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


def test_docker_documentation_outranks_docker_hub_for_installation_guides():
    query = "How do I install Docker Compose on Ubuntu?"
    documentation = {
        "title": "Install Docker Compose on Ubuntu",
        "url": "https://docs.docker.com/engine/install/ubuntu/",
        "content": "Official Docker installation documentation for Ubuntu",
    }
    hub = {
        "title": "Docker Compose container image",
        "url": "https://hub.docker.com/r/docker/compose",
        "content": "Docker Compose image overview",
    }
    assert search_gateway._candidate_score(
        query, documentation, 2
    ) > search_gateway._candidate_score(query, hub, 1)


def test_lexical_reranking_preserves_source_authority():
    documents = [
        {"text": "Docker Compose installation guide", "authority_score": -0.35},
        {"text": "Docker Compose installation guide", "authority_score": 2.6},
    ]
    ranked = search_gateway._lexical_rerank(
        "Docker Compose installation guide", documents, 2
    )
    assert ranked[0]["authority_score"] == 2.6


def test_url_canonicalization_and_source_matching_are_strict():
    normalized = search_gateway._normalize_search_result(
        {
            "title": "Guide",
            "url": "HTTPS://Example.COM:443/guide#section",
            "content": "Useful guide",
        },
        "guide",
        1,
    )
    assert normalized is not None
    assert normalized["url"] == "https://example.com/guide"
    assert search_gateway._canonical_url("https://example.com/" + "a" * 8192) is None
    assert search_gateway._source_adjustment("notdocs.example.com") == 0.0
    assert search_gateway._source_adjustment("docs.example.com") > 0.0
    assert search_gateway._root_domain("docs.example.co.uk") == "example.co.uk"


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


@pytest.mark.asyncio
async def test_reranker_uses_source_authority_for_equal_relevance(monkeypatch):
    response = httpx.Response(
        200,
        json=[{"index": 0, "score": 0.5}, {"index": 1, "score": 0.5}],
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
        {"text": "Docker Compose installation", "authority_score": -0.35},
        {"text": "Docker Compose installation", "authority_score": 2.6},
    ]
    ranked, status = await search_gateway._rerank(
        "Docker Compose installation", docs, 2
    )
    assert status == "ok"
    assert ranked[0]["authority_score"] == 2.6


@pytest.mark.asyncio
async def test_reranker_rejects_non_finite_scores(monkeypatch):
    response = httpx.Response(
        200,
        content=b'[{"index":0,"score":NaN},{"index":1,"score":0.5}]',
        headers={"content-type": "application/json"},
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
        {"text": "first answer", "candidate_index": 0},
        {"text": "second answer", "candidate_index": 1},
    ]
    ranked, status = await search_gateway._rerank("answer", docs, 2)
    assert status == "partial"
    assert all(math.isfinite(item["rerank_score"]) for item in ranked)


@pytest.mark.asyncio
async def test_crawl_deadline_keeps_completed_pages(monkeypatch):
    candidates = [
        {"title": "Fast", "url": "https://fast.example/"},
        {"title": "Slow", "url": "https://slow.example/"},
    ]

    async def fetch(url, *args, **kwargs):
        if "slow" in url:
            await asyncio.sleep(2)
        return {"url": url, "content": "evidence", "links": []}

    monkeypatch.setattr(search_gateway, "fetch_page", fetch)
    pages, failures = await search_gateway._crawl_candidates(
        candidates, "evidence", search_gateway.time.monotonic() + 1.05
    )
    assert [page["url"] for page in pages] == ["https://fast.example/"]
    assert failures == [{"url": "https://slow.example/", "error": "deadline"}]


@pytest.mark.asyncio
async def test_follow_links_canonicalizes_and_deduplicates_fragments(monkeypatch):
    captured = []

    async def crawl(candidates, query, deadline):
        captured.extend(candidates)
        return [], []

    pages = [
        {
            "url": "https://docs.example.co.uk/start",
            "title": "Start",
            "links": [
                {
                    "url": "https://docs.example.co.uk:443/install#one",
                    "anchor": "Install guide",
                },
                {
                    "url": "https://docs.example.co.uk/install#two",
                    "anchor": "Install guide duplicate",
                },
            ],
        }
    ]
    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)
    await search_gateway._follow_relevant_links(
        pages, "install guide", 5, search_gateway.time.monotonic() + 10
    )
    assert [item["url"] for item in captured] == [
        "https://docs.example.co.uk/install"
    ]


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
    assert result["diagnostics"]["partial"] is True


@pytest.mark.asyncio
async def test_final_rerank_timeout_keeps_crawled_evidence(monkeypatch):
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
                    "title": "Crawled guide",
                    "url": "https://example.com/guide",
                    "domain": "example.com",
                    "snippet": "short search snippet",
                    "search_rank": 1,
                    "discovery_score": 1.0,
                    "published_at": None,
                    "engines": ["bing"],
                }
            ],
            [],
        )

    async def rerank(query, docs, top_k):
        if docs and "page_index" in docs[0]:
            await asyncio.sleep(1)
            return [], "late"
        return [dict(docs[0], rerank_score=1.0)], "ok"

    async def crawl(candidates, query, deadline):
        return (
            [
                {
                    "url": candidates[0]["url"],
                    "title": candidates[0]["title"],
                    "content": "Detailed crawled installation evidence. " * 20,
                    "content_chars": 800,
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
    monkeypatch.setitem(
        search_gateway.MODE_BUDGETS,
        "quick",
        search_gateway.Budget(4, 1, 0, 2, 0.05),
    )
    monkeypatch.setattr(search_gateway, "FINALIZATION_RESERVE_SECONDS", 0.01)
    monkeypatch.setattr(search_gateway, "CANDIDATE_RERANKER_TIMEOUT_SECONDS", 0.01)

    result = await search_gateway.research(
        search_gateway.SearchRequest(query="installation evidence", mode="quick")
    )
    assert result["results"][0]["engine"] == "search-gateway"
    assert "Detailed crawled installation evidence" in result["results"][0]["content"]
    assert result["diagnostics"]["reranker"] == "fallback:deadline"


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
