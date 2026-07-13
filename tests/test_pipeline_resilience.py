import asyncio
from unittest.mock import AsyncMock

import pytest

import pipelines
from searching import SearchResults


@pytest.mark.asyncio
async def test_crawl_and_ingest_preserves_search_provenance(monkeypatch):
    captured = []

    async def ingest(request):
        captured.append(request)
        return {"stored": 1}

    monkeypatch.setattr(
        pipelines,
        "crawl_url_impl",
        AsyncMock(
            return_value={
                "url": "https://news.example/story",
                "title": "Current story",
                "content": "Substantive current reporting. " * 100,
                "extraction_method": "direct_http",
            }
        ),
    )
    monkeypatch.setattr(pipelines, "rag_ingest_impl", ingest)

    result = await pipelines.crawl_and_ingest(
        {
            "title": "Current story",
            "url": "https://news.example/story",
            "domain": "news.example",
            "published_at": "2026-07-13T08:30:00+00:00",
            "freshness_status": "exact_match",
            "engine": "google news",
            "search_rank": 2,
            "score": 8.5,
            "score_reasons": ["publication date matches requested day"],
        },
        query="today's news",
        persist_source_artifacts=False,
    )

    metadata = captured[0].metadata
    assert metadata["published_at"] == "2026-07-13T08:30:00+00:00"
    assert metadata["freshness_status"] == "exact_match"
    assert metadata["search_engine"] == "google news"
    assert metadata["search_rank"] == 2
    assert result["published_at"] == metadata["published_at"]
    assert result["freshness_status"] == "exact_match"


@pytest.mark.asyncio
async def test_crawl_and_ingest_keeps_extraction_when_memory_indexing_fails(
    monkeypatch,
):
    monkeypatch.setattr(
        pipelines,
        "crawl_url_impl",
        AsyncMock(
            return_value={
                "url": "https://docs.example/install",
                "title": "Install guide",
                "content": "Authoritative installation steps. " * 100,
                "extraction_method": "direct_http",
            }
        ),
    )
    monkeypatch.setattr(
        pipelines,
        "rag_ingest_impl",
        AsyncMock(side_effect=RuntimeError("Qdrant unavailable")),
    )

    result = await pipelines.crawl_and_ingest(
        {
            "title": "Install guide",
            "url": "https://docs.example/install",
            "domain": "docs.example",
        },
        query="install the product",
        persist_source_artifacts=False,
    )

    assert result["ok"] is True
    assert result["memory_indexed"] is False
    assert result["stored_chunks"] == 0
    assert result["evidence_text"].startswith("Authoritative installation steps")
    assert result["errors"] == ["memory indexing failed: Qdrant unavailable"]


@pytest.mark.asyncio
async def test_vector_query_failure_returns_extracted_redirect_without_snippet(
    monkeypatch,
):
    candidate = {
        "title": "Current guide",
        "url": "https://origin.example/guide",
        "domain": "origin.example",
        "snippet": "Discovery-only summary",
        "score": 10,
        "score_reasons": [],
    }

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return [candidate]

    async def crawl(_semaphore, _candidate, **_kwargs):
        return {
            "ok": True,
            "title": "Current guide",
            "url": "https://final.example/guide",
            "requested_url": candidate["url"],
            "domain": "final.example",
            "evidence_text": "Extracted installation instructions",
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(side_effect=RuntimeError("Qdrant query unavailable")),
    )

    result = await pipelines.research_pipeline(
        "install the product",
        mode="balanced",
        max_sources=1,
        verify=True,
        persist_source_artifacts=False,
    )

    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["evidence_type"] == "extracted_page_content"
    assert result["evidence"][0]["url"] == "https://final.example/guide"
    assert result["evidence"][0]["requested_url"] == candidate["url"]
    assert "limitations" not in result["evidence"][0]
    assert result["source_coverage"]["extracted_evidence_items"] == 1
    assert result["source_coverage"]["search_snippet_evidence_items"] == 0
    assert result["verification"]["excluded_search_snippet_evidence_items"] == 0


@pytest.mark.asyncio
async def test_research_backfills_a_failed_top_source(monkeypatch):
    candidates = [
        {
            "title": f"Story {index}",
            "url": f"https://source{index}.example/story",
            "domain": f"source{index}.example",
            "snippet": f"Current reporting from source {index}",
            "published_at": "2026-07-13T08:00:00+00:00",
            "freshness_status": "exact_match",
            "score": 10 - index,
            "score_reasons": [],
        }
        for index in range(1, 4)
    ]
    attempts = []

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return candidates

    async def crawl(_semaphore, candidate, **_kwargs):
        attempts.append(candidate["url"])
        if len(attempts) == 1:
            return {
                "ok": False,
                "url": candidate["url"],
                "domain": candidate["domain"],
                "reason": "publisher blocked extraction",
            }
        return {
            "ok": True,
            "url": candidate["url"],
            "domain": candidate["domain"],
            "published_at": candidate["published_at"],
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "current reporting",
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert attempts == [candidates[0]["url"], candidates[1]["url"]]
    assert len(result["selected_for_crawl"]) == 2
    assert len(result["failed_sources"]) == 1
    assert len(result["crawled_sources"]) == 1
    assert result["crawled_sources"][0]["url"] == candidates[1]["url"]
    assert result["evidence"][0]["evidence_type"] == "search_result_snippet"
    assert result["evidence"][0]["confidence"] == "low"


def test_search_snippet_evidence_is_bounded_deduplicated_and_excludes_stale():
    candidates = [
        {
            "title": "Already extracted",
            "url": "https://one.example/story",
            "domain": "one.example",
            "snippet": "Duplicate snippet",
        },
        {
            "title": "Stale result",
            "url": "https://stale.example/story",
            "domain": "stale.example",
            "snippet": "Old reporting",
            "freshness_status": "outside_window",
        },
        {
            "title": "Current result",
            "url": "https://current.example/story",
            "domain": "current.example",
            "snippet": "Current reporting " * 200,
            "published_at": "2026-07-13T09:00:00+00:00",
            "freshness_status": "exact_match",
            "engine": "news",
            "search_rank": 1,
        },
    ]

    evidence = pipelines.build_search_snippet_evidence(
        candidates,
        [{"url": "https://one.example/story", "quote": "Extracted"}],
        limit=2,
    )

    assert len(evidence) == 1
    assert evidence[0]["url"] == "https://current.example/story"
    assert len(evidence[0]["quote"]) == 1600
    assert evidence[0]["published_at"] == "2026-07-13T09:00:00+00:00"
    assert evidence[0]["evidence_type"] == "search_result_snippet"


def test_page_evidence_preserves_publication_and_search_metadata():
    evidence = pipelines.build_evidence_pack(
        [
            {
                "text": "Extracted evidence",
                "url": "https://news.example/story",
                "published_at": "2026-07-13T08:30:00+00:00",
                "freshness_status": "exact_match",
                "search_engine": "google news",
                "search_rank": 3,
            }
        ]
    )

    assert evidence[0]["published_at"] == "2026-07-13T08:30:00+00:00"
    assert evidence[0]["freshness_status"] == "exact_match"
    assert evidence[0]["search_engine"] == "google news"
    assert evidence[0]["search_rank"] == 3
    assert evidence[0]["evidence_type"] == "extracted_page_content"


@pytest.mark.asyncio
async def test_research_uses_one_date_safe_recovery_and_surfaces_diagnostics(
    monkeypatch,
):
    calls = []
    retrieval_context = {
        "retrieved_at_utc": "2026-07-13T12:00:00+00:00",
        "current_date_utc": "2026-07-13",
        "timezone": "America/New_York",
        "current_date_local": "2026-07-13",
        "freshness": "runtime_retrieved",
    }

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": ["today's AI news"]}

    async def search(*, query, max_results, mode, policy):
        calls.append(policy)
        if len(calls) == 1:
            return SearchResults(
                [
                    {
                        "title": "Undated discovery result",
                        "url": "https://undated.example/story",
                        "domain": "undated.example",
                        "snippet": "A potentially current but undated AI report",
                        "published_at": None,
                        "freshness_status": "undated",
                        "score": 4,
                        "score_reasons": [],
                    }
                ],
                diagnostics={
                    "search_policy": policy.to_dict(),
                    "counts": {
                        "raw_results": 1,
                        "returned_results": 1,
                        "exact_match_results": 0,
                        "undated_results": 1,
                        "unresponsive_engines": 1,
                    },
                    "unresponsive_engines": [
                        {"engine": "news-engine", "reason": "timeout"}
                    ],
                },
                policy=policy,
            )
        return SearchResults(
            [
                {
                    "title": "Confirmed current report",
                    "url": "https://current.example/story",
                    "domain": "current.example",
                    "snippet": "AI reporting confirmed for the requested date",
                    "published_at": "2026-07-13T11:00:00+00:00",
                    "freshness_status": "exact_match",
                    "score": 12,
                    "score_reasons": ["publication date exact match"],
                }
            ],
            diagnostics={
                "search_policy": policy.to_dict(),
                "counts": {
                    "raw_results": 1,
                    "returned_results": 1,
                    "exact_match_results": 1,
                    "undated_results": 0,
                    "unresponsive_engines": 0,
                },
                "unresponsive_engines": [],
            },
            policy=policy,
        )

    async def crawl(_semaphore, candidate, **_kwargs):
        return {
            "ok": True,
            "url": candidate["url"],
            "domain": candidate["domain"],
            "published_at": candidate["published_at"],
            "freshness_status": candidate["freshness_status"],
        }

    monkeypatch.setattr(pipelines, "runtime_retrieval_context", lambda: retrieval_context)
    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Tell me today's AI news. Choose the top three articles.",
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert len(calls) == 2
    assert calls[0].time_range == "day"
    assert calls[1].time_range is None
    assert calls[0].strict_date is calls[1].strict_date is True
    assert calls[0].target_date == calls[1].target_date
    assert result["search_fallback"]["reason"] == "insufficient_exact_date_coverage"
    assert result["search_fallback"]["policy_relaxation"] == "engine_time_range_only"
    assert result["search_fallback"]["exact_matches_after"] == 1
    assert [item["phase"] for item in result["search_diagnostics"]] == [
        "initial",
        "freshness_recovery",
    ]
    assert result["search_diagnostics"][0]["unresponsive_engines"] == [
        {"engine": "news-engine", "reason": "timeout"}
    ]
    assert result["evidence"][0]["url"] == "https://current.example/story"
    assert result["evidence"][0]["freshness_status"] == "exact_match"


@pytest.mark.asyncio
async def test_historical_exact_date_does_not_repeat_an_identical_search(monkeypatch):
    query = "AI news on 2026-06-01"
    calls = []

    async def plan(_query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(*, query, max_results, mode, policy):
        calls.append((query, policy))
        return SearchResults([], policy=policy)

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "fallback_search_query", lambda *_args, **_kwargs: query)
    monkeypatch.setattr(pipelines, "searxng_search", search)

    result = await pipelines.research_pipeline(
        query,
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert len(calls) == 1
    assert calls[0][1].strict_date is True
    assert calls[0][1].time_range is None
    assert "search_fallback" not in result


@pytest.mark.asyncio
async def test_fresh_crawl_preview_wins_over_stale_same_url_memory(monkeypatch):
    candidate = {
        "title": "Current installation guide",
        "url": "https://docs.example/install",
        "domain": "docs.example",
        "snippet": "Current discovery snippet",
        "score": 10,
        "score_reasons": [],
    }

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": [query],
            "query_intent_ids": ["install"],
        }

    async def search(**_kwargs):
        return [candidate]

    async def crawl(_semaphore, _candidate, **_kwargs):
        return {
            "ok": True,
            "title": candidate["title"],
            "url": candidate["url"],
            "requested_url": candidate["url"],
            "domain": candidate["domain"],
            "evidence_text": "Fresh instructions extracted from the current page.",
        }

    async def rag_query(request):
        if request.research_run_id:
            return {"results": []}
        return {
            "results": [
                {
                    "title": "Old installation guide",
                    "url": candidate["url"],
                    "domain": candidate["domain"],
                    "text": "Stale instructions retained from an older run.",
                    "research_run_id": "old-run",
                }
            ]
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(pipelines, "rag_query_impl", rag_query)

    result = await pipelines.research_pipeline(
        "install the current release",
        max_sources=1,
        include_memory=True,
        verify=False,
        persist_source_artifacts=False,
    )

    assert [item["quote"] for item in result["evidence"]] == [
        "Fresh instructions extracted from the current page."
    ]
    assert result["evidence"][0]["evidence_type"] == "extracted_page_content"


@pytest.mark.asyncio
async def test_unrelated_memory_does_not_reduce_current_snippet_coverage(monkeypatch):
    candidates = [
        {
            "title": f"Current source {index}",
            "url": f"https://current{index}.example/story",
            "domain": f"current{index}.example",
            "snippet": f"Current discovery evidence {index}",
            "score": 10 - index,
            "score_reasons": [],
        }
        for index in range(2)
    ]

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return candidates

    async def crawl(_semaphore, candidate, **_kwargs):
        return {
            "ok": False,
            "url": candidate["url"],
            "domain": candidate["domain"],
            "reason": "publisher blocked extraction",
        }

    async def rag_query(request):
        if request.research_run_id:
            return {"results": []}
        return {
            "results": [
                {
                    "url": f"https://memory{index}.example/old",
                    "domain": f"memory{index}.example",
                    "text": f"Unrelated retained memory {index}",
                }
                for index in range(6)
            ]
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(pipelines, "rag_query_impl", rag_query)

    result = await pipelines.research_pipeline(
        "current information",
        max_sources=2,
        include_memory=True,
        verify=False,
        persist_source_artifacts=False,
    )

    snippets = [
        item
        for item in result["evidence"]
        if item["evidence_type"] == "search_result_snippet"
    ]
    assert [item["url"] for item in snippets] == [
        candidate["url"] for candidate in candidates
    ]
    assert len(result["evidence"]) == 8


@pytest.mark.asyncio
async def test_crawl_selection_reserves_coverage_for_each_planned_intent(monkeypatch):
    news_candidates = [
        {
            "title": f"AI story {index}",
            "url": f"https://news{index}.example/story",
            "domain": f"news{index}.example",
            "snippet": f"AI news {index}",
            "score": 20 - index,
            "score_reasons": [],
        }
        for index in range(12)
    ]
    install_candidate = {
        "title": "Docker installation documentation",
        "url": "https://docs.example/docker/install",
        "domain": "docs.example",
        "snippet": "Official Docker installation steps",
        "score": 1,
        "score_reasons": [],
    }

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": ["today's AI news", "install Docker"],
            "query_intent_ids": ["news", "install"],
        }

    async def search(*, query, **_kwargs):
        return news_candidates if "news" in query else [install_candidate]

    async def crawl(_semaphore, candidate, **_kwargs):
        return {
            "ok": True,
            "title": candidate["title"],
            "url": candidate["url"],
            "requested_url": candidate["url"],
            "domain": candidate["domain"],
            "evidence_text": candidate["snippet"],
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Tell me today's AI news and explain how to install Docker",
        max_sources=2,
        verify=False,
        persist_source_artifacts=False,
    )

    selected_urls = {item["url"] for item in result["selected_for_crawl"]}
    assert install_candidate["url"] in selected_urls
    assert len(selected_urls & {item["url"] for item in news_candidates}) == 1


def test_extraction_fallback_preview_prefers_query_relevant_lines():
    content = "\n".join(
        [f"Generic introduction line {index}." for index in range(100)]
        + [
            "Docker installation",
            "Create the compose file in the application directory.",
            "Run docker compose up -d to start the service.",
            "Verify the container health before configuring the client.",
        ]
    )

    preview = pipelines._query_focused_evidence_preview(
        content,
        "How do I install Docker with Compose?",
    )

    assert "docker compose up -d" in preview.lower()
    assert len(preview) <= pipelines.CRAWLED_EVIDENCE_PREVIEW_LIMIT
    assert not preview.startswith("Generic introduction line 0.")


@pytest.mark.asyncio
async def test_persistence_uses_source_concurrency_limit(monkeypatch):
    candidates = [
        {
            "title": f"Source {index}",
            "url": f"https://source{index}.example/doc",
            "domain": f"source{index}.example",
            "snippet": f"Source {index} discovery",
            "score": 10 - index,
            "score_reasons": [],
        }
        for index in range(4)
    ]
    active = 0
    maximum_active = 0
    persisted = []

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return candidates

    async def crawl(_semaphore, candidate, **_kwargs):
        return {
            "ok": True,
            "title": candidate["title"],
            "url": candidate["url"],
            "requested_url": candidate["url"],
            "domain": candidate["domain"],
            "evidence_text": candidate["snippet"],
        }

    async def persist(source, **_kwargs):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(0.02)
            persisted.append(source["url"])
            return source
        finally:
            active -= 1

    monkeypatch.setitem(
        pipelines.RESEARCH_MODE_CONFIG,
        "balanced",
        {"max_urls": 4, "search_results": 4, "top_k": 0, "crawl_budget": 1},
    )
    monkeypatch.setattr(pipelines, "RESEARCH_SOURCE_CONCURRENCY", 2)
    monkeypatch.setattr(pipelines, "_persistence_budget_seconds", lambda _budget: 1)
    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(pipelines, "persist_crawled_source", persist)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "compare all sources",
        max_sources=4,
        verify=False,
        persist_source_artifacts=False,
    )

    assert maximum_active == 2
    assert set(persisted) == {candidate["url"] for candidate in candidates}
    assert len(result["crawled_sources"]) == 4


@pytest.mark.asyncio
async def test_persistence_timeout_returns_extraction_and_invalidates_attempt(monkeypatch):
    candidate = {
        "title": "Current guide",
        "url": "https://docs.example/current",
        "domain": "docs.example",
        "snippet": "Current guide discovery",
        "score": 10,
        "score_reasons": [],
    }
    completed_candidate = {
        "title": "Completed guide",
        "url": "https://docs.example/completed",
        "domain": "docs.example",
        "snippet": "Completed guide discovery",
        "score": 9,
        "score_reasons": [],
    }
    persistence_cancelled = asyncio.Event()

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return [candidate, completed_candidate]

    async def crawl(_semaphore, source, **_kwargs):
        return {
            "ok": True,
            "title": source["title"],
            "url": source["url"],
            "requested_url": source["url"],
            "domain": source["domain"],
            "evidence_text": "Fresh installation instructions from the extracted page.",
        }

    async def persist(source, **_kwargs):
        if source["url"] == completed_candidate["url"]:
            return {
                **source,
                "stored_chunks": 3,
                "memory_indexed": True,
            }
        try:
            await asyncio.Event().wait()
        finally:
            persistence_cancelled.set()

    invalidate = AsyncMock(return_value={"invalidated": 0})
    rag_query = AsyncMock(return_value={"results": []})
    monkeypatch.setitem(
        pipelines.RESEARCH_MODE_CONFIG,
        "balanced",
        {"max_urls": 2, "search_results": 2, "top_k": 4, "crawl_budget": 1},
    )
    monkeypatch.setattr(pipelines, "_persistence_budget_seconds", lambda _budget: 0.02)
    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(pipelines, "persist_crawled_source", persist)
    monkeypatch.setattr(pipelines, "invalidate_ingestion_attempt_impl", invalidate)
    monkeypatch.setattr(pipelines, "rag_query_impl", rag_query)

    result = await pipelines.research_pipeline(
        "install the current release",
        max_sources=2,
        verify=False,
        persist_source_artifacts=False,
        ingestion_attempt_id="persistence-timeout-attempt",
    )

    assert persistence_cancelled.is_set()
    assert result["evidence"][0]["quote"].startswith("Fresh installation instructions")
    assert any(
        "persistence exceeded" in error
        for source in result["crawled_sources"]
        for error in source.get("errors", [])
    )
    assert all(source["stored_chunks"] == 0 for source in result["crawled_sources"])
    assert all(
        source["memory_indexed"] is False for source in result["crawled_sources"]
    )
    assert {
        source["memory_index_state"] for source in result["crawled_sources"]
    } == {"revoked"}
    assert result["persistence"] == {
        "budget_seconds": 0.02,
        "timed_out": True,
        "completed_tasks": 1,
        "timed_out_tasks": 1,
        "invalidation": {
            "status": "succeeded",
            "invalidated": 0,
        },
    }
    invalidate.assert_awaited_once_with(
        "persistence-timeout-attempt",
        reason="research_persistence_timed_out",
    )
    rag_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_bounded_invalidation_reports_failure_without_raising(monkeypatch):
    async def fail_invalidation(_attempt_id, *, reason):
        assert reason == "research_persistence_timed_out"
        raise RuntimeError("API_TOKEN=super-secret-value")

    monkeypatch.setattr(
        pipelines,
        "invalidate_ingestion_attempt_impl",
        fail_invalidation,
    )

    result = await pipelines._invalidate_ingestion_attempt_bounded(
        "failed-attempt",
        reason="research_persistence_timed_out",
        timeout_seconds=1,
    )

    assert result == {
        "status": "failed",
        "error": "API_TOKEN=[REDACTED]",
    }


@pytest.mark.asyncio
async def test_cancellation_during_persistence_reaches_invalidation_promptly(monkeypatch):
    candidate = {
        "title": "Current guide",
        "url": "https://docs.example/current",
        "domain": "docs.example",
        "snippet": "Current guide discovery",
        "score": 10,
        "score_reasons": [],
    }
    persistence_started = asyncio.Event()
    persistence_cancelled = asyncio.Event()
    invalidation_started = asyncio.Event()
    release_invalidation = asyncio.Event()
    invalidation_finished = asyncio.Event()

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return [candidate]

    async def crawl(_semaphore, _candidate, **_kwargs):
        return {
            "ok": True,
            "title": candidate["title"],
            "url": candidate["url"],
            "requested_url": candidate["url"],
            "domain": candidate["domain"],
            "evidence_text": "Fresh extracted instructions.",
        }

    async def persist(_source, **_kwargs):
        persistence_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            persistence_cancelled.set()

    async def invalidate(attempt_id, *, reason):
        assert attempt_id == "cancel-during-persistence"
        assert reason == "research_request_cancelled"
        invalidation_started.set()
        await release_invalidation.wait()
        invalidation_finished.set()
        return {"invalidated": 0}

    monkeypatch.setitem(
        pipelines.RESEARCH_MODE_CONFIG,
        "balanced",
        {"max_urls": 1, "search_results": 1, "top_k": 4, "crawl_budget": 1},
    )
    monkeypatch.setattr(pipelines, "_persistence_budget_seconds", lambda _budget: 10)
    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(pipelines, "persist_crawled_source", persist)
    monkeypatch.setattr(pipelines, "invalidate_ingestion_attempt_impl", invalidate)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    task = asyncio.create_task(
        pipelines.research_pipeline(
            "install the current release",
            max_sources=1,
            verify=False,
            persist_source_artifacts=False,
            ingestion_attempt_id="cancel-during-persistence",
        )
    )
    await persistence_started.wait()
    task.cancel()
    await asyncio.wait_for(invalidation_started.wait(), timeout=0.2)
    task.cancel()
    release_invalidation.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert persistence_cancelled.is_set()
    assert invalidation_finished.is_set()


@pytest.mark.asyncio
async def test_inline_cancellation_waits_for_attempt_invalidation(monkeypatch):
    request_started = asyncio.Event()
    invalidation_started = asyncio.Event()
    release_invalidation = asyncio.Event()
    invalidation_finished = asyncio.Event()

    async def implementation(**_kwargs):
        request_started.set()
        await asyncio.Event().wait()

    async def invalidate(attempt_id, *, reason):
        assert attempt_id == "inline-attempt"
        assert reason == "research_request_cancelled"
        invalidation_started.set()
        await release_invalidation.wait()
        invalidation_finished.set()
        return {"invalidated": 0}

    monkeypatch.setattr(pipelines, "_research_pipeline_impl", implementation)
    monkeypatch.setattr(pipelines, "invalidate_ingestion_attempt_impl", invalidate)

    task = asyncio.create_task(
        pipelines.research_pipeline(
            "cancelled request",
            ingestion_attempt_id="inline-attempt",
        )
    )
    await request_started.wait()
    task.cancel()
    await invalidation_started.wait()
    task.cancel()
    release_invalidation.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert invalidation_finished.is_set()
