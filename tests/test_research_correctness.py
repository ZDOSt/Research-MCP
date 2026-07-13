from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import pytest
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

import pipelines
import planner
import searching
import shared
from searching import compact_search_results, estimate_source_owner_domain
from shared import IngestRequest, QueryRequest


def test_memory_cleaning_preserves_code_and_table_alignment():
    source = "key:\n  nested: value\nname\tvalue\nalpha\t  10"
    assert shared.clean_text(source) == source


@pytest.mark.parametrize(
    "chunk_size,overlap", [(0, 0), (100, -1), (100, 100), (100, 101)]
)
def test_chunking_rejects_non_progressing_configuration(chunk_size, overlap):
    with pytest.raises(ValueError):
        shared.split_long_text("content" * 100, chunk_size, overlap)


@pytest.mark.asyncio
async def test_url_pipeline_preserves_code_and_table_alignment(monkeypatch):
    async def fake_crawl(_url):
        return {"content": "config:\n  nested: true", "url": "https://example.com"}

    async def fake_browser(*_args, **_kwargs):
        return {
            "content": "name\tvalue\nalpha\t  10",
            "profile": "balanced",
            "errors": [],
        }

    monkeypatch.setattr(pipelines, "crawl_url_impl", fake_crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", fake_browser)
    monkeypatch.setattr(
        pipelines, "extraction_sufficient", lambda *_args, **_kwargs: True
    )

    result = await pipelines.explore_url_pipeline(
        "https://example.com",
        "read configuration",
        mode="balanced",
    )

    assert "config:\n  nested: true" in result["full_text_preview"]
    assert "name\tvalue\nalpha\t  10" in result["full_text_preview"]


@pytest.mark.asyncio
async def test_url_pipeline_bounds_and_offloads_result_analysis(monkeypatch):
    observed_lengths = []
    to_thread_calls = 0

    async def fake_crawl(_url):
        return {"content": "x" * 20_000, "url": "https://example.com"}

    def record_sections(content, _labels):
        observed_lengths.append(len(content))
        return {}

    def record_rows(content, **_kwargs):
        observed_lengths.append(len(content))
        return []

    def record_relevant(content, **_kwargs):
        observed_lengths.append(len(content))
        return []

    async def inline_to_thread(function, *args):
        nonlocal to_thread_calls
        to_thread_calls += 1
        return function(*args)

    monkeypatch.setattr(pipelines, "crawl_url_impl", fake_crawl)
    monkeypatch.setattr(pipelines, "extract_sections_from_text", record_sections)
    monkeypatch.setattr(pipelines, "extract_table_like_rows", record_rows)
    monkeypatch.setattr(pipelines, "extract_relevant_lines", record_relevant)
    monkeypatch.setattr(
        pipelines, "extraction_sufficient", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(pipelines.asyncio, "to_thread", inline_to_thread)

    result = await pipelines.explore_url_pipeline(
        "https://example.com",
        "find details",
        mode="targeted",
        max_chars=10_000,
    )

    assert observed_lengths == [10_000, 10_000, 10_000]
    assert to_thread_calls == 1
    assert result["truncated"] is True


def test_error_details_redact_query_credentials():
    secret = "secret-token-value-123456"
    error = ValueError(f"request failed: https://example.com/?access_token={secret}")

    assert secret not in pipelines._safe_error_detail(error)
    assert secret not in shared.safe_error_detail(error)


def test_run_scoped_point_ids_do_not_collide():
    memory_id = shared.point_id_for("https://example.com/", 0, "ns", "v1")
    first_run = shared.point_id_for(
        "https://example.com/", 0, "ns", "v1", research_run_id="run-a"
    )
    second_run = shared.point_id_for(
        "https://example.com/", 0, "ns", "v1", research_run_id="run-b"
    )

    assert len({memory_id, first_run, second_run}) == 3
    assert first_run == shared.point_id_for(
        "https://example.com/", 0, "ns", "v1", research_run_id="run-a"
    )


@pytest.mark.asyncio
async def test_ingest_marks_other_source_ingestions_superseded(monkeypatch):
    upserts = []
    supersede_calls = []

    async def fake_embed(texts):
        return [[0.1] * shared.VECTOR_SIZE for _ in texts]

    async def fake_upsert(points):
        upserts.append(points)

    async def fake_supersede(**kwargs):
        supersede_calls.append(kwargs)
        return {"is_latest_version": True, "lifecycle_status": "active"}

    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", False)
    monkeypatch.setattr(shared, "init_qdrant", lambda: None)
    monkeypatch.setattr(shared, "embed_texts_async", fake_embed)
    monkeypatch.setattr(shared, "qdrant_upsert_async", fake_upsert)
    monkeypatch.setattr(shared, "supersede_source_versions_async", fake_supersede)

    def request(run_id):
        return IngestRequest(
            text="One source snapshot.",
            metadata={
                "source": "https://example.com/source",
                "namespace": "project-a",
                "research_run_id": run_id,
            },
        )

    first = await shared.rag_ingest_impl(request("run-a"))
    second = await shared.rag_ingest_impl(request("run-b"))

    assert upserts[0][0].id != upserts[1][0].id
    assert all(not batch[0].payload["is_latest_version"] for batch in upserts)
    assert all(not batch[0].payload["ingestion_committed"] for batch in upserts)
    assert all(batch[0].payload["lifecycle_status"] == "pending" for batch in upserts)
    assert first["ingestion_id"] != second["ingestion_id"]
    assert first["ingestion_order_ns"] < second["ingestion_order_ns"]
    assert supersede_calls[-1]["active_ingestion_id"] == second["ingestion_id"]


@pytest.mark.asyncio
async def test_older_ingestion_finishing_last_cannot_replace_newer_latest(monkeypatch):
    client = QdrantClient(":memory:")
    collection = "lifecycle-race"
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    source = "https://example.com/source"
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=1,
                vector=[0.1],
                payload={
                    "namespace": "ns",
                    "source": source,
                    "ingestion_id": "older",
                    "ingestion_order_ns": 100,
                    "source_version": "v1",
                    "chunk_index": 0,
                    "ingestion_chunk_count": 1,
                    "ingestion_committed": False,
                    "is_latest_version": False,
                    "lifecycle_status": "pending",
                },
            ),
            PointStruct(
                id=2,
                vector=[0.2],
                payload={
                    "namespace": "ns",
                    "source": source,
                    "ingestion_id": "newer",
                    "ingestion_order_ns": 200,
                    "source_version": "v2",
                    "chunk_index": 0,
                    "ingestion_chunk_count": 1,
                    "ingestion_committed": False,
                    "is_latest_version": False,
                    "lifecycle_status": "pending",
                },
            ),
            PointStruct(
                id=3,
                vector=[0.3],
                payload={
                    "namespace": "ns",
                    "source": source,
                    "source_version": "legacy",
                },
            ),
        ],
    )
    monkeypatch.setattr(shared, "COLLECTION_NAME", collection)
    monkeypatch.setattr(shared, "get_qdrant_client", lambda: client)

    newer = await shared.supersede_source_versions_async(
        namespace="ns",
        source=source,
        active_ingestion_id="newer",
        active_source_version="v2",
        active_ingestion_order_ns=200,
        superseded_at="2026-01-02T00:00:00+00:00",
    )
    older = await shared.supersede_source_versions_async(
        namespace="ns",
        source=source,
        active_ingestion_id="older",
        active_source_version="v1",
        active_ingestion_order_ns=100,
        superseded_at="2026-01-01T00:00:00+00:00",
    )

    points, _ = client.scroll(collection_name=collection, limit=10, with_payload=True)
    payloads = {
        point.payload.get("ingestion_id", "legacy"): point.payload for point in points
    }
    assert newer["is_latest_version"] is True
    assert older["is_latest_version"] is False
    assert payloads["newer"]["is_latest_version"] is True
    assert payloads["older"]["is_latest_version"] is False
    assert payloads["legacy"]["is_latest_version"] is False


@pytest.mark.asyncio
async def test_embedding_cardinality_mismatch_fails_ingest(monkeypatch):
    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", False)
    monkeypatch.setattr(shared, "init_qdrant", lambda: None)

    async def missing_vectors(_texts):
        return []

    monkeypatch.setattr(shared, "embed_texts_async", missing_vectors)
    with pytest.raises(HTTPException, match="0 vectors for 1 chunks"):
        await shared.rag_ingest_impl(
            IngestRequest(text="content", metadata={"source": "manual"})
        )


@pytest.mark.asyncio
async def test_memory_and_run_queries_filter_uncommitted_or_obsolete_evidence(
    monkeypatch,
):
    filters = []
    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", False)
    monkeypatch.setattr(shared, "init_qdrant", lambda: None)

    async def fake_embed(_texts):
        return [[0.1] * shared.VECTOR_SIZE]

    async def fake_query(query_vec, limit, query_filter=None):
        assert query_vec and limit
        filters.append(query_filter)
        return []

    monkeypatch.setattr(shared, "embed_texts_async", fake_embed)
    monkeypatch.setattr(shared, "qdrant_query_points_async", fake_query)

    await shared.rag_query_impl(QueryRequest(query="q", namespace="ns"))
    await shared.rag_query_impl(
        QueryRequest(
            query="q",
            namespace="ns",
            research_run_id="run-a",
            ingestion_attempt_id="attempt-a",
        )
    )

    assert "is_latest_version" in str(filters[0])
    assert "ingestion_committed" in str(filters[0])
    assert "is_latest_version" in str(filters[1])
    assert "ingestion_committed" in str(filters[1])
    assert "run-a" in str(filters[1])
    assert "attempt-a" in str(filters[1])


@pytest.mark.asyncio
async def test_include_memory_queries_even_when_search_selects_no_urls(monkeypatch):
    captured = []

    async def fake_plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def fake_search(**_kwargs):
        return []

    async def fake_rag(request):
        captured.append(request)
        return {
            "results": [
                {
                    "text": "Stored evidence about the requested topic.",
                    "url": "https://docs.example.com/item",
                    "domain": "docs.example.com",
                }
            ]
        }

    monkeypatch.setattr(pipelines, "build_research_plan", fake_plan)
    monkeypatch.setattr(pipelines, "searxng_search", fake_search)
    monkeypatch.setattr(pipelines, "rag_query_impl", fake_rag)

    result = await pipelines.research_pipeline("topic", include_memory=True)

    assert captured and captured[0].research_run_id is None
    assert result["evidence"]
    assert result["selected_for_crawl"] == []
    assert result["verification"]["claim_verification_performed"] is False


def test_search_urls_are_canonicalized_before_deduplication():
    raw_url = "HTTPS://Example.COM:443/a/./b/../path/%7e?b=2&utm_source=x&a=1#section"
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "First",
                    "url": raw_url,
                    "content": "result",
                },
                {
                    "title": "Duplicate",
                    "url": "https://example.com/a/path/~?a=1&b=2",
                    "content": "result",
                },
            ]
        },
        query="result",
    )

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/a/path/~?a=1&b=2"
    assert results[0]["url"] == shared.normalize_url(raw_url)


@pytest.mark.asyncio
async def test_searxng_response_is_streamed_under_byte_cap(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"x" * 11

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(searching, "SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setattr(searching, "SEARXNG_MAX_RESPONSE_BYTES", 10)
    monkeypatch.setattr(searching.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(ValueError, match="SEARXNG_MAX_RESPONSE_BYTES"):
        await searching.searxng_search("query")


@pytest.mark.asyncio
async def test_searxng_url_rejects_embedded_credentials(monkeypatch):
    monkeypatch.setattr(searching, "SEARXNG_URL", "http://user:secret@searxng:8080")

    with pytest.raises(ValueError, match="without credentials"):
        await searching.searxng_search("query")


def test_source_owner_estimate_does_not_count_subdomains_as_independent():
    assert estimate_source_owner_domain("docs.example.com") == "example.com"
    assert estimate_source_owner_domain("blog.example.com") == "example.com"

    coverage = pipelines.build_source_coverage(
        [
            {"domain": "docs.example.com"},
            {"domain": "blog.example.com"},
        ]
    )
    assert coverage["distinct_hosts"] == 2
    assert coverage["distinct_source_owners_estimate"] == 1
    assert "do not establish organizational independence" in coverage["note"]


def test_verification_reports_topical_overlap_without_claiming_fact_verification():
    metadata = pipelines.build_verification_metadata(
        [
            {
                "evidence_id": 1,
                "domain": "one.example",
                "quote": "Release 4.2 adds signed package verification support.",
            },
            {
                "evidence_id": 2,
                "domain": "two.example",
                "quote": "Signed package verification support is included in release 4.2.",
            },
        ],
        requested=True,
    )

    assert metadata["status"] == "cross_source_topical_overlap_observed"
    assert metadata["cross_source_topical_overlap_pairs"]
    assert metadata["claim_verification_performed"] is False
    assert "not claim-level" in metadata["limitations"]


def test_verification_selection_prefers_distinct_owner_domains():
    candidates = [
        {"url": "https://docs.example.com/a", "domain": "docs.example.com"},
        {"url": "https://blog.example.com/b", "domain": "blog.example.com"},
        {"url": "https://other.example.net/c", "domain": "other.example.net"},
    ]
    selected = pipelines._select_candidates(candidates, 2, prefer_owner_diversity=True)
    assert [item["url"] for item in selected] == [
        "https://docs.example.com/a",
        "https://other.example.net/c",
    ]


@pytest.mark.asyncio
async def test_redirected_source_uses_final_url_and_domain(monkeypatch):
    captured_request = None

    async def fake_crawl(_url):
        return {
            "url": "https://final.example.org/article",
            "content": "Final page content",
            "title": "Final title",
            "extraction_method": "direct",
        }

    async def fake_ingest(request):
        nonlocal captured_request
        captured_request = request
        return {"stored": 1, "source_version": "v1", "snapshot_id": "s1"}

    monkeypatch.setattr(pipelines, "crawl_url_impl", fake_crawl)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", fake_ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://start.example.com/link", "domain": "start.example.com"},
        query="question",
        ingestion_attempt_id="c" * 64,
        ingestion_order_ns=987654,
    )

    assert captured_request.metadata["source"] == "https://final.example.org/article"
    assert (
        captured_request.metadata["requested_url"] == "https://start.example.com/link"
    )
    assert captured_request.metadata["domain"] == "final.example.org"
    assert captured_request.metadata["ingestion_attempt_id"] == "c" * 64
    assert captured_request.metadata["ingestion_order_ns"] == 987654
    assert result["url"] == "https://final.example.org/article"
    assert result["domain"] == "final.example.org"


@pytest.mark.asyncio
async def test_source_artifact_persistence_can_be_disabled(monkeypatch):
    async def fake_crawl(_url):
        return {
            "url": "https://example.org/article",
            "content": "Page content",
            "title": "Title",
            "extraction_method": "direct",
        }

    captured_request = None

    async def fake_ingest(request):
        nonlocal captured_request
        captured_request = request
        return {"stored": 1, "source_version": "v1", "snapshot_id": "s1"}

    write_text = AsyncMock()
    monkeypatch.setattr(pipelines, "crawl_url_impl", fake_crawl)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", fake_ingest)
    monkeypatch.setattr(
        pipelines,
        "get_artifact_store",
        lambda: type("Store", (), {"write_text": write_text})(),
    )

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/article", "domain": "example.org"},
        query="question",
        research_run_id=uuid.uuid4().hex,
        persist_source_artifacts=False,
    )

    write_text.assert_not_awaited()
    assert captured_request.metadata["artifact_id"] is None
    assert captured_request.metadata["artifact_path"] is None
    assert result["artifact_reference"] is None


@pytest.mark.asyncio
async def test_browser_fallback_reuses_initial_crawl(monkeypatch):
    crawl = AsyncMock(
        return_value={
            "url": "https://example.org/article",
            "content": "Thin initial content",
            "title": "Initial title",
            "extraction_method": "direct",
        }
    )
    browser = AsyncMock(
        return_value={
            "url": "https://example.org/article",
            "final_url": "https://example.org/article",
            "title": "Rendered title",
            "content": "Rendered content " * 100,
            "content_chars": 1700,
            "profile": "targeted",
            "errors": [],
        }
    )
    captured_request = None

    async def fake_ingest(request):
        nonlocal captured_request
        captured_request = request
        return {"stored": 1, "source_version": "v1", "snapshot_id": "s1"}

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", fake_ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/article", "domain": "example.org"},
        query="find article details",
        use_browser_fallback=True,
    )

    crawl.assert_awaited_once_with("https://example.org/article")
    browser.assert_awaited_once()
    assert "Thin initial content" in captured_request.text
    assert "Rendered content" in captured_request.text
    assert result["browser_fallback_used"] is True


@pytest.mark.asyncio
async def test_low_confidence_long_direct_result_forces_browser_fallback(monkeypatch):
    captured_request = None
    rendered_content = "\n".join(
        f"Dynamic application content detail {index}"
        for index in range(12)
    )
    crawl = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "content": "SHELL-CONTENT " * 12_000,
            "title": "Application shell",
            "extraction_method": "direct_http_fallback",
            "_direct_low_confidence": True,
        }
    )
    browser = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "final_url": "https://example.org/app",
            "title": "Rendered application",
            "content": rendered_content,
            "content_chars": len(rendered_content),
            "profile": "targeted",
            "errors": [],
        }
    )

    async def fake_ingest(request):
        nonlocal captured_request
        captured_request = request
        return {"stored": 1, "source_version": "v1", "snapshot_id": "s1"}

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", fake_ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/app", "domain": "example.org"},
        query="find dynamic application content",
        use_browser_fallback=True,
    )

    crawl.assert_awaited_once_with("https://example.org/app")
    browser.assert_awaited_once()
    assert result["browser_fallback_used"] is True
    assert captured_request.text == rendered_content
    assert "SHELL-CONTENT" not in captured_request.text
    assert result["title"] == "Rendered application"


@pytest.mark.asyncio
async def test_rejected_shell_does_not_inflate_rendered_sufficiency(monkeypatch):
    shell = "\n".join(
        f"container error fix irrelevant navigation item {index}"
        for index in range(20)
    )
    browser = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "final_url": "https://example.org/app",
            "title": "Rendered application",
            "content": "Rendered",
            "content_chars": 8,
            "profile": "targeted",
            "errors": [],
        }
    )

    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)

    result = await pipelines.explore_url_pipeline(
        "https://example.org/app",
        "fix container error",
        mode="targeted",
        initial_crawl_data={
            "url": "https://example.org/app",
            "content": shell,
            "title": "Application shell",
            "extraction_method": "direct_http_fallback",
            "_direct_low_confidence": True,
        },
    )

    browser.assert_awaited_once()
    assert result["full_text_preview"] == "Rendered"
    assert result["extraction_sufficient"] is False
    assert result["strategy_attempts"][-1]["sufficient"] is False
    assert "irrelevant navigation" not in result["full_text_preview"]
    assert result["title"] == "Rendered application"


@pytest.mark.asyncio
async def test_insufficient_rendered_content_does_not_replace_rejected_shell(monkeypatch):
    crawl = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "content": "Checking your browser. " + ("placeholder " * 400),
            "title": "Just a moment",
            "extraction_method": "direct_http_fallback",
            "_direct_low_confidence": True,
        }
    )
    browser = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "final_url": "https://example.org/app",
            "title": "Rendered application",
            "content": "Rendered",
            "content_chars": 8,
            "profile": "targeted",
            "errors": [],
        }
    )
    ingest = AsyncMock()

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/app", "domain": "example.org"},
        query="fix container error",
        use_browser_fallback=True,
    )

    browser.assert_awaited_once()
    ingest.assert_not_awaited()
    assert result["ok"] is False
    assert "Rendered extraction did not meet the quality threshold" in result["reason"]


@pytest.mark.parametrize("mode", ["quick", "balanced", "web_only"])
@pytest.mark.asyncio
async def test_default_modes_do_not_ingest_low_confidence_direct_content(
    monkeypatch,
    mode,
):
    async def fake_plan(query, selected_mode):
        return {"query": query, "mode": selected_mode, "queries": [query]}

    async def fake_search(query, max_results, mode, policy=None):
        return [
            {
                "title": "Challenge page",
                "url": "https://example.org/app",
                "domain": "example.org",
                "snippet": query,
                "score": 1,
                "score_reasons": [],
            }
        ]

    crawl = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "content": "Checking your browser. " + ("placeholder " * 400),
            "title": "Just a moment",
            "extraction_method": "direct_http_fallback",
            "_direct_low_confidence": True,
        }
    )
    browser = AsyncMock()
    ingest = AsyncMock()

    monkeypatch.setattr(pipelines, "build_research_plan", fake_plan)
    monkeypatch.setattr(pipelines, "searxng_search", fake_search)
    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", ingest)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "current container error",
        mode=mode,
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    browser.assert_not_awaited()
    ingest.assert_not_awaited()
    assert result["crawled_sources"] == []
    assert len(result["failed_sources"]) == 1
    assert "browser fallback was disabled" in result["failed_sources"][0]["reason"]


@pytest.mark.asyncio
async def test_low_confidence_shell_is_not_ingested_when_browser_fails(monkeypatch):
    crawl = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "content": "Navigation and legal links " * 150,
            "title": "Application shell",
            "extraction_method": "direct_http_fallback",
            "_direct_low_confidence": True,
        }
    )
    browser = AsyncMock(side_effect=RuntimeError("rendered browser unavailable"))
    ingest = AsyncMock()

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/app", "domain": "example.org"},
        query="find dynamic application content",
        use_browser_fallback=True,
    )

    browser.assert_awaited_once()
    ingest.assert_not_awaited()
    assert result["ok"] is False
    assert "rendered browser unavailable" in result["reason"]


@pytest.mark.asyncio
async def test_low_confidence_shell_is_cleared_if_exploration_crashes(monkeypatch):
    crawl = AsyncMock(
        return_value={
            "url": "https://example.org/app",
            "content": "Navigation and legal links " * 150,
            "title": "Application shell",
            "extraction_method": "direct_http_fallback",
            "_direct_low_confidence": True,
        }
    )
    explore = AsyncMock(side_effect=RuntimeError("exploration pipeline crashed"))
    ingest = AsyncMock()

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "explore_url_pipeline", explore)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/app", "domain": "example.org"},
        query="find dynamic application content",
        use_browser_fallback=True,
    )

    explore.assert_awaited_once()
    ingest.assert_not_awaited()
    assert result["ok"] is False
    assert "exploration pipeline crashed" in result["reason"]


@pytest.mark.asyncio
async def test_browser_fallback_does_not_retry_failed_crawl(monkeypatch):
    crawl = AsyncMock(side_effect=RuntimeError("initial crawl failed"))
    browser = AsyncMock(
        return_value={
            "url": "https://example.org/article",
            "final_url": "https://example.org/article",
            "title": "Rendered title",
            "content": "Rendered content " * 100,
            "content_chars": 1700,
            "profile": "targeted",
            "errors": [],
        }
    )

    async def fake_ingest(_request):
        return {"stored": 1, "source_version": "v1", "snapshot_id": "s1"}

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)
    monkeypatch.setattr(pipelines, "playwright_explore_page", browser)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", fake_ingest)

    result = await pipelines.crawl_and_ingest(
        {"url": "https://example.org/article", "domain": "example.org"},
        query="find article details",
        use_browser_fallback=True,
    )

    crawl.assert_awaited_once_with("https://example.org/article")
    browser.assert_awaited_once()
    assert result["browser_fallback_used"] is True
    assert result["errors"] == ["initial crawl failed"]


@pytest.mark.asyncio
async def test_source_artifact_name_is_isolated_by_job_attempt(monkeypatch):
    async def fake_crawl(_url):
        return {
            "url": "https://example.org/article",
            "content": "Page content",
            "title": "Title",
            "extraction_method": "direct",
        }

    async def fake_ingest(request):
        return {
            "stored": 1,
            "source_version": "v1",
            "snapshot_id": "s1",
            "artifact_id": request.metadata["artifact_id"],
            "artifact_path": request.metadata["artifact_path"],
        }

    names = []

    async def write_text(_owner, _content, *, name, metadata):
        assert metadata["url"] == "https://example.org/article"
        names.append(name)
        return {"artifact_id": name, "relative_path": f"owner/{name}.txt"}

    monkeypatch.setattr(pipelines, "crawl_url_impl", fake_crawl)
    monkeypatch.setattr(pipelines, "rag_ingest_impl", fake_ingest)
    monkeypatch.setattr(
        pipelines,
        "get_artifact_store",
        lambda: SimpleNamespace(write_text=write_text),
    )
    run_id = uuid.uuid4().hex

    for attempt_id in ("a" * 64, "b" * 64):
        await pipelines.crawl_and_ingest(
            {"url": "https://example.org/article", "domain": "example.org"},
            query="question",
            research_run_id=run_id,
            ingestion_attempt_id=attempt_id,
            ingestion_order_ns=123,
        )

    assert names[0] != names[1]
    assert names[0].endswith("a" * 16)
    assert names[1].endswith("b" * 16)


@pytest.mark.asyncio
async def test_partial_reranker_response_falls_back_to_vector_order(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b'[{"index":1,"score":0.99}]'

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(shared.httpx, "AsyncClient", lambda **_kwargs: Client())
    docs = [{"text": "first"}, {"text": "second"}]
    assert await shared.rerank_docs("q", docs, 2) == docs


@pytest.mark.asyncio
async def test_reranker_response_is_streamed_under_byte_cap(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"x" * 11

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(shared, "RERANKER_MAX_RESPONSE_BYTES", 10)
    monkeypatch.setattr(shared.httpx, "AsyncClient", lambda **_kwargs: Client())
    docs = [{"text": "first"}, {"text": "second"}]

    assert await shared.rerank_docs("q", docs, 2) == docs


@pytest.mark.asyncio
async def test_remote_management_uses_remote_rag_routes(monkeypatch):
    calls = []

    async def fake_remote(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return {"path": path}

    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", True)
    monkeypatch.setattr(shared, "_remote_rag_request", fake_remote)

    await shared.list_sources_impl(namespace="ns")
    await shared.source_stats_impl(namespace="ns")
    await shared.delete_source_impl("https://example.com/", namespace="ns")

    assert [(method, path) for method, path, _ in calls] == [
        ("POST", "/rag/sources"),
        ("GET", "/rag/source-stats"),
        ("POST", "/rag/delete-source"),
    ]


def test_remote_rag_public_endpoint_requires_token(monkeypatch):
    monkeypatch.setattr(shared, "RESEARCH_API_URL", "https://rag.example.com")
    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "")
    with pytest.raises(HTTPException, match="RESEARCH_API_TOKEN"):
        shared._remote_rag_headers()

    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "secret")
    assert shared._remote_rag_headers() == {"Authorization": "Bearer secret"}


def test_remote_rag_public_http_requires_explicit_insecure_opt_in(monkeypatch):
    monkeypatch.setattr(shared, "RESEARCH_API_URL", "http://rag.example.com")
    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "secret")
    monkeypatch.setattr(shared, "RESEARCH_API_ALLOW_INSECURE_HTTP", False)

    with pytest.raises(HTTPException, match="must use HTTPS"):
        shared._remote_rag_headers()

    monkeypatch.setattr(shared, "RESEARCH_API_ALLOW_INSECURE_HTTP", True)
    assert shared._remote_rag_headers() == {"Authorization": "Bearer secret"}


def test_remote_rag_internal_http_remains_supported(monkeypatch):
    monkeypatch.setattr(shared, "RESEARCH_API_URL", "http://research-api:8000")
    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "")
    monkeypatch.setattr(shared, "RESEARCH_API_ALLOW_INSECURE_HTTP", False)

    assert shared._remote_rag_headers() == {}


@pytest.mark.asyncio
async def test_remote_rag_response_is_streamed_under_a_byte_limit(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"123456789"

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(shared, "RESEARCH_API_URL", "https://rag.example.com")
    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "secret")
    monkeypatch.setattr(shared, "RESEARCH_API_MAX_RESPONSE_BYTES", 8)
    monkeypatch.setattr(shared.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(HTTPException, match="RESEARCH_API_MAX_RESPONSE_BYTES"):
        await shared._remote_rag_request("GET", "/rag/source-stats")


@pytest.mark.asyncio
async def test_remote_rag_transport_error_does_not_echo_credentials(monkeypatch):
    class FailingStream:
        async def __aenter__(self):
            raise shared.httpx.ConnectError("Authorization: Bearer top-secret")

        async def __aexit__(self, *_args):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return FailingStream()

    monkeypatch.setattr(shared, "RESEARCH_API_URL", "https://rag.example.com")
    monkeypatch.setattr(shared, "RESEARCH_API_TOKEN", "top-secret")
    monkeypatch.setattr(shared.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(HTTPException) as caught:
        await shared._remote_rag_request("GET", "/rag/source-stats")
    assert "top-secret" not in caught.value.detail
    assert caught.value.detail == "Remote RAG transport failed for /rag/source-stats"


def test_collection_schema_is_validated():
    good = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(size=shared.VECTOR_SIZE, distance=Distance.COSINE)
            )
        ),
        payload_schema={},
    )
    shared._validate_collection_schema(good)

    bad = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors=VectorParams(
                    size=shared.VECTOR_SIZE + 1, distance=Distance.COSINE
                )
            )
        ),
        payload_schema={},
    )
    with pytest.raises(shared.QdrantSchemaError, match="vector schema"):
        shared._validate_collection_schema(bad)


def test_payload_index_type_is_validated():
    client = SimpleNamespace(create_payload_index=lambda **_kwargs: None)
    collection = SimpleNamespace(
        payload_schema={
            "namespace": SimpleNamespace(data_type=PayloadSchemaType.INTEGER),
        }
    )
    with pytest.raises(shared.QdrantSchemaError, match="payload index 'namespace'"):
        shared._ensure_payload_indexes(client, collection)


def test_missing_payload_indexes_are_created():
    calls = []
    client = SimpleNamespace(create_payload_index=lambda **kwargs: calls.append(kwargs))
    shared._ensure_payload_indexes(client, SimpleNamespace(payload_schema={}))
    assert {call["field_name"] for call in calls} == set(shared.PAYLOAD_INDEXES)
    assert all(call["wait"] is True for call in calls)


@pytest.mark.asyncio
async def test_source_listing_exposes_lifecycle_and_truncation(monkeypatch):
    points = [
        SimpleNamespace(
            payload={
                "source": "https://example.com/",
                "is_latest_version": True,
                "source_version": "v2",
                "ingested_at": "2026-01-02T00:00:00+00:00",
            }
        ),
        SimpleNamespace(
            payload={
                "source": "https://example.com/",
                "is_latest_version": False,
                "source_version": "v1",
                "ingested_at": "2026-01-01T00:00:00+00:00",
            }
        ),
    ]
    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", False)
    monkeypatch.setattr(shared, "collect_points", lambda **_kwargs: (points, True))

    result = await shared.list_sources_impl(namespace="ns")
    source = result["sources"][0]
    assert result["truncated"] is True
    assert source["chunks"] == 1
    assert source["total_chunks"] == 2
    assert source["superseded_chunks"] == 1
    assert source["version_count"] == 2


def test_planner_url_policy_rejects_credentials_and_plain_http(monkeypatch):
    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://user:pass@example.com/v1")
    with pytest.raises(RuntimeError, match="credentials"):
        planner._validated_planner_base_url()

    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "http://planner.internal/v1")
    monkeypatch.setattr(planner, "PLANNER_ALLOW_INSECURE_HTTP", False)
    with pytest.raises(RuntimeError, match="PLANNER_ALLOW_INSECURE_HTTP"):
        planner._validated_planner_base_url()

    monkeypatch.setattr(planner, "PLANNER_ALLOW_INSECURE_HTTP", True)
    assert planner._validated_planner_base_url() == "http://planner.internal/v1"


@pytest.mark.asyncio
async def test_planner_stream_is_bounded_before_json_parse(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"x" * 11

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://planner.example.com/v1")
    monkeypatch.setattr(planner, "PLANNER_MODEL", "model")
    monkeypatch.setattr(planner, "PLANNER_MAX_RESPONSE_BYTES", 10)
    monkeypatch.setattr(planner.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(ValueError, match="PLANNER_MAX_RESPONSE_BYTES"):
        await planner._chat([{"role": "user", "content": "q"}])


@pytest.mark.asyncio
async def test_synthesis_rejects_invalid_citations_and_reports_validated_ids(
    monkeypatch,
):
    evidence = [
        {
            "evidence_id": 1,
            "title": "Source",
            "url": "https://example.com",
            "quote": "Fact",
        }
    ]
    monkeypatch.setattr(planner, "PLANNER_ENABLE_SYNTHESIS", True)
    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://planner.example.com/v1")
    monkeypatch.setattr(planner, "PLANNER_MODEL", "model")

    async def invalid_chat(*_args, **_kwargs):
        return "Unsupported claim [E99]"

    monkeypatch.setattr(planner, "_chat", invalid_chat)
    assert await planner.synthesize_report("q", evidence) is None

    async def valid_chat(*_args, **_kwargs):
        return "Supported claim [E1]"

    monkeypatch.setattr(planner, "_chat", valid_chat)
    report = await planner.synthesize_report("q", evidence)
    assert report["citation_validation"]["valid"] is True
    assert report["citation_validation"]["cited_evidence_ids"] == [1]


def test_evidence_pack_explains_artifact_reference_lifecycle():
    evidence = pipelines.build_evidence_pack(
        [
            {
                "text": "Evidence",
                "artifact_id": "job:source",
                "artifact_path": "job/source.txt",
            }
        ]
    )
    reference = evidence[0]["artifact_reference"]
    assert (
        reference["lifecycle"] == "retention_managed_independently_from_vector_memory"
    )
    assert reference["availability"] == "not_guaranteed_after_retention_cleanup"
