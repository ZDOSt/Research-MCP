import pytest

import pipelines
from github_connector import normalize_repository
from planner import deterministic_plan
from redaction import redact_sensitive_text
from searching import domain_adjustment, domain_matches
from shared import QueryRequest, normalize_namespace, point_id_for


def test_namespace_and_point_ids_are_stable_and_scoped():
    assert normalize_namespace(" My VPS / Production ") == "My-VPS-Production"
    first = point_id_for("https://example.com", 0, "one", "version")
    assert first == point_id_for("https://example.com", 0, "one", "version")
    assert first != point_id_for("https://example.com", 0, "two", "version")


def test_deterministic_technical_plan_uses_multiple_queries():
    plan = deterministic_plan("fix Docker networking error", "technical")
    assert plan["queries"][0] == "fix Docker networking error"
    assert any("official documentation" in item for item in plan["queries"])
    assert any("GitHub issues" in item for item in plan["queries"])


def test_github_repository_normalization_and_validation():
    assert normalize_repository("https://github.com/SillyTavern/SillyTavern.git") == "SillyTavern/SillyTavern"
    with pytest.raises(ValueError):
        normalize_repository("https://example.com/not/github")


def test_redaction_covers_common_operational_secrets():
    content = "API_TOKEN=super-secret-value\nAuthorization: Bearer abcdefghijklmnopqrstuvwxyz"
    redacted, count = redact_sensitive_text(content)
    assert "super-secret-value" not in redacted
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert count == 2


def test_domain_matching_applies_to_subdomains():
    assert domain_matches("docs.example.com", "example.com")
    adjustment, owner = domain_adjustment("docs.example.com", {"example.com": 2.5})
    assert adjustment == 2.5
    assert owner == "example.com"


@pytest.mark.parametrize("value", ["1", "2", "4"])
def test_source_concurrency_accepts_bounded_values(value):
    assert pipelines._validated_source_concurrency(value) == int(value)


@pytest.mark.parametrize("value", ["0", "5", "many"])
def test_source_concurrency_rejects_unsafe_values(value):
    with pytest.raises(ValueError, match="RESEARCH_SOURCE_CONCURRENCY"):
        pipelines._validated_source_concurrency(value)


@pytest.mark.asyncio
async def test_research_pipeline_deduplicates_queries_and_scopes_retrieval(monkeypatch):
    crawl_kwargs = []
    async def fake_plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query, f"{query} docs"]}

    async def fake_search(query, max_results, mode):
        return [
            {
                "title": "Official docs",
                "url": "https://docs.example.com/answer",
                "domain": "docs.example.com",
                "snippet": query,
                "score": 3 if query.endswith("docs") else 2,
                "score_reasons": [],
            }
        ]

    async def fake_crawl(semaphore, result, **kwargs):
        crawl_kwargs.append(kwargs)
        return {"ok": True, "url": result["url"], "domain": result["domain"]}

    captured_request = None

    async def fake_rag(request: QueryRequest):
        nonlocal captured_request
        captured_request = request
        return {
            "results": [
                {
                    "text": "Scoped evidence",
                    "url": "https://docs.example.com/answer",
                    "domain": "docs.example.com",
                    "research_run_id": request.research_run_id,
                }
            ]
        }

    monkeypatch.setattr(pipelines, "build_research_plan", fake_plan)
    monkeypatch.setattr(pipelines, "searxng_search", fake_search)
    monkeypatch.setattr(pipelines, "crawl_and_ingest_limited", fake_crawl)
    monkeypatch.setattr(pipelines, "rag_query_impl", fake_rag)

    result = await pipelines.research_pipeline(
        "test question",
        mode="balanced",
        max_sources=1,
        namespace="project-a",
        ingestion_attempt_id="b" * 64,
        ingestion_order_ns=123456,
    )

    assert len(result["searched"]) == 1
    assert result["searched"][0]["matched_queries"] == ["test question", "test question docs"]
    assert captured_request is not None
    assert captured_request.namespace == "project-a"
    assert captured_request.research_run_id == result["research_run_id"]
    assert captured_request.ingestion_attempt_id == "b" * 64
    assert crawl_kwargs[0]["ingestion_attempt_id"] == "b" * 64
    assert crawl_kwargs[0]["ingestion_order_ns"] == 123456
