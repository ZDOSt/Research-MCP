import json
from unittest.mock import AsyncMock

import pytest

import pipelines
from github_connector import normalize_repository
from planner import (
    compact_search_queries,
    compact_search_query,
    deterministic_plan,
    fallback_search_query,
)
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


VERBOSE_AI_NEWS_QUERY = (
    "Today's most important AI news. Identify and rank the top three substantive AI news "
    "articles published today, prioritizing major developments in AI models, companies, "
    "regulation, chips/infrastructure, safety, or research. For each, provide headline, "
    "publisher, publication date/time if available, concise summary, why it matters, and "
    "source URL. Avoid opinion pieces, duplicate syndicated coverage, and minor product "
    "announcements. Today means the current date at time of search."
)


def test_verbose_instruction_is_compacted_for_search_engines(monkeypatch):
    monkeypatch.setattr(
        "planner.runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )
    plan = deterministic_plan(VERBOSE_AI_NEWS_QUERY, "balanced")

    assert plan["query"] == VERBOSE_AI_NEWS_QUERY
    assert plan["queries"][0] == "Today's most important AI news 2026-07-12"
    assert "AI news today 2026-07-12" in plan["queries"]
    assert any("official documentation" in item for item in plan["queries"])
    assert all(len(item) <= 220 for item in plan["queries"])
    assert all("For each" not in item for item in plan["queries"])
    assert fallback_search_query(VERBOSE_AI_NEWS_QUERY) == "AI news today"


@pytest.mark.parametrize(
    ("source_query", "expected"),
    [
        ("Tell me today's AI news.", "today's AI news"),
        ("Can you find the Docker Compose installation guide?", "the Docker Compose installation guide"),
        ("Determine the current PostgreSQL version.", "the current PostgreSQL version"),
    ],
)
def test_short_request_language_is_removed_from_search_query(source_query, expected):
    assert compact_search_query(source_query) == expected


@pytest.mark.parametrize(
    ("source_query", "expected"),
    [
        ("Find out why Docker Compose cannot resolve a service name.", "why Docker Compose cannot resolve a service name"),
        ("Identify and rank current PostgreSQL backup tools.", "current PostgreSQL backup tools"),
    ],
)
def test_compaction_removes_complete_leading_request_phrases(source_query, expected):
    assert compact_search_query(source_query) == expected


@pytest.mark.parametrize(
    ("source_query", "required_terms"),
    [
        (
            "Find the current installation guide for LibreChat 0.8.1 on Ubuntu 24.04. "
            "Return commands, prerequisites, and links, then summarize common pitfalls.",
            {"LibreChat", "0.8.1", "Ubuntu", "24.04"},
        ),
        (
            'Research how to fix Docker error "network app-network not found" with Compose v2.29. '
            "Provide safe diagnostic steps and cite official documentation.",
            {'"network app-network not found"', "Docker", "v2.29"},
        ),
        (
            "Determine the OpenAI API rate limits since June 1, 2026. Include a table and "
            "exclude unofficial estimates.",
            {"OpenAI", "since June 1, 2026"},
        ),
        (
            "Check the exact behavior described at https://docs.example.com/v3/setup. "
            "Return a short explanation and preserve issue CVE-2026-1234.",
            {"https://docs.example.com/v3/setup", "CVE-2026-1234"},
        ),
        (
            "Find releases between June 1, 2026 and 07/10/2026. Return them as JSON.",
            {"June 1, 2026", "07/10/2026"},
        ),
    ],
)
def test_compaction_preserves_entities_versions_errors_and_dates(source_query, required_terms):
    compact = compact_search_query(source_query)

    assert len(compact) <= 180
    assert all(term in compact for term in required_terms)
    assert "Provide safe diagnostic" not in compact
    assert "Include a table" not in compact


def test_long_instruction_preserves_late_exact_anchors_without_cutting_them():
    source_query = (
        "Explain the detailed operational background and every diagnostic consideration that an "
        "administrator should understand before attempting to resolve a complicated production "
        "deployment failure involving several container services and network layers. "
        'The exact error is "network app-network not found" in Docker Compose v2.29. '
        "Use https://docs.docker.com/compose/networking/ and only cover changes since June 1, 2026."
    )

    compact = compact_search_query(source_query)

    assert len(compact) <= 180
    assert '"network app-network not found"' in compact
    assert "v2.29" in compact
    assert "https://docs.docker.com/compose/networking/" in compact
    assert "since June 1, 2026" in compact
    assert not compact.endswith("https://docs.docker.com/compose/networ")


@pytest.mark.parametrize(
    ("source_query", "required_terms"),
    [
        (
            "Please give me a detailed, careful, step-by-step explanation of all prerequisites "
            "and caveats involved in setting up a secure production deployment of the newest "
            "version of LibreChat using Docker Compose on Ubuntu 24.04, including changes since "
            "version 0.8.1 and official sources.",
            {"LibreChat", "Docker", "Compose", "Ubuntu", "24.04", "0.8.1"},
        ),
        (
            "The first sentence is generic context about my environment and contains a lot of "
            "irrelevant framing that is verbose enough to dominate query selection. The actual "
            "issue is that libssl.so.3 cannot be found in Alpine 3.20 when running widgetctl "
            "v9.4.2.",
            {"libssl.so.3", "cannot be found", "Alpine", "3.20", "widgetctl", "v9.4.2"},
        ),
    ],
)
def test_verbose_general_requests_retain_the_actual_subject(source_query, required_terms):
    compact = compact_search_query(source_query)
    fallback = fallback_search_query(source_query)

    assert len(compact) <= 180
    assert all(term in compact for term in required_terms)
    assert {"LibreChat", "Docker"}.intersection(required_terms).issubset(set(fallback.split()))


def test_substantive_lowercase_question_outranks_generic_preamble(monkeypatch):
    source_query = (
        "Please provide a comprehensive, current, source-backed answer that covers prerequisites, "
        "compatibility, security, failure cases, and any important caveats before giving concise "
        "steps. How can I install home assistant on a raspberry pi?"
    )
    monkeypatch.setattr(
        "planner.runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )

    compact = compact_search_query(source_query)
    fallback = fallback_search_query(source_query)
    plan = deterministic_plan(source_query, "balanced")

    assert "install home assistant on a raspberry pi" in compact.lower()
    assert "home assistant" in fallback.lower()
    assert all("please current" not in query.lower() for query in plan["queries"])


def test_later_lowercase_error_question_preserves_subject_and_status_code():
    source_query = (
        "I need a detailed explanation with context, caveats, and safe production advice before "
        "any commands. Why does nginx return 502 bad gateway behind docker compose?"
    )

    compact = compact_search_query(source_query)
    fallback = fallback_search_query(source_query)

    assert "nginx" in compact.lower()
    assert "502 bad gateway" in compact.lower()
    assert "docker compose" in compact.lower()
    assert "502" in fallback


def test_quality_instruction_preamble_does_not_outrank_error_question():
    source_query = (
        "Before answering, carefully verify current documentation, account for version "
        "differences, and include safe diagnostic and rollback steps. Why does pip say "
        "externally managed environment on debian?"
    )

    compact = compact_search_query(source_query)
    fallback = fallback_search_query(source_query)

    assert "pip" in compact.lower()
    assert "externally managed environment" in compact.lower()
    assert "debian" in compact.lower()
    assert "pip" in fallback.lower()


def test_multisentence_error_keeps_diagnostic_context_and_fix_intent():
    source_query = (
        "I'm on Ubuntu 24.04. Docker says permission denied while trying to connect to the "
        "Docker daemon socket. How do I fix it safely?"
    )

    compact = compact_search_query(source_query)
    queries = compact_search_queries(source_query)

    assert "permission denied" in compact.lower()
    assert "docker daemon socket" in compact.lower()
    assert "fix it safely" in compact.lower()
    assert "Ubuntu" in compact
    assert any("permission denied" in query.lower() for query in queries)
    assert any("fix it safely" in query.lower() for query in queries)


def test_multi_intent_request_produces_queries_for_each_intent(monkeypatch):
    source_query = (
        "What is the current Docker release? "
        "How do I migrate from version 26 to version 27?"
    )
    monkeypatch.setattr(
        "planner.runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )

    plan = deterministic_plan(source_query, "balanced")

    assert any("current Docker release" in query for query in plan["queries"])
    assert any("migrate from version 26 to version 27" in query for query in plan["queries"])
    assert len(plan["queries"]) <= 3


def test_three_intent_request_uses_the_query_budget_for_each_intent(monkeypatch):
    source_query = (
        "What is the current Docker release? "
        "How do I migrate from version 26 to version 27? "
        "What breaking changes are in the latest Docker Compose?"
    )
    monkeypatch.setattr(
        "planner.runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )

    plan = deterministic_plan(source_query, "balanced")

    assert len(plan["queries"]) == 3
    assert any("current Docker release" in query for query in plan["queries"])
    assert any("migrate from version 26 to version 27" in query for query in plan["queries"])
    assert any("breaking changes" in query for query in plan["queries"])


def test_relative_date_context_does_not_leak_between_independent_intents():
    queries = compact_search_queries(
        "What happened in AI today? How do I install Docker?",
        current_date="2026-07-12",
    )

    assert any("AI today" in query and "2026-07-12" in query for query in queries)
    docker_queries = [query for query in queries if "install Docker" in query and "AI" not in query]
    assert docker_queries == ["install Docker"]


def test_long_quoted_error_is_kept_as_a_search_anchor():
    quoted_error = (
        '"this is a deliberately very long exact error message that contains the only '
        "discriminating details including errno EHOSTUNREACH and service alpha-backend-v27\""
    )
    source_query = (
        "Research this issue in depth with full context and explain all possible causes, "
        "diagnostic steps, safe remediations, rollback considerations, production risks, and "
        f"official references for the exact error {quoted_error}"
    )

    compact = compact_search_query(source_query)

    assert quoted_error in compact
    assert len(compact) <= 180


def test_oversized_quoted_error_keeps_bounded_head_and_discriminating_tail():
    quoted_error = (
        '"connection attempt failed while processing a deliberately oversized diagnostic '
        + "message with repeated generic context " * 5
        + "and the decisive errno EHOSTUNREACH for service alpha-backend-v27\""
    )

    compact = compact_search_query(f"Investigate the exact failure {quoted_error}")

    assert len(compact) <= 180
    assert "connection attempt failed" in compact
    assert "EHOSTUNREACH" in compact
    assert "alpha-backend-v27" in compact


def test_oversized_url_and_identifier_never_become_empty_or_suffix_only(monkeypatch):
    monkeypatch.setattr(
        "planner.runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )
    long_url = "https://example.com/" + "path-segment-" * 30
    scheme_less_url = "docs.example.com/" + "path-segment-" * 30 + "distinctive-tail"
    private_urls = [
        "docs.example.com:8443/" + "path-segment-" * 30 + "distinctive-tail",
        "localhost:8080/" + "path-segment-" * 30 + "distinctive-tail",
        "192.168.1.20:8080/" + "path-segment-" * 30 + "distinctive-tail",
        "[2001:db8::1]:8080/" + "path-segment-" * 30 + "distinctive-tail",
    ]
    opaque_identifier = "prefix-" + "x" * 300 + "-distinctive-suffix"

    url_query = compact_search_query(f"The current release notes are here: {long_url}")
    scheme_less_query = compact_search_query(f"Find release notes at {scheme_less_url}")
    identifier_query = compact_search_query(opaque_identifier)
    url_plan = deterministic_plan(long_url, "balanced")
    identifier_plan = deterministic_plan(opaque_identifier, "balanced")

    assert 0 < len(url_query) <= 180
    assert "https://example.com" in url_query
    assert "path-segment" in url_query
    assert scheme_less_query.startswith("docs.example.com/")
    assert scheme_less_query.endswith("distinctive-tail")
    assert len(scheme_less_query) <= 180
    for private_url in private_urls:
        private_query = compact_search_query(f"Find release notes at {private_url}")
        assert private_query.startswith(private_url.split("/", 1)[0])
        assert private_query.endswith("distinctive-tail")
        assert len(private_query) <= 180
    assert 0 < len(identifier_query) <= 180
    assert identifier_query.startswith("prefix-")
    assert identifier_query.endswith("-distinctive-suffix")
    assert all(query.strip() != "official documentation" for query in url_plan["queries"])
    assert all(query.strip() != "official documentation" for query in identifier_plan["queries"])


@pytest.mark.parametrize("limit", [1, 2, 3, 8, 20])
@pytest.mark.parametrize(
    "source_query",
    [
        "Docker permission denied while connecting to the daemon socket",
        "https://example.com/" + "path-segment-" * 30,
        "prefix-" + "x" * 300 + "-distinctive-suffix",
    ],
)
def test_nonempty_queries_remain_nonempty_and_bounded_at_tiny_limits(source_query, limit):
    compact = compact_search_query(source_query, limit=limit)

    assert compact
    assert len(compact) <= limit


def test_zero_query_budget_emits_no_diagnostic_search_queries():
    assert deterministic_plan("local memory only", "local_only")["queries"] == []


@pytest.mark.parametrize("period", ["last week", "next month", "past year"])
def test_compaction_preserves_common_relative_date_periods(period):
    source_query = (
        "Explain the background and operational implications in enough detail to support a careful "
        f"comparison. Only include Docker security advisories from the {period}."
    )

    assert period in compact_search_query(source_query)


def test_unique_queries_deduplicates_after_length_cap():
    planner = pytest.importorskip("planner")
    shared_prefix = "x" * 500

    assert planner._unique_queries([shared_prefix + "a", shared_prefix + "b"], 3) == [shared_prefix]


def test_non_english_instruction_remains_nonempty_and_bounded():
    source_query = "Dockerのインストール方法を調べてください。Ubuntu 24.04向けの最新手順を説明してください。"

    compact = compact_search_query(source_query)
    fallback = fallback_search_query(source_query)

    assert compact
    assert fallback
    assert len(compact) <= 180
    assert "Ubuntu 24.04" in compact


def test_mixed_language_fallback_retains_non_ascii_research_intent():
    source_query = (
        "Dockerの最新のインストール方法を詳しく調べ、公式ドキュメントに基づいて"
        "Ubuntu 24.04での手順と注意点を説明してください。"
    )

    fallback = fallback_search_query(source_query)

    assert "Docker" in fallback
    assert "Ubuntu" in fallback
    assert "インストール" in fallback


def test_terminal_url_punctuation_is_not_duplicated():
    source_query = "Check https://docs.example.com/v3/setup. Return verified facts."

    compact = compact_search_query(source_query)

    assert compact == "https://docs.example.com/v3/setup"
    assert compact.count("https://") == 1


@pytest.mark.asyncio
async def test_research_pipeline_retries_compact_query_after_zero_results(monkeypatch):
    calls = []

    async def fake_plan(query, mode):
        return {"query": query, "mode": mode, "queries": ["verbose instruction query"]}

    async def fake_search(query, max_results, mode):
        calls.append(query)
        if query == "verbose instruction query":
            return []
        return [
            {
                "title": "Current AI news",
                "url": "https://example.com/news",
                "domain": "example.com",
                "snippet": "Current reporting",
                "score": 2,
                "score_reasons": [],
            }
        ]

    monkeypatch.setattr(pipelines, "build_research_plan", fake_plan)
    monkeypatch.setattr(pipelines, "searxng_search", fake_search)
    monkeypatch.setattr(
        pipelines,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )
    monkeypatch.setattr(
        pipelines,
        "crawl_and_ingest_limited",
        AsyncMock(
            return_value={"ok": True, "url": "https://example.com/news", "domain": "example.com"}
        ),
    )
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        VERBOSE_AI_NEWS_QUERY,
        mode="balanced",
        max_sources=1,
        persist_source_artifacts=False,
    )

    assert calls == ["verbose instruction query", "AI news today 2026-07-12"]
    assert result["searched"][0]["url"] == "https://example.com/news"
    assert result["search_fallback"] == {
        "triggered": True,
        "reason": "initial_queries_returned_no_results",
        "query": "AI news today 2026-07-12",
        "produced_results": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [RuntimeError("search unavailable"), [{"url": ""}]])
async def test_research_pipeline_does_not_fallback_on_error_or_raw_results(
    monkeypatch,
    initial,
):
    calls = []

    async def fake_plan(query, mode):
        return {"query": query, "mode": mode, "queries": ["initial query"]}

    async def fake_search(query, max_results, mode):
        calls.append(query)
        if isinstance(initial, Exception):
            raise initial
        return initial

    monkeypatch.setattr(pipelines, "build_research_plan", fake_plan)
    monkeypatch.setattr(pipelines, "searxng_search", fake_search)

    result = await pipelines.research_pipeline(
        VERBOSE_AI_NEWS_QUERY,
        mode="quick",
        max_sources=0,
        persist_source_artifacts=False,
    )

    assert calls == ["initial query"]
    assert "search_fallback" not in result


@pytest.mark.asyncio
async def test_configured_planner_compacts_request_and_preserves_query_budget(monkeypatch):
    planner = pytest.importorskip("planner")
    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://planner.example")
    monkeypatch.setattr(planner, "PLANNER_MODEL", "private-planner")
    monkeypatch.setattr(
        planner,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )
    monkeypatch.setattr(
        planner,
        "_chat",
        AsyncMock(
            return_value=json.dumps(
                {
                    "queries": [
                        VERBOSE_AI_NEWS_QUERY,
                        "AI news today",
                        "latest AI model releases",
                        "AI regulation news",
                    ],
                    "subquestions": [],
                }
            )
        ),
    )

    plan = await planner.build_research_plan(VERBOSE_AI_NEWS_QUERY, "balanced")

    assert len(plan["queries"]) == 3
    assert plan["queries"][0] == "Today's most important AI news 2026-07-12"
    assert VERBOSE_AI_NEWS_QUERY not in plan["queries"]
    assert "AI news today 2026-07-12" in plan["queries"]
    assert plan["generated_by"] == "model:private-planner"


@pytest.mark.asyncio
async def test_configured_planner_preserves_full_deterministic_intent_coverage(monkeypatch):
    planner = pytest.importorskip("planner")
    source_query = (
        "What is the current Docker release? "
        "How do I migrate from version 26 to version 27? "
        "What breaking changes are in the latest Docker Compose?"
    )
    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://planner.example")
    monkeypatch.setattr(planner, "PLANNER_MODEL", "private-planner")
    monkeypatch.setattr(
        planner,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-07-12"},
    )
    planner_chat = AsyncMock(
        return_value=json.dumps(
            {
                "queries": [
                    "Docker 26 to 27 migration official guide",
                    "Docker release history",
                    "Docker Compose latest breaking changes release notes",
                ],
                "subquestions": [],
            }
        )
    )
    monkeypatch.setattr(planner, "_chat", planner_chat)

    plan = await planner.build_research_plan(source_query, "balanced")

    assert len(plan["queries"]) == 3
    assert any("current Docker release" in query for query in plan["queries"])
    assert any("migrate from version 26 to version 27" in query for query in plan["queries"])
    assert any("breaking changes" in query for query in plan["queries"])
    assert plan["generated_by"] == "deterministic"
    planner_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_configured_planner_output_remains_deterministic(monkeypatch):
    planner = pytest.importorskip("planner")
    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://planner.example")
    monkeypatch.setattr(planner, "PLANNER_MODEL", "private-planner")
    monkeypatch.setattr(planner, "_chat", AsyncMock(return_value='{"queries": [], "subquestions": []}'))

    plan = await planner.build_research_plan("How do I install Docker?", "balanced")

    assert plan["generated_by"] == "deterministic"


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
