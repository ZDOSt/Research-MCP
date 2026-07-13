import json
from datetime import date, datetime, timezone

import pytest

import searching
from planner import deterministic_plan
from searching import (
    SearchResults,
    compact_search_results,
    infer_search_policy,
    normalize_published_at,
    parse_published_datetime,
)


def test_search_policy_targets_current_news_without_overfiltering_other_research():
    news = infer_search_policy(
        "today's AI news 2026-07-13",
        "balanced",
        current_date="2026-07-13",
    )
    ordinary = infer_search_policy("how TCP congestion control works", "balanced")
    technical = infer_search_policy("current Docker installation guide", "technical")
    academic = infer_search_policy("systematic review of sleep research", "academic")

    assert news.categories == ("news", "general")
    assert news.time_range == "day"
    assert news.strict_date is True
    assert news.target_date == date(2026, 7, 13)
    assert ordinary.categories == ("general",)
    assert ordinary.time_range is None
    assert technical.categories == ("it", "general")
    assert technical.time_range is None
    assert academic.categories == ("science", "general")
    assert academic.time_range is None


@pytest.mark.parametrize(
    ("query", "expected_days", "expected_time_range"),
    [
        ("Docker security advisories from the past 24 hours", 1, "day"),
        ("Docker security advisories from the last week", 7, "week"),
        ("Docker security advisories from the last 7 days", 7, "week"),
    ],
)
def test_relative_windows_map_to_bounded_freshness_policies(
    query,
    expected_days,
    expected_time_range,
):
    policy = infer_search_policy(query, current_date="2026-07-13")

    assert policy.temporal_intent == "recent"
    assert policy.freshness_max_age_days == expected_days
    assert policy.time_range == expected_time_range


def test_academic_intent_is_inferred_without_requiring_the_client_to_set_mode():
    policy = infer_search_policy(
        "Latest research papers on API security",
        mode="balanced",
        current_date="2026-07-13",
    )

    assert policy.categories == ("science", "general")
    assert policy.news_intent is False


@pytest.mark.parametrize(
    "query",
    ["best paper shredder", "how to make a paper airplane", "study desk buying guide"],
)
def test_weak_academic_words_do_not_force_science_search(query):
    assert infer_search_policy(query, mode="balanced").categories == ("general",)


def test_explicit_technical_intent_beats_weak_journal_word():
    policy = infer_search_policy("Journal app installation", mode="balanced")

    assert policy.categories == ("it", "general")


def test_balanced_academic_intent_uses_academic_ranking_boosts():
    policy = infer_search_policy("Latest research papers on API security", "balanced")
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "API security preprint",
                    "url": "https://arxiv.org/abs/2607.12345",
                    "content": "Research on API security",
                    "engine": "arxiv",
                }
            ]
        },
        "Latest research papers on API security",
        mode="balanced",
        policy=policy,
    )

    reasons = results[0]["score_reasons"]
    assert "domain boost: arxiv.org" in reasons
    assert "engine boost: arxiv" in reasons


def test_balanced_technical_intent_uses_technical_ranking_boosts():
    query = "current Docker installation guide"
    policy = infer_search_policy(query, mode="balanced", current_date="2026-07-13")
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Install Docker Engine",
                    "url": "https://docs.docker.com/engine/install/ubuntu/",
                    "content": "Official Docker installation documentation",
                },
                {
                    "title": "Docker installation issue",
                    "url": "https://github.com/docker/docs/issues/1",
                    "content": "Docker installation troubleshooting",
                    "engine": "github",
                },
                {
                    "title": "Docker installation answer",
                    "url": "https://stackoverflow.com/questions/1/docker-installation",
                    "content": "Docker installation troubleshooting",
                    "engine": "stackoverflow",
                },
            ]
        },
        query,
        mode="balanced",
        policy=policy,
    )

    assert policy.categories == ("it", "general")
    reasons_by_url = {item["url"]: item["score_reasons"] for item in results}
    assert "domain boost: docs.docker.com" in reasons_by_url[
        "https://docs.docker.com/engine/install/ubuntu/"
    ]
    assert "domain boost: github.com" in reasons_by_url[
        "https://github.com/docker/docs/issues/1"
    ]
    assert "engine boost: github" in reasons_by_url[
        "https://github.com/docker/docs/issues/1"
    ]
    assert "domain boost: stackoverflow.com" in reasons_by_url[
        "https://stackoverflow.com/questions/1/docker-installation"
    ]
    assert "engine boost: stackoverflow" in reasons_by_url[
        "https://stackoverflow.com/questions/1/docker-installation"
    ]


def test_hacker_news_api_documentation_is_technical_not_news():
    policy = infer_search_policy("Hacker News API documentation", mode="balanced")

    assert policy.categories == ("it", "general")
    assert policy.news_intent is False
    assert policy.time_range is None


@pytest.mark.parametrize(
    "query",
    [
        "today's GitHub news",
        "latest news about Docker installation",
    ],
)
def test_explicit_news_beats_incidental_technical_terms(query):
    policy = infer_search_policy(query, mode="balanced", current_date="2026-07-13")

    assert policy.categories == ("news", "general")
    assert policy.news_intent is True


def test_hacker_news_documentation_is_technical_in_either_word_order():
    policy = infer_search_policy(
        "API documentation for Hacker News",
        mode="balanced",
    )

    assert policy.categories == ("it", "general")
    assert policy.news_intent is False


def test_historical_as_of_date_does_not_apply_a_current_day_filter():
    policy = infer_search_policy(
        "AI news as of 2024-01-01",
        current_date="2026-07-13",
    )

    assert policy.temporal_intent == "as_of"
    assert policy.cutoff_date == date(2024, 1, 1)
    assert policy.target_date is None
    assert policy.strict_date is False
    assert policy.time_range is None
    assert policy.freshness_max_age_days is None


def test_as_of_cutoff_retains_sources_from_any_earlier_date():
    policy = infer_search_policy(
        "AI regulation as of 2024-01-01",
        current_date="2026-07-13",
    )
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Foundational source",
                    "url": "https://example.com/foundational",
                    "content": "Evidence available well before the cutoff",
                    "publishedDate": "2020-01-01",
                },
                {
                    "title": "After the cutoff",
                    "url": "https://example.com/later",
                    "content": "Later evidence",
                    "publishedDate": "2024-01-02",
                },
            ]
        },
        "AI regulation as of 2024-01-01",
        policy=policy,
    )

    assert [item["title"] for item in results] == ["Foundational source"]
    assert results[0]["freshness_status"] == "within_window"
    assert results.diagnostics["counts"]["outside_window_dropped"] == 1


def test_explicit_on_date_uses_exact_date_filtering():
    policy = infer_search_policy(
        "AI news on 2026-07-01",
        current_date="2026-07-13",
    )

    assert policy.temporal_intent == "exact_date"
    assert policy.target_date == date(2026, 7, 1)
    assert policy.strict_date is True
    assert policy.time_range is None


def test_ambiguous_technical_on_date_is_not_a_publication_date_filter():
    policy = infer_search_policy(
        "Docker documentation on 2026-07-01 networking changes",
        current_date="2026-07-13",
    )

    assert policy.categories == ("it", "general")
    assert policy.target_date is None
    assert policy.strict_date is False
    assert policy.temporal_intent == "event_date"
    assert policy.event_start_date == date(2026, 7, 1)
    assert policy.event_end_date == date(2026, 7, 1)
    assert policy.cutoff_date is None


@pytest.mark.parametrize(
    "query",
    [
        "AI news on 2026-99-99",
        "AI news as of 2026-02-30",
        "Docker released on 2026-02-30",
    ],
)
def test_invalid_explicit_date_does_not_fall_back_to_the_runtime_date(query):
    policy = infer_search_policy(query, current_date="2026-07-13")

    assert policy.target_date is None
    assert policy.cutoff_date is None
    assert policy.strict_date is False
    assert policy.temporal_intent == "none"


@pytest.mark.parametrize(
    ("query", "intent", "start", "cutoff"),
    [
        (
            "AI regulation as of January 1, 2024",
            "as_of",
            None,
            date(2024, 1, 1),
        ),
        (
            "AI regulation as of 01/31/2024",
            "as_of",
            None,
            date(2024, 1, 31),
        ),
        (
            "AI news since March 2024",
            "publication_since",
            date(2024, 3, 1),
            None,
        ),
        (
            "AI news after March 2024",
            "publication_since",
            date(2024, 4, 1),
            None,
        ),
        (
            "AI news before 2024",
            "publication_before",
            None,
            date(2023, 12, 31),
        ),
        (
            "AI news from January 2024 through March 2024",
            "publication_range",
            date(2024, 1, 1),
            date(2024, 3, 31),
        ),
        (
            "AI news between 01/01/2024 and 03/31/2024",
            "publication_range",
            date(2024, 1, 1),
            date(2024, 3, 31),
        ),
        (
            "AI news in 2024",
            "publication_range",
            date(2024, 1, 1),
            date(2024, 12, 31),
        ),
    ],
)
def test_natural_date_scopes_are_normalized(query, intent, start, cutoff):
    policy = infer_search_policy(query, current_date="2026-07-13")

    assert policy.temporal_intent == intent
    assert policy.start_date == start
    assert policy.cutoff_date == cutoff
    assert policy.reference_date == date(2026, 7, 13)


def test_publication_range_discards_results_outside_both_bounds():
    query = "AI news from January 2024 through March 2024"
    policy = infer_search_policy(query, current_date="2026-07-13")
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Before",
                    "url": "https://example.com/before",
                    "publishedDate": "2023-12-31",
                },
                {
                    "title": "Inside",
                    "url": "https://example.com/inside",
                    "publishedDate": "2024-02-15",
                },
                {
                    "title": "After",
                    "url": "https://example.com/after",
                    "publishedDate": "2024-04-01",
                },
            ]
        },
        query,
        policy=policy,
    )

    assert [item["title"] for item in results] == ["Inside"]
    assert results.diagnostics["counts"]["outside_window_dropped"] == 2


@pytest.mark.parametrize(
    "query",
    [
        "What did Docker release on July 1, 2024?",
        "News about the Docker release on July 1, 2024",
    ],
)
def test_event_date_does_not_filter_by_source_publication_date(query):
    policy = infer_search_policy(query, current_date="2026-07-13")
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Authoritative follow-up",
                    "url": "https://example.com/follow-up",
                    "content": "A report published after the release event",
                    "publishedDate": "2024-07-02",
                }
            ]
        },
        query,
        policy=policy,
    )

    assert policy.temporal_intent == "event_date"
    assert policy.target_date is None
    assert policy.strict_date is False
    assert results[0]["freshness_status"] == "not_evaluated"


@pytest.mark.parametrize(
    "query",
    [
        "Find news about the earthquake on July 1, 2024",
        "Find news about the election on July 1, 2024",
        "Find news of the election on July 1, 2024",
        "Find headlines regarding the outage on July 1, 2024",
    ],
)
def test_news_about_an_event_date_keeps_later_follow_up_sources_for_every_variant(query):
    plan = deterministic_plan(query, "balanced")

    for search_query in plan["queries"]:
        policy = infer_search_policy(search_query, current_date="2026-07-13")
        results = compact_search_results(
            {
                "results": [
                    {
                        "title": "Authoritative follow-up",
                        "url": "https://example.com/follow-up",
                        "content": "A report published after the event",
                        "publishedDate": "2024-07-02",
                    }
                ]
            },
            search_query,
            policy=policy,
        )

        assert policy.temporal_intent == "event_date"
        assert policy.strict_date is False
        assert policy.event_start_date == date(2024, 7, 1)
        assert [item["title"] for item in results] == ["Authoritative follow-up"]
        assert results[0]["freshness_status"] == "not_evaluated"


def test_explicit_publication_wording_wins_inside_news_about_event_phrase():
    query = "Find news about the earthquake report published on July 1, 2024"
    policy = infer_search_policy(query, current_date="2026-07-13")

    assert policy.temporal_intent == "exact_date"
    assert policy.strict_date is True
    assert policy.target_date == date(2024, 7, 1)


def test_explicit_publication_date_remains_strict():
    query = "Research papers published on July 1, 2024"
    policy = infer_search_policy(query, current_date="2026-07-13")
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Exact publication",
                    "url": "https://example.com/exact",
                    "publishedDate": "2024-07-01",
                },
                {
                    "title": "Later publication",
                    "url": "https://example.com/later",
                    "publishedDate": "2024-07-02",
                },
            ]
        },
        query,
        policy=policy,
    )

    assert policy.temporal_intent == "exact_date"
    assert policy.strict_date is True
    assert [item["title"] for item in results] == ["Exact publication"]


@pytest.mark.parametrize(
    ("relative_day", "target_date"),
    [
        ("today", date(2026, 7, 13)),
        ("yesterday", date(2026, 7, 12)),
    ],
)
def test_relative_publication_semantics_survive_every_query_variant(
    relative_day,
    target_date,
):
    plan = deterministic_plan(
        f"Find reports published {relative_day} about earnings",
        "balanced",
    )

    assert len(plan["queries"]) == 3
    for search_query in plan["queries"]:
        policy = infer_search_policy(search_query, current_date="2026-07-13")
        assert policy.strict_date is True
        assert policy.target_date == target_date


def test_as_of_date_rejects_later_results():
    policy = infer_search_policy(
        "AI news as of 2024-01-01",
        current_date="2026-07-13",
    )
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Available at cutoff",
                    "url": "https://example.com/at-cutoff",
                    "content": "Historical reporting",
                    "publishedDate": "2024-01-01T12:00:00Z",
                },
                {
                    "title": "Published after cutoff",
                    "url": "https://example.com/after-cutoff",
                    "content": "Later reporting",
                    "publishedDate": "2024-01-02T12:00:00Z",
                },
            ]
        },
        "AI news as of 2024-01-01",
        policy=policy,
    )

    assert [item["title"] for item in results] == ["Available at cutoff"]
    assert results[0]["freshness_status"] == "within_window"
    assert results.diagnostics["counts"]["outside_window_dropped"] == 1


def test_yesterday_uses_wider_engine_window_then_exact_date_filter():
    policy = infer_search_policy(
        "yesterday's major releases 2026-07-12",
        current_date="2026-07-13",
    )

    assert policy.time_range == "week"
    assert policy.target_date == date(2026, 7, 12)
    assert policy.strict_date is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-07-13T02:03:04Z", "2026-07-13T02:03:04+00:00"),
        ("Sun, 13 Jul 2026 02:03:04 GMT", "2026-07-13T02:03:04+00:00"),
        ("July 13, 2026", "2026-07-13T00:00:00+00:00"),
        (1783908184000, "2026-07-13T02:03:04+00:00"),
    ],
)
def test_publication_dates_are_normalized(value, expected):
    assert normalize_published_at(value) == expected


def test_relative_publication_date_parsing_is_bounded_to_supplied_clock():
    now = datetime(2026, 7, 13, 12, tzinfo=timezone.utc)

    assert parse_published_datetime("3 hours ago", now=now) == datetime(
        2026,
        7,
        13,
        9,
        tzinfo=timezone.utc,
    )


def test_strict_date_discards_known_stale_results_but_retains_undated_fallbacks():
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Current report",
                    "url": "https://news.example/current",
                    "content": "AI news published today",
                    "publishedDate": "2026-07-13T01:00:00Z",
                },
                {
                    "title": "Old report",
                    "url": "https://news.example/old",
                    "content": "Archived AI report",
                    "publishedDate": "2025-07-13T01:00:00Z",
                },
                {
                    "title": "Undated report",
                    "url": "https://other.example/report",
                    "content": "Potentially relevant AI news",
                },
            ]
        },
        "today's AI news 2026-07-13",
    )

    assert isinstance(results, SearchResults)
    assert [item["title"] for item in results] == ["Current report", "Undated report"]
    assert results[0]["published_at"] == "2026-07-13T01:00:00+00:00"
    assert results[0]["freshness_status"] == "exact_match"
    assert results[1]["published_at"] is None
    assert results[1]["freshness_status"] == "undated"
    assert results.diagnostics["counts"]["outside_window_dropped"] == 1


def test_strict_date_is_compared_in_the_configured_research_timezone():
    policy = infer_search_policy(
        "today's AI news 2026-07-13",
        current_date="2026-07-13",
        timezone_name="America/New_York",
    )
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Late local report",
                    "url": "https://example.com/late-report",
                    "content": "Published late on the local calendar day",
                    "publishedDate": "2026-07-14T02:00:00Z",
                }
            ]
        },
        "today's AI news 2026-07-13",
        policy=policy,
    )

    assert results[0]["freshness_status"] == "exact_match"
    assert results.diagnostics["search_policy"]["timezone"] == "America/New_York"


def test_canonical_duplicate_keeps_later_exact_date_metadata():
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "Undated discovery copy",
                    "url": "https://news.example/story?utm_source=search",
                    "content": "Initial undated result",
                },
                {
                    "title": "Dated discovery copy",
                    "url": "https://news.example/story",
                    "content": "Later result with publication metadata",
                    "publishedDate": "2026-07-13T08:00:00Z",
                },
            ]
        },
        "today's AI news 2026-07-13",
    )

    assert len(results) == 1
    assert results[0]["title"] == "Dated discovery copy"
    assert results[0]["freshness_status"] == "exact_match"
    assert results[0]["published_at"] == "2026-07-13T08:00:00+00:00"


def test_news_ranking_preserves_searx_rank_and_does_not_globally_boost_docs():
    results = compact_search_results(
        {
            "results": [
                {
                    "title": "New AI model announced",
                    "url": "https://www.reuters.com/technology/new-model/",
                    "content": "Latest AI news",
                    "publishedDate": "2026-07-13T02:00:00Z",
                },
                {
                    "title": "AI documentation",
                    "url": "https://docs.github.com/en/copilot",
                    "content": "Official AI documentation",
                },
                {
                    "title": "AI news archive",
                    "url": "https://github.com/example/ai-news",
                    "content": "An old archive",
                    "publishedDate": "2022-01-01T00:00:00Z",
                },
            ]
        },
        "latest AI news",
        mode="balanced",
    )

    assert results[0]["url"].startswith("https://www.reuters.com/")
    assert results[0]["search_rank"] == 1
    assert results[0]["freshness_status"] == "within_window"
    assert not any(
        "domain boost" in reason
        for item in results
        if "github.com" in item["domain"]
        for reason in item["score_reasons"]
    )


@pytest.mark.asyncio
async def test_searx_request_applies_policy_and_surfaces_engine_diagnostics(monkeypatch):
    captured = []
    payload = {
        "results": [
            {
                "title": "Current report",
                "url": "https://example.com/current",
                "content": "Current AI news",
                "publishedDate": "2026-07-13T05:00:00Z",
            }
        ],
        "unresponsive_engines": [
            ["bing news", "timeout"],
            {"engine": "example engine", "reason": "rate limited"},
        ],
    }

    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield json.dumps(payload).encode()

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, method, url, **kwargs):
            captured.append({"method": method, "url": url, **kwargs})
            return Response()

    monkeypatch.setattr(searching, "SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setattr(searching.httpx, "AsyncClient", Client)

    results = await searching.searxng_search(
        "today's AI news 2026-07-13",
        current_date="2026-07-13",
    )

    assert captured[0]["params"] == {
        "q": "today's AI news 2026-07-13",
        "format": "json",
        "language": "auto",
        "engines": "reuters,bing news",
        "time_range": "day",
    }
    assert all("categories" not in request["params"] for request in captured)
    assert results.diagnostics["search_policy"]["target_date"] == "2026-07-13"
    assert results.diagnostics["counts"]["exact_match_results"] == 1
    assert results.diagnostics["unresponsive_engines"] == [
        {"engine": "bing news", "reason_code": "timeout"},
        {"engine": "example engine", "reason_code": "rate_limited"},
    ]
