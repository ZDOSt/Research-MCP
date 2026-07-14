import asyncio
import math
import time
from unittest.mock import AsyncMock

import pytest

import pipelines
from artifact_store import ArtifactStore
from searching import SearchResults


@pytest.mark.asyncio
async def test_proposed_primary_cannot_resolve_without_canonical_relevance(monkeypatch):
    calls = []
    planner_inputs = []
    proposed = "Docker container network fixes"
    canonical = "install SillyTavern on Ubuntu using Docker"

    async def plan(query, mode, proposed_queries=None):
        planner_inputs.append(proposed_queries)
        return {
            "query": query,
            "mode": mode,
            "queries": [proposed, canonical],
            "query_intent_ids": ["install", "install"],
            "intent_contexts": {"install": canonical},
            "generated_by": "calling-model+deterministic",
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        if query == proposed:
            return [
                {
                    "title": "Docker bridge network fixes",
                    "snippet": "Troubleshoot generic container bridge networking",
                    "url": "https://network-one.example/guide",
                    "domain": "network-one.example",
                },
                {
                    "title": "Docker container network troubleshooting",
                    "snippet": "Repair bridge and DNS connectivity",
                    "url": "https://network-two.example/guide",
                    "domain": "network-two.example",
                },
            ]
        return [
            {
                "title": "Install SillyTavern on Ubuntu with Docker",
                "snippet": "SillyTavern Docker installation steps for Ubuntu",
                "url": "https://install-one.example/guide",
                "domain": "install-one.example",
            },
            {
                "title": "SillyTavern Docker setup on Ubuntu",
                "snippet": "Deploy SillyTavern using Docker on Ubuntu",
                "url": "https://install-two.example/guide",
                "domain": "install-two.example",
            },
        ]

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        canonical,
        mode="balanced",
        max_sources=0,
        verify=False,
        persist_source_artifacts=False,
        proposed_queries=[proposed],
    )

    assert planner_inputs == [[proposed]]
    assert calls == [proposed, canonical]
    assert result["plan"]["intent_contexts"] == {"install": canonical}
    assert all("network-" not in item["url"] for item in result["searched"])


@pytest.mark.parametrize(
    ("canonical", "executed_query"),
    [
        ("vegan dinner recipes", "chicken dinner recipes"),
        ("wireless headphones", "wired headphones"),
        ("free project management software", "paid project management software"),
        ("indoor security cameras", "outdoor security cameras"),
        ("beginner Python tutorials", "advanced Python tutorials"),
        ("cat food recommendations", "dog food recommendations"),
        ("Android TV boxes", "Android TV remote apps"),
    ],
)
def test_noncanonical_result_cannot_substitute_canonical_topic_qualifier(
    canonical,
    executed_query,
):
    analysis = pipelines._canonical_result_relevance(
        {
            "title": executed_query,
            "snippet": f"A detailed guide to {executed_query}",
            "url": "https://drift.example/result",
        },
        canonical,
        executed_query=executed_query,
    )

    assert analysis["is_relevant"] is False
    assert analysis["reason"] == "conflicting_topic_qualifier"


@pytest.mark.parametrize(
    ("canonical", "executed_query"),
    [
        (
            "Find free password managers for families",
            "free password managers for Windows",
        ),
        (
            "accounting software for small businesses",
            "accounting software for students",
        ),
        (
            "plants for low-light bathrooms",
            "plants for sunny offices",
        ),
    ],
)
def test_noncanonical_result_must_cover_the_distinctive_canonical_intent(
    canonical,
    executed_query,
):
    analysis = pipelines._canonical_result_relevance(
        {
            "title": executed_query,
            "snippet": f"Detailed recommendations for {executed_query}",
            "url": "https://drift.example/result",
        },
        canonical,
        executed_query=executed_query,
    )

    assert analysis["is_relevant"] is False
    assert analysis["reason"] in {
        "insufficient_canonical_distinctive_overlap",
        "insufficient_topic_overlap",
    }


def test_noncanonical_result_allows_a_complete_safe_reformulation():
    analysis = pipelines._canonical_result_relevance(
        {
            "title": "SillyTavern Docker installation on Ubuntu",
            "snippet": "Official documentation for installing SillyTavern with Docker",
            "url": "https://docs.example/installation",
        },
        "install SillyTavern on Ubuntu using Docker",
        executed_query="SillyTavern Docker installation Ubuntu official documentation",
    )

    assert analysis["is_relevant"] is True


@pytest.mark.asyncio
async def test_search_batch_caps_upstream_query_concurrency(monkeypatch):
    active = 0
    peak_active = 0

    async def search(*, query, **_kwargs):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.02)
            return [query]
        finally:
            active -= 1

    monkeypatch.setattr(pipelines, "SEARCH_QUERY_CONCURRENCY", 2)
    monkeypatch.setattr(pipelines, "searxng_search", search)

    outcomes = await pipelines._run_search_batch_bounded(
        ["one", "two", "three", "four"],
        [None, None, None, None],
        max_results=4,
        mode="balanced",
        timeout_seconds=1,
    )

    assert outcomes == [["one"], ["two"], ["three"], ["four"]]
    assert peak_active == 2


@pytest.mark.asyncio
async def test_search_batch_cancellation_stops_owned_searches(monkeypatch):
    active = 0
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def search(**_kwargs):
        nonlocal active
        active += 1
        if active == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
            if active == 0:
                stopped.set()

    monkeypatch.setattr(pipelines, "SEARCH_QUERY_CONCURRENCY", 2)
    monkeypatch.setattr(pipelines, "searxng_search", search)

    task = asyncio.create_task(
        pipelines._run_search_batch_bounded(
            ["one", "two"],
            [None, None],
            max_results=4,
            mode="balanced",
            timeout_seconds=10,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(stopped.wait(), timeout=0.2)
    assert active == 0


def test_search_query_waves_keep_one_primary_and_one_reserve_per_intent():
    primary, reserve_waves = pipelines._partition_search_query_waves(
        ["news", "news primary", "install", "news overview", "install docs"],
        [1, 2, 3, 4, 5],
        ["news", "news", "install", "news", "install"],
    )

    assert primary == [("news", 1, "news"), ("install", 3, "install")]
    assert reserve_waves == [
        [
            ("news primary", 2, "news"),
            ("install docs", 5, "install"),
        ]
    ]


def test_deep_search_retains_ordered_reserve_waves():
    primary, reserve_waves = pipelines._partition_search_query_waves(
        ["topic", "topic docs", "topic primary", "topic independent", "topic overview"],
        [1, 2, 3, 4, 5],
        ["topic"] * 5,
        max_reserve_per_intent=4,
    )

    assert primary == [("topic", 1, "topic")]
    assert reserve_waves == [
        [("topic docs", 2, "topic")],
        [("topic primary", 3, "topic")],
        [("topic independent", 4, "topic")],
        [("topic overview", 5, "topic")],
    ]


def test_role_aware_search_pairs_expansion_with_canonical_anchor():
    primary, reserve_waves = pipelines._partition_search_query_waves(
        [
            "calling-model wording",
            "semantic expansion",
            "canonical anchor",
            "deterministic reserve",
        ],
        [1, 2, 3, 4],
        ["topic"] * 4,
        query_roles=[
            "calling_model",
            "semantic_expansion",
            "deterministic",
            "deterministic",
        ],
    )

    assert primary == [
        ("semantic expansion", 2, "topic"),
        ("canonical anchor", 3, "topic"),
    ]
    assert reserve_waves == [[("deterministic reserve", 4, "topic")]]


@pytest.mark.parametrize(
    "query_roles",
    [
        ["semantic_expansion"],
        ["semantic_expansion", "calling_model"],
        ["semantic_expansion", "unsupported"],
        ["semantic_expansion", {"role": "deterministic"}],
        ["semantic_expansion", "deterministic", "calling_model"],
    ],
)
def test_malformed_or_unanchored_query_roles_use_exact_legacy_waves(query_roles):
    arguments = (
        ["first", "second"],
        [1, 2],
        ["topic", "topic"],
    )

    legacy = pipelines._partition_search_query_waves(*arguments)
    role_aware = pipelines._partition_search_query_waves(
        *arguments,
        query_roles=query_roles,
    )

    assert role_aware == legacy


@pytest.mark.parametrize(
    "raw_intent_ids",
    [
        ["topic"],
        ["", "", ""],
    ],
)
@pytest.mark.asyncio
async def test_malformed_intent_metadata_disables_query_roles(
    monkeypatch,
    raw_intent_ids,
):
    calls = []

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": ["Docker", "Docker docs", "Docker releases"],
            "query_intent_ids": raw_intent_ids,
            "query_roles": [
                "semantic_expansion",
                "deterministic",
                "deterministic",
            ],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return [
            {
                "title": f"Docker result {index}",
                "snippet": "Docker container documentation",
                "url": f"https://docker-{index}.example/{query.replace(' ', '-')}",
                "domain": f"docker-{index}.example",
            }
            for index in range(2)
        ]

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    await pipelines.research_pipeline(
        "Docker",
        mode="balanced",
        max_sources=0,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls == ["Docker"]


@pytest.mark.asyncio
async def test_expansion_only_coverage_runs_deterministic_reserve(monkeypatch):
    expansion = "SillyTavern Linux Docker setup guide"
    canonical = "install SillyTavern on Ubuntu using Docker"
    deterministic_reserve = "SillyTavern Ubuntu Docker official documentation"
    calls = []

    async def plan(query, mode, proposed_queries=None):
        return {
            "query": query,
            "mode": mode,
            "queries": [expansion, canonical, deterministic_reserve],
            "query_intent_ids": ["install"] * 3,
            "query_roles": [
                "semantic_expansion",
                "deterministic",
                "deterministic",
            ],
            "intent_contexts": {"install": canonical},
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        if query != expansion:
            return []
        return [
            {
                "title": "Install SillyTavern on Ubuntu with Docker",
                "snippet": "SillyTavern Docker installation steps for Ubuntu",
                "url": f"https://install-{index}.example/guide",
                "domain": f"install-{index}.example",
            }
            for index in range(2)
        ]

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
    monkeypatch.setattr(pipelines, "SEARCH_RERANKER_ENABLED", False)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        canonical,
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls[:2] == [expansion, canonical]
    assert calls[2] == deterministic_reserve
    assert result["completion"]["resolved_intents"] == []
    assert result["completion"]["unresolved_intents"] == ["install"]
    assert all(
        association.get("query_role") == "semantic_expansion"
        for item in result["searched"]
        for association in item.get("matched_query_intents") or ()
    )


def test_candidate_pool_reserves_only_topically_relevant_intents():
    candidates = [
        {
            "url": "https://shared.example/result",
            "score": 10,
            "matched_intents": ["one", "two"],
            "topical_relevance": {"relevant_intents": ["one"]},
        },
        {
            "url": "https://one.example/result",
            "score": 9,
            "matched_intents": ["one"],
            "topical_relevance": {"relevant_intents": ["one"]},
        },
        {
            "url": "https://two.example/result",
            "score": 1,
            "matched_intents": ["two"],
            "topical_relevance": {"relevant_intents": ["two"]},
        },
    ]

    selected = pipelines._build_candidate_pool(candidates, 2, ["one", "two"])

    assert {item["url"] for item in selected} == {
        "https://shared.example/result",
        "https://two.example/result",
    }


@pytest.mark.asyncio
async def test_local_only_requested_verification_does_not_overstate_completion(
    monkeypatch,
):
    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": []}

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Stored note",
                        "url": "https://memory.example/note",
                        "domain": "memory.example",
                        "text": "Relevant stored research evidence.",
                    },
                    {
                        "title": "Independent note",
                        "url": "https://archive.example/item",
                        "domain": "archive.example",
                        "text": "Separate archival material about the subject.",
                    },
                ]
            }
        ),
    )

    result = await pipelines.research_pipeline(
        "stored research",
        mode="local_only",
        verify=True,
    )

    assert (
        result["verification"]["status"] == "multiple_sources_without_detected_overlap"
    )
    assert result["completion"]["status"] == "partial"
    assert result["completion"]["reason"] == "local_memory_retrieved_with_limitations"
    assert result["completion"]["reasons"] == ["verification_inconclusive"]


def test_search_outcome_coverage_requires_distinct_source_owners():
    same_owner = [
        {
            "title": "Powerful Android TV box comparison",
            "snippet": "Android streaming box benchmarks and specifications",
            "url": "https://docs.example.com/one",
            "domain": "docs.example.com",
        },
        {
            "title": "Best high-performance Android TV devices",
            "snippet": "Compare Android TV box performance",
            "url": "https://www.example.com/two",
            "domain": "www.example.com",
        },
    ]
    distinct_owners = same_owner + [
        {
            "title": "Android TV box benchmark results",
            "snippet": "Independent performance measurements for Android boxes",
            "url": "https://other.example.net/three",
            "domain": "other.example.net",
        }
    ]
    irrelevant = [
        {
            "title": "Traditional German cider guide",
            "snippet": "Regional apple varieties used to make Most",
            "url": "https://cider.example.org/most",
            "domain": "cider.example.org",
        },
        {
            "title": "Apple harvest festival",
            "snippet": "A calendar of local orchard events",
            "url": "https://festival.example.net/events",
            "domain": "festival.example.net",
        },
    ]

    query = "powerful Android TV box comparison"
    assert (
        pipelines._search_outcome_has_source_coverage(
            same_owner,
            query=query,
        )
        is False
    )
    assert (
        pipelines._search_outcome_has_source_coverage(
            distinct_owners,
            query=query,
        )
        is True
    )
    assert (
        pipelines._search_outcome_has_source_coverage(
            irrelevant,
            query=query,
        )
        is False
    )


@pytest.mark.asyncio
async def test_relevant_primary_results_skip_reserve_and_recovery(monkeypatch):
    calls = []
    candidates = [
        {
            "title": "High-performance Android TV box benchmark",
            "snippet": "Android TV box performance compared with Nvidia Shield",
            "url": "https://benchmarks.example/device",
            "domain": "benchmarks.example",
            "score": 9,
        },
        {
            "title": "Best Nvidia Shield alternatives",
            "snippet": "Current powerful Android TV streaming boxes compared",
            "url": "https://reviews.example/android-tv",
            "domain": "reviews.example",
            "score": 8,
        },
    ]

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": [
                "powerful Android TV box Nvidia Shield alternative",
                "Android TV box performance benchmark",
            ],
            "query_intent_ids": ["android-tv", "android-tv"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
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

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Find a powerful Android TV box that is an Nvidia Shield alternative",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls == ["powerful Android TV box Nvidia Shield alternative"]
    assert result["topical_relevance"]["status"] == "sufficient"
    assert result["topical_relevance"]["accepted_candidates"] == 2
    assert result["completion"]["status"] == "complete"
    assert "search_fallback" not in result


@pytest.mark.asyncio
async def test_low_relevance_runs_one_internal_repair_and_filters_noise(monkeypatch):
    calls = []
    irrelevant = [
        {
            "title": "Traditional German cider guide",
            "snippet": "Regional apple varieties used to make Most",
            "url": "https://cider.example/most",
            "domain": "cider.example",
            "score": 20,
        },
        {
            "title": "Orchard harvest calendar",
            "snippet": "Apple festivals and cider tastings this year",
            "url": "https://orchards.example/events",
            "domain": "orchards.example",
            "score": 19,
        },
    ]
    repaired = [
        {
            "title": "High-performance Android TV box benchmark",
            "snippet": "Android TV box performance compared with Nvidia Shield",
            "url": "https://benchmarks.example/android-tv",
            "domain": "benchmarks.example",
            "score": 9,
        },
        {
            "title": "Best Nvidia Shield alternatives",
            "snippet": "Powerful Android TV streaming boxes reviewed",
            "url": "https://reviews.example/shield-alternative",
            "domain": "reviews.example",
            "score": 8,
        },
    ]
    repair_query = "best high performance Android TV box Nvidia Shield alternative"

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": [
                'most powerful "shield-killer"',
                "Android TV box performance benchmark",
            ],
            "query_intent_ids": ["android-tv", "android-tv"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return repaired if query == repair_query else irrelevant

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
    monkeypatch.setattr(
        pipelines, "fallback_search_query", lambda *_args, **_kwargs: repair_query
    )
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Find a powerful Android TV box that is an Nvidia Shield alternative",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls == [
        'most powerful "shield-killer"',
        "Android TV box performance benchmark",
        repair_query,
    ]
    assert result["search_fallback"]["reason"] == "low_topical_relevance"
    assert result["topical_relevance"]["recovery_attempted"] is True
    assert result["topical_relevance"]["accepted_candidates"] == 2
    assert {item["domain"] for item in result["searched"]} == {
        "benchmarks.example",
        "reviews.example",
    }
    assert result["completion"]["status"] == "complete"


@pytest.mark.asyncio
async def test_low_relevance_after_repair_is_reported_as_insufficient(monkeypatch):
    calls = []
    irrelevant = [
        {
            "title": "Traditional German cider guide",
            "snippet": "Regional apple varieties used to make Most",
            "url": "https://cider.example/most",
            "domain": "cider.example",
            "score": 9,
        },
        {
            "title": "Orchard harvest calendar",
            "snippet": "Apple festivals and cider tastings this year",
            "url": "https://orchards.example/events",
            "domain": "orchards.example",
            "score": 8,
        },
    ]
    repair_query = "best high performance Android TV box Nvidia Shield alternative"

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": ["powerful shield killer"],
            "query_intent_ids": ["android-tv"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return irrelevant

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(
        pipelines, "fallback_search_query", lambda *_args, **_kwargs: repair_query
    )
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Find a powerful Android TV box that is an Nvidia Shield alternative",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls == ["powerful shield killer", repair_query]
    assert result["searched"] == []
    assert result["evidence"] == []
    assert result["topical_relevance"]["status"] == "low_relevance"
    assert result["completion"]["status"] == "insufficient"
    assert result["completion"]["reason"] == "low_topical_relevance"
    assert "low_web_topical_relevance" in result["completion"]["reasons"]
    assert any(
        "Do not repeat the same research_web request" in instruction
        for instruction in result["answering_instructions"]
    )


@pytest.mark.asyncio
async def test_multi_intent_completion_tracks_unresolved_intent_and_recovery(
    monkeypatch,
):
    calls = []
    android_results = [
        {
            "title": "Android TV box benchmark",
            "snippet": "Powerful Nvidia Shield alternative performance",
            "url": f"https://android{index}.example/review",
            "domain": f"android{index}.example",
            "score": 10 - index,
        }
        for index in range(2)
    ]
    irrelevant = [
        {
            "title": "Traditional cider guide",
            "snippet": "Apple varieties and orchard events",
            "url": f"https://cider{index}.example/guide",
            "domain": f"cider{index}.example",
            "score": 8 - index,
        }
        for index in range(2)
    ]

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": [
                "powerful Android TV box",
                "Android TV box benchmark",
                "install Docker Engine",
                "Docker Engine setup guide",
            ],
            "query_intent_ids": ["android", "android", "docker", "docker"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return android_results if "Android" in query else irrelevant

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
    monkeypatch.setattr(
        pipelines,
        "fallback_search_query",
        lambda source, **_kwargs: f"{source} repaired",
    )
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Find a powerful Android TV box and explain how to install Docker Engine",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert "Android TV box benchmark" not in calls
    assert "Docker Engine setup guide" in calls
    assert "install Docker Engine repaired" in calls
    assert result["completion"]["status"] == "partial"
    assert result["completion"]["resolved_intents"] == ["android"]
    assert result["completion"]["unresolved_intents"] == ["docker"]
    assert result["intent_coverage"]["android"]["status"] == "resolved"
    assert result["intent_coverage"]["docker"]["status"] == "unresolved"
    assert result["search_recoveries"][0]["intent_id"] == "docker"
    assert result["search_recoveries"][0]["reserve_executed"] is True


@pytest.mark.asyncio
async def test_low_web_relevance_with_memory_is_partial_not_insufficient(monkeypatch):
    irrelevant = [
        {
            "title": "Traditional cider guide",
            "snippet": "Apple varieties and orchard events",
            "url": "https://cider.example/guide",
            "domain": "cider.example",
            "score": 8,
        }
    ]

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": ["powerful Android TV box"],
            "query_intent_ids": ["android"],
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(
        pipelines,
        "fallback_search_query",
        lambda *_args, **_kwargs: "Android TV box benchmark",
    )
    monkeypatch.setattr(
        pipelines,
        "searxng_search",
        AsyncMock(return_value=irrelevant),
    )
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Stored Android TV research",
                        "url": "https://memory.example/android-tv",
                        "domain": "memory.example",
                        "text": "Previously stored Android TV box benchmark evidence.",
                    }
                ]
            }
        ),
    )

    result = await pipelines.research_pipeline(
        "Find a powerful Android TV box",
        max_sources=1,
        include_memory=True,
        verify=False,
        persist_source_artifacts=False,
    )

    assert result["completion"]["status"] == "partial"
    assert (
        result["completion"]["reason"]
        == "memory_evidence_without_relevant_web_evidence"
    )
    assert result["evidence"][0]["url"] == "https://memory.example/android-tv"
    assert any(
        "no topically relevant web evidence" in instruction.lower()
        for instruction in result["answering_instructions"]
    )


@pytest.mark.asyncio
async def test_strict_date_low_relevance_uses_distinct_semantic_repair(monkeypatch):
    calls = []
    source_query = "today's AI news"
    irrelevant = [
        {
            "title": "Traditional cider guide",
            "snippet": "Apple varieties used to make Most",
            "url": "https://cider.example/most",
            "domain": "cider.example",
            "score": 8,
        }
    ]
    relevant = [
        {
            "title": "Today's AI news from primary sources",
            "snippet": "Current AI model and infrastructure news today",
            "url": "https://news.example/ai",
            "domain": "news.example",
            "freshness_status": "exact_match",
            "score": 10,
        }
    ]

    async def plan(query, mode):
        return {
            "query": query,
            "mode": mode,
            "queries": [source_query],
            "query_intent_ids": ["news"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return irrelevant if query == source_query else relevant

    async def crawl(_semaphore, candidate, **_kwargs):
        return {
            "ok": True,
            "title": candidate["title"],
            "url": candidate["url"],
            "requested_url": candidate["url"],
            "domain": candidate["domain"],
            "evidence_text": candidate["snippet"],
            "freshness_status": candidate["freshness_status"],
        }

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(
        pipelines,
        "fallback_search_query",
        lambda *_args, **_kwargs: source_query,
    )
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        source_query,
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert len(calls) == 2
    assert calls[1] != calls[0]
    assert "today" in calls[1].lower()
    assert result["search_recoveries"][0]["phase"] == "relevance_recovery"
    assert "policy_relaxation" not in result["search_recoveries"][0]
    assert result["completion"]["status"] == "complete"


@pytest.mark.asyncio
async def test_missing_intent_ids_cannot_execute_every_planner_variant(monkeypatch):
    calls = []

    async def plan(_query, mode):
        return {
            "query": "question",
            "mode": mode,
            "queries": ["question", "question docs", "question overview"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    await pipelines.research_pipeline(
        "question",
        max_sources=0,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls == ["question", "question docs", "question primary sources"]


@pytest.mark.asyncio
async def test_quick_zero_result_uses_one_distinct_internal_repair(monkeypatch):
    calls = []

    async def plan(_query, mode):
        return {
            "query": "Android TV box benchmark",
            "mode": mode,
            "queries": ["Android TV box benchmark"],
            "query_intent_ids": ["android"],
        }

    async def search(*, query, **_kwargs):
        calls.append(query)
        return []

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "Android TV box benchmark",
        mode="quick",
        max_sources=0,
        verify=False,
        persist_source_artifacts=False,
    )

    assert calls == [
        "Android TV box benchmark",
        "Android TV box benchmark primary sources",
    ]
    assert result["search_fallback"]["reason"] == (
        "initial_queries_returned_no_results"
    )


@pytest.mark.asyncio
async def test_backend_failure_is_not_reported_as_an_ordinary_empty_search(monkeypatch):
    async def plan(_query, mode):
        return {"query": "question", "mode": mode, "queries": ["question"]}

    async def search(**_kwargs):
        return SearchResults(
            [],
            diagnostics={
                "acquisition_status": "failed",
                "failure_class": "transient",
                "acquisition_error": {
                    "code": "search_backend_unavailable",
                    "successful_responses": 0,
                    "responsive_engines": 0,
                },
                "engine_policy": "general",
                "unresponsive_engines": [
                    {
                        "engine": "bing",
                        "reason_code": "rate_limited",
                        "retry_after_seconds": 900,
                        "reason": "raw upstream details must not escape",
                    }
                ],
                "cache": {"status": "miss"},
                "search_stages": [
                    {
                        "stage": 1,
                        "status": "service_circuit_open",
                        "engines": [],
                        "retry_after_seconds": 42,
                        "skipped_cooldowns": [],
                    }
                ],
            },
        )

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )

    result = await pipelines.research_pipeline(
        "question",
        max_sources=0,
        verify=False,
        persist_source_artifacts=False,
    )

    assert result["completion"]["status"] == "insufficient"
    assert "search_failures" in result["completion"]["reasons"]
    assert result["search_errors"] == [
        {
            "query": "question",
            "phase": "initial",
            "error": "search_backend_unavailable",
        }
    ]
    diagnostics = result["search_diagnostics"][0]
    assert diagnostics["acquisition_status"] == "failed"
    assert diagnostics["failure_class"] == "transient"
    assert diagnostics["unresponsive_engines"] == [
        {
            "engine": "bing",
            "reason_code": "rate_limited",
            "retry_after_seconds": 900.0,
        }
    ]
    assert diagnostics["search_stages"][0]["retry_after_seconds"] == 42


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
        "title": "Current product guide",
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
            "title": "Current product guide",
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
    assert result["completion"]["status"] == "partial"
    assert "verification_inconclusive" in result["completion"]["reasons"]


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
        {
            "title": "Cached fallback result",
            "url": "https://cached.example/story",
            "domain": "cached.example",
            "snippet": "Previously retrieved discovery metadata",
            "retrieved_at_utc": "2026-07-13T08:00:00+00:00",
            "search_cached_at_utc": "2026-07-13T08:00:00+00:00",
            "search_cache_status": "stale_fallback",
            "freshness": "stale_cache_unverified",
            "freshness_unverified": True,
        },
    ]

    evidence = pipelines.build_search_snippet_evidence(
        candidates,
        [{"url": "https://one.example/story", "quote": "Extracted"}],
        limit=2,
    )

    assert len(evidence) == 2
    assert evidence[0]["url"] == "https://current.example/story"
    assert len(evidence[0]["quote"]) == 1600
    assert evidence[0]["published_at"] == "2026-07-13T09:00:00+00:00"
    assert evidence[0]["evidence_type"] == "search_result_snippet"
    assert evidence[1]["search_cache_status"] == "stale_fallback"
    assert evidence[1]["freshness_unverified"] is True
    assert "stale cache fallback" in evidence[1]["limitations"]


def test_page_evidence_preserves_publication_and_search_metadata():
    evidence = pipelines.build_evidence_pack(
        [
            {
                "text": "Extracted evidence",
                "url": "https://news.example/story",
                "published_at": "2026-07-13T08:30:00+00:00",
                "freshness_status": "exact_match",
                "freshness_unverified": True,
                "search_cache_status": "stale_fallback",
                "search_cached_at_utc": "2026-07-13T07:30:00+00:00",
                "search_engine": "google news",
                "search_rank": 3,
            }
        ]
    )

    assert evidence[0]["published_at"] == "2026-07-13T08:30:00+00:00"
    assert evidence[0]["freshness_status"] == "exact_match"
    assert evidence[0]["search_engine"] == "google news"
    assert evidence[0]["search_rank"] == 3
    assert evidence[0]["freshness_unverified"] is True
    assert evidence[0]["search_cache_status"] == "stale_fallback"
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
                        {"engine": "news-engine", "reason_code": "timeout"}
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

    monkeypatch.setattr(
        pipelines, "runtime_retrieval_context", lambda: retrieval_context
    )
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
        {"engine": "news-engine", "reason_code": "timeout"}
    ]
    assert result["evidence"][0]["url"] == "https://current.example/story"
    assert result["evidence"][0]["freshness_status"] == "exact_match"


@pytest.mark.asyncio
async def test_historical_exact_date_uses_a_distinct_date_safe_repair(monkeypatch):
    query = "AI news on 2026-06-01"
    calls = []

    async def plan(_query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(*, query, max_results, mode, policy):
        calls.append((query, policy))
        return SearchResults([], policy=policy)

    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(
        pipelines, "fallback_search_query", lambda *_args, **_kwargs: query
    )
    monkeypatch.setattr(pipelines, "searxng_search", search)

    result = await pipelines.research_pipeline(
        query,
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    assert [item[0] for item in calls] == [
        query,
        f"{query} primary sources",
    ]
    assert all(item[1].strict_date for item in calls)
    assert all(item[1].time_range is None for item in calls)
    assert result["search_fallback"]["reason"] == (
        "initial_queries_returned_no_results"
    )


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
            "snippet": f"Independent comparison of source {index}",
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
async def test_persistence_timeout_returns_extraction_and_invalidates_attempt(
    monkeypatch,
):
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
    assert {source["memory_index_state"] for source in result["crawled_sources"]} == {
        "revoked"
    }
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
async def test_cancellation_during_persistence_reaches_invalidation_promptly(
    monkeypatch,
):
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


@pytest.mark.asyncio
async def test_deferred_research_returns_fresh_evidence_without_qdrant_round_trip(
    monkeypatch,
    tmp_path,
):
    candidate = {
        "title": "Current installation guide",
        "url": "https://docs.example/install",
        "domain": "docs.example",
        "snippet": "Current installation instructions",
        "score": 10,
        "score_reasons": [],
    }

    async def plan(query, mode):
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs):
        return [candidate]

    async def crawl(_semaphore, source, **_kwargs):
        return {
            "ok": True,
            "title": source["title"],
            "url": source["url"],
            "requested_url": source["url"],
            "domain": source["domain"],
            "evidence_text": "Run the supported installer, then verify the service health.",
            "_content": "Run the supported installer, then verify the service health. "
            * 20,
        }

    store = ArtifactStore(tmp_path)
    rag_query = AsyncMock(return_value={"results": []})
    monkeypatch.setitem(
        pipelines.RESEARCH_MODE_CONFIG,
        "balanced",
        {
            "max_urls": 1,
            "search_results": 1,
            "top_k": 4,
            "planner_budget": 0.2,
            "search_budget": 0.2,
            "crawl_budget": 0.5,
            "total_budget": 1.0,
        },
    )
    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", crawl)
    monkeypatch.setattr(pipelines, "get_artifact_store", lambda: store)
    monkeypatch.setattr(pipelines, "rag_query_impl", rag_query)

    parent_id = "a" * 32
    result = await pipelines.research_pipeline(
        "install the current release",
        max_sources=1,
        verify=False,
        research_run_id=parent_id,
        defer_persistence=True,
        ingestion_attempt_id="b" * 64,
    )

    rag_query.assert_not_awaited()
    assert result["evidence"][0]["evidence_type"] == "extracted_page_content"
    assert result["results"] == []
    assert result["memory_results"] == []
    assert "_content" not in result["crawled_sources"][0]
    manifest = result["_deferred_persistence"]["sources"][0]
    assert manifest["job_id"] != parent_id
    assert manifest["artifact_owner_id"] == manifest["job_id"]
    assert manifest["artifact_path"].startswith(f"{manifest['job_id']}/")
    assert await store.exists(manifest["artifact_path"])
    assert result["persistence"] == {
        "mode": "deferred",
        "status": "prepared",
        "source_count": 1,
    }
    assert any(
        "answer now from evidence" in instruction
        for instruction in result["answering_instructions"]
    )


@pytest.mark.asyncio
async def test_discovery_reranker_reorders_candidates_and_falls_back_on_timeout(
    monkeypatch,
):
    candidates = [
        {"title": "First", "snippet": "one", "score": 10},
        {"title": "Second", "snippet": "two", "score": 5},
    ]

    async def rerank(_query, docs, _top_k):
        return [
            {**docs[1], "rerank_score": 0.9},
            {**docs[0], "rerank_score": 0.1},
        ]

    monkeypatch.setattr(pipelines, "SEARCH_RERANKER_ENABLED", True)
    monkeypatch.setattr(pipelines, "SEARCH_RERANKER_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(pipelines, "rerank_docs", rerank)

    ranked, diagnostics = await pipelines._rerank_search_candidates(
        "query",
        candidates,
        timeout_seconds=0.1,
    )
    assert [item["title"] for item in ranked] == ["Second", "First"]
    assert diagnostics["status"] == "applied"

    cancellation_seen = asyncio.Event()

    async def slow_rerank(_query, _docs, _top_k):
        try:
            await asyncio.Event().wait()
        finally:
            cancellation_seen.set()

    monkeypatch.setattr(pipelines, "SEARCH_RERANKER_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pipelines, "rerank_docs", slow_rerank)
    fallback, diagnostics = await pipelines._rerank_search_candidates(
        "query",
        candidates,
        timeout_seconds=0.02,
    )
    assert fallback == candidates
    assert diagnostics["status"] == "timed_out"
    assert cancellation_seen.is_set()


def test_relevance_gate_uses_reranker_only_when_it_has_a_strong_contrast():
    candidates = [
        {
            "title": "Android TV box benchmark",
            "snippet": "High-performance Nvidia Shield alternative",
            "url": "https://strong.example/android-tv",
            "domain": "strong.example",
            "rerank_score": 0.92,
        },
        {
            "title": "Android TV box product listing",
            "snippet": "Android streaming box sold as a Shield alternative",
            "url": "https://weak.example/android-tv",
            "domain": "weak.example",
            "rerank_score": 0.01,
        },
        {
            "title": "Android TV device performance review",
            "snippet": "Benchmarks for a powerful Nvidia Shield alternative",
            "url": "https://second.example/android-tv",
            "domain": "second.example",
            "rerank_score": 0.80,
        },
        {
            "title": "Android TV streaming box comparison",
            "snippet": "Current high-performance Shield alternatives",
            "url": "https://third.example/android-tv",
            "domain": "third.example",
            "rerank_score": 0.75,
        },
    ]
    query = "powerful Android TV box Nvidia Shield alternative"

    gated, diagnostics = pipelines._filter_relevant_search_candidates(
        candidates,
        query,
        reranking_status="applied",
    )
    deterministic, fallback_diagnostics = pipelines._filter_relevant_search_candidates(
        candidates,
        query,
        reranking_status="disabled",
    )
    multi_intent, multi_diagnostics = pipelines._filter_relevant_search_candidates(
        candidates,
        query,
        reranking_status="applied",
        allow_reranker_rejection=False,
    )

    assert [item["domain"] for item in gated] == [
        "strong.example",
        "second.example",
        "third.example",
    ]
    assert diagnostics["reranker_signal_used"] is True
    assert diagnostics["reranker_rejections"] == 1
    assert len(deterministic) == 4
    assert fallback_diagnostics["reranker_signal_used"] is False
    assert len(multi_intent) == 4
    assert multi_diagnostics["reranker_signal_used"] is False


def test_relevance_gate_does_not_reject_negligible_zero_dispersion_gap():
    candidates = [
        {
            "title": "Android TV box benchmark",
            "snippet": "Powerful Nvidia Shield alternative",
            "url": f"https://source{index}.example/android-tv",
            "domain": f"source{index}.example",
            "rerank_score": score,
        }
        for index, score in enumerate([1.0, 1.0, 1.0, 0.99])
    ]

    accepted, diagnostics = pipelines._filter_relevant_search_candidates(
        candidates,
        "powerful Android TV box Nvidia Shield alternative",
        reranking_status="applied",
    )

    assert len(accepted) == 4
    assert diagnostics["reranker_signal_used"] is False
    assert diagnostics["reranker_rejections"] == 0


def test_reranker_floor_ignores_deterministically_irrelevant_candidates():
    candidates = [
        {
            "title": f"Unrelated cider result {index}",
            "snippet": "Apple harvest and cider production",
            "url": f"https://noise{index}.example/result",
            "rerank_score": score,
        }
        for index, score in enumerate([0.99, 0.98, 0.97])
    ]
    candidates.append(
        {
            "title": "Android TV box benchmark",
            "snippet": "Nvidia Shield alternatives compared",
            "url": "https://relevant.example/android-tv",
            "rerank_score": 0.01,
        }
    )

    accepted, diagnostics = pipelines._filter_relevant_search_candidates(
        candidates,
        "powerful Android TV box Nvidia Shield alternative",
        reranking_status="applied",
    )

    assert [item["url"] for item in accepted] == ["https://relevant.example/android-tv"]
    assert diagnostics["reranker_signal_used"] is False


def test_relevance_gate_ignores_nonfinite_reranker_scores():
    candidates = [
        {
            "title": "Android TV box benchmark",
            "snippet": "Powerful Nvidia Shield alternative",
            "url": f"https://source{index}.example/android-tv",
            "domain": f"source{index}.example",
            "rerank_score": score,
        }
        for index, score in enumerate([0.9, 0.8, float("nan"), float("inf")])
    ]

    gated, diagnostics = pipelines._filter_relevant_search_candidates(
        candidates,
        "powerful Android TV box Nvidia Shield alternative",
        reranking_status="applied",
    )

    assert len(gated) == 4
    assert diagnostics["invalid_reranker_scores"] == 2
    assert all(
        not isinstance(item.get("rerank_score"), float)
        or math.isfinite(item["rerank_score"])
        for item in gated
    )


@pytest.mark.asyncio
async def test_interactive_deadline_returns_snippet_evidence_instead_of_hanging(
    monkeypatch,
):
    candidate = {
        "title": "Current release notes",
        "url": "https://vendor.example/releases/current",
        "domain": "vendor.example",
        "snippet": "Version 9 is the currently supported release.",
        "score": 10,
        "score_reasons": [],
    }

    async def slow_plan(_query, _mode):
        await asyncio.Event().wait()

    async def search(**_kwargs):
        return [candidate]

    crawl_cancelled = asyncio.Event()

    async def slow_crawl(_semaphore, _source, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            crawl_cancelled.set()

    monkeypatch.setitem(
        pipelines.RESEARCH_MODE_CONFIG,
        "balanced",
        {
            "max_urls": 1,
            "search_results": 1,
            "top_k": 0,
            "planner_budget": 0.01,
            "search_budget": 0.03,
            "crawl_budget": 0.03,
            "total_budget": 0.08,
        },
    )
    monkeypatch.setattr(pipelines, "build_research_plan", slow_plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_source_limited", slow_crawl)
    monkeypatch.setattr(pipelines, "CRAWL_CANCEL_GRACE_SECONDS", 0.005)

    started = time.monotonic()
    result = await pipelines.research_pipeline(
        "what is the current supported release?",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
        defer_persistence=True,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert crawl_cancelled.is_set()
    assert result["completion"]["status"] == "partial"
    assert result["evidence"][0]["evidence_type"] == "search_result_snippet"
    assert result["evidence"][0]["url"] == candidate["url"]


@pytest.mark.asyncio
async def test_successful_crawl_separates_stale_discovery_from_current_extraction(
    monkeypatch,
):
    current_context = {
        "retrieved_at_utc": "2026-07-13T12:00:00+00:00",
        "current_date_utc": "2026-07-13",
        "freshness": "runtime_retrieved",
    }
    candidate = {
        "title": "Cached discovery",
        "url": "https://example.com/current-page",
        "domain": "example.com",
        "retrieval_context": current_context,
        "retrieved_at_utc": "2026-07-13T08:00:00+00:00",
        "freshness": "stale_cache_unverified",
        "freshness_unverified": True,
        "freshness_status": "within_window",
        "search_cache_status": "stale_fallback",
        "search_cached_at_utc": "2026-07-13T08:00:00+00:00",
    }

    async def crawl(_url):
        return {
            "url": candidate["url"],
            "content": "Freshly fetched page content " * 100,
            "extraction_method": "direct",
        }

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)

    source = await pipelines.crawl_source(candidate, "current page")
    evidence = pipelines.build_crawled_source_evidence([source], [])

    assert source["ok"] is True
    assert source["retrieved_at_utc"] == current_context["retrieved_at_utc"]
    assert source["freshness"] == "runtime_retrieved"
    assert source["freshness_unverified"] is False
    assert source["search_cache_status"] == "stale_fallback"
    assert source["discovery_retrieved_at_utc"] == "2026-07-13T08:00:00+00:00"
    assert source["discovery_freshness"] == "stale_cache_unverified"
    assert source["discovery_freshness_unverified"] is True
    assert evidence[0]["freshness"] == "runtime_retrieved"
    assert evidence[0]["discovery_freshness_unverified"] is True


@pytest.mark.asyncio
async def test_failed_crawl_retains_stale_discovery_freshness(monkeypatch):
    candidate = {
        "title": "Cached discovery",
        "url": "https://example.com/unavailable",
        "domain": "example.com",
        "retrieval_context": {
            "retrieved_at_utc": "2026-07-13T12:00:00+00:00",
            "current_date_utc": "2026-07-13",
            "freshness": "runtime_retrieved",
        },
        "retrieved_at_utc": "2026-07-13T08:00:00+00:00",
        "freshness": "stale_cache_unverified",
        "freshness_unverified": True,
        "search_cache_status": "stale_fallback",
        "search_cached_at_utc": "2026-07-13T08:00:00+00:00",
    }

    async def crawl(_url):
        raise OSError("unavailable")

    monkeypatch.setattr(pipelines, "crawl_url_impl", crawl)

    source = await pipelines.crawl_source(candidate, "current page")

    assert source["ok"] is False
    assert source["retrieved_at_utc"] == "2026-07-13T08:00:00+00:00"
    assert source["freshness"] == "stale_cache_unverified"
    assert source["freshness_unverified"] is True
    assert source["discovery_freshness_unverified"] is True
