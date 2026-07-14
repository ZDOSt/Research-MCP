import asyncio
import json
import time

import fakeredis.aioredis as fakeredis
import httpx
import pytest

import searching


def _payload(count=4, *, unresponsive=None):
    return {
        "results": [
            {
                "title": f"Result {index}",
                "url": f"https://source{index}.example/article",
                "content": (
                    "Relevant current source material about a rate limit test, "
                    "Docker docs, stale fallback query, shared Redis cache, "
                    "concurrent scoped cache, shared circuit, partial cache "
                    "response, clean zero result, and fail open Redis circuit."
                ),
            }
            for index in range(count)
        ],
        "unresponsive_engines": unresponsive or [],
    }


class _Response:
    def __init__(self, payload, *, delay=0.0, tracker=None):
        self.payload = payload
        self.delay = delay
        self.tracker = tracker
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        if self.tracker is not None:
            self.tracker["active"] += 1
            self.tracker["maximum"] = max(
                self.tracker["maximum"], self.tracker["active"]
            )
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield json.dumps(self.payload).encode()
        finally:
            if self.tracker is not None:
                self.tracker["active"] -= 1


class _Client:
    def __init__(self, responder, captured, **_kwargs):
        self.responder = responder
        self.captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def stream(self, method, url, **kwargs):
        self.captured.append({"method": method, "url": url, **kwargs})
        return self.responder(kwargs["params"])


class _RateLimitedResponse:
    headers = {"retry-after": "90"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    def raise_for_status(self):
        request = httpx.Request("GET", "http://searxng:8080/search")
        response = httpx.Response(
            429,
            request=request,
            headers=self.headers,
        )
        raise httpx.HTTPStatusError(
            "rate limited",
            request=request,
            response=response,
        )


@pytest.fixture(autouse=True)
def _reset_search_runtime(monkeypatch):
    with searching._SEARCH_CACHE_LOCK:
        searching._SEARCH_CACHE.clear()
    with searching._ENGINE_HEALTH_LOCK:
        searching._ENGINE_HEALTH.clear()
    with searching._LOOP_LIMITERS_LOCK:
        searching._LOOP_LIMITERS.clear()
    monkeypatch.setattr(searching, "SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setattr(searching, "SEARCH_CACHE_REDIS_ENABLED", False)
    monkeypatch.setattr(searching, "_SEARCH_REDIS_URL", "")
    monkeypatch.setattr(searching, "_SEARCH_REDIS_CLIENT", None)
    monkeypatch.setattr(searching, "_SEARCH_REDIS_DISABLED_UNTIL", 0.0)
    yield
    with searching._SEARCH_CACHE_LOCK:
        searching._SEARCH_CACHE.clear()
    with searching._ENGINE_HEALTH_LOCK:
        searching._ENGINE_HEALTH.clear()


@pytest.mark.asyncio
async def test_stages_engines_without_category_union_and_opens_rate_limit_circuit(
    monkeypatch,
):
    captured = []
    responses = [
        _payload(
            0,
            unresponsive=[
                ["bing", "HTTP 429 rate limited; retry after 120"],
                ["qwant", "timeout"],
            ],
        ),
        _payload(4),
        _payload(4),
    ]

    def responder(_params):
        return _Response(responses.pop(0))

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    first = await searching.searxng_search("rate limit test one")
    second = await searching.searxng_search("rate limit test two")

    assert first
    assert second
    assert captured[0]["params"]["engines"] == "bing,qwant"
    assert captured[1]["params"]["engines"] == "startpage,mojeek"
    assert captured[2]["params"]["engines"] == "qwant"
    assert all("categories" not in request["params"] for request in captured)
    assert first.diagnostics["search_stages"][0]["status"] == "partial"
    skipped = second.diagnostics["search_stages"][0]["skipped_cooldowns"]
    assert skipped[0]["engine"] == "bing"
    assert skipped[0]["retry_after_seconds"] > 100


@pytest.mark.asyncio
async def test_normalized_cache_avoids_duplicate_search(monkeypatch):
    captured = []

    def responder(_params):
        return _Response(_payload())

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    first = await searching.searxng_search("  Docker   Docs  ")
    second = await searching.searxng_search("docker docs")

    assert first
    assert second
    assert len(captured) == 1
    assert second.diagnostics["cache"]["status"] == "fresh"


@pytest.mark.asyncio
async def test_http_retry_after_opens_primary_group_circuit(monkeypatch):
    captured = []
    responses = [_RateLimitedResponse()]

    def responder(_params):
        return responses.pop(0)

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    first = await searching.searxng_search("HTTP retry after one")
    second = await searching.searxng_search("HTTP retry after two")

    assert not first
    assert not second
    assert len(captured) == 1
    assert captured[0]["params"]["engines"] == "bing,qwant"
    assert second.diagnostics["search_stages"][0]["status"] == "service_circuit_open"
    assert second.diagnostics["search_stages"][0][
        "retry_after_seconds"
    ] > 80


@pytest.mark.asyncio
async def test_stale_cache_is_used_when_searxng_is_unavailable(monkeypatch):
    captured = []
    available = True

    def responder(params):
        if available:
            return _Response(_payload())
        request = httpx.Request("GET", "http://searxng:8080/search", params=params)
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr(searching, "SEARCH_CACHE_TTL_SECONDS", 1.0)
    monkeypatch.setattr(searching, "SEARCH_CACHE_STALE_TTL_SECONDS", 120.0)
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    first = await searching.searxng_search("stale fallback query")
    assert first
    with searching._SEARCH_CACHE_LOCK:
        for payload in searching._SEARCH_CACHE.values():
            payload["cached_at"] = time.time() - 10
    available = False

    second = await searching.searxng_search("stale fallback query")

    assert [item["url"] for item in second] == [item["url"] for item in first]
    assert all(item["freshness_unverified"] is True for item in second)
    assert all(item["freshness"] == "stale_cache_unverified" for item in second)
    assert second.diagnostics["cache"]["status"] == "stale_fallback"
    assert second.diagnostics["cache"]["freshness_unverified"] is True
    assert second.diagnostics["stale_fallback_attempt"]["search_stages"][0][
        "status"
    ] == "service_unavailable"


@pytest.mark.asyncio
async def test_cache_can_be_reused_from_redis_after_local_eviction(monkeypatch):
    captured = []
    redis_client = fakeredis.FakeRedis(decode_responses=True)

    def responder(_params):
        return _Response(_payload())

    monkeypatch.setattr(searching, "SEARCH_CACHE_REDIS_ENABLED", True)
    monkeypatch.setattr(searching, "_SEARCH_REDIS_URL", "redis://cache.invalid:6379/0")
    monkeypatch.setattr(searching, "_SEARCH_REDIS_CLIENT", redis_client)
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    await searching.searxng_search("shared redis cache")
    with searching._SEARCH_CACHE_LOCK:
        searching._SEARCH_CACHE.clear()
    cached = await searching.searxng_search("SHARED REDIS CACHE")

    assert cached
    assert len(captured) == 1
    assert cached.diagnostics["cache"]["status"] == "fresh"
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_overlapping_engine_groups_are_serialized(monkeypatch):
    captured = []
    tracker = {"active": 0, "maximum": 0}

    def responder(_params):
        return _Response(_payload(), delay=0.02, tracker=tracker)

    monkeypatch.setattr(searching, "SEARCH_MAX_CONCURRENT_REQUESTS", 4)
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    await asyncio.gather(
        searching.searxng_search("concurrent query one"),
        searching.searxng_search("concurrent query two"),
    )

    assert len(captured) == 2
    assert tracker["maximum"] == 1


@pytest.mark.asyncio
async def test_query_control_tokens_are_escaped_but_literal_important_is_preserved(
    monkeypatch,
):
    captured = []

    def responder(_params):
        return _Response(_payload())

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    await searching.searxng_search(
        "CSS !bing !general :fr <1 !!g !! !important",
    )

    assert captured[0]["params"]["q"] == (
        "CSS \\!bing \\!general \\:fr \\<1 \\!!g \\!! !important"
    )
    assert captured[0]["params"]["engines"] == "bing,qwant"
    assert "categories" not in captured[0]["params"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", [(), ("not-an-enabled-engine",)])
async def test_empty_or_invalid_engine_stage_never_reaches_searxng(
    monkeypatch,
    stage,
):
    captured = []

    def responder(_params):
        return _Response(_payload())

    monkeypatch.setitem(searching._ENGINE_STAGES, "general", (stage,))
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    results = await searching.searxng_search("invalid configured engine stage")

    assert not results
    assert captured == []
    assert results.diagnostics["acquisition_status"] == "failed"
    assert results.diagnostics["failure_class"] == "configuration"


@pytest.mark.asyncio
async def test_cache_is_partitioned_by_internal_scope(monkeypatch):
    captured = []

    def responder(_params):
        return _Response(_payload())

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    await searching.searxng_search("scoped cache", cache_scope="owner-a")
    await searching.searxng_search("scoped cache", cache_scope="owner-b")
    third = await searching.searxng_search("scoped cache", cache_scope="owner-a")

    assert len(captured) == 2
    assert third.diagnostics["cache"]["status"] == "fresh"


@pytest.mark.asyncio
async def test_provider_cooldown_is_shared_through_redis(monkeypatch):
    captured = []
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    responses = [
        _payload(4, unresponsive=[["bing", "captcha challenge"]]),
        _payload(4),
    ]

    def responder(_params):
        return _Response(responses.pop(0))

    monkeypatch.setattr(searching, "SEARCH_CACHE_REDIS_ENABLED", False)
    monkeypatch.setattr(searching, "SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED", True)
    monkeypatch.setattr(searching, "_SEARCH_REDIS_URL", "redis://circuit.invalid:6379/0")
    monkeypatch.setattr(searching, "_SEARCH_REDIS_CLIENT", redis_client)
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    await searching.searxng_search("shared circuit first")
    with searching._ENGINE_HEALTH_LOCK:
        searching._ENGINE_HEALTH.clear()
    second = await searching.searxng_search("shared circuit second")

    assert captured[0]["params"]["engines"] == "bing,qwant"
    assert captured[1]["params"]["engines"] == "qwant"
    skipped = second.diagnostics["search_stages"][0]["skipped_cooldowns"]
    assert skipped[0]["engine"] == "bing"
    assert skipped[0]["shared"] is True
    assert skipped[0]["retry_after_seconds"] > 800
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_clean_zero_result_search_does_not_resurrect_stale_cache(monkeypatch):
    captured = []
    payloads = [_payload(), _payload(0), _payload(0)]

    def responder(_params):
        return _Response(payloads.pop(0))

    monkeypatch.setattr(searching, "SEARCH_CACHE_TTL_SECONDS", 1.0)
    monkeypatch.setattr(searching, "SEARCH_CACHE_STALE_TTL_SECONDS", 120.0)
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    cached = await searching.searxng_search("clean zero result")
    assert cached
    with searching._SEARCH_CACHE_LOCK:
        for payload in searching._SEARCH_CACHE.values():
            payload["cached_at"] = time.time() - 10

    current = await searching.searxng_search("clean zero result")

    assert not current
    assert current.diagnostics["acquisition_status"] == "succeeded"
    assert current.diagnostics["cache"]["status"] == "miss"


@pytest.mark.asyncio
async def test_partial_engine_response_is_not_cached_as_fresh(monkeypatch):
    captured = []

    def responder(_params):
        return _Response(_payload(4, unresponsive=[["bing", "timeout"]]))

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    first = await searching.searxng_search("partial cache response")
    second = await searching.searxng_search("partial cache response")

    assert first.diagnostics["cache"]["status"] == "bypassed_partial"
    assert second.diagnostics["cache"]["status"] == "bypassed_partial"
    assert len(captured) == 2


@pytest.mark.asyncio
async def test_shared_cooldown_is_atomic_max_and_success_does_not_erase_it(monkeypatch):
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(searching, "SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED", True)
    monkeypatch.setattr(searching, "_SEARCH_REDIS_URL", "redis://circuit.invalid:6379/0")
    monkeypatch.setattr(searching, "_SEARCH_REDIS_CLIENT", redis_client)

    await searching._record_engine_failure("bing", "captcha challenge")
    member = searching.hashlib.sha256(b"bing").hexdigest()[:24]
    initial = await redis_client.zscore(searching._SEARCH_ENGINE_COOLDOWN_ZSET, member)
    await searching._record_engine_success("bing")
    assert searching._ENGINE_HEALTH["bing"].cooldown_until > time.monotonic()
    with searching._ENGINE_HEALTH_LOCK:
        searching._ENGINE_HEALTH.clear()
    await searching._record_engine_failure("bing", "timeout")
    after_shorter = await redis_client.zscore(
        searching._SEARCH_ENGINE_COOLDOWN_ZSET,
        member,
    )
    await searching._record_engine_success("bing")
    after_success = await redis_client.zscore(
        searching._SEARCH_ENGINE_COOLDOWN_ZSET,
        member,
    )

    assert initial is not None
    assert after_shorter == initial
    assert after_success == initial
    await redis_client.aclose()


@pytest.mark.asyncio
async def test_redis_circuit_timeout_fails_open(monkeypatch):
    captured = []

    class SlowPipeline:
        def zscore(self, *_args):
            return self

        async def execute(self):
            await asyncio.sleep(0.05)
            return []

    class SlowRedis:
        def pipeline(self, **_kwargs):
            return SlowPipeline()

    def responder(_params):
        return _Response(_payload())

    monkeypatch.setattr(searching, "SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED", True)
    monkeypatch.setattr(searching, "SEARCH_ENGINE_REDIS_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(searching, "_SEARCH_REDIS_URL", "redis://slow.invalid:6379/0")
    monkeypatch.setattr(searching, "_SEARCH_REDIS_CLIENT", SlowRedis())
    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    results = await searching.searxng_search("fail open redis circuit")

    assert results
    assert len(captured) == 1


def test_topical_relevance_rejects_cider_but_accepts_android_tv_results():
    query = "most powerful Android TV box Nvidia Shield alternative"
    cider = searching.search_result_relevance(
        {
            "title": "Most: Austrian cider and perry",
            "url": "https://cider.example/most",
            "snippet": "Traditional apple and pear cider called Most in German.",
        },
        query,
    )
    android = searching.search_result_relevance(
        {
            "title": "High-performance Android TV boxes compared",
            "url": "https://tech.example/android-tv-boxes",
            "snippet": "Benchmarks of current Nvidia Shield alternatives.",
        },
        query,
    )

    assert cider["is_relevant"] is False
    assert cider["matched_terms"] == []
    assert android["is_relevant"] is True
    assert {"android", "tv", "box", "nvidia", "shield", "alternative"}.issubset(
        set(android["matched_terms"])
    )


def test_topical_relevance_normalizes_common_research_intent_synonyms():
    installation = searching.search_result_relevance(
        {
            "title": "Current product guide",
            "url": "https://docs.example/product/guide",
            "snippet": "A maintained product manual.",
        },
        "install the product",
    )
    comparison = searching.search_result_relevance(
        {
            "title": "Independent device benchmark",
            "url": "https://reviews.example/device-benchmark",
            "snippet": "Performance review of the device.",
        },
        "compare the device",
    )
    generic_installation = searching.search_result_relevance(
        {
            "title": "Installation guide",
            "url": "https://unrelated.example/installation-guide",
            "snippet": "A general setup manual for unrelated software.",
        },
        "install the product",
    )

    assert installation["is_relevant"] is True
    assert "install" in installation["matched_terms"]
    assert comparison["is_relevant"] is True
    assert "compare" in comparison["matched_terms"]
    assert generic_installation["is_relevant"] is False
    assert generic_installation["reason"] == "generic_only_topic_overlap"


@pytest.mark.parametrize(
    ("query", "drifted_title", "requested", "substituted"),
    [
        ("vegan dinner recipes", "chicken dinner recipes", "vegan", "chicken"),
        ("wireless headphones", "wired headphones", "wireless", "wired"),
        (
            "free project management software",
            "paid project management software",
            "free",
            "paid",
        ),
        ("indoor security cameras", "outdoor security cameras", "indoor", "outdoor"),
        ("beginner Python tutorials", "advanced Python tutorials", "beginner", "advanced"),
        ("cat food recommendations", "dog food recommendations", "cat", "dog"),
        ("Android TV boxes", "Android TV remote apps", "box", "app"),
    ],
)
def test_topical_relevance_rejects_explicit_qualifier_substitution(
    query,
    drifted_title,
    requested,
    substituted,
):
    analysis = searching.search_result_relevance(
        {
            "title": drifted_title,
            "url": "https://drift.example/result",
        },
        query,
        threshold=0.42,
    )

    assert analysis["is_relevant"] is False
    assert analysis["reason"] == "conflicting_topic_qualifier"
    assert any(
        requested in conflict["requested"]
        and substituted in conflict["substituted"]
        for conflict in analysis["topic_conflicts"]
    )


@pytest.mark.parametrize(
    ("query", "reformulated_title"),
    [
        ("vegan dinner recipes", "plant-based dinner recipes"),
        ("wireless headphones", "cordless Bluetooth headphones"),
        ("free project management software", "no-cost project management software"),
        ("beginner Python tutorials", "introductory Python guides"),
        ("cat food recommendations", "feline food recommendations"),
        ("Android TV boxes", "Android TV streaming devices"),
        (
            "wireless versus wired headphones",
            "wired headphones compared with wireless models",
        ),
    ],
)
def test_topical_relevance_preserves_equivalent_or_comparative_reformulations(
    query,
    reformulated_title,
):
    analysis = searching.search_result_relevance(
        {
            "title": reformulated_title,
            "url": "https://reformulation.example/result",
        },
        query,
        threshold=0.42,
    )

    assert analysis["is_relevant"] is True
    assert analysis["reason"] == "relevant_topic_overlap"


def test_topical_relevance_splits_mixed_japanese_and_latin_scripts():
    analysis = searching.search_result_relevance(
        {
            "title": "Ubuntu に Docker Engine をインストールする",
            "snippet": "Docker Engine の公式インストール手順",
            "url": "https://docs.docker.com/engine/install/ubuntu/",
        },
        "DockerをUbuntuにインストールする方法",
    )

    assert analysis["is_relevant"] is True
    assert {"docker", "ubuntu", "インストール"}.issubset(analysis["matched_terms"])


def test_topical_relevance_matches_reordered_chinese_character_runs():
    analysis = searching.search_result_relevance(
        {
            "title": "\u5bb6\u5ead\u81ea\u52a8\u5316\u7cfb\u7edf\u5b89\u88c5\u6307\u5357",
            "snippet": "\u914d\u7f6e\u548c\u5b89\u5168\u5efa\u8bae",
            "url": "https://docs.example/home-automation",
        },
        "\u5982\u4f55\u5b89\u88c5\u5bb6\u5ead\u81ea\u52a8\u5316\u7cfb\u7edf",
    )

    assert analysis["is_relevant"] is True
    assert analysis["reason"] == "relevant_cjk_bigram_overlap"
    assert analysis["cjk_overlap"]["matched_bigram_count"] >= 2
    assert analysis["cjk_overlap"]["coverage"] >= 0.30


def test_topical_relevance_matches_reordered_kanji_character_runs():
    analysis = searching.search_result_relevance(
        {
            "title": "\u5c0e\u5165\u65b9\u6cd5: \u5bb6\u5ead\u5411\u3051\u81ea\u52d5\u5316",
            "snippet": "\u73fe\u5728\u306e\u8a2d\u5b9a\u624b\u9806",
            "url": "https://docs.example/home-automation-ja",
        },
        "\u5bb6\u5ead\u81ea\u52d5\u5316\u5c0e\u5165\u65b9\u6cd5",
    )

    assert analysis["is_relevant"] is True
    assert analysis["reason"] == "relevant_cjk_bigram_overlap"
    assert analysis["cjk_overlap"]["coverage"] >= 0.30


def test_topical_relevance_matches_korean_spacing_variants():
    analysis = searching.search_result_relevance(
        {
            "title": "\ucd5c\uc2e0\ub274\uc2a4\uc778\uacf5\uc9c0\ub2a5\ub3d9\ud5a5",
            "snippet": "AI \ubaa8\ub378\uacfc \uc0b0\uc5c5 \ub3d9\ud5a5",
            "url": "https://news.example/ai",
        },
        "\uc778\uacf5\uc9c0\ub2a5\ub274\uc2a4\uc54c\ub824\uc918",
    )

    assert analysis["is_relevant"] is True
    assert analysis["reason"] == "relevant_cjk_bigram_overlap"
    assert analysis["cjk_overlap"]["coverage"] >= 0.30


def test_cjk_bigram_fallback_rejects_weak_generic_overlap():
    analysis = searching.search_result_relevance(
        {
            "title": "\u82f9\u679c\u6c34\u679c\u8425\u517b\u6307\u5357",
            "snippet": "\u82f9\u679c\u7684\u8425\u517b\u4ef7\u503c",
            "url": "https://food.example/apple",
        },
        "\u82f9\u679c\u624b\u673a\u8bc4\u6d4b",
    )

    assert analysis["is_relevant"] is False
    assert analysis["reason"] == "insufficient_topic_overlap"
    assert analysis["cjk_overlap"]["matched_bigram_count"] == 1
    assert analysis["cjk_overlap"]["used"] is False


def test_generic_publication_overlap_does_not_count_as_topic_relevance():
    analysis = searching.search_result_relevance(
        {
            "title": "University calendar published today",
            "url": "https://university.example/calendar",
            "snippet": "The updated academic calendar was published this morning.",
        },
        "AI news published today",
    )

    assert analysis["matched_terms"] == ["published"]
    assert analysis["distinctive_query_terms"] == ["ai"]
    assert analysis["matched_distinctive_terms"] == []
    assert analysis["is_relevant"] is False
    assert analysis["reason"] == "generic_only_topic_overlap"


def test_relevant_results_are_not_truncated_behind_high_ranked_noise():
    results = searching.compact_search_results(
        {
            "results": [
                {
                    "title": "Traditional German cider festival",
                    "url": "https://cider.example/most",
                    "content": "Apple varieties and regional Most tasting events.",
                    "score": 100,
                },
                {
                    "title": "Android TV box benchmarks",
                    "url": "https://benchmarks.example/android-tv",
                    "content": "Nvidia Shield alternatives compared by streaming performance.",
                    "score": 1,
                },
            ]
        },
        "powerful Android TV box Nvidia Shield alternative",
        max_results=1,
    )

    assert [item["domain"] for item in results] == ["benchmarks.example"]
    assert results[0]["topical_relevance"]["is_relevant"] is True


@pytest.mark.asyncio
async def test_irrelevant_owner_diversity_does_not_stop_later_engine_stage(
    monkeypatch,
):
    captured = []
    payloads = [
        {
            "results": [
                {
                    "title": f"Most cider producer {index}",
                    "url": f"https://cider{index}.example/most",
                    "content": "Austrian apple cider and perry production.",
                }
                for index in range(4)
            ]
        },
        {
            "results": [
                {
                    "title": "Best Android TV boxes",
                    "url": "https://android.example/tv-boxes",
                    "content": "Current Nvidia Shield alternative models.",
                },
                {
                    "title": "Android TV box benchmark comparison",
                    "url": "https://bench.example/android-tv",
                    "content": "Performance results for streaming boxes.",
                },
                {
                    "title": "High performance Google TV boxes",
                    "url": "https://streaming.example/google-tv-box",
                    "content": "Android streaming hardware compared.",
                },
                {
                    "title": "Nvidia Shield alternatives",
                    "url": "https://media.example/shield-alternatives",
                    "content": "Android TV replacement boxes reviewed.",
                },
            ]
        },
    ]

    def responder(_params):
        return _Response(payloads.pop(0))

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    results = await searching.searxng_search(
        "most powerful Android TV box Nvidia Shield alternative"
    )

    assert len(captured) == 2
    stages = results.diagnostics["search_stages"]
    assert stages[0]["coverage_sufficient"] is False
    assert stages[0]["topical_relevance"]["relevant_count"] == 0
    assert stages[1]["coverage_sufficient"] is True
    assert stages[1]["topical_relevance"]["relevant_count"] >= 3


@pytest.mark.asyncio
async def test_generic_only_overlap_does_not_stop_later_engine_stage(monkeypatch):
    captured = []
    payloads = [
        {
            "results": [
                {
                    "title": f"University notice published today {index}",
                    "url": f"https://campus{index}.example/notices",
                    "content": "An academic calendar update published today.",
                }
                for index in range(4)
            ]
        },
        {
            "results": [
                {
                    "title": f"AI news report {index}",
                    "url": f"https://ainews{index}.example/report",
                    "content": "Artificial intelligence industry news published today.",
                }
                for index in range(4)
            ]
        },
    ]

    def responder(_params):
        return _Response(payloads.pop(0))

    monkeypatch.setattr(
        searching.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(responder, captured, **kwargs),
    )

    results = await searching.searxng_search("AI news published today")

    assert len(captured) == 2
    stages = results.diagnostics["search_stages"]
    assert stages[0]["coverage_sufficient"] is False
    assert stages[0]["topical_relevance"]["relevant_count"] == 0
    assert stages[1]["coverage_sufficient"] is True
    assert stages[1]["topical_relevance"]["relevant_count"] >= 3
