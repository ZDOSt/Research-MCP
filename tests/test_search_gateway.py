import asyncio
import math

import httpx
import pytest

import gateway_fetch
import search_gateway


@pytest.fixture(autouse=True)
def reset_adaptive_state(monkeypatch):
    search_gateway._ENGINE_HEALTH.clear()
    search_gateway._SUPPLEMENT_COOLDOWNS.clear()
    gateway_fetch._DOMAIN_FETCH_STATS.clear()
    monkeypatch.setattr(search_gateway, "ENABLE_KEYLESS_SUPPLEMENTS", False)
    monkeypatch.setattr(gateway_fetch, "REDIS_URL", "")


@pytest.mark.asyncio
async def test_fetch_resource_cleanup_closes_existing_client_without_recreating_it():
    class Client:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    client = Client()
    gateway_fetch._FETCH_REDIS = client
    gateway_fetch._FETCH_REDIS_FAILED_UNTIL = 123.0

    await gateway_fetch.close_fetch_resources()

    assert client.closed is True
    assert gateway_fetch._FETCH_REDIS is None
    assert gateway_fetch._FETCH_REDIS_FAILED_UNTIL == 0.0


def test_query_variants_are_bounded_and_intent_aware():
    variants = search_gateway._query_variants(
        "How do I install Docker Compose on Ubuntu?", "balanced"
    )
    assert variants[0] == "How do I install Docker Compose on Ubuntu?"
    assert any('"Docker Compose"' in item for item in variants[1:])
    assert any("official documentation" in item for item in variants[1:])
    assert all("site:" not in item for item in variants)
    assert len(variants) <= 2


def test_generic_technical_query_uses_general_documentation_variant():
    variants = search_gateway._query_variants(
        "How do I install PostgreSQL on Ubuntu?", "balanced"
    )
    assert variants[0] == "How do I install PostgreSQL on Ubuntu?"
    assert any("PostgreSQL" in item for item in variants[1:])
    assert any("official documentation" in item for item in variants[1:])
    assert all("site:" not in item for item in variants)


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
    assert any("official documentation" in item for item in variants)


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
    assert search_gateway._search_categories("What is version control?", []) == [
        "general"
    ]


def test_search_engines_are_bounded_by_intent_and_exclude_repository_search():
    technical = search_gateway._search_engines(
        "How do I install Docker Compose on Ubuntu?",
        [],
        "install docker compose ubuntu",
    )
    targeted = search_gateway._search_engines(
        "How do I install Docker Compose on Ubuntu?",
        [],
        "install docker compose ubuntu site:docs.docker.com",
    )
    assert technical == [
        "bing",
        "brave",
        "startpage",
        "askubuntu",
    ]
    assert targeted == ["bing", "brave", "startpage"]
    assert "github" not in technical


def test_specialized_engines_require_matching_subject_intent():
    web = search_gateway._search_engines(
        "How does the browser InstallEvent Web API work?",
        [],
        "browser installevent web api",
    )
    programming = search_gateway._search_engines(
        "Python stack trace error in an async function", [], "python async error"
    )
    assert "mdn" in web
    assert "stackoverflow" in programming
    assert "mdn" not in programming


def test_zero_subject_match_is_strongly_penalized_without_being_hard_filtered():
    unrelated = search_gateway._normalize_search_result(
        {
            "title": "InstallEvent: InstallEvent() constructor",
            "url": "https://developer.mozilla.org/en-US/docs/Web/API/InstallEvent/InstallEvent",
            "content": "The InstallEvent constructor creates an event.",
            "engine": "mdn",
        },
        "How do I install Docker Compose on Ubuntu?",
        1,
    )
    relevant = search_gateway._normalize_search_result(
        {
            "title": "Install Docker Compose on Ubuntu",
            "url": "https://docs.example.com/docker-compose-ubuntu",
            "content": "Docker Compose plugin installation instructions for Ubuntu.",
            "engine": "bing",
        },
        "How do I install Docker Compose on Ubuntu?",
        2,
    )
    assert unrelated is not None
    assert relevant is not None
    assert unrelated["subject_coverage"] == 0
    assert relevant["discovery_score"] > unrelated["discovery_score"] + 5


def test_subject_matching_does_not_treat_partial_words_as_coverage():
    result = search_gateway._normalize_search_result(
        {
            "title": "Trusted package installation guide",
            "url": "https://developer.example.com/trusted-packages",
            "content": "Install a package from a trusted source.",
            "engine": "startpage",
        },
        "How do I install Rust?",
        1,
    )
    assert result is not None
    assert result["subject_coverage"] == 0


def test_recommendation_queries_preserve_the_models_subject_and_intent():
    variants = search_gateway._query_variants(
        "Best team composition in DragonSword Awakening", "balanced"
    )
    assert any("DragonSword" in item for item in variants)
    assert all("measurements" not in item for item in variants)

    equipment_variants = search_gateway._query_variants(
        "DragonSword Awakening best equipment Theresia Astria Roxy", "balanced"
    )
    assert all("measurements" not in item for item in equipment_variants)

    product_variants = search_gateway._query_variants(
        "Best Android TV box based on measurements", "balanced"
    )
    assert any("android" in item.casefold() for item in product_variants)
    assert all("strategy guide wiki" not in item for item in product_variants)


@pytest.mark.parametrize(
    ("query", "anchor"),
    [
        ("DragonSword Awakening best equipment", "DragonSword Awakening"),
        ("What are the best Alienware AW3426DW settings?", "Alienware AW3426DW"),
        ("Who is Taylor Swift?", "Taylor Swift"),
        ("Best beginner build for Path of Exile 2", "Path of Exile 2"),
        ("Compare Redis and Valkey licensing", "Redis and Valkey"),
        (
            'Fix the error "connection refused by upstream"',
            "connection refused by upstream",
        ),
    ],
)
def test_topic_anchor_is_domain_agnostic(query, anchor):
    assert search_gateway._topic_anchor(query) == anchor


@pytest.mark.parametrize(
    "query",
    [
        "Recent academic survey of retrieval augmented generation evaluation",
        "What is the recommended way to create a Python virtual environment?",
        "Important artificial intelligence regulation news today",
    ],
)
def test_broad_queries_are_not_forced_into_an_entity_anchor(query):
    assert search_gateway._topic_anchor(query) is None


def test_precise_queries_keep_an_exact_and_relaxed_form():
    inferred = search_gateway._query_variants(
        "DragonSword Awakening best equipment characters guide", "balanced"
    )
    explicit = search_gateway._query_variants(
        '"DragonSword Awakening" Theresia Astria Roxy', "balanced"
    )

    assert inferred[0].startswith("DragonSword Awakening")
    assert any(item.startswith('"DragonSword Awakening"') for item in inferred[1:])
    assert explicit[0].startswith('"DragonSword Awakening"')
    assert all(len(item) > len('"DragonSword Awakening"') for item in explicit)
    assert (
        search_gateway._topic_anchor('"DragonSword Awakening" Theresia Astria Roxy')
        == "DragonSword Awakening"
    )
    assert (
        search_gateway._topic_anchor("Persona 5 best team composition") == "Persona 5"
    )
    assert search_gateway._topic_anchor("Squid Game character build") == "Squid Game"


def test_topic_anchor_prefers_the_subject_over_later_feature_terms():
    assert (
        search_gateway._topic_anchor(
            "Best Android TV boxes with AV1, Dolby Vision, and gigabit Ethernet"
        )
        == "Android TV"
    )
    assert (
        search_gateway._topic_anchor("Path of Exile 2 best build") == "Path of Exile 2"
    )


@pytest.mark.parametrize(
    "query",
    [
        "Best 4K TVs for gaming",
        "Best HDR monitors",
        "Best USB microphones",
        "Best Wi-Fi routers",
        "Best Android TV boxes",
    ],
)
def test_generic_feature_anchors_are_soft_not_hard_filters(query):
    anchor = search_gateway._topic_anchor(query)
    assert anchor is not None
    assert not search_gateway._topic_anchor_is_strict(query, anchor)


def test_compact_anchor_matching_does_not_match_a_longer_numeric_version():
    assert search_gateway._anchor_matches(
        "Path of Exile 2", "Path-of-Exile-2 beginner build"
    )
    assert not search_gateway._anchor_matches(
        "Path of Exile 2", "Path of Exile 20 beginner build"
    )


def test_soft_feature_anchor_allows_a_natural_language_paraphrase():
    result = search_gateway._normalize_search_result(
        {
            "title": "Best televisions for gaming with high dynamic range",
            "url": "https://display.example/best-gaming-televisions",
            "content": "A comparison of 4K televisions and HDR support.",
            "engine": "bing",
        },
        "Best 4K TVs for gaming",
        1,
    )
    assert result is not None
    assert result["topic_strict"] is False


def test_entity_relevance_penalty_keeps_generic_word_matches_below_real_results():
    query = "DragonSword Awakening best equipment characters guide"
    unrelated = search_gateway._normalize_search_result(
        {
            "title": "Dragon - mythological creature",
            "url": "https://www.britannica.com/topic/dragon-mythological-creature",
            "content": "A dragon is a legendary character in folklore.",
            "engine": "bing",
        },
        query,
        1,
    )
    relevant = search_gateway._normalize_search_result(
        {
            "title": "Dragon Sword Awakening equipment guide",
            "url": "https://example.com/dragon-sword-awakening-guide",
            "content": "Recommended equipment for the playable characters.",
            "engine": "bing",
        },
        query,
        2,
    )

    assert unrelated is not None
    assert relevant is not None
    assert relevant["discovery_score"] > unrelated["discovery_score"] + 5
    assert relevant["topic_anchor"] == "DragonSword Awakening"
    assert relevant["topic_match"] is True


def test_spaced_entity_marks_a_single_ambiguous_word_match_as_partial_only():
    result = search_gateway._normalize_search_result(
        {
            "title": "Dragon - mythological creature",
            "url": "https://www.britannica.com/topic/dragon-mythological-creature",
            "content": "A dragon is a legendary character in folklore.",
            "engine": "bing",
        },
        "Dragon Sword Awakening game Astria",
        1,
    )
    assert result is not None
    assert result["topic_match"] is False
    assert result["topic_partial_match"] is False


def test_product_identifier_can_match_without_repeating_the_brand():
    result = search_gateway._normalize_search_result(
        {
            "title": "AW3426DW recommended HDR settings",
            "url": "https://display.example/AW3426DW/settings",
            "content": "Measured SDR and HDR configuration values.",
            "engine": "bing",
        },
        "Best Alienware AW3426DW settings",
        1,
    )
    assert result is not None
    assert result["topic_strict"] is True


def test_comparison_queries_keep_sources_covering_one_side():
    result = search_gateway._normalize_search_result(
        {
            "title": "Redis licensing and compatibility",
            "url": "https://redis.io/legal/licenses/",
            "content": "Redis licensing and migration details.",
            "engine": "bing",
        },
        "Compare Redis and Valkey licensing and migration",
        1,
    )
    assert result is not None
    assert result["topic_match"] is False
    assert result["topic_term_matches"] == 1


def test_application_compose_search_does_not_target_docker_documentation():
    variants = search_gateway._query_variants(
        "Find the best docker-compose file for Immich", "balanced"
    )
    assert all("site:docs.docker.com" not in item for item in variants)
    assert any("immich" in item.casefold() for item in variants)


def test_current_queries_get_a_freshness_variant_before_generic_documentation():
    variants = search_gateway._query_variants(
        "What is the current version of Docker?", "balanced"
    )
    assert any(item.endswith(" latest") for item in variants)
    assert all("official documentation" not in item for item in variants)


def test_variant_count_is_bounded_by_mode():
    query = "Best current settings for Alienware AW3426DW"
    assert len(search_gateway._query_variants(query, "balanced")) <= 2
    assert len(search_gateway._query_variants(query, "deep")) <= 3


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
    assert search_gateway._candidate_score(
        query, official, 2
    ) > search_gateway._candidate_score(query, weak, 1)


def test_crawl_selection_reserves_instructional_slot_and_diversifies_domains():
    candidates = [
        {
            "title": "Community answer one",
            "url": "https://stackoverflow.com/questions/1/one",
            "domain": "stackoverflow.com",
            "source_type": "technical_reference",
        },
        {
            "title": "Community answer two",
            "url": "https://stackoverflow.com/questions/2/two",
            "domain": "stackoverflow.com",
            "source_type": "technical_reference",
        },
        {
            "title": "Install Docker Compose plugin",
            "url": "https://docs.docker.com/compose/install/linux/",
            "domain": "docs.docker.com",
            "source_type": "documentation_candidate",
            "primary_source_candidate": True,
        },
        {
            "title": "Ubuntu package overview",
            "url": "https://packages.ubuntu.com/docker-compose-v2",
            "domain": "packages.ubuntu.com",
            "source_type": "general_web",
        },
    ]
    ranked = [
        {"candidate_index": 0},
        {"candidate_index": 1},
        {"candidate_index": 3},
        {"candidate_index": 2},
    ]

    selected = search_gateway._select_crawl_candidates(
        "How do I install Docker Compose on Ubuntu?",
        candidates,
        ranked,
        target_pages=3,
        limit=4,
    )

    assert selected[0]["domain"] == "docs.docker.com"
    assert [item["domain"] for item in selected[:3]] == [
        "docs.docker.com",
        "packages.ubuntu.com",
        "stackoverflow.com",
    ]


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


def test_instruction_query_penalizes_source_repository_without_repo_intent():
    repository = {
        "title": "Terraform GCP Ubuntu container ready VM",
        "url": "https://github.com/example/terraform-gcp-ubuntu-container-ready",
        "content": "Ubuntu with Docker and Docker Compose installed using Terraform",
    }
    install_query = "How do I install Docker Compose on Ubuntu?"
    repository_query = "Find the GitHub repository for Docker Compose"
    install_profile = search_gateway.source_profile(
        repository["url"],
        title=repository["title"],
        snippet=repository["content"],
        query=install_query,
    )
    repository_profile = search_gateway.source_profile(
        repository["url"],
        title=repository["title"],
        snippet=repository["content"],
        query=repository_query,
    )
    assert (
        search_gateway._intent_source_adjustment(install_query, install_profile)
        == -1.25
    )
    assert (
        search_gateway._intent_source_adjustment(repository_query, repository_profile)
        == 0.0
    )


@pytest.mark.asyncio
async def test_searx_uses_original_query_then_a_bounded_fallback_wave(monkeypatch):
    captured = []

    class Stream:
        def __init__(self, params):
            self.params = params

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params):
            captured.append(params)
            return Stream(params)

    async def read(response, max_bytes=0):
        query = response.params["q"]
        if query != "How do I install Docker Compose on Ubuntu?":
            return {
                "results": [
                    {
                        "title": "Install Docker Engine on Ubuntu",
                        "url": "https://docs.docker.com/engine/install/ubuntu/",
                        "content": "Official installation steps including Compose plugin",
                        "engine": "bing",
                    }
                ]
            }
        return {
            "results": [
                {
                    "title": f"Unrelated result {index}",
                    "url": f"https://github.com/example/unrelated-{index}",
                    "content": "Ubuntu Docker Compose container",
                    "engine": "github",
                }
                for index in range(1, 6)
            ]
        }

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)
    results, diagnostics = await search_gateway._searx_search(
        "How do I install Docker Compose on Ubuntu?",
        mode="balanced",
        max_results=10,
        language="auto",
        time_range=None,
        categories=[],
    )

    official = next(item for item in results if item["domain"] == "docs.docker.com")
    assert results[0] == official
    assert official["search_rank"] == 1
    assert captured[0]["q"] == "How do I install Docker Compose on Ubuntu?"
    assert captured[0]["engines"] == "bing,brave"
    targeted = captured[1]
    assert '"Docker Compose"' in targeted["q"]
    assert "github" not in targeted["engines"]
    assert all("categories" not in params for params in captured)
    summary = diagnostics[-1]
    assert summary["fallback_triggered"] is True
    assert summary["initial_quality"]["status"] == "weak"


@pytest.mark.asyncio
async def test_failed_search_wave_updates_health_before_fallback(monkeypatch):
    captured = []

    class Stream:
        def __init__(self, params, fail):
            self.params = params
            self.fail = fail

        async def __aenter__(self):
            if self.fail:
                raise httpx.ReadTimeout("initial wave timed out")
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params):
            captured.append(params)
            return Stream(params, len(captured) == 1)

    async def read(response, max_bytes=0):
        return {
            "results": [
                {
                    "title": "Nebula XZ900 setup guide",
                    "url": "https://docs.example.com/nebula-xz900",
                    "content": "Nebula XZ900 installation and configuration guide.",
                    "engine": "startpage",
                }
            ]
        }

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)

    results, diagnostics = await search_gateway._searx_search(
        "Nebula XZ900 setup guide",
        mode="balanced",
        max_results=3,
        language="auto",
        time_range=None,
        categories=[],
    )

    assert results[0]["domain"] == "docs.example.com"
    assert captured[0]["engines"] == "bing,brave"
    assert captured[1]["engines"].split(",")[0] == "startpage"
    failed = diagnostics[0]
    assert failed["status"] == "failed"
    assert failed["error"] == "ReadTimeout"
    assert all(item["failures"] == 1 for item in failed["engine_health"])


@pytest.mark.asyncio
async def test_supplement_failure_does_not_fail_the_core_search(monkeypatch):
    class Stream:
        def __init__(self, params):
            self.params = params

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params):
            return Stream(params)

    async def read(response, max_bytes=0):
        if response.params["q"] == "Nebula XZ900 setup guide":
            return {"results": []}
        return {
            "results": [
                {
                    "title": f"Nebula XZ900 setup guide {index}",
                    "url": f"https://guide{index}.example/xz900",
                    "content": "Nebula XZ900 installation and configuration details.",
                    "engine": "bing",
                }
                for index in range(1, 4)
            ]
        }

    async def failed_supplement(query, time_range, categories=None):
        raise ValueError("unexpected provider payload")

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)
    monkeypatch.setattr(search_gateway, "_supplemental_search", failed_supplement)

    results, diagnostics = await search_gateway._searx_search(
        "Nebula XZ900 setup guide",
        mode="balanced",
        max_results=3,
        language="auto",
        time_range=None,
        categories=[],
    )

    assert len(results) == 3
    assert any(
        item.get("provider") == "supplemental"
        and item.get("status") == "failed"
        and item.get("error") == "ValueError"
        for item in diagnostics
    )


@pytest.mark.asyncio
async def test_discovery_softly_penalizes_topic_drift_across_search_waves(monkeypatch):
    captured = []

    class Stream:
        def __init__(self, params):
            self.params = params

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params):
            captured.append(params["q"])
            return Stream(params)

    async def read(response, max_bytes=0):
        return {
            "results": [
                {
                    "title": "Dragon - mythological creature",
                    "url": "https://www.britannica.com/topic/dragon-mythological-creature",
                    "content": "A dragon is a legendary character in folklore.",
                    "engine": "bing",
                },
                {
                    "title": "Dragon Sword Awakening equipment guide",
                    "url": "https://guide.example/dragon-sword-awakening-equipment",
                    "content": "Character equipment recommendations and builds.",
                    "engine": "bing",
                },
            ]
        }

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)

    results, _ = await search_gateway._searx_search(
        "DragonSword Awakening best equipment characters guide",
        mode="balanced",
        max_results=10,
        language="auto",
        time_range=None,
        categories=[],
    )

    assert captured == [
        "DragonSword Awakening best equipment characters guide",
        '"DragonSword Awakening" best equipment characters guide',
    ]
    assert results[0]["domain"] == "guide.example"
    unrelated = next(item for item in results if item["domain"] == "britannica.com")
    assert results[0]["discovery_score"] > unrelated["discovery_score"]


@pytest.mark.asyncio
async def test_strong_initial_wave_skips_fallback(monkeypatch):
    captured = []

    class Stream:
        def __init__(self, params):
            self.params = params

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params):
            captured.append(params)
            return Stream(params)

    async def read(response, max_bytes=0):
        return {
            "results": [
                {
                    "title": f"Orchid Falcon XZ900 setup guide {index}",
                    "url": f"https://source{index}.example/guide",
                    "content": "Orchid Falcon XZ900 configuration and setup instructions.",
                    "engine": "bing",
                }
                for index in range(1, 4)
            ]
        }

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)
    results, diagnostics = await search_gateway._searx_search(
        "Orchid Falcon XZ900 setup guide",
        mode="balanced",
        max_results=3,
        language="auto",
        time_range=None,
        categories=[],
    )

    assert len(results) == 3
    assert len(captured) == 1
    assert diagnostics[-1]["fallback_triggered"] is False


@pytest.mark.asyncio
async def test_rate_limited_engine_enters_cooldown_before_fallback(monkeypatch):
    captured = []

    class Stream:
        def __init__(self, params):
            self.params = params

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params):
            captured.append(params)
            return Stream(params)

    async def read(response, max_bytes=0):
        if len(captured) == 1:
            return {
                "results": [
                    {
                        "title": "Unrelated page",
                        "url": "https://unrelated.example/",
                        "content": "No product information.",
                        "engine": "bing",
                    }
                ],
                "unresponsive_engines": [["brave", "too many requests"]],
            }
        return {
            "results": [
                {
                    "title": f"Nebula XZ900 setup guide {index}",
                    "url": f"https://guide{index}.example/xz900",
                    "content": "Nebula XZ900 installation and configuration details.",
                    "engine": "bing",
                }
                for index in range(1, 4)
            ]
        }

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)
    _, diagnostics = await search_gateway._searx_search(
        "Nebula XZ900 setup guide",
        mode="balanced",
        max_results=3,
        language="auto",
        time_range=None,
        categories=[],
    )

    assert "brave" in captured[0]["engines"].split(",")
    assert "brave" not in captured[1]["engines"].split(",")
    assert (
        search_gateway._engine_health_snapshot("brave")["cooldown_remaining_seconds"]
        > 0
    )
    assert diagnostics[-1]["fallback_triggered"] is True


def test_rrf_fusion_rewards_independent_query_and_engine_agreement():
    shared = {
        "title": "Shared result",
        "url": "https://example.com/shared",
        "discovery_score": 3.0,
        "search_rank": 1,
        "engines": ["bing"],
        "query_variant": "original query",
        "retrieval_rank": 1,
        "retrieval_weight": 1.0,
        "retrieval_source": "searxng",
    }
    occurrences = [
        shared,
        {
            **shared,
            "engines": ["startpage"],
            "query_variant": "relaxed query",
            "retrieval_rank": 2,
        },
        {
            **shared,
            "title": "Single result",
            "url": "https://other.example/single",
            "retrieval_rank": 1,
        },
    ]

    fused = search_gateway._fuse_candidates(occurrences)
    assert fused[0]["url"] == "https://example.com/shared"
    assert fused[0]["query_consensus"] == 2
    assert fused[0]["engine_consensus"] == 2


def test_fusion_preserves_prefetched_api_evidence_for_duplicate_urls():
    shared = {
        "title": "Shared question",
        "url": "https://stackoverflow.com/questions/1/shared",
        "domain": "stackoverflow.com",
        "search_rank": 1,
        "discovery_score": 4.0,
        "engines": ["bing"],
        "query_variant": "specific error",
        "retrieval_rank": 1,
        "retrieval_weight": 1.0,
    }
    fused = search_gateway._fuse_candidates(
        [
            {**shared, "retrieval_source": "searxng"},
            {
                **shared,
                "retrieval_source": "stackexchange",
                "prefetched_content": "Question body from the API",
                "prefetched_content_method": "stackexchange-api-question",
                "prefetched_low_confidence": True,
            },
        ]
    )

    assert fused[0]["prefetched_content"] == "Question body from the API"
    assert fused[0]["prefetched_low_confidence"] is True


def test_specialized_sources_are_supplements_not_technical_hard_routes(monkeypatch):
    monkeypatch.setattr(search_gateway, "ENABLE_KEYLESS_SUPPLEMENTS", True)
    assert search_gateway._supplement_sources_for("Python async stack trace error") == [
        "stackexchange"
    ]
    assert "github" not in search_gateway._supplement_sources_for(
        "How do I install PostgreSQL?"
    )
    assert search_gateway._supplement_sources_for("How do I install PostgreSQL?") == []
    assert "github" in search_gateway._supplement_sources_for(
        "Find the GitHub repository for this package"
    )


def test_explicit_community_request_enables_stackexchange_supplement(monkeypatch):
    monkeypatch.setattr(search_gateway, "ENABLE_KEYLESS_SUPPLEMENTS", True)
    assert search_gateway._supplement_sources_for(
        "What do users on Stack Overflow recommend for this Python problem?"
    ) == ["stackexchange"]


def test_general_technical_question_keeps_source_neutral_wikipedia_supplement(
    monkeypatch,
):
    monkeypatch.setattr(search_gateway, "ENABLE_KEYLESS_SUPPLEMENTS", True)
    assert search_gateway._supplement_sources_for("What is Python?") == ["wikipedia"]


@pytest.mark.asyncio
async def test_stackexchange_supplement_preserves_question_body_as_prefetched_evidence(
    monkeypatch,
):
    captured = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, params, headers):
            captured.update(params)
            return Stream()

    async def read(response, max_bytes=0):
        return {
            "items": [
                {
                    "title": "Example failure",
                    "link": "https://stackoverflow.com/questions/1/example",
                    "body": "<p>The command fails with a specific error.</p>",
                }
            ]
        }

    monkeypatch.setattr(search_gateway, "ENABLE_KEYLESS_SUPPLEMENTS", True)
    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)

    results, _ = await search_gateway._supplemental_search(
        "Python command failed with an error", None
    )

    assert captured["filter"] == "withbody"
    assert "specific error" in results[0]["prefetched_content"]
    assert results[0]["prefetched_low_confidence"] is True


@pytest.mark.asyncio
async def test_image_search_does_not_add_text_only_supplements(monkeypatch):
    monkeypatch.setattr(search_gateway, "ENABLE_KEYLESS_SUPPLEMENTS", True)

    results, diagnostics = await search_gateway._supplemental_search(
        "Nebula wallpaper", None, ["images"]
    )

    assert results == []
    assert diagnostics == []


@pytest.mark.asyncio
async def test_planner_invalid_output_falls_back_to_deterministic_queries(monkeypatch):
    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, *args, **kwargs):
            return Stream()

    async def read(response, max_bytes=0):
        return {"choices": [{"message": {"content": "not json"}}]}

    monkeypatch.setattr(search_gateway, "PLANNER_BASE_URL", "http://planner/v1")
    monkeypatch.setattr(search_gateway, "PLANNER_MODEL", "planner-model")
    monkeypatch.setattr(search_gateway, "PLANNER_MODES", {"deep"})
    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(search_gateway, "_read_json_response", read)

    variants, diagnostics = await search_gateway._planner_query_variants(
        "Nebula XZ900 setup guide", "deep"
    )
    assert variants == []
    assert diagnostics["status"] == "fallback"


def test_planner_rejects_non_string_query_variants():
    assert (
        search_gateway._planner_variant_is_safe(
            "Nebula XZ900 setup guide",
            {"query": "Nebula XZ900 setup guide"},
        )
        is None
    )


def test_deep_discovery_timeout_reserves_time_for_enabled_planner(monkeypatch):
    monkeypatch.setattr(search_gateway, "REQUEST_TIMEOUT_SECONDS", 60.0)
    monkeypatch.setattr(search_gateway, "SEARCH_TIMEOUT_SECONDS", 7.0)
    monkeypatch.setattr(search_gateway, "PLANNER_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(search_gateway, "PLANNER_BASE_URL", "http://planner/v1")
    monkeypatch.setattr(search_gateway, "PLANNER_MODEL", "planner-model")
    monkeypatch.setattr(search_gateway, "PLANNER_MODES", {"deep"})

    assert search_gateway._discovery_timeout_seconds("deep") == 28.0
    assert search_gateway._discovery_timeout_seconds("balanced") == 16.0


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
    assert (
        search_gateway._canonical_url(
            "https://example.com/guide?utm_source=test&item=2"
        )
        == "https://example.com/guide?item=2"
    )
    signed = "https://example.com/file?X-Amz-Signature=A%2FB&z=2&a=1"
    assert search_gateway._canonical_url(signed) == signed
    signed_path = (
        "https://EXAMPLE.com:443/files//release.bin?sig=A%2FB&token=preserve"
    )
    assert search_gateway._canonical_url(signed_path) == signed_path
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

    monkeypatch.setattr(
        search_gateway.httpx, "AsyncClient", lambda **kwargs: FailingClient()
    )
    docs = [
        {"text": "unrelated text", "candidate_index": 0},
        {"text": "AW3426DW recommended HDR settings", "candidate_index": 1},
    ]
    ranked, status = await search_gateway._rerank("AW3426DW HDR settings", docs, 2)
    assert status.startswith("fallback:")
    assert ranked[0]["candidate_index"] == 1


@pytest.mark.asyncio
async def test_reranker_falls_back_without_queueing_when_at_capacity(monkeypatch):
    class UnexpectedClient:
        async def __aenter__(self):
            raise AssertionError("capacity fallback must not contact the reranker")

        async def __aexit__(self, *args):
            return False

    semaphore = asyncio.Semaphore(1)
    await semaphore.acquire()
    monkeypatch.setattr(search_gateway, "_RERANK_ADMISSION", semaphore)
    monkeypatch.setattr(search_gateway, "RERANKER_ADMISSION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        search_gateway.httpx, "AsyncClient", lambda **kwargs: UnexpectedClient()
    )
    docs = [
        {"text": "unrelated text", "document_index": 0},
        {"text": "apt-get install docker-ce", "document_index": 1},
    ]

    ranked, status = await search_gateway._rerank("install docker-ce", docs, 1)
    semaphore.release()

    assert status == "fallback:capacity"
    assert ranked[0]["document_index"] == 1


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
async def test_reranker_chunks_large_frontend_batches(monkeypatch):
    calls = []

    class Stream:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return False

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            texts = kwargs["json"]["texts"]
            calls.append(texts)
            rows = [
                {"index": index, "score": int(text.rsplit(" ", 1)[1]) / 1000}
                for index, text in enumerate(texts)
            ]
            response = httpx.Response(
                200,
                json=rows,
                request=httpx.Request(method, url),
            )
            return Stream(response)

    monkeypatch.setattr(search_gateway.httpx, "AsyncClient", lambda **kwargs: Client())
    docs = [
        {"text": f"candidate {index}", "candidate_index": index} for index in range(202)
    ]

    ranked, status = await search_gateway._rerank("candidate", docs, 5)

    assert status == "ok"
    assert [len(batch) for batch in calls] == [32, 32, 32, 32, 32, 32, 10]
    assert [item["candidate_index"] for item in ranked] == [201, 200, 199, 198, 197]


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
async def test_adaptive_crawl_backfills_a_failed_top_candidate(monkeypatch):
    calls = []

    async def crawl(candidates, query, deadline):
        calls.append([item["url"] for item in candidates])
        candidate = candidates[0]
        if "blocked" in candidate["url"]:
            return [], [{"url": candidate["url"], "error": "blocked"}]
        return [
            {
                "url": candidate["url"],
                "content": "Useful extracted evidence. " * 60,
                "content_chars": 1600,
                "links": [],
                "search": candidate,
            }
        ], []

    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)
    pages, failures, batches = await search_gateway._adaptive_crawl_candidates(
        [
            {"title": "Blocked", "url": "https://blocked.example/"},
            {"title": "Working", "url": "https://working.example/"},
        ],
        "useful evidence",
        1,
        search_gateway.time.monotonic() + 5,
    )

    assert calls == [
        ["https://blocked.example/"],
        ["https://working.example/"],
    ]
    assert pages[0]["url"] == "https://working.example/"
    assert failures[0]["error"] == "blocked"
    assert batches[1]["backfill"] is True


@pytest.mark.asyncio
async def test_adaptive_crawl_backfills_low_confidence_pages(monkeypatch):
    calls = []

    async def crawl(candidates, query, deadline):
        candidate = candidates[0]
        calls.append(candidate["url"])
        return [
            {
                "url": candidate["url"],
                "content": "evidence",
                "content_chars": 8,
                "links": [],
                "low_confidence": "weak" in candidate["url"],
                "search": candidate,
            }
        ], []

    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)
    pages, _, _ = await search_gateway._adaptive_crawl_candidates(
        [
            {"title": "Weak", "url": "https://weak.example/"},
            {"title": "Strong", "url": "https://strong.example/"},
        ],
        "evidence",
        1,
        search_gateway.time.monotonic() + 5,
    )

    assert calls == ["https://weak.example/", "https://strong.example/"]
    assert pages[0]["url"] == "https://strong.example/"


@pytest.mark.asyncio
async def test_adaptive_crawl_skips_remaining_pages_from_a_blocked_domain(monkeypatch):
    calls = []

    async def crawl(candidates, query, deadline):
        candidate = candidates[0]
        calls.append(candidate["url"])
        if "blocked.example" in candidate["url"]:
            return [], [
                {
                    "url": candidate["url"],
                    "error": "PageExtractionError",
                    "detail": "direct:HTTP-403,playwright:challenge-or-interstitial",
                }
            ]
        return [
            {
                "url": candidate["url"],
                "content": "Useful extracted evidence. " * 60,
                "content_chars": 1600,
                "links": [],
                "search": candidate,
            }
        ], []

    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)
    pages, failures, _ = await search_gateway._adaptive_crawl_candidates(
        [
            {"title": "Blocked one", "url": "https://blocked.example/one"},
            {"title": "Blocked two", "url": "https://blocked.example/two"},
            {"title": "Working", "url": "https://working.example/guide"},
        ],
        "useful evidence",
        1,
        search_gateway.time.monotonic() + 5,
    )

    assert calls == [
        "https://blocked.example/one",
        "https://working.example/guide",
    ]
    assert pages[0]["url"] == "https://working.example/guide"
    assert any(
        failure["error"] == "domain-blocked-after-prior-failure" for failure in failures
    )


@pytest.mark.asyncio
async def test_prefetched_supplement_content_avoids_protected_page_fetch(monkeypatch):
    async def unexpected_fetch(*args, **kwargs):
        raise AssertionError("prefetched API content must not trigger a page fetch")

    monkeypatch.setattr(search_gateway, "fetch_page", unexpected_fetch)
    pages, failures = await search_gateway._crawl_candidates(
        [
            {
                "title": "Protected question",
                "url": "https://stackoverflow.com/questions/1/protected",
                "prefetched_content": "Question context from the API. " * 20,
                "prefetched_content_method": "stackexchange-api-question",
                "prefetched_low_confidence": True,
            }
        ],
        "specific error",
        search_gateway.time.monotonic() + 5,
    )

    assert failures == []
    assert pages[0]["extraction_method"] == "stackexchange-api-question"
    assert pages[0]["low_confidence"] is True


def test_passage_spans_reproduce_the_exact_source_text():
    source = (
        "  Introduction with useful context.\r\n\r\n"
        "## Setup\r\nInstall the package, then verify the service.\r\n\r\n"
        "Troubleshooting details follow.  "
    )
    chunks = search_gateway._chunk_text_with_spans(source)
    assert chunks
    for chunk in chunks:
        assert source[chunk["start_char"] : chunk["end_char"]] == chunk["text"]


@pytest.mark.asyncio
async def test_domain_fetch_learning_prefers_a_successful_fallback_after_failures():
    domain = "learned.example"
    await gateway_fetch._record_domain_outcome(
        domain, "direct", success=False, latency_seconds=1.0, reason="timeout"
    )
    await gateway_fetch._record_domain_outcome(
        domain, "direct", success=False, latency_seconds=1.0, reason="timeout"
    )
    await gateway_fetch._record_domain_outcome(
        domain, "crawl4ai", success=True, latency_seconds=0.5, reason="usable"
    )
    order, preferred, _ = await gateway_fetch._method_order(
        "https://learned.example/guide"
    )
    assert preferred == "crawl4ai"
    assert order[0] == "crawl4ai"


@pytest.mark.asyncio
async def test_recent_direct_failures_outweigh_old_success_history():
    domain = "changed.example"
    gateway_fetch._DOMAIN_FETCH_STATS[domain] = {
        "direct": {
            "attempts": 102,
            "successes": 100,
            "failures": 2,
            "consecutive_failures": 2,
            "latency_total": 20.0,
            "updated_at": int(gateway_fetch.time.time()),
        },
        "crawl4ai": {
            "attempts": 2,
            "successes": 2,
            "failures": 0,
            "consecutive_failures": 0,
            "latency_total": 2.0,
            "updated_at": int(gateway_fetch.time.time()),
        },
    }

    order, preferred, _ = await gateway_fetch._method_order(
        "https://changed.example/guide"
    )

    assert preferred == "crawl4ai"
    assert order[0] == "crawl4ai"


@pytest.mark.asyncio
async def test_domain_fetch_learning_reprobes_direct_after_temporary_failures():
    domain = "recovered.example"
    await gateway_fetch._record_domain_outcome(
        domain, "direct", success=False, latency_seconds=1.0, reason="timeout"
    )
    await gateway_fetch._record_domain_outcome(
        domain, "direct", success=False, latency_seconds=1.0, reason="timeout"
    )
    await gateway_fetch._record_domain_outcome(
        domain, "playwright", success=True, latency_seconds=0.5, reason="usable"
    )
    gateway_fetch._DOMAIN_FETCH_STATS[domain]["direct"]["updated_at"] = int(
        gateway_fetch.time.time() - gateway_fetch.FETCH_REPROBE_SECONDS - 1
    )

    order, preferred, _ = await gateway_fetch._method_order(
        "https://recovered.example/guide"
    )

    assert preferred == "playwright"
    assert order[0] == "direct"

    await gateway_fetch._record_domain_outcome(
        domain, "direct", success=True, latency_seconds=0.2, reason="usable"
    )
    order, _, stats = await gateway_fetch._method_order(
        "https://recovered.example/guide"
    )
    assert stats["direct"]["consecutive_failures"] == 0
    assert order[0] == "direct"


@pytest.mark.asyncio
async def test_domain_fetch_learning_keeps_direct_fast_paths_first():
    urls = (
        ("downloads.example", "https://downloads.example/manual.pdf"),
        ("downloads.example", "https://downloads.example/releases.json"),
        ("github.com", "https://github.com/example/project"),
        (
            "github.com",
            "https://github.com/example/project/blob/main/docs/install.md",
        ),
    )
    for domain in {domain for domain, _ in urls}:
        await gateway_fetch._record_domain_outcome(
            domain, "direct", success=False, latency_seconds=1.0, reason="timeout"
        )
        await gateway_fetch._record_domain_outcome(
            domain, "direct", success=False, latency_seconds=1.0, reason="timeout"
        )
        await gateway_fetch._record_domain_outcome(
            domain, "playwright", success=True, latency_seconds=0.5, reason="usable"
        )

    for _, url in urls:
        order, preferred, _ = await gateway_fetch._method_order(url)
        assert preferred == "playwright"
        assert order[0] == "direct"
        if gateway_fetch._github_fast_path(url):
            assert order[1] == "github_raw"


def test_long_documentation_about_access_denied_is_not_a_challenge():
    page = {
        "title": "Troubleshooting access errors",
        "content": (
            "This documentation explains Access Denied errors, their causes, "
            "and the configuration steps used to resolve them. "
        )
        * 40,
        "body_format": "html",
    }

    assessment = gateway_fetch.assess_content(
        page, "How do I resolve an Access Denied configuration error?"
    )

    assert assessment["status"] == "usable"
    assert assessment["usable"] is True


def test_long_documentation_with_one_decisive_marker_is_not_a_challenge():
    page = {
        "title": "Diagnosing reverse proxy errors",
        "content": (
            "This guide explains how administrators can interpret a Cloudflare Ray ID "
            "while diagnosing proxy configuration and origin connectivity. "
            + "Detailed configuration examples and verification steps follow. " * 55
        ),
        "body_format": "html",
    }

    assessment = gateway_fetch.assess_content(
        page, "How do I diagnose reverse proxy configuration errors?"
    )

    assert assessment["status"] == "usable"
    assert assessment["usable"] is True


@pytest.mark.asyncio
async def test_fetch_page_never_returns_a_challenge_shell(monkeypatch):
    async def validate(url):
        return url

    async def challenge(*args, **kwargs):
        return {
            "url": "https://challenge.example/",
            "title": "Attention Required",
            "content": "CAPTCHA. Please verify you are human. " * 80,
            "content_chars": 3040,
            "body_format": "html",
            "links": [],
            "extraction_method": "test",
        }

    monkeypatch.setattr(gateway_fetch, "validate_public_url", validate)
    monkeypatch.setattr(gateway_fetch, "direct_fetch", challenge)
    monkeypatch.setattr(gateway_fetch, "crawl4ai_fetch", challenge)
    monkeypatch.setattr(gateway_fetch, "browser_fetch", challenge)

    with pytest.raises(gateway_fetch.PageExtractionError):
        await gateway_fetch.fetch_page(
            "https://challenge.example/", "specific answer", 12
        )


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
    assert [item["url"] for item in captured] == ["https://docs.example.co.uk/install"]


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
async def test_fetch_page_reserves_time_for_playwright_after_crawl_failure(monkeypatch):
    calls = []

    async def validate(url):
        return url

    async def direct(url, timeout):
        return {"content": "thin", "content_chars": 4}

    async def crawl(url, timeout):
        calls.append(("crawl4ai", timeout))
        raise httpx.HTTPStatusError(
            "blocked",
            request=httpx.Request("POST", "http://crawl4ai/crawl"),
            response=httpx.Response(400),
        )

    async def browser(url, query, timeout):
        calls.append(("playwright", timeout))
        return {
            "content": "rendered evidence " * 100,
            "content_chars": 1800,
            "extraction_method": "playwright",
        }

    monkeypatch.setattr(gateway_fetch, "validate_public_url", validate)
    monkeypatch.setattr(gateway_fetch, "direct_fetch", direct)
    monkeypatch.setattr(gateway_fetch, "crawl4ai_fetch", crawl)
    monkeypatch.setattr(gateway_fetch, "browser_fetch", browser)

    result = await gateway_fetch.fetch_page("https://example.com", "query", 20)

    assert result["extraction_method"] == "playwright"
    assert calls[0][0] == "crawl4ai"
    assert calls[0][1] <= 10
    assert calls[1][0] == "playwright"
    assert calls[1][1] >= 7


def test_http_extraction_failure_preserves_status_code_for_domain_backoff():
    error = httpx.HTTPStatusError(
        "forbidden",
        request=httpx.Request("GET", "https://blocked.example/"),
        response=httpx.Response(403),
    )
    assert gateway_fetch._exception_reason(error) == "HTTP-403"


@pytest.mark.asyncio
async def test_integrated_budget_preserves_requested_search_mode(monkeypatch):
    search_gateway._CACHE.clear()
    search_gateway._QUERY_LOCKS.clear()
    search_gateway._QUERY_LOCK_USERS.clear()
    captured = {}

    async def no_cache(_):
        return None

    async def ignore_cache(*_):
        return None

    async def search(*args, **kwargs):
        captured["mode"] = kwargs["mode"]
        return (
            [
                {
                    "title": "Deep evidence",
                    "url": "https://example.com/deep",
                    "domain": "example.com",
                    "snippet": "Detailed evidence for the requested subject.",
                    "search_rank": 1,
                    "discovery_score": 4.0,
                    "published_at": None,
                    "engines": ["bing"],
                }
            ],
            [],
        )

    async def rerank(query, documents, top_k, timeout_seconds):
        return documents[:top_k], "ok"

    async def crawl(candidates, query, target_pages, deadline):
        return [], [], []

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_rerank_bounded", rerank)
    monkeypatch.setattr(search_gateway, "_adaptive_crawl_candidates", crawl)

    response = await search_gateway.research(
        search_gateway.SearchRequest(
            query="Deep investigation of a subject",
            mode="deep",
            max_results=1,
        ),
        budget_override=search_gateway.Budget(8, 1, 0, 1, 5),
        pipeline="integrated",
    )

    assert captured["mode"] == "deep"
    assert response["diagnostics"]["mode"] == "deep"
    assert response["diagnostics"]["pipeline"] == "integrated"


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
                    "content": "Docker Compose installation steps and configuration. "
                    * 80,
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
    evidence = response["results"][0]["evidence"][0]
    assert evidence["id"].startswith(response["results"][0]["evidence_id"])
    assert evidence["end_char"] > evidence["start_char"]


@pytest.mark.asyncio
async def test_research_direct_url_bypasses_searxng_and_keeps_citation_metadata(
    monkeypatch,
):
    search_gateway._QUERY_LOCKS.clear()
    search_gateway._QUERY_LOCK_USERS.clear()

    async def no_cache(_):
        return None

    async def ignore_cache(*_):
        return None

    async def unexpected_search(*args, **kwargs):
        raise AssertionError("direct URL research must bypass SearXNG")

    async def rerank(query, documents, top_k):
        ranked = []
        for index, document in enumerate(documents[:top_k]):
            item = dict(document)
            item["rerank_score"] = 1.0 - index * 0.01
            item["ranking_score"] = item["rerank_score"]
            ranked.append(item)
        return ranked, "ok"

    async def crawl(candidates, query, deadline):
        assert candidates[0]["url"] == "https://docs.example.com/install"
        return (
            [
                {
                    "url": candidates[0]["url"],
                    "title": "Example install guide",
                    "content": "Install Example with the supported package manager. "
                    * 80,
                    "content_chars": 4320,
                    "links": [],
                    "extraction_method": "direct",
                    "metadata": {"modifiedDate": "2026-08-01"},
                    "search": candidates[0],
                }
            ],
            [],
        )

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", unexpected_search)
    monkeypatch.setattr(search_gateway, "_rerank", rerank)
    monkeypatch.setattr(search_gateway, "_crawl_candidates", crawl)

    response = await search_gateway.research(
        search_gateway.SearchRequest(
            query="Read https://docs.example.com/install and summarize it",
            max_results=1,
        )
    )
    result = response["results"][0]
    assert response["diagnostics"]["direct_url_count"] == 1
    assert result["citation_url"] == "https://docs.example.com/install"
    assert result["evidence_id"].startswith("ev-")
    assert result["modifiedDate"] == "2026-08-01"


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
    assert response["diagnostics"]["evidence_status"] == "weak"
    assert response["evidence_summary"]["status"] == "weak"


@pytest.mark.asyncio
async def test_integrated_failure_pattern_keeps_official_candidate_first(monkeypatch):
    search_gateway._CACHE.clear()
    search_gateway._QUERY_LOCKS.clear()
    search_gateway._QUERY_LOCK_USERS.clear()
    captured = {}

    async def no_cache(_):
        return None

    async def ignore_cache(*_):
        return None

    def candidate(title, url, source_type, rank, primary=False):
        return {
            "title": title,
            "url": url,
            "domain": search_gateway._domain(url),
            "snippet": "Docker Compose installation instructions for Ubuntu.",
            "search_rank": rank,
            "discovery_score": 5.0 - rank * 0.1,
            "source_authority": 1.0 if primary else 0.5,
            "source_type": source_type,
            "source_tier": 1 if primary else 2,
            "authority_score": 0.9 if primary else 0.7,
            "primary_source_candidate": primary,
            "source_classification_method": "test",
            "subject_coverage": 0.8,
            "topic_partial_match": True,
            "published_at": None,
            "modified_at": None,
            "freshness_score": 0.0,
            "version_context": [],
            "evidence_id": search_gateway.stable_evidence_id(url),
            "citation_url": url,
            "engines": ["bing"],
            "image_url": None,
            "thumbnail_url": None,
        }

    candidates = [
        candidate(
            "Installing latest Docker Compose on Ubuntu",
            "https://stackoverflow.com/questions/1/install",
            "technical_reference",
            1,
        ),
        candidate(
            "Another Docker Compose discussion",
            "https://askubuntu.com/questions/2/install",
            "technical_reference",
            2,
        ),
        candidate(
            "Install the Docker Compose plugin",
            "https://docs.docker.com/compose/install/linux/",
            "documentation_candidate",
            3,
            primary=True,
        ),
    ]

    async def search(*args, **kwargs):
        return candidates, []

    async def rerank(query, documents, top_k, timeout_seconds):
        return [
            {"candidate_index": 0},
            {"candidate_index": 1},
            {"candidate_index": 2},
        ], "ok"

    async def crawl(selected, query, target_pages, deadline):
        captured["selected"] = selected
        return (
            [],
            [
                {"url": item["url"], "error": "PageExtractionError"}
                for item in selected
            ],
            [],
        )

    monkeypatch.setattr(search_gateway, "_cache_get", no_cache)
    monkeypatch.setattr(search_gateway, "_cache_set", ignore_cache)
    monkeypatch.setattr(search_gateway, "_searx_search", search)
    monkeypatch.setattr(search_gateway, "_rerank_bounded", rerank)
    monkeypatch.setattr(search_gateway, "_adaptive_crawl_candidates", crawl)

    response = await search_gateway.research(
        search_gateway.SearchRequest(
            query="How do I install Docker Compose on Ubuntu?", max_results=3
        ),
        pipeline="integrated",
    )

    official_url = "https://docs.docker.com/compose/install/linux/"
    assert captured["selected"][0]["url"] == official_url
    assert response["results"][0]["url"] == official_url
    assert response["diagnostics"]["evidence_status"] == "weak"


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
            await asyncio.sleep(2)
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
        search_gateway.Budget(4, 1, 0, 2, 1.6),
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
