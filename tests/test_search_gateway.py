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
    assert any("official documentation" in item for item in variants)
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
        "startpage",
        "bing",
        "brave",
        "askubuntu",
    ]
    assert targeted == ["startpage", "bing", "brave"]
    assert "github" not in technical


def test_specialized_engines_require_matching_subject_intent():
    web = search_gateway._search_engines(
        "How does the browser InstallEvent Web API work?", [], "browser installevent web api"
    )
    programming = search_gateway._search_engines(
        "Python stack trace error in an async function", [], "python async error"
    )
    assert "mdn" in web
    assert "stackoverflow" in programming
    assert "mdn" not in programming


def test_zero_subject_match_is_rejected_before_authority_can_boost_it():
    result = search_gateway._normalize_search_result(
        {
            "title": "InstallEvent: InstallEvent() constructor",
            "url": "https://developer.mozilla.org/en-US/docs/Web/API/InstallEvent/InstallEvent",
            "content": "The InstallEvent constructor creates an event.",
            "engine": "mdn",
        },
        "How do I install Docker Compose on Ubuntu?",
        1,
    )
    assert result is None


def test_subject_matching_does_not_accept_partial_words():
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
    assert result is None


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
        ('Fix the error "connection refused by upstream"', "connection refused by upstream"),
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

    assert inferred[0].startswith('"DragonSword Awakening"')
    assert any(item.startswith("DragonSword Awakening") for item in inferred[1:])
    assert explicit[0].startswith('"DragonSword Awakening"')
    assert all(len(item) > len('"DragonSword Awakening"') for item in explicit)
    assert search_gateway._topic_anchor(
        '"DragonSword Awakening" Theresia Astria Roxy'
    ) == "DragonSword Awakening"
    assert search_gateway._topic_anchor(
        "Persona 5 best team composition"
    ) == "Persona 5"
    assert search_gateway._topic_anchor(
        "Squid Game character build"
    ) == "Squid Game"


def test_topic_anchor_prefers_the_subject_over_later_feature_terms():
    assert search_gateway._topic_anchor(
        "Best Android TV boxes with AV1, Dolby Vision, and gigabit Ethernet"
    ) == "Android TV"
    assert search_gateway._topic_anchor("Path of Exile 2 best build") == "Path of Exile 2"


@pytest.mark.parametrize("query", [
    "Best 4K TVs for gaming",
    "Best HDR monitors",
    "Best USB microphones",
    "Best Wi-Fi routers",
    "Best Android TV boxes",
])
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


def test_entity_filter_rejects_generic_word_matches_before_authority_ranking():
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

    assert unrelated is None
    assert relevant is not None
    assert relevant["topic_anchor"] == "DragonSword Awakening"
    assert relevant["topic_match"] is True


def test_spaced_entity_rejects_a_single_ambiguous_word_match():
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
    assert result is None


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
    assert search_gateway._intent_source_adjustment(
        install_query, install_profile
    ) == -1.25
    assert search_gateway._intent_source_adjustment(
        repository_query, repository_profile
    ) == 0.0


@pytest.mark.asyncio
async def test_searx_variants_use_local_rank_and_targeted_engine_routing(monkeypatch):
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
        if "site:docs.docker.com" in query:
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
    targeted = next(item for item in captured if "site:docs.docker.com" in item["q"])
    assert targeted["engines"] == "startpage,bing,brave"
    assert all("categories" not in params for params in captured)
    assert diagnostics[1]["requested_engines"] == ["startpage", "bing", "brave"]


@pytest.mark.asyncio
async def test_discovery_rejects_topic_drift_across_exact_and_relaxed_queries(monkeypatch):
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
        '"DragonSword Awakening" best equipment characters guide',
        "DragonSword Awakening best equipment characters guide",
    ]
    assert [item["domain"] for item in results] == ["guide.example"]


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
        {"text": f"candidate {index}", "candidate_index": index}
        for index in range(202)
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
                    "content": "Install Example with the supported package manager. " * 80,
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
