"""Deterministic, SearXNG-compatible evidence search gateway."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from html import unescape
from typing import Any, Literal
from urllib.parse import unquote_plus, urlsplit, urlunsplit

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from evidence_quality import (
    evidence_summary,
    extract_version_markers,
    freshness_score,
    normalize_date,
    normalize_page_metadata,
    source_profile,
    stable_evidence_id,
    temporal_requirement,
)
from extractors import html_to_text
from gateway_fetch import close_fetch_resources, fetch_page
from request_limits import RequestBodyLimitMiddleware


LOGGER = logging.getLogger("search-gateway")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000").rstrip("/")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
REQUEST_TIMEOUT_SECONDS = max(
    5.0, float(os.getenv("GATEWAY_REQUEST_TIMEOUT_SECONDS", "30"))
)
SEARCH_TIMEOUT_SECONDS = max(
    2.0, float(os.getenv("GATEWAY_SEARCH_TIMEOUT_SECONDS", "7"))
)
PAGE_TIMEOUT_SECONDS = max(3.0, float(os.getenv("GATEWAY_PAGE_TIMEOUT_SECONDS", "16")))
MAX_SEARCH_RESULTS = max(8, int(os.getenv("GATEWAY_MAX_SEARCH_RESULTS", "30")))
MAX_CRAWL_PAGES = max(1, int(os.getenv("GATEWAY_MAX_CRAWL_PAGES", "6")))
MAX_FOLLOW_LINKS = max(0, int(os.getenv("GATEWAY_MAX_FOLLOW_LINKS", "2")))
MAX_CONCURRENT_FETCHES = max(1, int(os.getenv("GATEWAY_MAX_CONCURRENT_FETCHES", "4")))
MAX_FINAL_RESULTS = max(1, int(os.getenv("GATEWAY_MAX_FINAL_RESULTS", "8")))
MAX_PASSAGE_CHARS = max(800, int(os.getenv("GATEWAY_MAX_PASSAGE_CHARS", "2600")))
MAX_CONTENT_CHARS = max(1000, int(os.getenv("GATEWAY_MAX_CONTENT_CHARS", "7000")))
CACHE_TTL_SECONDS = max(10, int(os.getenv("GATEWAY_CACHE_TTL_SECONDS", "900")))
CACHE_STALE_SECONDS = max(
    CACHE_TTL_SECONDS, int(os.getenv("GATEWAY_CACHE_STALE_SECONDS", "86400"))
)
CACHE_MAX_ENTRIES = max(16, int(os.getenv("GATEWAY_CACHE_MAX_ENTRIES", "256")))
RERANKER_TIMEOUT_SECONDS = max(
    0.5, float(os.getenv("GATEWAY_RERANKER_TIMEOUT_SECONDS", "5"))
)
RERANKER_MAX_RESPONSE_BYTES = max(
    65_536, int(os.getenv("GATEWAY_RERANKER_MAX_RESPONSE_BYTES", "2097152"))
)
# The pinned CPU TEI image rejects client batches larger than 32. Keep the
# gateway-side value capped even if an operator accidentally configures more.
RERANKER_MAX_BATCH_SIZE = min(
    32, max(1, int(os.getenv("GATEWAY_RERANKER_MAX_BATCH_SIZE", "32")))
)
MAX_CONCURRENT_RERANKS = max(1, int(os.getenv("GATEWAY_MAX_CONCURRENT_RERANKS", "2")))
RERANKER_ADMISSION_TIMEOUT_SECONDS = max(
    0.01, float(os.getenv("GATEWAY_RERANKER_ADMISSION_TIMEOUT_SECONDS", "0.25"))
)
MAX_REQUEST_BYTES = max(4096, int(os.getenv("GATEWAY_MAX_REQUEST_BYTES", "262144")))
QUERY_MAX_CHARS = max(256, int(os.getenv("GATEWAY_QUERY_MAX_CHARS", "4000")))
MAX_CONCURRENT_REQUESTS = max(1, int(os.getenv("GATEWAY_MAX_CONCURRENT_REQUESTS", "4")))
ADMISSION_TIMEOUT_SECONDS = max(
    0.05, float(os.getenv("GATEWAY_ADMISSION_TIMEOUT_SECONDS", "2"))
)
FINALIZATION_RESERVE_SECONDS = max(
    0.25, float(os.getenv("GATEWAY_FINALIZATION_RESERVE_SECONDS", "3"))
)
CANDIDATE_RERANKER_TIMEOUT_SECONDS = max(
    0.25, float(os.getenv("GATEWAY_CANDIDATE_RERANKER_TIMEOUT_SECONDS", "2"))
)
PRIMARY_ENGINE_COUNT = max(
    1, min(3, int(os.getenv("GATEWAY_PRIMARY_ENGINE_COUNT", "2")))
)
FALLBACK_ENGINE_COUNT = max(1, int(os.getenv("GATEWAY_FALLBACK_ENGINE_COUNT", "4")))
ENGINE_COOLDOWN_SECONDS = max(
    30.0, float(os.getenv("GATEWAY_ENGINE_COOLDOWN_SECONDS", "900"))
)
ENGINE_HEALTH_MAX_ENTRIES = max(
    16, int(os.getenv("GATEWAY_ENGINE_HEALTH_MAX_ENTRIES", "128"))
)
RRF_K = max(1.0, float(os.getenv("GATEWAY_RRF_K", "60")))
ENABLE_KEYLESS_SUPPLEMENTS = os.getenv(
    "GATEWAY_ENABLE_KEYLESS_SUPPLEMENTS", "true"
).strip().casefold() in {"1", "true", "yes", "on"}
SUPPLEMENT_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("GATEWAY_SUPPLEMENT_TIMEOUT_SECONDS", "4"))
)
SUPPLEMENT_COOLDOWN_SECONDS = max(
    30.0, float(os.getenv("GATEWAY_SUPPLEMENT_COOLDOWN_SECONDS", "900"))
)
PLANNER_BASE_URL = os.getenv("GATEWAY_PLANNER_BASE_URL", "").strip().rstrip("/")
PLANNER_API_KEY = os.getenv("GATEWAY_PLANNER_API_KEY", "").strip()
PLANNER_MODEL = os.getenv("GATEWAY_PLANNER_MODEL", "").strip()
PLANNER_TIMEOUT_SECONDS = max(
    1.0, float(os.getenv("GATEWAY_PLANNER_TIMEOUT_SECONDS", "5"))
)
PLANNER_MODES = {
    item.strip().casefold()
    for item in os.getenv("GATEWAY_PLANNER_MODES", "deep").split(",")
    if item.strip()
}

# Compatibility routes deliberately have smaller budgets than deep research.
# They are intended for frontends that expect a quick search or one-page scrape.
INTEGRATED_TIMEOUT_SECONDS = max(
    5.0, float(os.getenv("GATEWAY_INTEGRATED_TIMEOUT_SECONDS", "20"))
)
INTEGRATED_MAX_SEARCH_RESULTS = max(
    4, int(os.getenv("GATEWAY_INTEGRATED_MAX_SEARCH_RESULTS", "16"))
)
INTEGRATED_MAX_CRAWL_PAGES = max(
    1, int(os.getenv("GATEWAY_INTEGRATED_MAX_CRAWL_PAGES", "3"))
)
INTEGRATED_MAX_RESULTS = max(1, int(os.getenv("GATEWAY_INTEGRATED_MAX_RESULTS", "5")))
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "").strip()
FIRECRAWL_TIMEOUT_SECONDS = max(
    5.0, float(os.getenv("FIRECRAWL_TIMEOUT_SECONDS", "30"))
)
FIRECRAWL_MAX_TIMEOUT_SECONDS = max(
    FIRECRAWL_TIMEOUT_SECONDS,
    float(os.getenv("FIRECRAWL_MAX_TIMEOUT_SECONDS", "60")),
)
FIRECRAWL_MAX_CONTENT_CHARS = max(
    10_000, int(os.getenv("FIRECRAWL_MAX_CONTENT_CHARS", "300000"))
)
FIRECRAWL_MAX_RESPONSE_BYTES = max(
    65_536, int(os.getenv("FIRECRAWL_MAX_RESPONSE_BYTES", "4194304"))
)
FIRECRAWL_MAX_RESULTS = max(1, int(os.getenv("FIRECRAWL_MAX_RESULTS", "20")))
CACHE_SCHEMA_VERSION = "adaptive-v2"


_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_LOCK = asyncio.Lock()
_QUERY_LOCKS: dict[str, asyncio.Lock] = {}
_QUERY_LOCK_USERS: dict[str, int] = {}
_ADMISSION = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
_RERANK_ADMISSION = asyncio.Semaphore(MAX_CONCURRENT_RERANKS)
_REDIS = None
_ENGINE_HEALTH: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SUPPLEMENT_COOLDOWNS: dict[str, float] = {}


TECHNICAL_RE = re.compile(
    r"\b(?:api|compose|config(?:uration)?|container|docker|error|exception|fix|github|"
    r"install|kubernetes|linux|log|manual|nginx|postgres|python|redis|server|setup|"
    r"stack trace|vps|virtual environment)\b",
    re.I,
)
INSTRUCTION_RE = re.compile(
    r"\b(?:configure|configuration|fix|guide|how\s+to|install(?:ation)?|manual|"
    r"setup|steps?|troubleshoot(?:ing)?|upgrade)\b",
    re.I,
)
REPOSITORY_INTENT_RE = re.compile(
    r"\b(?:commit|github|gitlab|issue|pull request|repository|repo|source code|tag)\b",
    re.I,
)
WEB_PLATFORM_RE = re.compile(
    r"\b(?:browser|css|dom|html|javascript|service worker|typescript|web api|webextension)\b",
    re.I,
)
PROGRAMMING_RE = re.compile(
    r"\b(?:c\+\+|compile|compiler|function|golang|java|javascript|node\.js|python|"
    r"rust|source code|stack trace|typescript)\b",
    re.I,
)
UBUNTU_RE = re.compile(r"\b(?:apt|debian|linux mint|ubuntu)\b", re.I)
TROUBLESHOOT_RE = re.compile(
    r"\b(?:bug|crash|error|exception|failed|failure|fix|problem|stack trace|troubleshoot)\b",
    re.I,
)
COMMUNITY_INTENT_RE = re.compile(
    r"\b(?:askubuntu|community|discussion|forum|reddit|stack\s*exchange|"
    r"stackoverflow|user reports?|what (?:do|did) (?:people|users))\b",
    re.I,
)
CURRENT_RE = re.compile(
    r"\b(?:current|latest|new|news|patch|release|recent|today|tonight)\b",
    re.I,
)
ACADEMIC_RE = re.compile(
    r"\b(?:academic|clinical|journal|paper|preprint|research|scientific|study)\b",
    re.I,
)
IMAGE_RE = re.compile(
    r"\b(?:image|images|photo|photos|picture|pictures|wallpaper)\b", re.I
)
RECOMMENDATION_RE = re.compile(
    r"\b(?:best|compare|comparison|recommend|recommended|settings|team composition|tier list)\b",
    re.I,
)
COMPARISON_INTENT_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?)(?:\b|$)", re.I
)
ERROR_QUOTE_RE = re.compile(r"(?:error|exception|failed|failure)[:\s]+(.{8,220})", re.I)
URL_IN_QUERY_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
QUOTED_PHRASE_RE = re.compile(r'"([^"\r\n]{2,160})"')
WORD_RE = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)
STOP_WORDS = frozenset(
    "a an and are as at be best by can could do does find for from give how i in information "
    "is it me my of on or please search should tell that the this to what when where which who "
    "why with would you your".split()
)
SOURCE_LABEL_BOOSTS = {
    "docs": 1.2,
    "developer": 1.0,
    "support": 0.9,
}
SOURCE_PENALTIES = {
    "hub.docker.com": -0.35,
    "pinterest.com": -4.0,
    "facebook.com": -3.0,
    "instagram.com": -3.0,
    "tiktok.com": -3.0,
    "fandom.com": -1.0,
}
AUTHORITY_RANK_WEIGHT = 0.08
BROAD_WEB_ENGINES = ("bing", "brave", "startpage")
NEWS_ENGINES = ("startpage news", "bing news", "brave.news", "reuters")
ACADEMIC_ENGINES = ("arxiv", "pubmed", "semantic scholar", "openalex", "crossref")
IMAGE_ENGINES = (
    "startpage images",
    "bing images",
    "brave.images",
    "wikicommons.images",
)
GENERIC_INTENT_TERMS = frozenset(
    {
        "best",
        "compare",
        "comparison",
        "configure",
        "configuration",
        "current",
        "error",
        "fix",
        "guide",
        "install",
        "installation",
        "latest",
        "manual",
        "news",
        "recommend",
        "recommended",
        "release",
        "settings",
        "setup",
        "steps",
        "today",
        "troubleshoot",
        "troubleshooting",
        "upgrade",
        "version",
    }
)
ANCHOR_CONNECTORS = frozenset({"and", "of", "the", "vs", "versus"})
ANCHOR_EXCLUDED_TERMS = frozenset(
    {
        *STOP_WORDS,
        *GENERIC_INTENT_TERMS,
        "current",
        "latest",
        "new",
        "recent",
        "today",
        "tonight",
        "use",
        "using",
        "read",
        "explain",
        "show",
        "help",
    }
)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=QUERY_MAX_CHARS)
    mode: Literal["auto", "quick", "balanced", "deep"] = "auto"
    max_results: int = Field(default=5, ge=1, le=MAX_FINAL_RESULTS)
    language: str = Field(default="auto", min_length=1, max_length=32)
    time_range: Literal["day", "week", "month", "year"] | None = None
    categories: list[str] = Field(default_factory=list, max_length=8)


class DiscoveryRequest(SearchRequest):
    """Search-only request that may ask for more candidates than final evidence."""

    max_results: int = Field(default=5, ge=1, le=MAX_SEARCH_RESULTS)


class FirecrawlScrapeRequest(BaseModel):
    """Subset of Firecrawl v2 scrape options used by supported frontends."""

    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=1, max_length=8192)
    formats: list[str] = Field(default_factory=lambda: ["markdown"], max_length=8)
    timeout: int | None = Field(default=None, ge=1000, le=120_000)


class FirecrawlSearchRequest(BaseModel):
    """Subset of Firecrawl v2 search options mapped to SearXNG discovery."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=QUERY_MAX_CHARS)
    limit: int = Field(default=5, ge=1, le=FIRECRAWL_MAX_RESULTS)
    lang: str = Field(default="auto", min_length=1, max_length=32)
    tbs: str | None = Field(default=None, max_length=16)


class JinaRerankRequest(BaseModel):
    """Subset of Jina's rerank request used by LibreChat web search."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(default="BAAI/bge-reranker-base", min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=QUERY_MAX_CHARS)
    documents: list[str] = Field(min_length=1, max_length=1000)
    top_n: int = Field(default=5, ge=1, le=20)
    return_documents: bool = True


class GatewayBusyError(RuntimeError):
    """Raised when all expensive retrieval slots remain occupied."""


@dataclass(frozen=True)
class Budget:
    search_results: int
    crawl_pages: int
    follow_links: int
    final_results: int
    total_seconds: float


MODE_BUDGETS = {
    "quick": Budget(12, 1, 0, 4, min(12.0, REQUEST_TIMEOUT_SECONDS)),
    "balanced": Budget(24, 3, 1, 6, REQUEST_TIMEOUT_SECONDS),
    "deep": Budget(30, 5, 2, 8, max(REQUEST_TIMEOUT_SECONDS, 40.0)),
}


def _mode_for(query: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if TECHNICAL_RE.search(query) or RECOMMENDATION_RE.search(query):
        return "balanced"
    return "quick" if len(query.split()) <= 9 else "balanced"


def _tokens(value: object) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for token in WORD_RE.findall(str(value or "").casefold()):
        token = token.strip("-.")
        if (
            (len(token) < 2 and not token.isdigit())
            or token in STOP_WORDS
            or token in seen
        ):
            continue
        seen.add(token)
        output.append(token)
    return output[:80]


def _topic_anchor(query: str) -> str | None:
    """Extract a high-confidence subject without requiring domain-specific labels.

    This is deliberately conservative. A model-provided quoted phrase, a product
    identifier, acronym, version number, or proper-name phrase is useful for
    relevance gating. Ordinary natural-language queries remain unanchored so the
    gateway does not reject valid paraphrases.
    """

    quoted = [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in QUOTED_PHRASE_RE.finditer(query)
    ]
    quoted = [
        phrase
        for phrase in quoted
        if len(re.sub(r"[\W_]+", "", phrase, flags=re.UNICODE)) >= 2
    ]
    if quoted:
        return quoted[0]

    words = list(WORD_RE.finditer(query))
    if not words:
        return None

    def is_identifier(word: str) -> bool:
        letters = re.sub(r"[^a-zA-Z]", "", word)
        return bool(
            re.search(r"\d", word)
            or re.search(r"[a-z][A-Z]|[A-Z][a-z].*[A-Z]", word)
            or (letters and letters.isupper() and len(letters) >= 2)
        )

    def is_candidate(word: str, index: int) -> bool:
        normalized = word.casefold().strip("-.")
        if not normalized or normalized in ANCHOR_EXCLUDED_TERMS:
            return False
        return is_identifier(word) or (
            index > 0 and word[:1].isupper() and word[1:].islower()
        )

    candidates = [
        is_candidate(word.group(0).strip("-."), index)
        for index, word in enumerate(words)
    ]
    for index, word in enumerate(words):
        if (
            candidates[index]
            or word.group(0).casefold() in ANCHOR_EXCLUDED_TERMS
            or not word.group(0)[:1].isupper()
            or not word.group(0)[1:].islower()
        ):
            continue
        cursor = index + 1
        while (
            cursor < len(words)
            and words[cursor].group(0).casefold() in ANCHOR_CONNECTORS
        ):
            cursor += 1
        if cursor < len(words) and candidates[cursor]:
            candidates[index] = True
    groups: list[tuple[list[str], int, int]] = []
    for start, word in enumerate(words):
        if not candidates[start]:
            continue
        group = [word.group(0).strip("-.")]
        identifiers = int(is_identifier(group[0]))
        cursor = start + 1
        while cursor < len(words):
            if candidates[cursor]:
                group.append(words[cursor].group(0).strip("-."))
                identifiers += int(is_identifier(group[-1]))
                cursor += 1
                continue
            if (
                words[cursor].group(0).casefold() in ANCHOR_CONNECTORS
                and cursor + 1 < len(words)
                and candidates[cursor + 1]
            ):
                group.extend(
                    [
                        words[cursor].group(0).casefold(),
                        words[cursor + 1].group(0).strip("-."),
                    ]
                )
                identifiers += int(is_identifier(group[-1]))
                cursor += 2
                continue
            break
        if len(_tokens(" ".join(group))) >= 2:
            groups.append((group, identifiers, start))

    if groups:
        group, _, _ = min(
            groups,
            key=lambda item: (
                item[2],
                not bool(item[1]),
                -item[1],
                -len(_tokens(" ".join(item[0]))),
            ),
        )
        return " ".join(group)

    # A single identifier can still be a precise subject (AW3426DW, HTTP, NIST).
    return next(
        (
            word.group(0).strip("-.")
            for index, word in enumerate(words)
            if candidates[index] and is_identifier(word.group(0))
        ),
        None,
    )


def _anchor_matches(anchor: str, text: str) -> bool:
    """Match compact, spaced, and multi-entity spellings without substrings."""

    anchor_words = WORD_RE.findall(anchor)
    anchor_parts: list[str] = []
    for word in anchor_words:
        # Treat CamelCase subjects (for example DragonSword) as either a
        # compact or spaced spelling, while keeping numeric versions bounded.
        anchor_parts.extend(
            re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", word)
        )
    anchor_parts = anchor_parts[:32]
    compact_pattern = r"[\W_]{0,16}".join(re.escape(part) for part in anchor_parts)
    if compact_pattern and re.search(rf"(?<!\w){compact_pattern}(?!\w)", text, re.I):
        return True
    terms = _tokens(anchor)
    identifiers = [
        term
        for term in terms
        if len(re.sub(r"[^a-z0-9]", "", term)) >= 4
        and re.search(r"[a-z]", term, re.I)
        and re.search(r"\d", term)
    ]
    if identifiers and any(_contains_term(text, term) for term in identifiers):
        return True
    return bool(terms) and all(_contains_term(text, term) for term in terms)


def _anchor_query(anchor: str, query: str) -> str:
    """Quote the subject while preserving the model's wording and intent."""

    if any(
        anchor.casefold() == phrase.casefold()
        for phrase in QUOTED_PHRASE_RE.findall(query)
    ):
        return query
    return re.sub(re.escape(anchor), f'"{anchor}"', query, count=1, flags=re.I)


def _topic_anchor_is_strict(query: str, anchor: str | None) -> bool:
    """Return whether a topic is precise enough for hard relevance filtering."""

    if not anchor:
        return False
    if any(
        anchor.casefold() == phrase.casefold()
        for phrase in QUOTED_PHRASE_RE.findall(query)
    ):
        return True
    terms = WORD_RE.findall(anchor)
    if not terms:
        return False
    # Short feature labels (4K, HDR, USB, Wi-Fi) are useful soft signals but
    # have many natural-language paraphrases. Keep them out of hard filtering.
    if len(terms) == 1:
        term = terms[0]
        compact = re.sub(r"[\W_]+", "", term)
        if re.search(r"\d", term):
            return len(compact) >= 4
        if re.search(r"[a-z][A-Z]", term) and not re.search(r"[-.]", term):
            return len(compact) >= 6
        return term.isupper() and len(compact) >= 4
    # A multiword subject containing a version/model marker is precise enough
    # to require a topic match (for example iPhone 17 or Path of Exile 2).
    return any(bool(re.search(r"\d", term)) for term in terms)


def _query_variants(query: str, mode: str) -> list[str]:
    """Build conservative fallbacks while preserving the user's query first."""

    normalized = re.sub(r"\s+", " ", query).strip()
    topic_anchor = _topic_anchor(normalized)
    variants = [normalized]
    error = ERROR_QUOTE_RE.search(normalized)
    if error and not topic_anchor:
        variants.append(f'"{error.group(1).strip()}"')
    elif CURRENT_RE.search(normalized):
        variants.append(f"{normalized} latest")
    elif TECHNICAL_RE.search(normalized) and INSTRUCTION_RE.search(normalized):
        instructional = (
            _anchor_query(topic_anchor, normalized) if topic_anchor else normalized
        )
        variants.append(f"{instructional} official documentation")
    elif topic_anchor:
        variants.append(_anchor_query(topic_anchor, normalized))
    if mode == "deep" and len(variants) < 3:
        if error and all(error.group(1).strip() not in item for item in variants[1:]):
            variants.append(f'{normalized} "{error.group(1).strip()}"')
        elif CURRENT_RE.search(normalized):
            variants.append(f"{normalized} recent authoritative sources")
        elif TECHNICAL_RE.search(normalized):
            variants.append(f"{normalized} official documentation")
        else:
            variants.append(f"{normalized} authoritative sources")
    limit = 3 if mode == "deep" else 2
    return list(dict.fromkeys(variants))[:limit]


def _search_categories(query: str, categories: list[str]) -> list[str]:
    if categories:
        return categories
    inferred = ["general"]
    if TECHNICAL_RE.search(query):
        inferred.append("it")
    if CURRENT_RE.search(query):
        inferred.append("news")
    if ACADEMIC_RE.search(query):
        inferred.append("science")
    if IMAGE_RE.search(query):
        inferred.append("images")
    return inferred


def _search_engines(
    query: str,
    categories: list[str],
    variant: str,
) -> list[str]:
    """Route each query variant to bounded engines suited to its intent."""

    effective = set(_search_categories(query, categories))
    if effective == {"images"}:
        return list(IMAGE_ENGINES)

    engines = list(BROAD_WEB_ENGINES)
    site_restricted = bool(re.search(r"(?:^|\s)site:[^\s]+", variant, re.I))
    if "it" in effective and not site_restricted:
        if WEB_PLATFORM_RE.search(query):
            engines.append("mdn")
        if PROGRAMMING_RE.search(query) or TROUBLESHOOT_RE.search(query):
            engines.append("stackoverflow")
        if UBUNTU_RE.search(query):
            engines.append("askubuntu")
    if "news" in effective:
        engines.extend(NEWS_ENGINES)
    if "science" in effective:
        engines.extend(ACADEMIC_ENGINES)
    if "images" in effective:
        engines.extend(IMAGE_ENGINES)
    return list(dict.fromkeys(engines))


def _domain(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _root_domain(domain: str) -> str:
    parts = [part for part in domain.rstrip(".").split(".") if part]
    if len(parts) < 2:
        return domain
    common_second_level = {"ac", "co", "com", "edu", "gov", "net", "org"}
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in common_second_level:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _canonical_url(value: object) -> str | None:
    url = str(value or "").strip()
    if not url or len(url) > 8192:
        return None
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").rstrip(".").lower()
        port = parsed.port
    except (UnicodeError, ValueError):
        return None
    if (
        scheme not in {"http", "https"}
        or not host
        or parsed.username
        or parsed.password
    ):
        return None
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    display_host = f"[{host}]" if ":" in host else host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        display_host = f"{display_host}:{port}"
    tracking_names = {
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref_src",
        "spm",
    }
    raw_query_parts = parsed.query.split("&") if parsed.query else []
    decoded_keys = [
        unquote_plus(part.partition("=")[0]).casefold() for part in raw_query_parts
    ]
    signed_query = any(
        key in {"signature", "sig", "token", "x-goog-signature"}
        or key.startswith("x-amz-")
        for key in decoded_keys
    )
    if signed_query:
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path or "/",
                parsed.query,
                "",
            )
        )
    query_string = "&".join(
        part
        for part, key in zip(raw_query_parts, decoded_keys)
        if not key.startswith("utm_") and key not in tracking_names
    )
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    return urlunsplit((scheme, display_host, path, query_string, ""))


def _extract_query_urls(query: str) -> list[str]:
    """Return unique HTTP(S) URLs explicitly supplied in a query."""

    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_IN_QUERY_RE.finditer(query):
        candidate = match.group(0).rstrip(".,;:!?")
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
            while candidate.endswith(closing) and candidate.count(
                closing
            ) > candidate.count(opening):
                candidate = candidate[:-1]
        canonical = _canonical_url(candidate)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        urls.append(canonical)
    return urls


def _direct_url_candidates(query: str, max_results: int) -> list[dict[str, Any]]:
    """Build deterministic discovery rows for URLs explicitly supplied by the client."""

    candidates: list[dict[str, Any]] = []
    for rank, url in enumerate(_extract_query_urls(query)[:max_results], start=1):
        parsed = urlsplit(url)
        domain = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/") or "/"
        title = f"{domain}{path}"
        profile = source_profile(url, title=title, query=query)
        source_authority = _source_adjustment(domain) + float(
            profile["authority_adjustment"]
        )
        candidates.append(
            {
                "title": title,
                "url": url,
                "domain": domain,
                "snippet": "User-supplied URL selected for direct page extraction.",
                "search_rank": rank,
                "discovery_score": round(10.0 + source_authority, 4),
                "source_authority": round(source_authority, 4),
                "source_type": profile["source_type"],
                "source_tier": profile["source_tier"],
                "authority_score": profile["authority_score"],
                "primary_source_candidate": profile["primary_source_candidate"],
                "source_classification_method": profile["classification_method"],
                "published_at": None,
                "modified_at": None,
                "freshness_score": 0.0,
                "version_context": extract_version_markers(title),
                "evidence_id": stable_evidence_id(url),
                "citation_url": url,
                "engines": ["direct-url"],
                "image_url": None,
                "thumbnail_url": None,
                "category": "general",
                "template": None,
                "parsed_url": None,
            }
        )
    return candidates


def _source_adjustment(domain: str) -> float:
    first_label = domain.split(".", 1)[0]
    adjustment = SOURCE_LABEL_BOOSTS.get(first_label, 0.0)
    for candidate, value in SOURCE_PENALTIES.items():
        if domain == candidate or domain.endswith("." + candidate):
            adjustment += value
    return adjustment


def _lexical_score(query: str, text: str) -> float:
    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    lowered = text.casefold()
    matched = sum(1 for term in query_terms if term in lowered)
    phrase = 1.0 if re.sub(r"\s+", " ", query.casefold()).strip() in lowered else 0.0
    return matched / len(query_terms) + 0.25 * phrase


def _subject_terms(query: str) -> list[str]:
    return [term for term in _tokens(query) if term not in GENERIC_INTENT_TERMS]


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.I) is not None


def _subject_coverage(query: str, text: str) -> float:
    terms = _subject_terms(query)
    if not terms:
        return 1.0
    return sum(_contains_term(text, term) for term in terms) / len(terms)


def _candidate_score(
    query: str,
    item: dict[str, Any],
    rank: int,
    time_range: str | None = None,
) -> float:
    domain = _domain(str(item.get("url") or ""))
    text = f"{item.get('title', '')} {item.get('content', '')}"
    subject_coverage = _subject_coverage(
        query,
        f"{text} {item.get('url', '')}",
    )
    profile = source_profile(
        item.get("url"),
        title=str(item.get("title") or ""),
        snippet=str(item.get("content") or ""),
        query=query,
    )
    requirement = temporal_requirement(query, time_range)
    published = normalize_date(item.get("publishedDate") or item.get("published_at"))
    modified = normalize_date(item.get("modifiedDate") or item.get("modified_at"))
    score = 4.0 * _lexical_score(query, text)
    score += 2.0 * subject_coverage
    topic_anchor = _topic_anchor(query)
    if topic_anchor:
        if _anchor_matches(topic_anchor, f"{text} {item.get('url', '')}"):
            score += 1.5
        elif _topic_anchor_is_strict(query, topic_anchor):
            score -= 2.5
        else:
            score -= 0.75
    if _subject_terms(query) and subject_coverage == 0:
        score -= 2.0
    score += 0.75 / max(1, rank)
    score += 0.2 * _source_adjustment(domain)
    score += 0.25 * float(profile["authority_adjustment"])
    score += 0.35 * _intent_source_adjustment(query, profile)
    score += 0.5 * max(
        freshness_score(published, requirement),
        freshness_score(modified, requirement),
    )
    return score


def _intent_source_adjustment(query: str, profile: dict[str, Any]) -> float:
    """Prefer instructional sources for how-to intent without harming repo queries."""

    if not INSTRUCTION_RE.search(query):
        return 0.0
    source_type = str(profile.get("source_type") or "")
    if source_type == "documentation_candidate":
        return 0.9
    if source_type in {"government", "standard", "technical_reference"}:
        return 0.35
    if source_type == "source_repository" and not REPOSITORY_INTENT_RE.search(query):
        return -1.25
    return 0.0


def _variant_source_adjustment(
    variant: str,
    result: dict[str, Any],
) -> float:
    domain = str(result.get("domain") or "")
    site = re.search(r"(?:^|\s)site:([^\s]+)", variant, re.I)
    if site:
        expected = site.group(1).strip(".").casefold()
        if domain == expected or domain.endswith("." + expected):
            return 0.8
    if (
        "official documentation" in variant.casefold()
        and result.get("source_type") == "documentation_candidate"
    ):
        return 0.4
    return 0.0


def _ranking_score(relevance: float, authority: object = 0.0) -> float:
    try:
        authority_value = float(authority or 0.0)
    except (TypeError, ValueError):
        authority_value = 0.0
    if not math.isfinite(authority_value):
        authority_value = 0.0
    authority_value = max(-4.0, min(4.0, authority_value))
    return relevance + AUTHORITY_RANK_WEIGHT * authority_value


def _document_ranking_score(document: dict[str, Any]) -> float:
    value = document.get("ranking_score", document.get("rerank_score", 0.0))
    try:
        score = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _normalize_search_result(
    item: dict[str, Any],
    query: str,
    rank: int,
    time_range: str | None = None,
) -> dict[str, Any] | None:
    url = _canonical_url(item.get("url"))
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
    if not url or not title:
        return None
    domain = _domain(url)
    if not domain or any(
        domain == blocked or domain.endswith("." + blocked)
        for blocked in SOURCE_PENALTIES
        if SOURCE_PENALTIES[blocked] <= -3
    ):
        return None
    snippet = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
    searchable_text = f"{title} {snippet} {url}"
    topic_anchor = _topic_anchor(query)
    topic_match = not topic_anchor or _anchor_matches(topic_anchor, searchable_text)
    topic_strict = _topic_anchor_is_strict(query, topic_anchor)
    anchor_terms = _tokens(topic_anchor) if topic_anchor else []
    anchor_match_count = sum(
        _contains_term(searchable_text, term) for term in anchor_terms
    )
    if topic_anchor and not topic_match:
        soft_anchor = any(
            (term.isupper() and len(re.sub(r"[\W_]+", "", term)) <= 3)
            or re.search(r"[-.]", term)
            or (re.search(r"\d", term) and len(re.sub(r"[\W_]+", "", term)) < 4)
            for term in WORD_RE.findall(topic_anchor)
        )
        required_anchor_terms = (
            0
            if soft_anchor and not topic_strict
            else 1
            if COMPARISON_INTENT_RE.search(query)
            else min(2, len(anchor_terms))
        )
        topic_partial_match = anchor_match_count >= required_anchor_terms
    else:
        topic_partial_match = topic_match
    subject_coverage = _subject_coverage(query, searchable_text)
    profile = source_profile(url, title=title, snippet=snippet, query=query)
    published = normalize_date(item.get("publishedDate") or item.get("published_at"))
    modified = normalize_date(item.get("modifiedDate") or item.get("modified_at"))
    requirement = temporal_requirement(query, time_range)
    version_context = extract_version_markers(title, snippet)
    intent_adjustment = _intent_source_adjustment(query, profile)
    source_authority = (
        _source_adjustment(domain)
        + float(profile["authority_adjustment"])
        + intent_adjustment
    )
    return {
        "title": title[:500],
        "url": url,
        "domain": domain,
        "snippet": snippet[:1500],
        "search_rank": rank,
        "discovery_score": round(_candidate_score(query, item, rank, time_range), 4),
        "source_authority": round(source_authority, 4),
        "source_type": profile["source_type"],
        "source_tier": profile["source_tier"],
        "authority_score": profile["authority_score"],
        "primary_source_candidate": profile["primary_source_candidate"],
        "source_classification_method": profile["classification_method"],
        "intent_source_adjustment": intent_adjustment,
        "subject_coverage": round(subject_coverage, 4),
        "topic_anchor": topic_anchor,
        "topic_match": topic_match,
        "topic_partial_match": topic_partial_match,
        "topic_strict": topic_strict,
        "topic_term_matches": anchor_match_count,
        "published_at": published,
        "modified_at": modified,
        "freshness_score": max(
            freshness_score(published, requirement),
            freshness_score(modified, requirement),
        ),
        "version_context": version_context,
        "evidence_id": stable_evidence_id(url),
        "citation_url": url,
        "engines": item.get("engines")
        or ([item.get("engine")] if item.get("engine") else []),
        "image_url": item.get("img_src") or item.get("image_url"),
        "thumbnail_url": item.get("thumbnail_src") or item.get("thumbnail"),
        "category": item.get("category"),
        "template": item.get("template"),
        "parsed_url": item.get("parsed_url"),
        "query_variants": item.get("query_variants") or [],
        "retrieval_sources": item.get("retrieval_sources") or [],
        "query_consensus": item.get("query_consensus", 0),
        "engine_consensus": item.get("engine_consensus", 0),
        "fusion_score": item.get("fusion_score", 0.0),
        "prefetched_content": item.get("prefetched_content"),
        "prefetched_content_method": item.get("prefetched_content_method"),
        "prefetched_low_confidence": bool(item.get("prefetched_low_confidence")),
    }


async def _read_json_value(
    response: httpx.Response, max_bytes: int = 4 * 1024 * 1024
) -> Any:
    declared = response.headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
            if declared_size < 0 or declared_size > max_bytes:
                raise ValueError("Upstream response exceeds the configured limit")
        except ValueError as exc:
            raise ValueError("Upstream returned an invalid Content-Length") from exc
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_bytes:
            raise ValueError("Upstream response exceeds the configured limit")
        body.extend(chunk)
    return json.loads(body.decode("utf-8"))


async def _read_json_response(
    response: httpx.Response, max_bytes: int = 4 * 1024 * 1024
) -> dict[str, Any]:
    payload = await _read_json_value(response, max_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Upstream returned non-object JSON")
    return payload


def _planner_is_enabled(mode: str) -> bool:
    return bool(PLANNER_BASE_URL and PLANNER_MODEL and mode.casefold() in PLANNER_MODES)


def _discovery_timeout_seconds(mode: str) -> float:
    wave_count = 1 if mode == "quick" else (3 if mode == "deep" else 2)
    planner_seconds = PLANNER_TIMEOUT_SECONDS if _planner_is_enabled(mode) else 0.0
    return min(
        REQUEST_TIMEOUT_SECONDS,
        SEARCH_TIMEOUT_SECONDS * wave_count + planner_seconds + 2.0,
    )


def _planner_variant_is_safe(original: str, candidate: object) -> str | None:
    if not isinstance(candidate, str):
        return None
    value = re.sub(r"\s+", " ", str(candidate or "")).strip()
    if not value or len(value) > min(500, QUERY_MAX_CHARS):
        return None
    if URL_IN_QUERY_RE.search(value) and not URL_IN_QUERY_RE.search(original):
        return None
    if re.search(r"(?:^|\s)site:[^\s]+", value, re.I) and not re.search(
        r"(?:^|\s)site:[^\s]+", original, re.I
    ):
        return None
    anchor = _topic_anchor(original)
    if anchor and not _anchor_matches(anchor, value):
        return None
    subject_terms = _subject_terms(original)
    if subject_terms:
        matched = sum(_contains_term(value, term) for term in subject_terms)
        if matched < max(1, math.ceil(len(subject_terms) * 0.4)):
            return None
    return value


async def _planner_query_variants(
    query: str, mode: str
) -> tuple[list[str], dict[str, Any]]:
    if not _planner_is_enabled(mode):
        return [], {
            "status": "disabled"
            if not (PLANNER_BASE_URL and PLANNER_MODEL)
            else "not-selected",
            "model": PLANNER_MODEL or None,
        }
    endpoint = (
        PLANNER_BASE_URL
        if PLANNER_BASE_URL.endswith("/chat/completions")
        else f"{PLANNER_BASE_URL}/chat/completions"
    )
    headers = {"Content-Type": "application/json"}
    if PLANNER_API_KEY:
        headers["Authorization"] = f"Bearer {PLANNER_API_KEY}"
    instruction = (
        "Create at most two concise web-search query alternatives for the user's request. "
        "Preserve named entities, product identifiers, quoted errors, and the actual intent. "
        "Do not choose a website, domain, or source type unless the user explicitly did. "
        'Return JSON only: {"queries":["..."]}.'
    )
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(PLANNER_TIMEOUT_SECONDS), trust_env=False
        ) as client:
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json={
                    "model": PLANNER_MODEL,
                    "temperature": 0,
                    "max_tokens": 180,
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": query},
                    ],
                },
            ) as response:
                response.raise_for_status()
                payload = await _read_json_response(response, 1_048_576)
        choices = payload.get("choices")
        content: Any = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict)
            )
        text = str(content or "").strip()
        match = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(match.group(0) if match else text)
        raw_queries = parsed.get("queries") if isinstance(parsed, dict) else None
        if not isinstance(raw_queries, list):
            raise ValueError("Planner response did not contain a query list")
        variants: list[str] = []
        for raw in raw_queries:
            candidate = _planner_variant_is_safe(query, raw)
            if (
                candidate
                and candidate.casefold() != query.casefold()
                and candidate not in variants
            ):
                variants.append(candidate)
            if len(variants) >= 2:
                break
        return variants, {
            "status": "ok",
            "model": PLANNER_MODEL,
            "variant_count": len(variants),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        return [], {
            "status": "fallback",
            "model": PLANNER_MODEL,
            "error": type(exc).__name__,
            "duration_seconds": round(time.monotonic() - started, 3),
        }


def _unresponsive_engine_map(value: object) -> dict[str, str]:
    output: dict[str, str] = {}
    if not isinstance(value, list):
        return output
    for row in value:
        if isinstance(row, (list, tuple)) and row:
            name = str(row[0]).strip().casefold()
            reason = str(row[1] if len(row) > 1 else "unresponsive").strip()
        elif isinstance(row, dict):
            name = str(row.get("engine") or row.get("name") or "").strip().casefold()
            reason = str(
                row.get("error") or row.get("reason") or "unresponsive"
            ).strip()
        else:
            continue
        if name:
            output[name] = reason[:300]
    return output


def _record_engine_outcomes(
    requested: list[str], unresponsive: object, latency_seconds: float
) -> None:
    now = time.monotonic()
    failed = _unresponsive_engine_map(unresponsive)
    for engine in requested:
        key = engine.casefold()
        state = dict(_ENGINE_HEALTH.get(key) or {})
        state.setdefault("successes", 0)
        state.setdefault("failures", 0)
        state.setdefault("consecutive_failures", 0)
        state.setdefault("latency_total", 0.0)
        state.setdefault("attempts", 0)
        state["attempts"] += 1
        state["latency_total"] += max(0.0, latency_seconds)
        if key in failed:
            reason = failed[key]
            state["failures"] += 1
            state["consecutive_failures"] += 1
            state["last_failure"] = reason
            state["last_failure_at"] = now
            if re.search(
                r"(?:429|captcha|rate|too many|suspend|blocked)", reason, re.I
            ):
                multiplier = min(4, int(state["consecutive_failures"]))
                state["cooldown_until"] = now + ENGINE_COOLDOWN_SECONDS * multiplier
        else:
            state["successes"] += 1
            state["consecutive_failures"] = 0
            state["last_success_at"] = now
            state["cooldown_until"] = 0.0
        _ENGINE_HEALTH[key] = state
        _ENGINE_HEALTH.move_to_end(key)
    while len(_ENGINE_HEALTH) > ENGINE_HEALTH_MAX_ENTRIES:
        _ENGINE_HEALTH.popitem(last=False)


def _engine_health_snapshot(engine: str) -> dict[str, Any]:
    state = _ENGINE_HEALTH.get(engine.casefold()) or {}
    attempts = max(1, int(state.get("attempts") or 0))
    return {
        "engine": engine,
        "successes": int(state.get("successes") or 0),
        "failures": int(state.get("failures") or 0),
        "average_latency_seconds": round(
            float(state.get("latency_total") or 0.0) / attempts, 3
        ),
        "cooldown_remaining_seconds": round(
            max(0.0, float(state.get("cooldown_until") or 0.0) - time.monotonic()),
            3,
        ),
        "last_failure": state.get("last_failure"),
    }


def _select_healthy_engines(
    engines: list[str], limit: int, *, excluded: set[str] | None = None
) -> tuple[list[str], list[str]]:
    now = time.monotonic()
    excluded = {item.casefold() for item in (excluded or set())}
    unique = list(dict.fromkeys(engines))
    ready: list[tuple[tuple[float, float, int], str]] = []
    cooling: list[tuple[float, str]] = []
    for index, engine in enumerate(unique):
        if engine.casefold() in excluded:
            continue
        state = _ENGINE_HEALTH.get(engine.casefold()) or {}
        cooldown_until = float(state.get("cooldown_until") or 0.0)
        if cooldown_until > now:
            cooling.append((cooldown_until, engine))
            continue
        attempts = max(1, int(state.get("attempts") or 0))
        failure_rate = float(state.get("failures") or 0) / attempts
        latency = float(state.get("latency_total") or 0.0) / attempts
        ready.append(((failure_rate, latency, index), engine))
    ready.sort(key=lambda item: item[0])
    selected = [engine for _, engine in ready[:limit]]
    skipped = [engine for _, engine in sorted(cooling)]
    if not selected and cooling:
        selected = [min(cooling)[1]]
        skipped = [engine for _, engine in cooling if engine != selected[0]]
    return selected, skipped


def _engines_for_wave(
    query: str,
    categories: list[str],
    variant: str,
    wave: int,
    *,
    used_engines: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    all_engines = _search_engines(query, categories, variant)
    effective = set(_search_categories(query, categories))
    image_primary = "images" in effective and not effective.intersection(
        {"it", "news", "science"}
    )
    if effective == {"images"} or image_primary:
        pool = [engine for engine in all_engines if engine in IMAGE_ENGINES]
    elif wave == 0:
        pool = [engine for engine in all_engines if engine in BROAD_WEB_ENGINES]
    else:
        used = {item.casefold() for item in (used_engines or set())}
        unused = [engine for engine in all_engines if engine.casefold() not in used]
        pool = [*unused, *[engine for engine in all_engines if engine not in unused]]
    limit = PRIMARY_ENGINE_COUNT if wave == 0 else FALLBACK_ENGINE_COUNT
    selected, skipped = _select_healthy_engines(pool, limit)
    if not selected:
        selected, skipped_retry = _select_healthy_engines(all_engines, 1)
        skipped = list(dict.fromkeys([*skipped, *skipped_retry]))
    return selected, skipped


def _fuse_candidates(occurrences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        url = str(occurrence.get("url") or "")
        if not url:
            continue
        rank = max(
            1,
            int(occurrence.get("retrieval_rank") or occurrence.get("search_rank") or 1),
        )
        weight = max(0.1, float(occurrence.get("retrieval_weight") or 1.0))
        rrf = weight * RRF_K / (RRF_K + rank)
        existing = fused.get(url)
        if existing is None:
            existing = dict(occurrence)
            existing["query_variants"] = []
            existing["retrieval_sources"] = []
            existing["rrf_score"] = 0.0
            existing["search_occurrences"] = 0
            existing["engines"] = []
            fused[url] = existing
        best_rank = min(
            int(existing.get("search_rank") or rank),
            int(occurrence.get("search_rank") or rank),
        )
        if float(occurrence.get("discovery_score") or 0.0) > float(
            existing.get("base_discovery_score", existing.get("discovery_score") or 0.0)
        ):
            preserved = {
                key: existing[key]
                for key in (
                    "query_variants",
                    "retrieval_sources",
                    "rrf_score",
                    "search_occurrences",
                    "engines",
                )
            }
            existing.update(occurrence)
            existing.update(preserved)
        existing["search_rank"] = best_rank
        existing["base_discovery_score"] = max(
            float(existing.get("base_discovery_score") or 0.0),
            float(occurrence.get("discovery_score") or 0.0),
        )
        existing["rrf_score"] += rrf
        existing["search_occurrences"] += 1
        variant = str(occurrence.get("query_variant") or "").strip()
        source = str(occurrence.get("retrieval_source") or "searxng").strip()
        if variant and variant not in existing["query_variants"]:
            existing["query_variants"].append(variant)
        if source and source not in existing["retrieval_sources"]:
            existing["retrieval_sources"].append(source)
        for engine in occurrence.get("engines") or []:
            if engine and engine not in existing["engines"]:
                existing["engines"].append(engine)
        if occurrence.get("prefetched_content") and not existing.get(
            "prefetched_content"
        ):
            existing["prefetched_content"] = occurrence["prefetched_content"]
            existing["prefetched_content_method"] = occurrence.get(
                "prefetched_content_method"
            )
            existing["prefetched_low_confidence"] = bool(
                occurrence.get("prefetched_low_confidence")
            )
    for item in fused.values():
        query_consensus = len(item["query_variants"])
        engine_consensus = len(item["engines"])
        consensus_bonus = 0.22 * max(0, min(3, engine_consensus) - 1)
        consensus_bonus += 0.3 * max(0, min(3, query_consensus) - 1)
        item["query_consensus"] = query_consensus
        item["engine_consensus"] = engine_consensus
        item["fusion_score"] = round(float(item["rrf_score"]) + consensus_bonus, 6)
        item["discovery_score"] = round(
            float(item.get("base_discovery_score") or 0.0)
            + 0.8 * float(item["rrf_score"])
            + consensus_bonus,
            4,
        )
    return sorted(
        fused.values(),
        key=lambda item: float(item.get("discovery_score") or 0.0),
        reverse=True,
    )


def _candidate_quality(
    query: str, candidates: list[dict[str, Any]], desired_results: int, mode: str
) -> dict[str, Any]:
    sample = candidates[: max(8, min(len(candidates), desired_results * 2))]
    domains = {_root_domain(str(item.get("domain") or "")) for item in sample}
    domains.discard("")
    topic_anchor = _topic_anchor(query)
    relevant = [
        item
        for item in sample
        if (bool(topic_anchor) and bool(item.get("topic_partial_match")))
        or float(item.get("subject_coverage") or 0.0) >= 0.34
        or _lexical_score(query, f"{item.get('title', '')} {item.get('snippet', '')}")
        >= 0.42
    ]
    topic_matches = (
        sum(bool(item.get("topic_partial_match")) for item in sample)
        if topic_anchor
        else 0
    )
    target = min(max(1, desired_results), 5 if mode == "deep" else 3)
    if mode == "deep" and desired_results > 1:
        target = max(3, target)
    required_domains = (
        1 if target == 1 else (3 if mode == "deep" and target >= 4 else 2)
    )
    reasons: list[str] = []
    if len(relevant) < target:
        reasons.append("too-few-relevant-candidates")
    if len(domains) < min(required_domains, max(1, len(sample))):
        reasons.append("limited-domain-diversity")
    if topic_anchor and topic_matches == 0:
        reasons.append("named-entity-not-covered")
    if not sample:
        reasons.append("no-candidates")
    return {
        "status": "sufficient" if not reasons else "weak",
        "candidate_count": len(candidates),
        "sample_count": len(sample),
        "relevant_candidate_count": len(relevant),
        "independent_domain_count": len(domains),
        "topic_match_count": topic_matches,
        "target_relevant_candidates": target,
        "reasons": reasons,
    }


def _supplement_sources_for(query: str) -> list[str]:
    if not ENABLE_KEYLESS_SUPPLEMENTS:
        return []
    sources: list[str] = []
    if ACADEMIC_RE.search(query):
        sources.append("crossref")
    elif TROUBLESHOOT_RE.search(query) or COMMUNITY_INTENT_RE.search(query):
        sources.append("stackexchange")
    elif not INSTRUCTION_RE.search(query):
        sources.append("wikipedia")
    if REPOSITORY_INTENT_RE.search(query):
        sources.append("github")
    now = time.monotonic()
    return [
        source
        for source in sources
        if float(_SUPPLEMENT_COOLDOWNS.get(source) or 0.0) <= now
    ]


async def _supplemental_search(
    query: str, time_range: str | None, categories: list[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if "images" in _search_categories(query, categories or []):
        return [], []
    sources = _supplement_sources_for(query)
    if not sources:
        return [], []

    async def request(source: str) -> tuple[str, dict[str, Any]]:
        if source == "stackexchange":
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params: dict[str, Any] = {
                "site": "stackoverflow",
                "q": query,
                "pagesize": 5,
                "order": "desc",
                "sort": "relevance",
                "filter": "withbody",
            }
        elif source == "github":
            url = "https://api.github.com/search/repositories"
            params = {"q": query, "per_page": 5}
        elif source == "crossref":
            url = "https://api.crossref.org/works"
            params = {"query": query, "rows": 5}
        else:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 5,
                "prop": "extracts|info",
                "exintro": "1",
                "explaintext": "1",
                "inprop": "url",
                "format": "json",
            }
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(SUPPLEMENT_TIMEOUT_SECONDS), trust_env=False
        ) as client:
            async with client.stream(
                "GET",
                url,
                params=params,
                headers={"User-Agent": "private-search-gateway/1.0"},
            ) as response:
                response.raise_for_status()
                return source, await _read_json_response(response, 2_097_152)

    outcomes = await asyncio.gather(
        *(request(source) for source in sources), return_exceptions=True
    )
    occurrences: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for source, outcome in zip(sources, outcomes):
        if isinstance(outcome, Exception):
            status_code = (
                outcome.response.status_code
                if isinstance(outcome, httpx.HTTPStatusError)
                else None
            )
            if status_code in {403, 429}:
                _SUPPLEMENT_COOLDOWNS[source] = (
                    time.monotonic() + SUPPLEMENT_COOLDOWN_SECONDS
                )
            diagnostics.append(
                {
                    "provider": source,
                    "wave": "supplemental",
                    "status": "failed",
                    "error": type(outcome).__name__,
                    "status_code": status_code,
                }
            )
            continue
        _, payload = outcome
        raw: list[dict[str, Any]] = []
        if source == "stackexchange":
            for item in payload.get("items") or []:
                if isinstance(item, dict):
                    body = html_to_text(str(item.get("body") or ""))
                    raw.append(
                        {
                            "title": unescape(str(item.get("title") or "")),
                            "url": item.get("link"),
                            "content": body[:1500]
                            or "Stack Overflow question with community answers.",
                            "prefetched_content": body or None,
                            "prefetched_content_method": "stackexchange-api-question",
                            "prefetched_low_confidence": True,
                            "publishedDate": item.get("creation_date"),
                            "modifiedDate": item.get("last_activity_date"),
                            "engines": ["stackexchange-api"],
                        }
                    )
        elif source == "github":
            for item in payload.get("items") or []:
                if isinstance(item, dict):
                    raw.append(
                        {
                            "title": item.get("full_name") or item.get("name"),
                            "url": item.get("html_url"),
                            "content": item.get("description") or "Source repository.",
                            "modifiedDate": item.get("updated_at"),
                            "engines": ["github-api"],
                        }
                    )
        elif source == "crossref":
            message = (
                payload.get("message")
                if isinstance(payload.get("message"), dict)
                else {}
            )
            for item in message.get("items") or []:
                if not isinstance(item, dict):
                    continue
                titles = item.get("title") or []
                title = titles[0] if isinstance(titles, list) and titles else titles
                raw.append(
                    {
                        "title": title,
                        "url": item.get("URL"),
                        "content": item.get("abstract")
                        or "Academic publication record.",
                        "engines": ["crossref-api"],
                    }
                )
        else:
            query_payload = (
                payload.get("query") if isinstance(payload.get("query"), dict) else {}
            )
            pages = (
                query_payload.get("pages")
                if isinstance(query_payload.get("pages"), dict)
                else {}
            )
            for item in pages.values():
                if isinstance(item, dict):
                    raw.append(
                        {
                            "title": item.get("title"),
                            "url": item.get("fullurl"),
                            "content": item.get("extract") or "Wikipedia article.",
                            "engines": ["wikipedia-api"],
                        }
                    )
        accepted = 0
        for rank, item in enumerate(raw, start=1):
            normalized = _normalize_search_result(item, query, rank, time_range)
            if normalized is None:
                continue
            normalized.update(
                {
                    "query_variant": query,
                    "retrieval_rank": rank,
                    "retrieval_weight": 0.7,
                    "retrieval_source": source,
                }
            )
            occurrences.append(normalized)
            accepted += 1
        diagnostics.append(
            {
                "provider": source,
                "wave": "supplemental",
                "status": "ok",
                "result_count": accepted,
            }
        )
    return occurrences, diagnostics


async def _searx_search(
    query: str,
    *,
    mode: str,
    max_results: int,
    language: str,
    time_range: str | None,
    categories: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    deterministic = _query_variants(query, mode)
    planned, planner_diagnostics = await _planner_query_variants(query, mode)
    variants = [
        deterministic[0],
        *[item for item in planned if item.casefold() != deterministic[0].casefold()],
        *deterministic[1:],
    ]
    variant_limit = 3 if mode == "deep" else 2
    variants = list(dict.fromkeys(variants))[:variant_limit]
    diagnostics: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    used_engines: set[str] = set()

    async def one(variant: str, wave: int) -> tuple[dict[str, Any], dict[str, Any]]:
        engines, skipped = _engines_for_wave(
            query,
            categories,
            variant,
            wave,
            used_engines=used_engines,
        )
        params: dict[str, str] = {
            "q": variant,
            "format": "json",
            "language": language or "auto",
            "safesearch": "0",
            "engines": ",".join(engines),
        }
        if time_range:
            params["time_range"] = time_range
        started = time.monotonic()
        diagnostic = {
            "query": variant,
            "provider": "searxng",
            "wave": "initial" if wave == 0 else "fallback",
            "requested_engines": engines,
            "cooldown_skipped_engines": skipped,
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(SEARCH_TIMEOUT_SECONDS), trust_env=False
            ) as client:
                async with client.stream(
                    "GET", f"{SEARXNG_URL}/search", params=params
                ) as response:
                    response.raise_for_status()
                    payload = await _read_json_response(response)
        except Exception as exc:
            latency = time.monotonic() - started
            status_code = (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError)
                else None
            )
            reason = type(exc).__name__
            if status_code is not None:
                reason = f"{reason}:{status_code}"
            _record_engine_outcomes(
                engines,
                [[engine, f"request-failed:{reason}"] for engine in engines],
                latency,
            )
            diagnostic.update(
                {
                    "status": "failed",
                    "error": type(exc).__name__,
                    "status_code": status_code,
                    "engine_health": [
                        _engine_health_snapshot(engine) for engine in engines
                    ],
                    "latency_seconds": round(latency, 3),
                }
            )
            return {}, diagnostic
        latency = time.monotonic() - started
        _record_engine_outcomes(engines, payload.get("unresponsive_engines"), latency)
        diagnostic.update(
            {
                "engine_health": [
                    _engine_health_snapshot(engine) for engine in engines
                ],
                "latency_seconds": round(latency, 3),
            }
        )
        return payload, diagnostic

    async def collect(variant: str, wave: int, weight: float) -> None:
        try:
            payload, diagnostic = await one(variant, wave)
        except Exception as exc:
            diagnostics.append(
                {
                    "query": variant,
                    "provider": "searxng",
                    "wave": "initial" if wave == 0 else "fallback",
                    "status": "failed",
                    "error": type(exc).__name__,
                }
            )
            return
        used_engines.update(diagnostic.get("requested_engines") or [])
        if diagnostic.get("status") == "failed":
            diagnostics.append(diagnostic)
            return
        raw_results = payload.get("results") if isinstance(payload, dict) else None
        diagnostic.update(
            {
                "status": "ok",
                "result_count": len(raw_results)
                if isinstance(raw_results, list)
                else 0,
                "unresponsive_engines": payload.get("unresponsive_engines", []),
            }
        )
        diagnostics.append(diagnostic)
        for rank, item in enumerate(raw_results or [], start=1):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_search_result(item, query, rank, time_range)
            if normalized is None:
                continue
            normalized["query_variant"] = variant
            normalized["retrieval_rank"] = rank
            normalized["retrieval_weight"] = weight
            normalized["retrieval_source"] = "searxng"
            normalized["discovery_score"] = round(
                normalized["discovery_score"]
                + _variant_source_adjustment(variant, normalized),
                4,
            )
            occurrences.append(normalized)

    await collect(variants[0], 0, 1.0)
    fused = _fuse_candidates(occurrences)
    initial_quality = _candidate_quality(query, fused, max_results, mode)
    fallback_triggered = mode != "quick" and initial_quality["status"] != "sufficient"
    if fallback_triggered:
        if "images" in _search_categories(query, categories):
            default_fallback = f"{query} high resolution"
        elif CURRENT_RE.search(query):
            default_fallback = f"{query} latest"
        else:
            default_fallback = f"{query} authoritative sources"
        fallback_variants = variants[1:] or [default_fallback]
        supplement_task = asyncio.create_task(
            _supplemental_search(query, time_range, categories)
        )
        supplemental: list[dict[str, Any]] = []
        supplemental_diagnostics: list[dict[str, Any]] = []
        try:
            for variant in fallback_variants[: (2 if mode == "deep" else 1)]:
                await collect(variant, 1, 0.9)
            try:
                supplemental, supplemental_diagnostics = await supplement_task
            except Exception as exc:
                supplemental_diagnostics = [
                    {
                        "provider": "supplemental",
                        "wave": "supplemental",
                        "status": "failed",
                        "error": type(exc).__name__,
                    }
                ]
        finally:
            if not supplement_task.done():
                supplement_task.cancel()
                await asyncio.gather(supplement_task, return_exceptions=True)
        occurrences.extend(supplemental)
        diagnostics.extend(supplemental_diagnostics)
        fused = _fuse_candidates(occurrences)
    final_quality = _candidate_quality(query, fused, max_results, mode)
    summary = {
        "provider": "search-gateway",
        "wave": "summary",
        "status": "ok",
        "planner": planner_diagnostics,
        "variants": variants,
        "fallback_triggered": fallback_triggered,
        "fallback_reasons": initial_quality["reasons"] if fallback_triggered else [],
        "initial_quality": initial_quality,
        "final_quality": final_quality,
    }
    diagnostics.append(summary)
    return fused[:max_results], diagnostics


def _chunk_text_with_spans(text: str) -> list[dict[str, Any]]:
    normalized = text or ""
    segments: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=(?:\r?\n){2,}|\s*\Z)", normalized, re.S):
        start, end = match.span()
        while end - start > MAX_PASSAGE_CHARS:
            upper = start + MAX_PASSAGE_CHARS
            window = normalized[start:upper]
            sentence_breaks = [
                item.end()
                for item in re.finditer(r"[.!?](?:\s+|$)", window)
                if item.end() >= int(MAX_PASSAGE_CHARS * 0.55)
            ]
            split_at = start + (sentence_breaks[-1] if sentence_breaks else len(window))
            if split_at <= start:
                split_at = upper
            segments.append((start, split_at))
            start = split_at
            while start < end and normalized[start].isspace():
                start += 1
        if start < end:
            segments.append((start, end))

    spans: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for start, end in segments:
        if current is None:
            current = (start, end)
        elif end - current[0] <= MAX_PASSAGE_CHARS:
            current = (current[0], end)
        else:
            spans.append(current)
            current = (start, end)
        if len(spans) >= 40:
            break
    if current is not None and len(spans) < 40:
        spans.append(current)

    output: list[dict[str, Any]] = []
    for start, end in spans[:40]:
        while start < end and normalized[start].isspace():
            start += 1
        while end > start and normalized[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        heading = None
        prefix = normalized[max(0, start - 500) : start]
        heading_matches = list(
            re.finditer(r"(?:^|\r?\n)#{1,6}\s+([^\r\n]{1,200})", prefix)
        )
        if heading_matches:
            heading = heading_matches[-1].group(1).strip()
        output.append(
            {
                "text": normalized[start:end],
                "start_char": start,
                "end_char": end,
                "section": heading,
            }
        )
    return output


def _chunk_text(text: str) -> list[str]:
    return [item["text"] for item in _chunk_text_with_spans(text)]


async def _rerank(
    query: str, documents: list[dict[str, Any]], top_k: int
) -> tuple[list[dict[str, Any]], str]:
    if not documents:
        return [], "empty"
    texts = [
        str(document.get("reranker_text") or document["text"])[:MAX_PASSAGE_CHARS]
        for document in documents
    ]
    try:
        await asyncio.wait_for(
            _RERANK_ADMISSION.acquire(), RERANKER_ADMISSION_TIMEOUT_SECONDS
        )
    except TimeoutError:
        return _lexical_rerank(query, documents, top_k), "fallback:capacity"
    try:
        deadline = time.monotonic() + RERANKER_TIMEOUT_SECONDS
        scored: list[dict[str, Any]] = []
        scored_indices: set[int] = set()
        failures: list[str] = []
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(RERANKER_TIMEOUT_SECONDS), trust_env=False
        ) as client:
            for offset in range(0, len(texts), RERANKER_MAX_BATCH_SIZE):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failures.append("deadline")
                    break
                batch = texts[offset : offset + RERANKER_MAX_BATCH_SIZE]
                try:
                    async with client.stream(
                        "POST",
                        f"{RERANKER_URL}/rerank",
                        json={"query": query, "texts": batch},
                        timeout=httpx.Timeout(max(0.1, remaining)),
                    ) as response:
                        response.raise_for_status()
                        payload = await _read_json_value(
                            response, RERANKER_MAX_RESPONSE_BYTES
                        )
                except Exception as exc:
                    failures.append(type(exc).__name__)
                    continue

                rows = payload.get("results") if isinstance(payload, dict) else payload
                if not isinstance(rows, list):
                    failures.append("invalid_response")
                    continue
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(
                        row.get("index"), int
                    ):
                        continue
                    local_index = row["index"]
                    index = offset + local_index
                    if not 0 <= local_index < len(batch) or index in scored_indices:
                        continue
                    score = float(row.get("score") or 0.0)
                    if not math.isfinite(score):
                        continue
                    item = dict(documents[index])
                    item["rerank_score"] = score
                    item["ranking_score"] = _ranking_score(
                        score, item.get("authority_score")
                    )
                    scored.append(item)
                    scored_indices.add(index)

        if not scored:
            reason = failures[0] if failures else "invalid_response"
            LOGGER.warning("Reranker unavailable; using lexical fallback: %s", reason)
            return _lexical_rerank(query, documents, top_k), f"fallback:{reason}"
        unscored: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            if index in scored_indices:
                continue
            item = dict(document)
            item["rerank_score"] = _lexical_score(
                query, item.get("reranker_text") or item["text"]
            )
            item["ranking_score"] = _ranking_score(
                item["rerank_score"], item.get("authority_score")
            )
            unscored.append(item)
        scored.sort(key=lambda item: item["ranking_score"], reverse=True)
        unscored.sort(key=lambda item: item["ranking_score"], reverse=True)
        status = "ok" if len(scored_indices) == len(documents) else "partial"
        return [*scored, *unscored][:top_k], status
    except Exception as exc:
        LOGGER.warning(
            "Reranker unavailable; using lexical fallback: %s", type(exc).__name__
        )
        return _lexical_rerank(
            query, documents, top_k
        ), f"fallback:{type(exc).__name__}"
    finally:
        _RERANK_ADMISSION.release()


def _lexical_rerank(
    query: str, documents: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    fallback = []
    for document in documents:
        item = dict(document)
        item["rerank_score"] = _lexical_score(
            query, item.get("reranker_text") or item["text"]
        )
        item["ranking_score"] = _ranking_score(
            item["rerank_score"], item.get("authority_score")
        )
        fallback.append(item)
    fallback.sort(key=lambda item: item["ranking_score"], reverse=True)
    return fallback[:top_k]


async def _rerank_bounded(
    query: str,
    documents: list[dict[str, Any]],
    top_k: int,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], str]:
    if timeout_seconds <= 0:
        return _lexical_rerank(query, documents, top_k), "fallback:deadline"
    try:
        async with asyncio.timeout(timeout_seconds):
            return await _rerank(query, documents, top_k)
    except TimeoutError:
        return _lexical_rerank(query, documents, top_k), "fallback:deadline"


def _select_crawl_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    ranked_documents: list[dict[str, Any]],
    target_pages: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep relevance order while reserving early crawl slots for source diversity."""

    ranked_indices: list[int] = []
    for document in ranked_documents:
        try:
            index = int(document.get("candidate_index", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(candidates) and index not in ranked_indices:
            ranked_indices.append(index)
    ranked_indices.extend(
        index for index in range(len(candidates)) if index not in ranked_indices
    )

    instructional = bool(INSTRUCTION_RE.search(query))
    explicit_community = bool(
        TROUBLESHOOT_RE.search(query) or COMMUNITY_INTENT_RE.search(query)
    )

    def is_primary(index: int) -> bool:
        candidate = candidates[index]
        source_type = str(candidate.get("source_type") or "")
        candidate_text = f"{candidate.get('title', '')} {candidate.get('snippet', '')}"
        relevant = (
            bool(candidate.get("topic_partial_match"))
            or float(candidate.get("subject_coverage") or 0.0) >= 0.34
            or _lexical_score(query, candidate_text) >= 0.35
        )
        if not relevant:
            return False
        if source_type in {
            "documentation_candidate",
            "government",
            "release_notes",
            "standard",
        }:
            return True
        return bool(candidate.get("primary_source_candidate")) and (
            source_type != "source_repository" or REPOSITORY_INTENT_RE.search(query)
        )

    def is_community(index: int) -> bool:
        return str(candidates[index].get("source_type") or "") in {
            "technical_community",
            "technical_reference",
        }

    ordered_indices = ranked_indices
    if instructional:
        primary = [index for index in ranked_indices if is_primary(index)]
        remainder = [index for index in ranked_indices if index not in primary]
        if not explicit_community:
            non_community = [index for index in remainder if not is_community(index)]
            community = [index for index in remainder if is_community(index)]
            remainder = [*non_community, *community]
        ordered_indices = [*primary, *remainder]

    early_unique_limit = min(limit, max(target_pages * 2, target_pages + 2))
    selected_indices: list[int] = []
    deferred_indices: list[int] = []
    seen_domains: set[str] = set()
    for index in ordered_indices:
        domain = _root_domain(str(candidates[index].get("domain") or ""))
        if len(selected_indices) < early_unique_limit and domain not in seen_domains:
            selected_indices.append(index)
            if domain:
                seen_domains.add(domain)
        else:
            deferred_indices.append(index)
    selected_indices.extend(deferred_indices)
    return [candidates[index] for index in selected_indices[:limit]]


BLOCKING_CRAWL_FAILURE_RE = re.compile(
    r"(?:HTTP[- ]?(?:403|429)|access denied|captcha|challenge|forbidden|"
    r"interstitial|rate.?limit|robots?|temporarily blocked|too many requests)",
    re.I,
)


def _failure_blocks_domain(failure: dict[str, Any]) -> bool:
    detail = f"{failure.get('error', '')} {failure.get('detail', '')}"
    return BLOCKING_CRAWL_FAILURE_RE.search(detail) is not None


async def _crawl_candidates(
    candidates: list[dict[str, Any]], query: str, deadline: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def one(
        candidate_index: int, candidate: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        async with semaphore:
            prefetched = str(candidate.get("prefetched_content") or "").strip()
            if prefetched:
                return candidate, {
                    "url": candidate["url"],
                    "title": candidate.get("title"),
                    "content": prefetched,
                    "content_chars": len(prefetched),
                    "body_format": "text",
                    "links": [],
                    "status_code": 200,
                    "extraction_method": candidate.get("prefetched_content_method")
                    or "supplemental-api",
                    "metadata": normalize_page_metadata(
                        {
                            "publishedDate": candidate.get("published_at"),
                            "modifiedDate": candidate.get("modified_at"),
                        }
                    ),
                    "low_confidence": bool(
                        candidate.get("prefetched_low_confidence", True)
                    ),
                }
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                return candidate, None
            try:
                page = await fetch_page(
                    candidate["url"],
                    query,
                    min(PAGE_TIMEOUT_SECONDS, remaining),
                    allow_expensive_fallback=bool(
                        candidate.get("_allow_expensive_fallback", candidate_index < 3)
                    ),
                )
                return candidate, page
            except Exception as exc:
                return candidate, {
                    "error": type(exc).__name__,
                    "detail": re.sub(r"[\r\n]+", " ", str(exc)).strip()[:300],
                }

    tasks = {
        asyncio.create_task(one(index, candidate)): (index, candidate)
        for index, candidate in enumerate(candidates)
    }
    if not tasks:
        return [], []
    done: set[asyncio.Task] = set()
    pending: set[asyncio.Task] = set(tasks)
    try:
        done, pending = await asyncio.wait(
            tasks, timeout=max(0.0, deadline - time.monotonic())
        )
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    pairs = [(tasks[task][0], *task.result()) for task in done if not task.cancelled()]
    pairs.sort(key=lambda pair: pair[0])
    pages: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = [
        {"url": tasks[task][1]["url"], "error": "deadline"} for task in pending
    ]
    for _, candidate, page in pairs:
        if page is None or page.get("error"):
            failure = {
                "url": candidate["url"],
                "error": (page or {}).get("error", "deadline"),
            }
            if (page or {}).get("detail"):
                failure["detail"] = page["detail"]
            failures.append(failure)
            continue
        page["search"] = candidate
        pages.append(page)
    return pages, failures


async def _adaptive_crawl_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    target_pages: int,
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch small batches and replace failed candidates until the target is met."""

    target_pages = max(1, target_pages)
    strong_pages: list[dict[str, Any]] = []
    weak_pages: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    blocked_domains: set[str] = set()
    cursor = 0
    while cursor < len(candidates) and len(strong_pages) < target_pages:
        if deadline - time.monotonic() <= 1.25:
            break
        needed = target_pages - len(strong_pages)
        batch_size = min(MAX_CONCURRENT_FETCHES, max(1, min(2, needed)))
        batch = []
        while cursor < len(candidates) and len(batch) < batch_size:
            index = cursor
            candidate = candidates[cursor]
            cursor += 1
            root = _root_domain(
                str(candidate.get("domain") or _domain(candidate["url"]))
            )
            if root and root in blocked_domains:
                failures.append(
                    {
                        "url": candidate["url"],
                        "error": "domain-blocked-after-prior-failure",
                    }
                )
                continue
            row = dict(candidate)
            row["_allow_expensive_fallback"] = index < max(3, target_pages)
            batch.append(row)
        if not batch:
            continue
        batch_pages, batch_failures = await _crawl_candidates(batch, query, deadline)
        batch_strong = [page for page in batch_pages if not page.get("low_confidence")]
        batch_weak = [page for page in batch_pages if page.get("low_confidence")]
        strong_pages.extend(batch_strong)
        weak_pages.extend(batch_weak)
        failures.extend(batch_failures)
        for failure in batch_failures:
            if not _failure_blocks_domain(failure):
                continue
            root = _root_domain(_domain(str(failure.get("url") or "")))
            if root:
                blocked_domains.add(root)
        batches.append(
            {
                "attempted": len(batch),
                "succeeded": len(batch_strong),
                "low_confidence": len(batch_weak),
                "failed": len(batch_failures),
                "backfill": len(batches) > 0,
            }
        )
    if cursor < len(candidates) and deadline - time.monotonic() <= 1.25:
        failures.extend(
            {"url": item["url"], "error": "deadline-not-attempted"}
            for item in candidates[cursor:]
        )
    pages = [*strong_pages, *weak_pages]
    return pages[:target_pages], failures, batches


def _follow_link_score(query: str, source_url: str, link: dict[str, str]) -> float:
    url = link.get("url", "")
    if _root_domain(_domain(url)) != _root_domain(_domain(source_url)):
        return -10.0
    path = urlsplit(url).path.casefold()
    text = f"{link.get('anchor', '')} {path}"
    score = 2.0 * _lexical_score(query, text)
    if re.search(
        r"\b(?:docs?|guide|install|setup|config|manual|troubleshoot|faq|support)\b",
        text,
        re.I,
    ):
        score += 0.7
    if re.search(r"(?:login|sign.?in|account|privacy|terms|contact|cart)", text, re.I):
        score -= 2.0
    return score


def _pages_need_follow_up(
    pages: list[dict[str, Any]], query: str, target_pages: int
) -> bool:
    if len(pages) < target_pages:
        return True
    for page in pages:
        content = str(page.get("content") or "")
        search = page.get("search") or {}
        if (
            len(content) < 1400
            and (
                float(search.get("subject_coverage") or 0.0) >= 0.25
                or _lexical_score(query, content) >= 0.2
            )
            and page.get("links")
        ):
            return True
    return False


async def _follow_relevant_links(
    pages: list[dict[str, Any]], query: str, limit: int, deadline: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if limit <= 0:
        return [], []
    candidates: list[tuple[float, dict[str, Any]]] = []
    visited = {
        canonical
        for page in pages
        if (canonical := _canonical_url(page.get("url"))) is not None
    }
    for page in pages:
        for link in page.get("links") or []:
            url = _canonical_url(link.get("url"))
            if not url or url in visited:
                continue
            normalized_link = dict(link)
            normalized_link["url"] = url
            score = _follow_link_score(
                query, str(page.get("url") or ""), normalized_link
            )
            if score > 0.35:
                visited.add(url)
                profile = source_profile(
                    url,
                    title=str(link.get("anchor") or ""),
                    query=query,
                )
                source_authority = _source_adjustment(_domain(url)) + float(
                    profile["authority_adjustment"]
                )
                candidates.append(
                    (
                        score,
                        {
                            "title": link.get("anchor") or url,
                            "url": url,
                            "domain": _domain(url),
                            "snippet": f"Relevant link from {page.get('title') or page.get('url')}",
                            "search_rank": 999,
                            "discovery_score": score,
                            "source_authority": source_authority,
                            "source_type": profile["source_type"],
                            "source_tier": profile["source_tier"],
                            "authority_score": profile["authority_score"],
                            "primary_source_candidate": profile[
                                "primary_source_candidate"
                            ],
                            "source_classification_method": profile[
                                "classification_method"
                            ],
                            "published_at": None,
                            "modified_at": None,
                            "freshness_score": 0.0,
                            "version_context": extract_version_markers(
                                link.get("anchor")
                            ),
                            "evidence_id": stable_evidence_id(url),
                            "citation_url": url,
                            "engines": ["bounded-link-follow"],
                        },
                    )
                )
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    selected = [item for _, item in candidates[:limit]]
    if not selected:
        return [], []
    return await _crawl_candidates(selected, query, deadline)


def _passage_documents(pages: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        search = page.get("search") or {}
        metadata = normalize_page_metadata(page.get("metadata"))
        context_parts = [
            f"Title: {page.get('title') or search.get('title') or ''}",
            f"Source type: {search.get('source_type') or 'general_web'}",
        ]
        if metadata.get("publishedDate") or search.get("published_at"):
            context_parts.append(
                f"Published: {metadata.get('publishedDate') or search.get('published_at')}"
            )
        if metadata.get("modifiedDate") or search.get("modified_at"):
            context_parts.append(
                f"Modified: {metadata.get('modifiedDate') or search.get('modified_at')}"
            )
        if metadata.get("declaredDate"):
            context_parts.append(f"Page-declared date: {metadata['declaredDate']}")
        versions = list(
            dict.fromkeys(
                [
                    *(search.get("version_context") or []),
                    *(metadata.get("version_context") or []),
                ]
            )
        )
        if versions:
            context_parts.append(f"Version context: {', '.join(versions[:8])}")
        reranker_context = "\n".join(context_parts)
        for passage_index, passage_record in enumerate(
            _chunk_text_with_spans(str(page.get("content") or ""))
        ):
            passage = str(passage_record["text"])
            lexical = _lexical_score(query, passage)
            if lexical <= 0 and passage_index > 6:
                continue
            documents.append(
                {
                    "text": passage,
                    "reranker_text": f"{reranker_context}\n\n{passage}",
                    "page_index": page_index,
                    "passage_index": passage_index,
                    "start_char": passage_record["start_char"],
                    "end_char": passage_record["end_char"],
                    "section": passage_record.get("section"),
                    "lexical_score": lexical,
                    "source_score": float(search.get("discovery_score") or 0.0),
                    "authority_score": float(search.get("source_authority") or 0.0),
                }
            )
    documents.sort(
        key=lambda item: item["lexical_score"] + 0.08 * item["source_score"],
        reverse=True,
    )
    return documents[:80]


def _assemble_results(
    pages: list[dict[str, Any]],
    ranked_passages: list[dict[str, Any]],
    limit: int,
    query: str = "",
) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict[str, Any]]] = {}
    for passage in ranked_passages:
        by_page.setdefault(int(passage["page_index"]), []).append(passage)
    ranked_pages = sorted(
        by_page,
        key=lambda index: max(_document_ranking_score(item) for item in by_page[index]),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    owners: set[str] = set()
    deferred: list[dict[str, Any]] = []
    for page_index in ranked_pages:
        page = pages[page_index]
        search = page.get("search") or {}
        passages = sorted(
            by_page[page_index],
            key=_document_ranking_score,
            reverse=True,
        )[:3]
        content = "\n\n".join(item["text"] for item in passages)[:MAX_CONTENT_CHARS]
        url = str(page.get("url") or search.get("url") or "")
        metadata = normalize_page_metadata(page.get("metadata"))
        profile = source_profile(
            url,
            title=str(page.get("title") or search.get("title") or ""),
            snippet=str(search.get("snippet") or ""),
            query=query,
        )
        published = metadata.get("publishedDate") or search.get("published_at")
        modified = metadata.get("modifiedDate") or search.get("modified_at")
        declared = metadata.get("declaredDate")
        version_context = list(
            dict.fromkeys(
                [
                    *(search.get("version_context") or []),
                    *(metadata.get("version_context") or []),
                    *extract_version_markers(page.get("title"), content),
                ]
            )
        )[:8]
        evidence_id = search.get("evidence_id") or stable_evidence_id(url)
        evidence_records = []
        for passage in passages:
            start_char = int(passage.get("start_char") or 0)
            end_char = int(passage.get("end_char") or start_char)
            passage_text = str(passage.get("text") or "")
            digest = hashlib.sha256(
                f"{evidence_id}:{start_char}:{end_char}:{passage_text}".encode("utf-8")
            ).hexdigest()[:12]
            evidence_records.append(
                {
                    "id": f"{evidence_id}-p-{digest}",
                    "source_id": evidence_id,
                    "url": url,
                    "passage_index": int(passage.get("passage_index") or 0),
                    "start_char": start_char,
                    "end_char": end_char,
                    "section": passage.get("section"),
                    "text": passage_text,
                    "score": round(_document_ranking_score(passage), 6),
                }
            )
        item = {
            "title": page.get("title") or search.get("title") or page.get("url"),
            "url": url,
            "content": content,
            "snippet": search.get("snippet", ""),
            "engine": "search-gateway",
            "engines": search.get("engines") or ["search-gateway"],
            "score": round(max(_document_ranking_score(p) for p in passages), 6),
            "publishedDate": published,
            "modifiedDate": modified,
            "declaredDate": declared,
            "source_type": search.get("source_type") or profile["source_type"],
            "source_tier": search.get("source_tier") or profile["source_tier"],
            "authority_score": search.get(
                "authority_score", profile["authority_score"]
            ),
            "primary_source_candidate": search.get(
                "primary_source_candidate", profile["primary_source_candidate"]
            ),
            "source_classification_method": search.get(
                "source_classification_method", profile["classification_method"]
            ),
            "freshness_score": search.get("freshness_score", 0.0),
            "version_context": version_context,
            "evidence_id": evidence_id,
            "citation_url": url,
            "citation": {"id": evidence_id, "url": url},
            "evidence": evidence_records,
            "extraction_method": page.get("extraction_method"),
            "content_chars": page.get("content_chars"),
            "fetch_strategy": page.get("fetch_strategy"),
            "fetch_assessment": page.get("fetch_assessment"),
            "extraction_failures": page.get("extraction_errors") or [],
            "low_confidence": bool(page.get("low_confidence")),
            "img_src": search.get("image_url"),
            "thumbnail_src": search.get("thumbnail_url"),
        }
        owner = _root_domain(_domain(str(item["url"])))
        if owner and owner in owners:
            deferred.append(item)
        else:
            owners.add(owner)
            output.append(item)
        if len(output) >= limit:
            break
    for item in deferred:
        if len(output) >= limit:
            break
        output.append(item)
    return output[:limit]


def _cache_key(request: SearchRequest, pipeline: str = "research") -> str:
    payload = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "pipeline": pipeline,
            "request": request.model_dump(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _redis_client():
    global _REDIS
    if not REDIS_URL:
        return None
    if _REDIS is None:
        try:
            import redis.asyncio as redis_async

            _REDIS = redis_async.from_url(REDIS_URL, decode_responses=True)
        except Exception:
            return None
    return _REDIS


async def _cache_get(key: str) -> dict[str, Any] | None:
    now = time.time()
    async with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached and now - cached["cached_at"] <= CACHE_STALE_SECONDS:
            _CACHE.move_to_end(key)
            return cached
    client = await _redis_client()
    if client is not None:
        try:
            raw = await asyncio.wait_for(client.get(f"search-gateway:v1:{key}"), 0.25)
            cached = json.loads(raw) if raw else None
            if (
                isinstance(cached, dict)
                and now - float(cached.get("cached_at", 0)) <= CACHE_STALE_SECONDS
            ):
                async with _CACHE_LOCK:
                    _CACHE[key] = cached
                return cached
        except Exception as exc:
            LOGGER.debug("Redis cache read failed: %s", type(exc).__name__)
    return None


async def _cache_set(key: str, result: dict[str, Any]) -> None:
    cached = {"cached_at": time.time(), "result": result}
    async with _CACHE_LOCK:
        _CACHE[key] = cached
        _CACHE.move_to_end(key)
        while len(_CACHE) > CACHE_MAX_ENTRIES:
            _CACHE.popitem(last=False)
    client = await _redis_client()
    if client is not None:
        with suppress(Exception):
            await asyncio.wait_for(
                client.set(
                    f"search-gateway:v1:{key}",
                    json.dumps(cached, ensure_ascii=False),
                    ex=CACHE_STALE_SECONDS,
                ),
                0.25,
            )


def _searx_result(item: dict[str, Any]) -> dict[str, Any]:
    """Map an internal candidate back to the fields SearXNG clients expect."""

    return {
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "content": item.get("snippet", ""),
        "snippet": item.get("snippet", ""),
        "engine": "search-gateway",
        "engines": item.get("engines") or ["search-gateway"],
        "score": item.get("discovery_score", 0.0),
        "publishedDate": item.get("published_at"),
        "modifiedDate": item.get("modified_at"),
        "source_type": item.get("source_type"),
        "source_tier": item.get("source_tier"),
        "authority_score": item.get("authority_score"),
        "primary_source_candidate": item.get("primary_source_candidate", False),
        "freshness_score": item.get("freshness_score", 0.0),
        "version_context": item.get("version_context") or [],
        "evidence_id": item.get("evidence_id") or stable_evidence_id(item.get("url")),
        "citation_url": item.get("citation_url") or item.get("url"),
        "img_src": item.get("image_url"),
        "thumbnail_src": item.get("thumbnail_url"),
        "category": item.get("category"),
        "template": item.get("template"),
        "parsed_url": item.get("parsed_url"),
    }


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _search_payload(
    query: str,
    candidates: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    started: float,
    *,
    cache: str,
) -> dict[str, Any]:
    results = [_searx_result(item) for item in candidates]
    return {
        "query": query,
        "number_of_results": len(results),
        "results": results,
        "suggestions": [],
        "answers": [],
        "corrections": [],
        "infoboxes": [],
        "unresponsive_engines": [
            item
            for search in diagnostics.get("searches", [])
            for item in search.get("unresponsive_engines", [])
        ],
        "diagnostics": diagnostics,
        "duration_seconds": round(time.monotonic() - started, 3),
        "cache": cache,
    }


def _attach_search_diagnostics(
    diagnostics: dict[str, Any], searches: list[dict[str, Any]]
) -> None:
    diagnostics["searches"] = searches
    summary = next(
        (
            item
            for item in reversed(searches)
            if item.get("provider") == "search-gateway"
            and item.get("wave") == "summary"
        ),
        None,
    )
    if not summary:
        return
    diagnostics["planner"] = summary.get("planner")
    diagnostics["query_variants"] = summary.get("variants") or []
    diagnostics["fallback_triggered"] = bool(summary.get("fallback_triggered"))
    diagnostics["fallback_reasons"] = summary.get("fallback_reasons") or []
    diagnostics["initial_search_quality"] = summary.get("initial_quality") or {}
    diagnostics["final_search_quality"] = summary.get("final_quality") or {}


async def discovery_search(request: SearchRequest) -> dict[str, Any]:
    """Return bounded SearXNG discovery results without crawling pages."""

    started = time.monotonic()
    direct_candidates = _direct_url_candidates(request.query, request.max_results)
    if direct_candidates:
        return _search_payload(
            request.query,
            direct_candidates,
            {
                "mode": "direct",
                "pipeline": "discovery",
                "partial": False,
                "direct_url_count": len(direct_candidates),
                "searches": [],
            },
            started,
            cache="bypass",
        )

    key = _cache_key(request, "search")
    cached = await _cache_get(key)
    if cached and time.time() - cached["cached_at"] <= CACHE_TTL_SECONDS:
        result = dict(cached["result"])
        result["cache"] = "fresh"
        return result

    lock = _QUERY_LOCKS.setdefault(key, asyncio.Lock())
    _QUERY_LOCK_USERS[key] = _QUERY_LOCK_USERS.get(key, 0) + 1
    try:
        async with lock:
            cached = await _cache_get(key)
            if cached and time.time() - cached["cached_at"] <= CACHE_TTL_SECONDS:
                result = dict(cached["result"])
                result["cache"] = "fresh-coalesced"
                return result

            acquired = False
            try:
                await asyncio.wait_for(_ADMISSION.acquire(), ADMISSION_TIMEOUT_SECONDS)
                acquired = True
            except TimeoutError as exc:
                raise GatewayBusyError("Search gateway is at capacity") from exc

            try:
                started = time.monotonic()
                mode = _mode_for(request.query, request.mode)
                diagnostics: dict[str, Any] = {
                    "mode": mode,
                    "pipeline": "discovery",
                    "partial": False,
                    "topic_anchor": _topic_anchor(request.query),
                    "topic_strict": _topic_anchor_is_strict(
                        request.query, _topic_anchor(request.query)
                    ),
                }
                timed_out = False
                try:
                    discovery_timeout = _discovery_timeout_seconds(mode)
                    async with asyncio.timeout(max(discovery_timeout, 3.0)):
                        candidates, search_diagnostics = await _searx_search(
                            request.query,
                            mode=mode,
                            max_results=min(MAX_SEARCH_RESULTS, request.max_results),
                            language=request.language,
                            time_range=request.time_range,
                            categories=request.categories,
                        )
                    _attach_search_diagnostics(diagnostics, search_diagnostics)
                except TimeoutError:
                    diagnostics["partial"] = True
                    diagnostics["deadline_exceeded"] = True
                    timed_out = True
                    candidates, search_diagnostics = [], []
                    _attach_search_diagnostics(diagnostics, search_diagnostics)
                if timed_out:
                    raise TimeoutError("SearXNG discovery deadline exceeded")
                searx_runs = [
                    item
                    for item in diagnostics.get("searches", [])
                    if item.get("provider") == "searxng"
                ]
                if (
                    candidates == []
                    and searx_runs
                    and all(item.get("status") == "failed" for item in searx_runs)
                ):
                    raise RuntimeError("SearXNG discovery failed")
                result = _search_payload(
                    request.query,
                    candidates[: request.max_results],
                    diagnostics,
                    started,
                    cache="miss",
                )
                await _cache_set(key, result)
                return result
            finally:
                if acquired:
                    _ADMISSION.release()
    except Exception as exc:
        if cached and cached.get("result"):
            result = dict(cached["result"])
            result["cache"] = "stale-fallback"
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["refresh_error"] = type(exc).__name__
            result["diagnostics"] = diagnostics
            return result
        raise
    finally:
        users = _QUERY_LOCK_USERS.get(key, 1) - 1
        if users <= 0:
            _QUERY_LOCK_USERS.pop(key, None)
            _QUERY_LOCKS.pop(key, None)
        else:
            _QUERY_LOCK_USERS[key] = users


def _scrape_cache_key(url: str) -> str:
    payload = json.dumps(
        {"schema": CACHE_SCHEMA_VERSION, "pipeline": "scrape", "url": url},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _firecrawl_scrape_payload(
    page: dict[str, Any], requested_url: str
) -> dict[str, Any]:
    source_url = str(page.get("url") or requested_url)
    page_metadata = normalize_page_metadata(page.get("metadata"))
    declared_lines = []
    if page_metadata.get("publishedDate"):
        declared_lines.append(f"Published: {page_metadata['publishedDate']}")
    if page_metadata.get("modifiedDate"):
        declared_lines.append(f"Modified: {page_metadata['modifiedDate']}")
    if page_metadata.get("declaredDate"):
        declared_lines.append(f"Declared date: {page_metadata['declaredDate']}")
    if page_metadata.get("version_context"):
        declared_lines.append(
            "Version context: " + ", ".join(page_metadata["version_context"][:8])
        )
    metadata_block = ""
    if declared_lines:
        metadata_block = (
            "> Page-declared metadata (not independently verified)\n> "
            + "\n> ".join(declared_lines)
            + "\n\n"
        )
    raw_content = metadata_block + str(page.get("content") or "")
    content = _truncate_utf8(
        raw_content[:FIRECRAWL_MAX_CONTENT_CHARS], FIRECRAWL_MAX_RESPONSE_BYTES
    )
    metadata = {
        "title": page.get("title") or "",
        "sourceURL": source_url,
        "statusCode": page.get("status_code"),
        "evidenceId": stable_evidence_id(source_url),
    }
    published = page_metadata.get("publishedDate")
    modified = page_metadata.get("modifiedDate")
    declared = page_metadata.get("declaredDate")
    if published:
        metadata["publishedTime"] = published
        metadata["article:published_time"] = published
    if modified:
        metadata["modifiedTime"] = modified
        metadata["article:modified_time"] = modified
    if declared:
        metadata["date"] = declared
    for key in ("language", "author", "description", "version_context"):
        if page_metadata.get(key):
            metadata[key] = page_metadata[key]
    data: dict[str, Any] = {"markdown": content, "metadata": metadata}
    links = page.get("links")
    if isinstance(links, list):
        data["links"] = [
            item.get("url")
            for item in links
            if isinstance(item, dict) and item.get("url")
        ][:500]
    return {"success": True, "data": data}


async def scrape_page(url: str, timeout_seconds: float) -> dict[str, Any]:
    """Fetch one public page with cache, coalescing, and admission control."""

    canonical_url = _canonical_url(url)
    if canonical_url is None:
        raise ValueError("Invalid scrape URL")
    key = _scrape_cache_key(canonical_url)
    cached = await _cache_get(key)
    if cached and time.time() - cached["cached_at"] <= CACHE_TTL_SECONDS:
        return dict(cached["result"])

    lock = _QUERY_LOCKS.setdefault(key, asyncio.Lock())
    _QUERY_LOCK_USERS[key] = _QUERY_LOCK_USERS.get(key, 0) + 1
    try:
        async with lock:
            cached = await _cache_get(key)
            if cached and time.time() - cached["cached_at"] <= CACHE_TTL_SECONDS:
                return dict(cached["result"])

            acquired = False
            try:
                await asyncio.wait_for(_ADMISSION.acquire(), ADMISSION_TIMEOUT_SECONDS)
                acquired = True
            except TimeoutError as exc:
                raise GatewayBusyError("Search gateway is at capacity") from exc
            try:
                page = await fetch_page(
                    canonical_url,
                    "Extract the useful information from this page.",
                    timeout_seconds,
                    allow_expensive_fallback=True,
                )
                result = _firecrawl_scrape_payload(page, canonical_url)
                await _cache_set(key, result)
                return result
            finally:
                if acquired:
                    _ADMISSION.release()
    except Exception:
        if cached and cached.get("result"):
            return dict(cached["result"])
        raise
    finally:
        users = _QUERY_LOCK_USERS.get(key, 1) - 1
        if users <= 0:
            _QUERY_LOCK_USERS.pop(key, None)
            _QUERY_LOCKS.pop(key, None)
        else:
            _QUERY_LOCK_USERS[key] = users


async def research(
    request: SearchRequest,
    *,
    budget_override: Budget | None = None,
    pipeline: str = "research",
) -> dict[str, Any]:
    key = _cache_key(request, pipeline)
    cached = await _cache_get(key)
    if cached and time.time() - cached["cached_at"] <= CACHE_TTL_SECONDS:
        result = dict(cached["result"])
        result["cache"] = "fresh"
        return result

    lock = _QUERY_LOCKS.setdefault(key, asyncio.Lock())
    _QUERY_LOCK_USERS[key] = _QUERY_LOCK_USERS.get(key, 0) + 1
    try:
        async with lock:
            cached = await _cache_get(key)
            if cached and time.time() - cached["cached_at"] <= CACHE_TTL_SECONDS:
                result = dict(cached["result"])
                result["cache"] = "fresh-coalesced"
                return result

            acquired = False
            try:
                await asyncio.wait_for(_ADMISSION.acquire(), ADMISSION_TIMEOUT_SECONDS)
                acquired = True
            except TimeoutError as exc:
                raise GatewayBusyError("Search gateway is at capacity") from exc

            try:
                started = time.monotonic()
                mode = _mode_for(request.query, request.mode)
                budget = budget_override or MODE_BUDGETS[mode]
                deadline = started + budget.total_seconds
                diagnostics: dict[str, Any] = {
                    "mode": mode,
                    "pipeline": pipeline,
                    "partial": False,
                }
                candidates: list[dict[str, Any]] = []
                fallback_candidates: list[dict[str, Any]] = []
                pages: list[dict[str, Any]] = []
                results: list[dict[str, Any]] = []
                try:
                    async with asyncio.timeout(budget.total_seconds):
                        direct_candidates = _direct_url_candidates(
                            request.query,
                            min(MAX_SEARCH_RESULTS, budget.search_results),
                        )
                        if direct_candidates:
                            candidates = direct_candidates
                            search_diagnostics = []
                            diagnostics["direct_url_count"] = len(candidates)
                        else:
                            candidates, search_diagnostics = await _searx_search(
                                request.query,
                                mode=mode,
                                max_results=min(
                                    MAX_SEARCH_RESULTS, budget.search_results
                                ),
                                language=request.language,
                                time_range=request.time_range,
                                categories=request.categories,
                            )
                        _attach_search_diagnostics(diagnostics, search_diagnostics)
                        if not candidates:
                            raise RuntimeError("SearXNG returned no usable candidates")
                        fallback_candidates = candidates

                        candidate_documents = []
                        for index, item in enumerate(candidates):
                            context = [
                                str(item["title"]),
                                str(item["snippet"]),
                                f"Source type: {item.get('source_type') or 'general_web'}",
                            ]
                            if item.get("published_at"):
                                context.append(f"Published: {item['published_at']}")
                            if item.get("modified_at"):
                                context.append(f"Modified: {item['modified_at']}")
                            if item.get("version_context"):
                                context.append(
                                    "Version context: "
                                    + ", ".join(item["version_context"][:8])
                                )
                            candidate_documents.append(
                                {
                                    "text": "\n".join(context),
                                    "candidate_index": index,
                                    "authority_score": item.get(
                                        "source_authority", 0.0
                                    ),
                                }
                            )
                        crawl_deadline = deadline - FINALIZATION_RESERVE_SECONDS
                        candidate_rerank_timeout = min(
                            CANDIDATE_RERANKER_TIMEOUT_SECONDS,
                            max(0.0, crawl_deadline - time.monotonic() - 1.0),
                        )
                        target_pages = min(
                            MAX_CRAWL_PAGES,
                            budget.crawl_pages,
                            max(1, request.max_results),
                        )
                        candidate_pool_size = min(
                            len(candidates), 15, max(target_pages * 5, 8)
                        )
                        ranked_candidates, candidate_reranker = await _rerank_bounded(
                            request.query,
                            candidate_documents,
                            candidate_pool_size,
                            candidate_rerank_timeout,
                        )
                        diagnostics["candidate_reranker"] = candidate_reranker
                        selected = _select_crawl_candidates(
                            request.query,
                            candidates,
                            ranked_candidates,
                            target_pages,
                            candidate_pool_size,
                        )
                        selected_urls = {item["url"] for item in selected}
                        fallback_candidates = [
                            *selected,
                            *[
                                item
                                for item in candidates
                                if item["url"] not in selected_urls
                            ],
                        ]
                        (
                            pages,
                            failures,
                            crawl_batches,
                        ) = await _adaptive_crawl_candidates(
                            selected,
                            request.query,
                            target_pages,
                            crawl_deadline,
                        )
                        diagnostics["crawl_failures"] = failures
                        diagnostics["crawl_batches"] = crawl_batches
                        diagnostics["crawl_target_pages"] = target_pages
                        diagnostics["crawl_successful_pages"] = len(pages)
                        follow_limit = min(MAX_FOLLOW_LINKS, budget.follow_links)
                        if (
                            pages
                            and follow_limit
                            and _pages_need_follow_up(
                                pages, request.query, target_pages
                            )
                            and crawl_deadline - time.monotonic() > 2
                        ):
                            followed, follow_failures = await _follow_relevant_links(
                                pages, request.query, follow_limit, crawl_deadline
                            )
                            pages.extend(followed)
                            diagnostics["follow_failures"] = follow_failures

                        documents = _passage_documents(pages, request.query)
                        if documents:
                            ranked, reranker_status = await _rerank_bounded(
                                request.query,
                                documents,
                                max(20, request.max_results * 5),
                                max(0.0, deadline - time.monotonic()),
                            )
                        else:
                            ranked, reranker_status = [], "empty"
                        diagnostics["reranker"] = reranker_status
                        results = _assemble_results(
                            pages,
                            ranked,
                            min(request.max_results, budget.final_results),
                            request.query,
                        )
                except TimeoutError:
                    diagnostics["partial"] = True
                    diagnostics["deadline_exceeded"] = True
                    documents = _passage_documents(pages, request.query)
                    if documents:
                        ranked = _lexical_rerank(
                            request.query, documents, max(20, request.max_results * 5)
                        )
                        diagnostics["reranker"] = "fallback:deadline"
                        results = _assemble_results(
                            pages,
                            ranked,
                            min(request.max_results, budget.final_results),
                            request.query,
                        )

                if len(results) < min(request.max_results, budget.final_results):
                    existing_urls = {str(item.get("url") or "") for item in results}
                    snippets = [
                        {
                            "title": item["title"],
                            "url": item["url"],
                            "content": item["snippet"],
                            "snippet": item["snippet"],
                            "engine": "search-gateway-snippet-fallback",
                            "engines": item["engines"],
                            "score": item["discovery_score"],
                            "publishedDate": item["published_at"],
                            "modifiedDate": item.get("modified_at"),
                            "source_type": item.get("source_type"),
                            "source_tier": item.get("source_tier"),
                            "authority_score": item.get("authority_score"),
                            "primary_source_candidate": item.get(
                                "primary_source_candidate", False
                            ),
                            "source_classification_method": item.get(
                                "source_classification_method"
                            ),
                            "freshness_score": item.get("freshness_score", 0.0),
                            "version_context": item.get("version_context") or [],
                            "evidence_id": item.get("evidence_id")
                            or stable_evidence_id(item["url"]),
                            "citation_url": item.get("citation_url") or item["url"],
                            "citation": {
                                "id": item.get("evidence_id")
                                or stable_evidence_id(item["url"]),
                                "url": item.get("citation_url") or item["url"],
                            },
                            "img_src": item.get("image_url"),
                            "thumbnail_src": item.get("thumbnail_url"),
                        }
                        for item in (fallback_candidates or candidates)
                        if item.get("snippet") and item["url"] not in existing_urls
                    ]
                    needed = min(request.max_results, budget.final_results) - len(
                        results
                    )
                    if snippets:
                        results.extend(snippets[:needed])
                        diagnostics["partial"] = True
                        diagnostics["fallback"] = "search-snippets"

                if not results:
                    raise RuntimeError("Search produced no usable evidence")

                coverage = evidence_summary(
                    results, request.query, time_range=request.time_range
                )
                strong_page_count = sum(
                    not bool(page.get("low_confidence")) for page in pages
                )
                if strong_page_count == 0:
                    coverage = dict(coverage)
                    coverage["status"] = "weak"
                    warnings = list(coverage.get("warnings") or [])
                    warning = (
                        "No high-confidence page content could be extracted; results "
                        "rely on search-engine snippets or low-confidence API content."
                    )
                    if warning not in warnings:
                        warnings.append(warning)
                    coverage["warnings"] = warnings
                diagnostics["evidence_status"] = coverage["status"]
                result = {
                    "query": request.query,
                    "number_of_results": len(results),
                    "results": results,
                    "suggestions": [],
                    "answers": [],
                    "corrections": [],
                    "infoboxes": [],
                    "unresponsive_engines": [
                        item
                        for search in diagnostics.get("searches", [])
                        for item in search.get("unresponsive_engines", [])
                    ],
                    "diagnostics": diagnostics,
                    "evidence_summary": coverage,
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "cache": "miss",
                }
                await _cache_set(key, result)
                return result
            finally:
                if acquired:
                    _ADMISSION.release()

    except Exception as exc:
        if cached and cached.get("result"):
            result = dict(cached["result"])
            result["cache"] = "stale-fallback"
            diagnostics = dict(result.get("diagnostics") or {})
            diagnostics["refresh_error"] = type(exc).__name__
            result["diagnostics"] = diagnostics
            return result
        raise
    finally:
        users = _QUERY_LOCK_USERS.get(key, 1) - 1
        if users <= 0:
            _QUERY_LOCK_USERS.pop(key, None)
            _QUERY_LOCKS.pop(key, None)
        else:
            _QUERY_LOCK_USERS[key] = users


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _REDIS
    try:
        yield
    finally:
        client = _REDIS
        _REDIS = None
        with suppress(Exception):
            if client is not None:
                await client.aclose()
        with suppress(Exception):
            await close_fetch_resources()


app = FastAPI(
    title="Private Evidence Search Gateway",
    version="1.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)


@app.get("/healthz")
async def healthz(http_response: Response) -> dict[str, Any]:
    checks = {}
    async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
        for name, url in (
            ("searxng", f"{SEARXNG_URL}/"),
            ("reranker", f"{RERANKER_URL}/health"),
        ):
            try:
                upstream = await client.get(url)
                checks[name] = upstream.status_code < 500
            except Exception:
                checks[name] = False
    ok = checks.get("searxng", False)
    if not ok:
        http_response.status_code = 503
    return {"ok": ok, "checks": checks}


@app.get("/search")
async def searx_compatible_search(
    q: str = Query(min_length=1, max_length=QUERY_MAX_CHARS),
    format: Literal["json"] = "json",
    language: str = Query(default="auto", min_length=1, max_length=32),
    time_range: Literal["day", "week", "month", "year"] | None = None,
    categories: str = Query(default="", max_length=200),
    pageno: int = Query(default=1, ge=1, le=1),
    max_results: int = Query(default=5, ge=1, le=MAX_FINAL_RESULTS),
    mode: Literal["auto", "quick", "balanced", "deep"] = "auto",
) -> dict[str, Any]:
    del format, pageno
    category_list = [item.strip() for item in categories.split(",") if item.strip()][:8]
    try:
        return await discovery_search(
            DiscoveryRequest(
                query=q,
                mode=mode,
                max_results=max_results,
                language=language,
                time_range=time_range,
                categories=category_list,
            )
        )
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Search gateway deadline exceeded"
        ) from exc
    except GatewayBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search gateway is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        LOGGER.warning("Search request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Search retrieval failed") from exc


@app.get("/integrated/search")
async def integrated_search(
    q: str = Query(min_length=1, max_length=QUERY_MAX_CHARS),
    format: Literal["json"] = "json",
    language: str = Query(default="auto", min_length=1, max_length=32),
    time_range: Literal["day", "week", "month", "year"] | None = None,
    categories: str = Query(default="", max_length=200),
    max_results: int = Query(default=5, ge=1, le=MAX_FINAL_RESULTS),
    mode: Literal["auto", "quick", "balanced", "deep"] = "auto",
) -> dict[str, Any]:
    """AnythingLLM-friendly bounded search plus page extraction."""

    del format
    category_list = [item.strip() for item in categories.split(",") if item.strip()][:8]
    request = SearchRequest(
        query=q,
        mode=mode,
        max_results=min(max_results, INTEGRATED_MAX_RESULTS),
        language=language,
        time_range=time_range,
        categories=category_list,
    )
    budget = Budget(
        search_results=min(MAX_SEARCH_RESULTS, INTEGRATED_MAX_SEARCH_RESULTS),
        crawl_pages=min(MAX_CRAWL_PAGES, INTEGRATED_MAX_CRAWL_PAGES),
        follow_links=0,
        final_results=min(MAX_FINAL_RESULTS, INTEGRATED_MAX_RESULTS),
        total_seconds=INTEGRATED_TIMEOUT_SECONDS,
    )
    try:
        return await research(request, budget_override=budget, pipeline="integrated")
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Integrated search deadline exceeded"
        ) from exc
    except GatewayBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search gateway is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        LOGGER.warning("Integrated search request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Integrated search failed") from exc


def _require_compat_auth(
    authorization: str | None,
    x_api_key: str | None,
) -> None:
    """Require the configured frontend credential for compatibility APIs."""

    if not FIRECRAWL_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gateway compatibility APIs are not configured",
        )
    supplied = ""
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() == "bearer":
            supplied = token.strip()
    if not supplied and x_api_key:
        supplied = x_api_key.strip()
    if not supplied or not hmac.compare_digest(supplied, FIRECRAWL_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid gateway API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/v1/rerank")
async def jina_compatible_rerank(
    request: JinaRerankRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Jina-compatible adapter backed by the stack's local BGE reranker."""

    _require_compat_auth(authorization, x_api_key)
    documents = [
        {"text": text, "document_index": index}
        for index, text in enumerate(request.documents)
    ]
    ranked, status = await _rerank(
        request.query,
        documents,
        min(request.top_n, len(documents)),
    )
    results: list[dict[str, Any]] = []
    for item in ranked:
        index = int(item["document_index"])
        row: dict[str, Any] = {
            "index": index,
            "relevance_score": float(item.get("rerank_score") or 0.0),
        }
        if request.return_documents:
            row["document"] = {"text": request.documents[index]}
        results.append(row)
    return {
        "model": RERANKER_MODEL,
        "results": results,
        "backend_status": status,
    }


@app.post("/v2/scrape")
async def firecrawl_scrape(
    request: FirecrawlScrapeRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Firecrawl v2-compatible scrape backed by the protected fetch pipeline."""

    _require_compat_auth(authorization, x_api_key)
    formats = {str(item).strip() for item in request.formats if str(item).strip()}
    if formats and "markdown" not in formats:
        raise HTTPException(
            status_code=422,
            detail="This compatible scraper supports the markdown format",
        )
    timeout_seconds = FIRECRAWL_TIMEOUT_SECONDS
    if request.timeout is not None:
        timeout_seconds = min(
            FIRECRAWL_MAX_TIMEOUT_SECONDS,
            max(1.0, request.timeout / 1000.0),
        )
    try:
        async with asyncio.timeout(timeout_seconds):
            return await scrape_page(request.url, timeout_seconds)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Scrape deadline exceeded") from exc
    except GatewayBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search gateway is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid scrape URL") from exc
    except Exception as exc:
        detail = re.sub(r"[\r\n]+", " ", str(exc)).strip()[:500]
        LOGGER.warning(
            "Firecrawl scrape failed: %s%s",
            type(exc).__name__,
            f": {detail}" if detail else "",
        )
        raise HTTPException(status_code=502, detail="Page scrape failed") from exc


@app.post("/v2/search")
async def firecrawl_search(
    request: FirecrawlSearchRequest,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict[str, Any]:
    """Firecrawl v2-compatible discovery search backed by SearXNG."""

    _require_compat_auth(authorization, x_api_key)
    search_request = DiscoveryRequest(
        query=request.query,
        mode="quick",
        max_results=min(request.limit, MAX_SEARCH_RESULTS),
        language=request.lang,
        time_range=None,
        categories=[],
    )
    try:
        result = await discovery_search(search_request)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Search deadline exceeded") from exc
    except GatewayBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search gateway is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        LOGGER.warning("Firecrawl search failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Search retrieval failed") from exc
    web = [
        {
            "url": item.get("url", ""),
            "title": item.get("title", ""),
            "description": item.get("snippet", ""),
        }
        for item in result.get("results", [])
    ]
    return {"success": True, "data": {"web": web}}


@app.post("/v1/research")
async def rich_research(request: SearchRequest) -> dict[str, Any]:
    try:
        return await research(request)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504, detail="Search gateway deadline exceeded"
        ) from exc
    except GatewayBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search gateway is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        LOGGER.warning("Research request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=502, detail="Research retrieval failed"
        ) from exc


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "private-evidence-search-gateway",
        "search": "/search?q=your+question&format=json",
        "integrated_search": "/integrated/search?q=your+question&format=json",
        "firecrawl": "/v2/scrape",
        "reranker": "/v1/rerank",
        "health": "/healthz",
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Docker publishes no host port; all-interface binding is container-internal.
    uvicorn.run(app, host="0.0.0.0", port=8080, proxy_headers=False)  # nosec B104
