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
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

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
from gateway_fetch import fetch_page
from request_limits import RequestBodyLimitMiddleware


LOGGER = logging.getLogger("search-gateway")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000").rstrip("/")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()
REQUEST_TIMEOUT_SECONDS = max(
    5.0, float(os.getenv("GATEWAY_REQUEST_TIMEOUT_SECONDS", "28"))
)
SEARCH_TIMEOUT_SECONDS = max(
    2.0, float(os.getenv("GATEWAY_SEARCH_TIMEOUT_SECONDS", "9"))
)
PAGE_TIMEOUT_SECONDS = max(
    3.0, float(os.getenv("GATEWAY_PAGE_TIMEOUT_SECONDS", "18"))
)
MAX_SEARCH_RESULTS = max(8, int(os.getenv("GATEWAY_MAX_SEARCH_RESULTS", "30")))
MAX_CRAWL_PAGES = max(1, int(os.getenv("GATEWAY_MAX_CRAWL_PAGES", "8")))
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
MAX_CONCURRENT_RERANKS = max(
    1, int(os.getenv("GATEWAY_MAX_CONCURRENT_RERANKS", "2"))
)
RERANKER_ADMISSION_TIMEOUT_SECONDS = max(
    0.01, float(os.getenv("GATEWAY_RERANKER_ADMISSION_TIMEOUT_SECONDS", "0.25"))
)
MAX_REQUEST_BYTES = max(4096, int(os.getenv("GATEWAY_MAX_REQUEST_BYTES", "262144")))
QUERY_MAX_CHARS = max(256, int(os.getenv("GATEWAY_QUERY_MAX_CHARS", "4000")))
MAX_CONCURRENT_REQUESTS = max(
    1, int(os.getenv("GATEWAY_MAX_CONCURRENT_REQUESTS", "4"))
)
ADMISSION_TIMEOUT_SECONDS = max(
    0.05, float(os.getenv("GATEWAY_ADMISSION_TIMEOUT_SECONDS", "2"))
)
FINALIZATION_RESERVE_SECONDS = max(
    0.25, float(os.getenv("GATEWAY_FINALIZATION_RESERVE_SECONDS", "3"))
)
CANDIDATE_RERANKER_TIMEOUT_SECONDS = max(
    0.25, float(os.getenv("GATEWAY_CANDIDATE_RERANKER_TIMEOUT_SECONDS", "2"))
)

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
INTEGRATED_MAX_RESULTS = max(
    1, int(os.getenv("GATEWAY_INTEGRATED_MAX_RESULTS", "5"))
)
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
FIRECRAWL_MAX_RESULTS = max(
    1, int(os.getenv("FIRECRAWL_MAX_RESULTS", "20"))
)
CACHE_SCHEMA_VERSION = "quality-v5"


_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CACHE_LOCK = asyncio.Lock()
_QUERY_LOCKS: dict[str, asyncio.Lock] = {}
_QUERY_LOCK_USERS: dict[str, int] = {}
_ADMISSION = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
_RERANK_ADMISSION = asyncio.Semaphore(MAX_CONCURRENT_RERANKS)
_REDIS = None


TECHNICAL_RE = re.compile(
    r"\b(?:api|compose|config(?:uration)?|container|docker|error|exception|fix|github|"
    r"install|kubernetes|linux|log|manual|postgres|server|setup|stack trace|vps)\b",
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
DOCKER_DOCUMENTATION_QUERY_RE = re.compile(
    r"(?:\b(?:configure|fix|install|setup|troubleshoot(?:ing)?)\s+"
    r"(?:docker(?:\s+compose)?|docker-compose)\b|"
    r"\b(?:docker(?:\s+compose)?|docker-compose)\s+"
    r"(?:config(?:uration)?|error|exception|fix|guide|install(?:ation)?|manual|"
    r"setup|troubleshoot(?:ing)?)\b|\bcompose\.ya?ml\b|"
    r"\bcompose\s+(?:file|plugin|stack)\b)",
    re.I,
)
CURRENT_RE = re.compile(
    r"\b(?:current|latest|new|news|patch|release|recent|today|tonight|version)\b",
    re.I,
)
ACADEMIC_RE = re.compile(
    r"\b(?:academic|clinical|journal|paper|preprint|research|scientific|study)\b",
    re.I,
)
IMAGE_RE = re.compile(r"\b(?:image|images|photo|photos|picture|pictures|wallpaper)\b", re.I)
GAME_RE = re.compile(
    r"\b(?:game|gaming|team composition|tier list|character build|party composition)\b",
    re.I,
)
GAME_CONTEXT_RE = re.compile(
    r"(?:\bawakening\b.*\b(?:armor|build|character|equipment|gear|karma|party|team|"
    r"weapon)\b|\b(?:armor|build|character|equipment|gear|karma|party|team|weapon)\b"
    r".*\bawakening\b)",
    re.I,
)
RECOMMENDATION_RE = re.compile(
    r"\b(?:best|compare|comparison|recommend|recommended|settings|team composition|tier list)\b",
    re.I,
)
ERROR_QUOTE_RE = re.compile(r"(?:error|exception|failed|failure)[:\s]+(.{8,220})", re.I)
URL_IN_QUERY_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
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
SOURCE_DOMAIN_BOOSTS = {
    "docs.docker.com": 1.4,
    "stackoverflow.com": 0.65,
    "serverfault.com": 0.65,
    "superuser.com": 0.55,
    "reddit.com": 0.15,
}
SOURCE_PENALTIES = {
    "hub.docker.com": -0.35,
    "pinterest.com": -4.0,
    "facebook.com": -3.0,
    "instagram.com": -3.0,
    "tiktok.com": -3.0,
    "fandom.com": -1.0,
}
AUTHORITY_RANK_WEIGHT = 0.12
BROAD_WEB_ENGINES = ("startpage", "bing", "brave")
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
    "quick": Budget(12, 3, 0, 4, min(14.0, REQUEST_TIMEOUT_SECONDS)),
    "balanced": Budget(24, 6, 1, 6, REQUEST_TIMEOUT_SECONDS),
    "deep": Budget(30, 8, 2, 8, max(REQUEST_TIMEOUT_SECONDS, 40.0)),
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
        if len(token) < 2 or token in STOP_WORDS or token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output[:80]


def _query_variants(query: str, mode: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", query).strip()
    terms = _tokens(normalized)
    compact = " ".join(terms[:16])
    variants = [compact or normalized]
    error = ERROR_QUOTE_RE.search(normalized)
    if error:
        variants.append(f'"{error.group(1).strip()}"')
    if TECHNICAL_RE.search(normalized):
        if DOCKER_DOCUMENTATION_QUERY_RE.search(normalized):
            variants.append(f"{compact or normalized} site:docs.docker.com")
        else:
            variants.append(f"{compact or normalized} official documentation guide")
    elif GAME_RE.search(normalized) or GAME_CONTEXT_RE.search(normalized):
        variants.append(f"{compact or normalized} strategy guide wiki")
    elif RECOMMENDATION_RE.search(normalized):
        variants.append(f"{compact or normalized} review measurements guide")
    elif CURRENT_RE.search(normalized):
        variants.append(f"{compact or normalized} latest")
    elif compact.casefold() != normalized.casefold():
        variants.append(normalized)
    if mode == "deep" and len(variants) < 3:
        variants.append(f"{compact or normalized} authoritative sources")
    return list(dict.fromkeys(variants))[:3]


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
    if scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
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
    return urlunsplit((scheme, display_host, parsed.path or "/", parsed.query, ""))


def _extract_query_urls(query: str) -> list[str]:
    """Return unique HTTP(S) URLs explicitly supplied in a query."""

    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_IN_QUERY_RE.finditer(query):
        candidate = match.group(0).rstrip(".,;:!?")
        for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
            while candidate.endswith(closing) and candidate.count(closing) > candidate.count(opening):
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
    for candidate, value in SOURCE_DOMAIN_BOOSTS.items():
        if domain == candidate or domain.endswith("." + candidate):
            adjustment += value
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
    score = 3.0 * _lexical_score(query, text)
    score += 1.5 * subject_coverage
    score += 1.0 / max(1, rank)
    score += _source_adjustment(domain)
    score += float(profile["authority_adjustment"])
    score += _intent_source_adjustment(query, profile)
    score += 0.55 * max(
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
    if "official documentation" in variant.casefold() and result.get(
        "source_type"
    ) == "documentation_candidate":
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
    subject_coverage = _subject_coverage(query, f"{title} {snippet} {url}")
    if _subject_terms(query) and subject_coverage == 0:
        return None
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
        "discovery_score": round(
            _candidate_score(query, item, rank, time_range), 4
        ),
        "source_authority": round(source_authority, 4),
        "source_type": profile["source_type"],
        "source_tier": profile["source_tier"],
        "authority_score": profile["authority_score"],
        "primary_source_candidate": profile["primary_source_candidate"],
        "source_classification_method": profile["classification_method"],
        "intent_source_adjustment": intent_adjustment,
        "subject_coverage": round(subject_coverage, 4),
        "published_at": published,
        "modified_at": modified,
        "freshness_score": max(
            freshness_score(published, requirement),
            freshness_score(modified, requirement),
        ),
        "version_context": version_context,
        "evidence_id": stable_evidence_id(url),
        "citation_url": url,
        "engines": item.get("engines") or ([item.get("engine")] if item.get("engine") else []),
        "image_url": item.get("img_src") or item.get("image_url"),
        "thumbnail_url": item.get("thumbnail_src") or item.get("thumbnail"),
        "category": item.get("category"),
        "template": item.get("template"),
        "parsed_url": item.get("parsed_url"),
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


async def _searx_search(
    query: str,
    *,
    mode: str,
    max_results: int,
    language: str,
    time_range: str | None,
    categories: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = _query_variants(query, mode)
    diagnostics: list[dict[str, Any]] = []

    async def one(variant: str) -> dict[str, Any]:
        engines = _search_engines(query, categories, variant)
        params: dict[str, str] = {
            "q": variant,
            "format": "json",
            "language": language or "auto",
            "safesearch": "0",
            "engines": ",".join(engines),
        }
        if time_range:
            params["time_range"] = time_range
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(SEARCH_TIMEOUT_SECONDS), trust_env=False
        ) as client:
            async with client.stream("GET", f"{SEARXNG_URL}/search", params=params) as response:
                response.raise_for_status()
                return await _read_json_response(response)

    outcomes = await asyncio.gather(*(one(variant) for variant in variants), return_exceptions=True)
    by_url: dict[str, dict[str, Any]] = {}
    for variant, outcome in zip(variants, outcomes):
        if isinstance(outcome, Exception):
            diagnostics.append({"query": variant, "status": "failed", "error": type(outcome).__name__})
            continue
        raw_results = outcome.get("results") if isinstance(outcome, dict) else None
        diagnostics.append(
            {
                "query": variant,
                "status": "ok",
                "result_count": len(raw_results) if isinstance(raw_results, list) else 0,
                "requested_engines": _search_engines(query, categories, variant),
                "unresponsive_engines": outcome.get("unresponsive_engines", []),
            }
        )
        for rank, item in enumerate(raw_results or [], start=1):
            if not isinstance(item, dict):
                continue
            normalized = _normalize_search_result(item, query, rank, time_range)
            if normalized is None:
                continue
            normalized["query_variant"] = variant
            normalized["discovery_score"] = round(
                normalized["discovery_score"]
                + _variant_source_adjustment(variant, normalized),
                4,
            )
            existing = by_url.get(normalized["url"])
            if existing is None or normalized["discovery_score"] > existing["discovery_score"]:
                by_url[normalized["url"]] = normalized
    results = sorted(by_url.values(), key=lambda item: item["discovery_score"], reverse=True)
    return results[:max_results], diagnostics


def _chunk_text(text: str) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text or "")
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > MAX_PASSAGE_CHARS:
            sentences = re.split(r"(?<=[.!?])\s+", block)
        else:
            sentences = [block]
        for part in sentences:
            if len(current) + len(part) + 2 <= MAX_PASSAGE_CHARS:
                current = f"{current}\n\n{part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part[:MAX_PASSAGE_CHARS]
            if len(chunks) >= 40:
                return chunks
    if current:
        chunks.append(current)
    return chunks[:40]


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
                    if not isinstance(row, dict) or not isinstance(row.get("index"), int):
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
        LOGGER.warning("Reranker unavailable; using lexical fallback: %s", type(exc).__name__)
        return _lexical_rerank(query, documents, top_k), f"fallback:{type(exc).__name__}"
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


async def _crawl_candidates(
    candidates: list[dict[str, Any]], query: str, deadline: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def one(
        candidate_index: int, candidate: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        async with semaphore:
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                return candidate, None
            try:
                page = await fetch_page(
                    candidate["url"],
                    query,
                    min(PAGE_TIMEOUT_SECONDS, remaining),
                    allow_expensive_fallback=candidate_index < 3,
                )
                return candidate, page
            except Exception as exc:
                return candidate, {"error": type(exc).__name__}

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

    pairs = [
        (tasks[task][0], *task.result())
        for task in done
        if not task.cancelled()
    ]
    pairs.sort(key=lambda pair: pair[0])
    pages: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = [
        {"url": tasks[task][1]["url"], "error": "deadline"}
        for task in pending
    ]
    for _, candidate, page in pairs:
        if page is None or page.get("error"):
            failures.append({"url": candidate["url"], "error": (page or {}).get("error", "deadline")})
            continue
        page["search"] = candidate
        pages.append(page)
    return pages, failures


def _follow_link_score(query: str, source_url: str, link: dict[str, str]) -> float:
    url = link.get("url", "")
    if _root_domain(_domain(url)) != _root_domain(_domain(source_url)):
        return -10.0
    path = urlsplit(url).path.casefold()
    text = f"{link.get('anchor', '')} {path}"
    score = 2.0 * _lexical_score(query, text)
    if re.search(r"\b(?:docs?|guide|install|setup|config|manual|troubleshoot|faq|support)\b", text, re.I):
        score += 0.7
    if re.search(r"(?:login|sign.?in|account|privacy|terms|contact|cart)", text, re.I):
        score -= 2.0
    return score


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
        for passage_index, passage in enumerate(_chunk_text(str(page.get("content") or ""))):
            lexical = _lexical_score(query, passage)
            if lexical <= 0 and passage_index > 1:
                continue
            documents.append(
                {
                    "text": passage,
                    "reranker_text": f"{reranker_context}\n\n{passage}",
                    "page_index": page_index,
                    "passage_index": passage_index,
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
            "authority_score": search.get("authority_score", profile["authority_score"]),
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
            "extraction_method": page.get("extraction_method"),
            "content_chars": page.get("content_chars"),
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
            if isinstance(cached, dict) and now - float(cached.get("cached_at", 0)) <= CACHE_STALE_SECONDS:
                async with _CACHE_LOCK:
                    _CACHE[key] = cached
                return cached
        except Exception:
            pass
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
                }
                timed_out = False
                try:
                    async with asyncio.timeout(
                        max(SEARCH_TIMEOUT_SECONDS + 1.0, 3.0)
                    ):
                        candidates, search_diagnostics = await _searx_search(
                            request.query,
                            mode=mode,
                            max_results=min(MAX_SEARCH_RESULTS, request.max_results),
                            language=request.language,
                            time_range=request.time_range,
                            categories=request.categories,
                        )
                    diagnostics["searches"] = search_diagnostics
                except TimeoutError:
                    diagnostics["partial"] = True
                    diagnostics["deadline_exceeded"] = True
                    timed_out = True
                    candidates, search_diagnostics = [], []
                    diagnostics["searches"] = search_diagnostics
                if timed_out:
                    raise TimeoutError("SearXNG discovery deadline exceeded")
                if candidates == [] and diagnostics["searches"] and all(
                    item.get("status") == "failed"
                    for item in diagnostics["searches"]
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
                await asyncio.wait_for(
                    _ADMISSION.acquire(), ADMISSION_TIMEOUT_SECONDS
                )
                acquired = True
            except TimeoutError as exc:
                raise GatewayBusyError("Search gateway is at capacity") from exc

            try:
                started = time.monotonic()
                mode = _mode_for(request.query, request.mode)
                budget = budget_override or MODE_BUDGETS[mode]
                if budget_override is not None:
                    mode = "integrated"
                deadline = started + budget.total_seconds
                diagnostics: dict[str, Any] = {"mode": mode, "partial": False}
                candidates: list[dict[str, Any]] = []
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
                        diagnostics["searches"] = search_diagnostics
                        if not candidates:
                            raise RuntimeError("SearXNG returned no usable candidates")

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
                        ranked_candidates, candidate_reranker = await _rerank_bounded(
                            request.query,
                            candidate_documents,
                            min(MAX_CRAWL_PAGES, budget.crawl_pages),
                            candidate_rerank_timeout,
                        )
                        diagnostics["candidate_reranker"] = candidate_reranker
                        selected = [
                            candidates[int(item["candidate_index"])]
                            for item in ranked_candidates
                            if 0
                            <= int(item.get("candidate_index", -1))
                            < len(candidates)
                        ]
                        pages, failures = await _crawl_candidates(
                            selected, request.query, crawl_deadline
                        )
                        diagnostics["crawl_failures"] = failures
                        follow_limit = min(MAX_FOLLOW_LINKS, budget.follow_links)
                        if pages and follow_limit and crawl_deadline - time.monotonic() > 2:
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
                        for item in candidates
                        if item.get("snippet") and item["url"] not in existing_urls
                    ]
                    needed = min(request.max_results, budget.final_results) - len(results)
                    if snippets:
                        results.extend(snippets[:needed])
                        diagnostics["partial"] = True
                        diagnostics["fallback"] = "search-snippets"

                if not results:
                    raise RuntimeError("Search produced no usable evidence")

                coverage = evidence_summary(
                    results, request.query, time_range=request.time_range
                )
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
    yield
    client = await _redis_client()
    if client is not None:
        with suppress(Exception):
            await client.aclose()


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
        raise HTTPException(status_code=504, detail="Search gateway deadline exceeded") from exc
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
        raise HTTPException(status_code=504, detail="Integrated search deadline exceeded") from exc
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
        raise HTTPException(status_code=504, detail="Search gateway deadline exceeded") from exc
    except GatewayBusyError as exc:
        raise HTTPException(
            status_code=503,
            detail="Search gateway is busy",
            headers={"Retry-After": "1"},
        ) from exc
    except Exception as exc:
        LOGGER.warning("Research request failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Research retrieval failed") from exc


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
