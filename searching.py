import asyncio
import ipaddress
import json
import os
import re
from typing import Any, Dict, List
from urllib.parse import urlsplit

import httpx

from shared import SEARXNG_URL, get_domain
from url_identity import canonicalize_web_url


SEARXNG_MAX_RESPONSE_BYTES = max(
    1024,
    int(os.getenv("SEARXNG_MAX_RESPONSE_BYTES", "4194304")),
)
SEARXNG_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("SEARXNG_TIMEOUT_SECONDS", "30")),
)

RESEARCH_MODE_CONFIG = {
    "quick": {"max_urls": 2, "search_results": 6, "top_k": 4},
    "balanced": {"max_urls": 4, "search_results": 10, "top_k": 6},
    "deep": {"max_urls": 8, "search_results": 16, "top_k": 10},
    "technical": {"max_urls": 6, "search_results": 14, "top_k": 8},
    "academic": {"max_urls": 6, "search_results": 14, "top_k": 8},
    "local_only": {"max_urls": 0, "search_results": 0, "top_k": 8},
    "web_only": {"max_urls": 5, "search_results": 12, "top_k": 0},
}

DOMAIN_BOOSTS = {
    "github.com": 3.0,
    "docs.python.org": 3.0,
    "developer.mozilla.org": 3.0,
    "kubernetes.io": 3.0,
    "docs.docker.com": 3.0,
    "docs.github.com": 3.0,
    "learn.microsoft.com": 2.5,
    "cloud.google.com": 2.2,
    "docs.aws.amazon.com": 2.2,
    "stackoverflow.com": 2.0,
    "serverfault.com": 2.0,
    "superuser.com": 1.8,
    "unix.stackexchange.com": 2.0,
    "askubuntu.com": 1.8,
    "wiki.archlinux.org": 2.5,
    "man7.org": 2.3,
    "mankier.com": 2.0,
    "arxiv.org": 2.2,
    "semanticscholar.org": 2.0,
    "pubmed.ncbi.nlm.nih.gov": 2.0,
    "wikipedia.org": 0.7,
    "fleetguard.com": 2.5,
    "cummins.com": 2.5,
}

DOMAIN_PENALTIES = {
    "pinterest.com": -5.0,
    "quora.com": -2.0,
    "medium.com": -0.8,
    "dev.to": -0.3,
    "fandom.com": -4.0,
    "fiction.live": -5.0,
    "archiveofourown.org": -5.0,
    "reddit.com": -0.5,
    "x.com": -2.0,
    "twitter.com": -2.0,
    "facebook.com": -4.0,
    "instagram.com": -4.0,
    "tiktok.com": -4.0,
}

BLOCKED_DOMAINS = {
    "pinterest.com",
    "fandom.com",
    "fiction.live",
    "archiveofourown.org",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
}

TECHNICAL_DOMAINS = {
    "github.com",
    "stackoverflow.com",
    "serverfault.com",
    "superuser.com",
    "unix.stackexchange.com",
    "askubuntu.com",
    "wiki.archlinux.org",
    "docs.docker.com",
    "kubernetes.io",
    "docs.python.org",
    "developer.mozilla.org",
    "learn.microsoft.com",
    "man7.org",
    "mankier.com",
}

ACADEMIC_DOMAINS = {
    "arxiv.org",
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "doi.org",
    "crossref.org",
}

COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac.uk",
    "co.jp",
    "co.nz",
    "co.uk",
    "com.au",
    "com.br",
    "com.mx",
    "com.sg",
    "edu.au",
    "gov.au",
    "gov.uk",
    "net.au",
    "org.au",
    "org.uk",
}


def normalize_domain(domain: str) -> str:
    domain = (domain or "").lower().strip()
    return domain[4:] if domain.startswith("www.") else domain


def normalize_search_url(url: str) -> str:
    return canonicalize_web_url(url)


def estimate_source_owner_domain(domain: str) -> str:
    """Estimate an owner-level domain without claiming organizational independence."""
    normalized = normalize_domain(domain).rstrip(".")
    host = normalized
    if normalized.startswith("[") and "]" in normalized:
        host = normalized[1 : normalized.index("]")]
    elif normalized.count(":") == 1:
        host = normalized.split(":", 1)[0]
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        normalized = host
    labels = [label for label in normalized.split(".") if label]
    if len(labels) <= 2:
        return normalized
    suffix = ".".join(labels[-2:])
    if suffix in COMMON_SECOND_LEVEL_SUFFIXES and len(labels) >= 3:
        return ".".join(labels[-3:])
    return suffix


def strip_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def domain_matches(domain: str, candidate: str) -> bool:
    return domain == candidate or domain.endswith(f".{candidate}")


def domain_adjustment(
    domain: str, adjustments: Dict[str, float]
) -> tuple[float, str | None]:
    matches = [name for name in adjustments if domain_matches(domain, name)]
    if not matches:
        return 0.0, None
    best = max(matches, key=len)
    return adjustments[best], best


def score_search_result(
    result: Dict[str, Any], query: str, mode: str = "balanced"
) -> Dict[str, Any]:
    title = result.get("title") or ""
    url = result.get("url") or ""
    snippet = result.get("content") or result.get("snippet") or ""
    engine = result.get("engine")
    domain = normalize_domain(get_domain(url))

    score = 1.0
    reasons = []

    boost, boost_domain = domain_adjustment(domain, DOMAIN_BOOSTS)
    if boost_domain:
        score += boost
        reasons.append(f"domain boost: {boost_domain}")

    penalty, penalty_domain = domain_adjustment(domain, DOMAIN_PENALTIES)
    if penalty_domain:
        score += penalty
        reasons.append(f"domain penalty: {penalty_domain}")

    if domain.endswith(".gov"):
        score += 2.5
        reasons.append("government primary source")
    elif domain.endswith(".edu"):
        score += 1.5
        reasons.append("academic institution source")

    if mode == "technical" and any(
        domain_matches(domain, item) for item in TECHNICAL_DOMAINS
    ):
        score += 2.0
        reasons.append("technical source")

    if mode == "academic" and any(
        domain_matches(domain, item) for item in ACADEMIC_DOMAINS
    ):
        score += 2.0
        reasons.append("academic source")

    lowered = f"{title} {snippet} {url}".lower()
    query_terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_\-\.]{3,}", query)]

    if query_terms:
        matches = sum(1 for term in query_terms if term in lowered)
        score += min(matches * 0.25, 2.0)
        if matches:
            reasons.append(f"query term matches: {matches}")

    if "/wiki/portal:current_events" in url.lower():
        score -= 4.0
        reasons.append("current-events portal penalty")

    if "sandbox" in lowered or "alternate history" in lowered or "fiction" in lowered:
        score -= 3.0
        reasons.append("fiction/sandbox penalty")

    if not snippet:
        score -= 0.5
        reasons.append("missing snippet penalty")

    if engine in {"github", "stackoverflow", "arxiv"}:
        score += 0.8
        reasons.append(f"engine boost: {engine}")

    result["score"] = round(score, 3)
    result["score_reasons"] = reasons
    return result


def compact_search_results(
    data: dict, query: str, max_results: int = 10, mode: str = "balanced"
) -> List[dict]:
    seen_urls = set()
    results = []

    for item in data.get("results", []):
        url = normalize_search_url(item.get("url"))
        title = item.get("title")
        content = item.get("content") or ""

        if not url or not title:
            continue

        if url in seen_urls:
            continue

        domain = normalize_domain(get_domain(url))
        if any(domain_matches(domain, item) for item in BLOCKED_DOMAINS):
            continue

        seen_urls.add(url)

        result = {
            "title": strip_text(title),
            "url": url,
            "domain": domain,
            "snippet": strip_text(content)[:900],
            "engine": item.get("engine"),
            "published_at": item.get("publishedDate") or item.get("published_at"),
            "content_trust": "untrusted_external_content",
        }

        results.append(score_search_result(result, query=query, mode=mode))

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results[:max_results]


async def searxng_search(
    query: str, max_results: int = 10, mode: str = "balanced"
) -> List[dict]:
    base_url = SEARXNG_URL.rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "SEARXNG_URL must be an HTTP(S) base URL without credentials, query, or fragment"
        )

    async with asyncio.timeout(SEARXNG_TIMEOUT_SECONDS):
        async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT_SECONDS) as client:
            async with client.stream(
                "GET",
                f"{base_url}/search",
                params={"q": query, "format": "json"},
            ) as response:
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_length = int(content_length)
                    except ValueError as exc:
                        raise ValueError(
                            "SearXNG returned an invalid Content-Length"
                        ) from exc
                    if declared_length > SEARXNG_MAX_RESPONSE_BYTES:
                        raise ValueError(
                            "SearXNG response exceeded SEARXNG_MAX_RESPONSE_BYTES"
                        )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > SEARXNG_MAX_RESPONSE_BYTES:
                        raise ValueError(
                            "SearXNG response exceeded SEARXNG_MAX_RESPONSE_BYTES"
                        )
                    body.extend(chunk)

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SearXNG returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("SearXNG returned a non-object JSON response")

    return compact_search_results(data, query=query, max_results=max_results, mode=mode)
