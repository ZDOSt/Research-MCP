"""Bounded, SSRF-resistant page retrieval for the search gateway."""

from __future__ import annotations

import asyncio
import json
import ipaddress
import os
import re
import ssl
import time
from collections import OrderedDict
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
import redis.asyncio as redis

from egress_policy import (
    DEFAULT_ALLOWED_PORTS,
    DestinationPolicyError,
    parse_allowed_ports,
    parse_denied_networks,
    resolve_public_addresses,
    validate_http_url_without_dns,
    validate_public_address,
)
from evidence_quality import normalize_page_metadata
from extractors import (
    extract_html_metadata,
    html_to_text,
    parse_maybe_json_text,
)


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
}
REDIRECT_CODES = {301, 302, 303, 307, 308}
CHALLENGE_MARKERS = (
    "are you a human",
    "attention required",
    "access denied",
    "captcha",
    "cf-chl-",
    "cloudflare ray id",
    "checking your browser",
    "checking if the site connection is secure",
    "ddos protection",
    "enable cookies",
    "enable javascript",
    "just a moment",
    "please verify you are human",
    "robot check",
    "security check required",
    "temporarily blocked",
    "unusual traffic",
    "verify you are not a robot",
)
ALLOWED_PORTS = parse_allowed_ports(
    os.getenv("SAFE_EGRESS_ALLOWED_PORTS", DEFAULT_ALLOWED_PORTS)
)
DENIED_NETWORKS = parse_denied_networks(os.getenv("SAFE_EGRESS_DENY_CIDRS", ""))
DNS_TIMEOUT_SECONDS = max(0.1, float(os.getenv("SAFE_EGRESS_DNS_TIMEOUT_SECONDS", "5")))
DIRECT_TIMEOUT_SECONDS = max(
    2.0, float(os.getenv("GATEWAY_DIRECT_TIMEOUT_SECONDS", "5"))
)
DIRECT_MAX_BYTES = max(
    65_536, int(os.getenv("GATEWAY_DIRECT_MAX_BYTES", str(8 * 1024 * 1024)))
)
PAGE_MAX_CHARS = max(20_000, int(os.getenv("GATEWAY_PAGE_MAX_CHARS", "300000")))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
FETCH_LEARNING_TTL_SECONDS = max(
    300, int(os.getenv("GATEWAY_FETCH_LEARNING_TTL_SECONDS", "2592000"))
)
FETCH_LEARNING_MAX_DOMAINS = max(
    32, int(os.getenv("GATEWAY_FETCH_LEARNING_MAX_DOMAINS", "1024"))
)
FETCH_REPROBE_SECONDS = max(
    300, int(os.getenv("GATEWAY_FETCH_REPROBE_SECONDS", "21600"))
)
WEB_RUNNER_SOCKET = os.getenv(
    "WEB_RUNNER_SOCKET", "/run/search-gateway-web/runner.sock"
).strip()
PDF_RUNNER_SOCKET = os.getenv(
    "PDF_RUNNER_SOCKET", "/run/search-gateway-pdf/runner.sock"
).strip()


UnsafeURLError = DestinationPolicyError


class PageExtractionError(RuntimeError):
    """Raised when every permitted extraction method returns unusable content."""


def _exception_reason(exc: Exception) -> str:
    """Keep actionable HTTP failure details for gateway-level backoff decisions."""

    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP-{exc.response.status_code}"
    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return "timeout"
    return type(exc).__name__


_DOMAIN_FETCH_STATS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_FETCH_REDIS: Any = None
_FETCH_REDIS_FAILED_UNTIL = 0.0


async def close_fetch_resources() -> None:
    """Close the optional fetch-strategy Redis client without creating one."""

    global _FETCH_REDIS, _FETCH_REDIS_FAILED_UNTIL
    client = _FETCH_REDIS
    _FETCH_REDIS = None
    _FETCH_REDIS_FAILED_UNTIL = 0.0
    if client is not None:
        await client.aclose()


class _LinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a" or len(self.links) >= 500:
            return
        values = {key.lower(): value or "" for key, value in attrs}
        self._href = values.get("href")
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None and sum(map(len, self._parts)) < 500:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        href = self._href
        anchor = re.sub(r"\s+", " ", " ".join(self._parts)).strip()[:500]
        self._href = None
        self._parts = []
        try:
            absolute = urljoin(self.base_url, href)
            parsed = urlsplit(absolute)
        except ValueError:
            return
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return
        normalized = urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
        )
        self.links.append({"url": normalized, "anchor": anchor})


def _extract_links(raw_html: str, base_url: str) -> list[dict[str, str]]:
    try:
        parser = _LinkParser(base_url)
        parser.feed(raw_html)
        parser.close()
        seen: set[str] = set()
        output = []
        for link in parser.links:
            if link["url"] in seen:
                continue
            seen.add(link["url"])
            output.append(link)
        return output
    except Exception:
        return []


def _normalized_host(url: str) -> tuple[str, int]:
    return validate_http_url_without_dns(
        url,
        allowed_ports=ALLOWED_PORTS,
        denied_networks=DENIED_NETWORKS,
    )


async def validate_public_url(url: str) -> str:
    host, port = _normalized_host(url)
    await resolve_public_addresses(
        host,
        port,
        allowed_ports=ALLOWED_PORTS,
        denied_networks=DENIED_NETWORKS,
        dns_timeout_seconds=DNS_TIMEOUT_SECONDS,
    )
    return url


async def validate_proxy_url_safety(url: str) -> str:
    """Validate URL syntax before an isolated proxy performs DNS and egress checks."""
    _normalized_host(url)
    return url


async def _resolved_addresses(url: str) -> tuple[str, int, tuple[str, ...]]:
    host, port = _normalized_host(url)
    addresses = await resolve_public_addresses(
        host,
        port,
        allowed_ports=ALLOWED_PORTS,
        denied_networks=DENIED_NETWORKS,
        dns_timeout_seconds=DNS_TIMEOUT_SECONDS,
    )
    return host, port, addresses


def _pinned_request(
    url: str, address: str
) -> tuple[str, dict[str, str], dict[str, str]]:
    parsed = urlsplit(url)
    host, port = _normalized_host(url)
    ip = ipaddress.ip_address(address)
    pinned_host = f"[{ip}]" if ip.version == 6 else str(ip)
    default_port = 443 if parsed.scheme == "https" else 80
    netloc = pinned_host if port == default_port else f"{pinned_host}:{port}"
    headers = dict(BROWSER_HEADERS)
    display_host = f"[{host}]" if ":" in host else host
    headers["Host"] = display_host if port == default_port else f"{display_host}:{port}"
    pinned_url = urlunsplit(
        (parsed.scheme, netloc, parsed.path or "/", parsed.query, "")
    )
    extensions: dict[str, str] = {}
    if parsed.scheme == "https":
        extensions["sni_hostname"] = host
    return pinned_url, headers, extensions


def _validate_peer(response: httpx.Response, expected_address: str) -> None:
    stream = (response.extensions or {}).get("network_stream")
    peer = stream.get_extra_info("server_addr") if stream is not None else None
    if not isinstance(peer, (tuple, list)) or not peer:
        raise UnsafeURLError("Unable to verify the connected response peer")
    actual = validate_public_address(str(peer[0]), DENIED_NETWORKS)
    expected = validate_public_address(expected_address, DENIED_NETWORKS)
    if actual != expected:
        raise UnsafeURLError("Connected peer did not match the validated address")


async def _read_limited(response: httpx.Response, limit: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared:
        try:
            declared_size = int(declared)
            if declared_size < 0 or declared_size > limit:
                raise ValueError("Page response exceeds the configured byte limit")
        except ValueError as exc:
            raise ValueError(
                "Page returned an invalid or excessive Content-Length"
            ) from exc
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > limit:
            raise ValueError("Page response exceeds the configured byte limit")
        body.extend(chunk)
    return bytes(body)


def _decode(body: bytes, encoding: str | None) -> str:
    for candidate in (encoding, "utf-8"):
        if not candidate:
            continue
        try:
            return body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            pass
    return body.decode("utf-8", errors="replace")


def _extract_body(
    body: bytes,
    content_type: str,
    encoding: str | None,
    final_url: str,
) -> tuple[str, str | None, str, list[dict[str, str]], dict[str, Any]]:
    media_type = content_type.split(";", 1)[0].strip().lower()
    decoded = _decode(body, encoding)
    if "json" in media_type or decoded.lstrip().startswith(("{", "[")):
        return parse_maybe_json_text(decoded)[:PAGE_MAX_CHARS], None, "json", [], {}
    if "html" in media_type or "<html" in decoded[:2000].lower():
        metadata = extract_html_metadata(decoded)
        return (
            html_to_text(decoded)[:PAGE_MAX_CHARS],
            metadata.get("title"),
            "html",
            _extract_links(decoded, final_url),
            metadata,
        )
    return decoded[:PAGE_MAX_CHARS], None, "text", [], {}


async def _extract_pdf_isolated(body: bytes) -> tuple[str, str | None]:
    if not PDF_RUNNER_SOCKET:
        raise RuntimeError("The isolated PDF runner is not configured")
    transport = httpx.AsyncHTTPTransport(uds=PDF_RUNNER_SOCKET)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://pdf-runner",
        timeout=httpx.Timeout(DIRECT_TIMEOUT_SECONDS),
        trust_env=False,
    ) as client:
        response = await client.post(
            "/v1/extract", content=body, headers={"Content-Type": "application/pdf"}
        )
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("PDF runner returned an invalid response")
    text = str(payload.get("content") or "")[:PAGE_MAX_CHARS]
    title = payload.get("title")
    return text, str(title)[:2000] if title else None


async def direct_fetch(
    url: str, timeout_seconds: float | None = None
) -> dict[str, Any]:
    timeout_seconds = min(
        DIRECT_TIMEOUT_SECONDS, timeout_seconds or DIRECT_TIMEOUT_SECONDS
    )
    async with asyncio.timeout(max(0.5, timeout_seconds)):
        current_url = url
        redirects: list[str] = []
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(max(1.0, timeout_seconds)),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for _ in range(8):
                _, _, addresses = await _resolved_addresses(current_url)
                response_data = None
                last_error: Exception | None = None
                for address in addresses:
                    pinned_url, headers, extensions = _pinned_request(
                        current_url, address
                    )
                    try:
                        async with client.stream(
                            "GET", pinned_url, headers=headers, extensions=extensions
                        ) as response:
                            _validate_peer(response, address)
                            location = response.headers.get("location")
                            if response.status_code in REDIRECT_CODES and location:
                                response_data = (
                                    response.status_code,
                                    dict(response.headers),
                                    b"",
                                    None,
                                )
                            else:
                                response.raise_for_status()
                                body = await _read_limited(response, DIRECT_MAX_BYTES)
                                response_data = (
                                    response.status_code,
                                    dict(response.headers),
                                    body,
                                    response.encoding,
                                )
                            break
                    except (
                        httpx.ConnectError,
                        httpx.ConnectTimeout,
                        httpx.RemoteProtocolError,
                        ssl.SSLError,
                    ) as exc:
                        last_error = exc
                if response_data is None:
                    raise last_error or RuntimeError(
                        "No validated address accepted the request"
                    )
                status, headers, body, encoding = response_data
                location = headers.get("location")
                if status in REDIRECT_CODES and location:
                    current_url = urljoin(current_url, location)
                    await validate_public_url(current_url)
                    redirects.append(current_url)
                    continue
                content_type = headers.get("content-type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type == "application/pdf" or body.lstrip().startswith(
                    b"%PDF-"
                ):
                    text, title = await _extract_pdf_isolated(body)
                    body_format = "pdf"
                    links = []
                    metadata: dict[str, Any] = {}
                else:
                    text, title, body_format, links, metadata = await asyncio.to_thread(
                        _extract_body, body, content_type, encoding, current_url
                    )
                metadata = normalize_page_metadata(
                    metadata,
                    last_modified=headers.get("last-modified"),
                ) | {
                    key: value
                    for key, value in metadata.items()
                    if key
                    not in {
                        "publishedDate",
                        "modifiedDate",
                        "declaredDate",
                        "language",
                        "author",
                        "description",
                        "version_context",
                    }
                }
                return {
                    "url": current_url,
                    "title": title,
                    "content": text.strip(),
                    "content_chars": len(text.strip()),
                    "body_format": body_format,
                    "links": links,
                    "status_code": status,
                    "redirect_chain": redirects,
                    "extraction_method": "direct",
                    "metadata": metadata,
                }
        raise RuntimeError("Too many redirects")


def _content_from_crawl4ai(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in (
            "fit_markdown",
            "raw_markdown",
            "markdown_with_citations",
            "markdown",
        ):
            content = value.get(key)
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _normalize_crawl4ai(payload: Any, original_url: str) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        payload = payload["results"][0] if payload["results"] else {}
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("Crawl4AI returned an invalid response")
    content = _content_from_crawl4ai(payload.get("markdown"))
    if not content:
        content = str(
            payload.get("cleaned_html") or payload.get("extracted_content") or ""
        )
        if content.lstrip().startswith("<"):
            content = html_to_text(content)
        else:
            content = parse_maybe_json_text(content)
    metadata = (
        payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    )
    metadata = {
        **metadata,
        **{
            key: payload[key]
            for key in (
                "publishedDate",
                "publishedTime",
                "modifiedDate",
                "modifiedTime",
                "datePublished",
                "dateModified",
                "language",
                "author",
                "description",
                "softwareVersion",
            )
            if payload.get(key) not in (None, "", [], {})
        },
    }
    normalized_metadata = normalize_page_metadata(metadata)
    links: list[dict[str, str]] = []
    raw_links = payload.get("links")
    if isinstance(raw_links, dict):
        raw_links = [*raw_links.get("internal", []), *raw_links.get("external", [])]
    if isinstance(raw_links, list):
        for item in raw_links[:500]:
            if isinstance(item, dict) and item.get("href"):
                links.append(
                    {
                        "url": urljoin(original_url, str(item["href"])),
                        "anchor": str(item.get("text") or "")[:500],
                    }
                )
    return {
        "url": str(payload.get("url") or original_url),
        "title": str(payload.get("title") or metadata.get("title") or "") or None,
        "content": content[:PAGE_MAX_CHARS].strip(),
        "content_chars": len(content[:PAGE_MAX_CHARS].strip()),
        "body_format": "markdown",
        "links": links,
        "status_code": payload.get("status_code"),
        "extraction_method": "crawl4ai",
        "metadata": normalized_metadata,
    }


async def crawl4ai_fetch(url: str, timeout_seconds: float) -> dict[str, Any]:
    if not WEB_RUNNER_SOCKET:
        raise RuntimeError("The isolated web runner is not configured")
    transport = httpx.AsyncHTTPTransport(uds=WEB_RUNNER_SOCKET)
    async with asyncio.timeout(max(1.0, timeout_seconds)):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://web-runner",
            timeout=httpx.Timeout(max(2.0, timeout_seconds + 2.0)),
            trust_env=False,
        ) as client:
            response = await client.post(
                "/v1/crawl4ai/crawl",
                params={"timeout_seconds": timeout_seconds},
                json={"urls": [url], "browser_config": {}, "crawler_config": {}},
            )
            response.raise_for_status()
            return _normalize_crawl4ai(response.json(), url)


async def browser_fetch(url: str, query: str, timeout_seconds: float) -> dict[str, Any]:
    transport = httpx.AsyncHTTPTransport(uds=WEB_RUNNER_SOCKET)
    async with asyncio.timeout(max(1.0, timeout_seconds)):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://web-runner",
            timeout=httpx.Timeout(max(2.0, timeout_seconds + 2.0)),
            trust_env=False,
        ) as client:
            response = await client.post(
                "/v1/explore",
                json={
                    "url": url,
                    "task": query[:4000],
                    "labels": [],
                    "max_chars": PAGE_MAX_CHARS,
                    "profile": "targeted",
                    "timeout_ms": int(min(60.0, timeout_seconds) * 1000),
                },
            )
            response.raise_for_status()
            payload = response.json()
            return {
                "url": str(payload.get("final_url") or url),
                "title": payload.get("title"),
                "content": str(payload.get("content") or "")[:PAGE_MAX_CHARS].strip(),
                "content_chars": int(payload.get("content_chars") or 0),
                "body_format": "rendered_text",
                "links": [],
                "status_code": 200,
                "extraction_method": "playwright",
                "metadata": normalize_page_metadata(payload.get("metadata")),
            }


async def _fetch_redis_client() -> Any:
    global _FETCH_REDIS, _FETCH_REDIS_FAILED_UNTIL
    if not REDIS_URL or time.monotonic() < _FETCH_REDIS_FAILED_UNTIL:
        return None
    if _FETCH_REDIS is None:
        _FETCH_REDIS = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=0.2,
            socket_timeout=0.2,
        )
    return _FETCH_REDIS


async def _load_domain_stats(domain: str) -> dict[str, Any]:
    global _FETCH_REDIS_FAILED_UNTIL
    if domain in _DOMAIN_FETCH_STATS:
        _DOMAIN_FETCH_STATS.move_to_end(domain)
        return _DOMAIN_FETCH_STATS[domain]
    stats: dict[str, Any] = {}
    client = await _fetch_redis_client()
    if client is not None:
        try:
            raw = await asyncio.wait_for(
                client.get(f"search-gateway:fetch-strategy:v1:{domain}"), 0.25
            )
            parsed = json.loads(raw) if raw else {}
            if isinstance(parsed, dict):
                stats = parsed
        except Exception:
            _FETCH_REDIS_FAILED_UNTIL = time.monotonic() + 30.0
    _DOMAIN_FETCH_STATS[domain] = stats
    _DOMAIN_FETCH_STATS.move_to_end(domain)
    while len(_DOMAIN_FETCH_STATS) > FETCH_LEARNING_MAX_DOMAINS:
        _DOMAIN_FETCH_STATS.popitem(last=False)
    return stats


async def _persist_domain_stats(domain: str, stats: dict[str, Any]) -> None:
    global _FETCH_REDIS_FAILED_UNTIL
    client = await _fetch_redis_client()
    if client is None:
        return
    try:
        await asyncio.wait_for(
            client.setex(
                f"search-gateway:fetch-strategy:v1:{domain}",
                FETCH_LEARNING_TTL_SECONDS,
                json.dumps(stats, separators=(",", ":")),
            ),
            0.25,
        )
    except Exception:
        _FETCH_REDIS_FAILED_UNTIL = time.monotonic() + 30.0


async def _record_domain_outcome(
    domain: str,
    method: str,
    *,
    success: bool,
    latency_seconds: float,
    reason: str,
) -> None:
    stats = await _load_domain_stats(domain)
    row = dict(stats.get(method) or {})
    row["attempts"] = int(row.get("attempts") or 0) + 1
    row["successes"] = int(row.get("successes") or 0) + int(success)
    row["failures"] = int(row.get("failures") or 0) + int(not success)
    row["consecutive_failures"] = (
        0 if success else int(row.get("consecutive_failures") or 0) + 1
    )
    row["latency_total"] = float(row.get("latency_total") or 0.0) + max(
        0.0, latency_seconds
    )
    row["last_reason"] = reason[:120]
    row["updated_at"] = int(time.time())
    stats[method] = row
    _DOMAIN_FETCH_STATS[domain] = stats
    await _persist_domain_stats(domain, stats)


def _preferred_method(stats: dict[str, Any]) -> str | None:
    ranked: list[tuple[float, str]] = []
    for method, raw in stats.items():
        if not isinstance(raw, dict) or int(raw.get("successes") or 0) <= 0:
            continue
        attempts = max(1, int(raw.get("attempts") or 0))
        success_rate = int(raw.get("successes") or 0) / attempts
        average_latency = float(raw.get("latency_total") or 0.0) / attempts
        recent_failure_penalty = 0.35 * min(
            3, int(raw.get("consecutive_failures") or 0)
        )
        ranked.append(
            (
                success_rate
                - 0.015 * average_latency
                - recent_failure_penalty,
                method,
            )
        )
    return max(ranked)[1] if ranked else None


def _github_fast_path(url: str) -> str | None:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    if (parsed.hostname or "").casefold().removeprefix("www.") != "github.com":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 5 and parts[2] == "blob":
        owner, repository, _, revision, *path = parts
        return f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{'/'.join(path)}"
    if len(parts) == 2:
        return f"https://raw.githubusercontent.com/{parts[0]}/{parts[1]}/HEAD/README.md"
    return None


def _query_coverage(query: str, content: str) -> float:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
        "what",
        "with",
    }
    terms = list(
        dict.fromkeys(
            term
            for term in re.findall(r"[a-z0-9][a-z0-9_.-]+", query.casefold())
            if len(term) >= 2 and term not in stop
        )
    )[:40]
    if not terms:
        return 1.0
    lowered = content.casefold()
    return sum(term in lowered for term in terms) / len(terms)


def assess_content(
    page: dict[str, Any], query: str = "", minimum_chars: int = 900
) -> dict[str, Any]:
    content = re.sub(r"\s+", " ", str(page.get("content") or "")).strip()
    title = re.sub(r"\s+", " ", str(page.get("title") or "")).strip()
    preview = f"{title}\n{content[:3000]}".casefold()
    marker_hits = [marker for marker in CHALLENGE_MARKERS if marker in preview]
    decisive_markers = {
        "cf-chl-",
        "checking if the site connection is secure",
        "checking your browser",
        "cloudflare ray id",
        "please verify you are human",
        "security check required",
        "unusual traffic",
        "verify you are not a robot",
    }
    title_challenge = any(
        marker in title.casefold()
        for marker in {
            "access denied",
            "attention required",
            "just a moment",
            "robot check",
            "temporarily blocked",
        }
    )
    strong_hits = [marker for marker in marker_hits if marker in decisive_markers]
    marker_occurrences = sum(preview.count(marker) for marker in marker_hits)
    coverage = _query_coverage(query, f"{title} {content}") if query else 0.0
    body_format = str(page.get("body_format") or "").casefold()
    structured = body_format in {"json", "pdf", "text"}
    if (
        (title_challenge and (len(content) < 4000 or marker_occurrences >= 2))
        or (
            strong_hits
            and (len(content) < 2000 or marker_occurrences >= 2)
        )
        or (
            len(marker_hits) >= 2
            and (len(content) < 4000 or marker_occurrences >= 3)
        )
    ):
        return {
            "status": "challenge",
            "usable": False,
            "reason": "challenge-or-interstitial",
            "markers": marker_hits[:8],
            "content_chars": len(content),
            "query_coverage": round(coverage, 4),
        }
    usable = len(content) >= minimum_chars
    if structured and len(content) >= 120:
        usable = True
    if len(content) >= 250 and coverage >= 0.35:
        usable = True
    reason = (
        "usable"
        if usable
        else ("thin-content" if len(content) < minimum_chars else "low-relevance")
    )
    return {
        "status": "usable" if usable else "low-confidence",
        "usable": usable,
        "reason": reason,
        "markers": marker_hits[:8],
        "content_chars": len(content),
        "query_coverage": round(coverage, 4),
    }


def content_is_usable(page: dict[str, Any], minimum_chars: int = 900) -> bool:
    return bool(assess_content(page, minimum_chars=minimum_chars)["usable"])


async def _method_order(url: str) -> tuple[list[str], str | None, dict[str, Any]]:
    domain = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    stats = await _load_domain_stats(domain)
    preferred = _preferred_method(stats)
    order = ["direct"]
    if _github_fast_path(url):
        order.append("github_raw")
    order.extend(["crawl4ai", "playwright"])
    direct = stats.get("direct") if isinstance(stats.get("direct"), dict) else {}
    direct_failures = int(
        direct.get("consecutive_failures", direct.get("failures") or 0)
    )
    direct_updated_at = int(direct.get("updated_at") or 0)
    direct_reprobe_due = bool(
        direct_failures >= 2
        and direct_updated_at
        and time.time() - direct_updated_at >= FETCH_REPROBE_SECONDS
    )
    path = urlsplit(url).path.casefold()
    direct_fast_path = (
        domain in {
            "raw.githubusercontent.com",
            "api.github.com",
        }
        or bool(_github_fast_path(url))
        or path.endswith(
            (
                ".csv",
                ".json",
                ".md",
                ".pdf",
                ".tsv",
                ".txt",
                ".xml",
                ".yaml",
                ".yml",
            )
        )
    )
    if (
        not direct_fast_path
        and preferred in order
        and preferred != "direct"
        and direct_failures >= 2
        and not direct_reprobe_due
    ):
        order.remove(preferred)
        order.insert(0, preferred)
    return order, preferred, stats


async def fetch_page(
    url: str,
    query: str,
    timeout_seconds: float,
    *,
    allow_expensive_fallback: bool = True,
) -> dict[str, Any]:
    """Use learned, bounded extraction while rejecting challenge responses."""

    await validate_public_url(url)
    domain = (urlsplit(url).hostname or "").casefold().removeprefix("www.")
    order, preferred, _ = await _method_order(url)
    if not allow_expensive_fallback:
        order = [method for method in order if method in {"direct", "github_raw"}]
    errors: list[str] = []
    best: dict[str, Any] | None = None
    best_assessment: dict[str, Any] | None = None
    started = time.monotonic()

    for method in order:
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 1.0:
            errors.append(f"{method}:deadline")
            break
        method_started = time.monotonic()
        try:
            if method == "direct":
                reserve = (
                    3.5 if "playwright" in order[order.index(method) + 1 :] else 0.0
                )
                method_timeout = min(
                    DIRECT_TIMEOUT_SECONDS,
                    max(1.0, remaining - reserve),
                )
                page = await direct_fetch(url, method_timeout)
                minimum_chars = 900
            elif method == "github_raw":
                fast_url = _github_fast_path(url)
                if not fast_url:
                    continue
                await validate_public_url(fast_url)
                reserve = (
                    3.5 if "playwright" in order[order.index(method) + 1 :] else 0.0
                )
                page = await direct_fetch(
                    fast_url,
                    min(DIRECT_TIMEOUT_SECONDS, max(1.0, remaining - reserve)),
                )
                page["retrieved_url"] = page.get("url")
                page["url"] = url
                page["extraction_method"] = "github_raw"
                minimum_chars = 300
            elif method == "crawl4ai":
                reserve = (
                    3.5 if "playwright" in order[order.index(method) + 1 :] else 0.0
                )
                method_timeout = min(10.0, remaining - reserve)
                if method_timeout <= 1.0:
                    errors.append("crawl4ai:insufficient-time")
                    continue
                page = await crawl4ai_fetch(url, method_timeout)
                minimum_chars = 500
            else:
                if remaining <= 2.0:
                    errors.append("playwright:insufficient-time")
                    continue
                page = await browser_fetch(url, query, min(remaining, 12.0))
                minimum_chars = 400
            assessment = assess_content(page, query, minimum_chars)
            latency = time.monotonic() - method_started
            await _record_domain_outcome(
                domain,
                method,
                success=bool(assessment["usable"]),
                latency_seconds=latency,
                reason=str(assessment["reason"]),
            )
            page["fetch_assessment"] = assessment
            page["fetch_strategy"] = {
                "domain": domain,
                "attempt_order": order,
                "learned_preference": preferred,
                "selected_method": method,
            }
            if assessment["usable"]:
                if errors:
                    page["extraction_errors"] = errors
                return page
            errors.append(f"{method}:{assessment['reason']}")
            if assessment["status"] != "challenge" and (
                best is None
                or int(assessment["content_chars"])
                > int((best_assessment or {}).get("content_chars") or 0)
            ):
                best = page
                best_assessment = assessment
        except Exception as exc:
            latency = time.monotonic() - method_started
            reason = _exception_reason(exc)
            await _record_domain_outcome(
                domain,
                method,
                success=False,
                latency_seconds=latency,
                reason=reason,
            )
            errors.append(f"{method}:{reason}")

    if (
        best
        and best_assessment
        and best_assessment.get("status") != "challenge"
        and int(best_assessment.get("content_chars") or 0) >= 200
        and (
            float(best_assessment.get("query_coverage") or 0.0) >= 0.2
            or str(best.get("body_format") or "").casefold() in {"json", "pdf", "text"}
        )
    ):
        best["extraction_errors"] = errors
        best["low_confidence"] = True
        return best
    raise PageExtractionError("Page extraction failed: " + ",".join(errors))
