"""Bounded, SSRF-resistant page retrieval for the search gateway."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import ssl
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from egress_policy import (
    DEFAULT_ALLOWED_PORTS,
    DestinationPolicyError,
    parse_allowed_ports,
    parse_denied_networks,
    resolve_public_addresses,
    validate_http_url_without_dns,
    validate_public_address,
)
from extractors import extract_title_from_html, html_to_text, parse_maybe_json_text


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
    "access denied",
    "checking your browser",
    "enable javascript",
    "just a moment",
    "please verify you are human",
    "security check required",
    "unusual traffic",
    "verify you are not a robot",
)
ALLOWED_PORTS = parse_allowed_ports(
    os.getenv("SAFE_EGRESS_ALLOWED_PORTS", DEFAULT_ALLOWED_PORTS)
)
DENIED_NETWORKS = parse_denied_networks(os.getenv("SAFE_EGRESS_DENY_CIDRS", ""))
DNS_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("SAFE_EGRESS_DNS_TIMEOUT_SECONDS", "5"))
)
DIRECT_TIMEOUT_SECONDS = max(
    2.0, float(os.getenv("GATEWAY_DIRECT_TIMEOUT_SECONDS", "5"))
)
DIRECT_MAX_BYTES = max(
    65_536, int(os.getenv("GATEWAY_DIRECT_MAX_BYTES", str(8 * 1024 * 1024)))
)
PAGE_MAX_CHARS = max(20_000, int(os.getenv("GATEWAY_PAGE_MAX_CHARS", "300000")))
WEB_RUNNER_SOCKET = os.getenv(
    "WEB_RUNNER_SOCKET", "/run/search-gateway-web/runner.sock"
).strip()
PDF_RUNNER_SOCKET = os.getenv(
    "PDF_RUNNER_SOCKET", "/run/search-gateway-pdf/runner.sock"
).strip()


UnsafeURLError = DestinationPolicyError


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
        normalized = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
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


def _pinned_request(url: str, address: str) -> tuple[str, dict[str, str], dict[str, str]]:
    parsed = urlsplit(url)
    host, port = _normalized_host(url)
    ip = ipaddress.ip_address(address)
    pinned_host = f"[{ip}]" if ip.version == 6 else str(ip)
    default_port = 443 if parsed.scheme == "https" else 80
    netloc = pinned_host if port == default_port else f"{pinned_host}:{port}"
    headers = dict(BROWSER_HEADERS)
    display_host = f"[{host}]" if ":" in host else host
    headers["Host"] = display_host if port == default_port else f"{display_host}:{port}"
    pinned_url = urlunsplit((parsed.scheme, netloc, parsed.path or "/", parsed.query, ""))
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
            raise ValueError("Page returned an invalid or excessive Content-Length") from exc
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
) -> tuple[str, str | None, str, list[dict[str, str]]]:
    media_type = content_type.split(";", 1)[0].strip().lower()
    decoded = _decode(body, encoding)
    if "json" in media_type or decoded.lstrip().startswith(("{", "[")):
        return parse_maybe_json_text(decoded)[:PAGE_MAX_CHARS], None, "json", []
    if "html" in media_type or "<html" in decoded[:2000].lower():
        return (
            html_to_text(decoded)[:PAGE_MAX_CHARS],
            extract_title_from_html(decoded),
            "html",
            _extract_links(decoded, final_url),
        )
    return decoded[:PAGE_MAX_CHARS], None, "text", []


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


async def direct_fetch(url: str, timeout_seconds: float | None = None) -> dict[str, Any]:
    timeout_seconds = min(DIRECT_TIMEOUT_SECONDS, timeout_seconds or DIRECT_TIMEOUT_SECONDS)
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
                    pinned_url, headers, extensions = _pinned_request(current_url, address)
                    try:
                        async with client.stream(
                            "GET", pinned_url, headers=headers, extensions=extensions
                        ) as response:
                            _validate_peer(response, address)
                            location = response.headers.get("location")
                            if response.status_code in REDIRECT_CODES and location:
                                response_data = (response.status_code, dict(response.headers), b"", None)
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
                    raise last_error or RuntimeError("No validated address accepted the request")
                status, headers, body, encoding = response_data
                location = headers.get("location")
                if status in REDIRECT_CODES and location:
                    current_url = urljoin(current_url, location)
                    await validate_public_url(current_url)
                    redirects.append(current_url)
                    continue
                content_type = headers.get("content-type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type == "application/pdf" or body.lstrip().startswith(b"%PDF-"):
                    text, title = await _extract_pdf_isolated(body)
                    body_format = "pdf"
                    links = []
                else:
                    text, title, body_format, links = await asyncio.to_thread(
                        _extract_body, body, content_type, encoding, current_url
                    )
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
                }
        raise RuntimeError("Too many redirects")


def _content_from_crawl4ai(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("fit_markdown", "raw_markdown", "markdown_with_citations", "markdown"):
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
        content = str(payload.get("cleaned_html") or payload.get("extracted_content") or "")
        if content.lstrip().startswith("<"):
            content = html_to_text(content)
        else:
            content = parse_maybe_json_text(content)
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
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
            }


def content_is_usable(page: dict[str, Any], minimum_chars: int = 900) -> bool:
    content = re.sub(r"\s+", " ", str(page.get("content") or "")).strip()
    if len(content) < minimum_chars:
        return False
    preview = content[:3000].lower()
    marker_count = sum(marker in preview for marker in CHALLENGE_MARKERS)
    return marker_count < 2


async def fetch_page(
    url: str,
    query: str,
    timeout_seconds: float,
    *,
    allow_expensive_fallback: bool = True,
) -> dict[str, Any]:
    """Use cheap HTTP first, then Crawl4AI and Playwright only when needed."""
    await validate_public_url(url)
    errors: list[str] = []
    best: dict[str, Any] | None = None
    started = asyncio.get_running_loop().time()

    try:
        best = await direct_fetch(url, min(DIRECT_TIMEOUT_SECONDS, timeout_seconds))
        if content_is_usable(best):
            return best
    except Exception as exc:
        errors.append(f"direct:{type(exc).__name__}")

    elapsed = asyncio.get_running_loop().time() - started
    remaining = timeout_seconds - elapsed
    if allow_expensive_fallback and remaining > 2:
        try:
            crawled = await crawl4ai_fetch(url, min(remaining, 15.0))
            if content_is_usable(crawled, 500):
                return crawled
            if best is None or crawled.get("content_chars", 0) > best.get("content_chars", 0):
                best = crawled
        except Exception as exc:
            errors.append(f"crawl4ai:{type(exc).__name__}")

    elapsed = asyncio.get_running_loop().time() - started
    remaining = timeout_seconds - elapsed
    if allow_expensive_fallback and remaining > 3:
        try:
            rendered = await browser_fetch(url, query, min(remaining, 12.0))
            if content_is_usable(rendered, 400):
                return rendered
            if best is None or rendered.get("content_chars", 0) > best.get("content_chars", 0):
                best = rendered
        except Exception as exc:
            errors.append(f"playwright:{type(exc).__name__}")

    if best and best.get("content"):
        best["extraction_errors"] = errors
        best["low_confidence"] = True
        return best
    raise RuntimeError("Page extraction failed: " + ",".join(errors))
