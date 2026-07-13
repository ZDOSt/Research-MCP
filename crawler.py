import asyncio
import io
import ipaddress
import json
import os
import re
import sys
from contextlib import suppress
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from egress_policy import (
    DEFAULT_ALLOWED_PORTS,
    DestinationPolicyError,
    normalize_destination_host,
    parse_allowed_ports,
    parse_denied_networks,
    resolve_public_addresses,
    validate_destination_port,
    validate_public_address,
)
from extractors import extract_title_from_html, html_to_text, parse_maybe_json_text
from redaction import redact_sensitive_text
from shared import CRAWL4AI_URL

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

ALLOWED_URL_SCHEMES = {"http", "https"}
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
MAX_REDIRECTS = 10
SAFE_EGRESS_ALLOWED_PORTS = parse_allowed_ports(
    os.getenv("SAFE_EGRESS_ALLOWED_PORTS", DEFAULT_ALLOWED_PORTS)
)
SAFE_EGRESS_DENY_NETWORKS = parse_denied_networks(os.getenv("SAFE_EGRESS_DENY_CIDRS", ""))
SAFE_EGRESS_DNS_TIMEOUT_SECONDS = max(
    0.1, float(os.getenv("SAFE_EGRESS_DNS_TIMEOUT_SECONDS", "5"))
)
DIRECT_MAX_RESPONSE_BYTES = max(1024, int(os.getenv("DIRECT_MAX_RESPONSE_BYTES", str(16 * 1024 * 1024))))
CRAWL4AI_MAX_RESPONSE_BYTES = max(1024, int(os.getenv("CRAWL4AI_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))
DIRECT_TOTAL_TIMEOUT_SECONDS = max(1.0, float(os.getenv("DIRECT_TOTAL_TIMEOUT_SECONDS", "75")))
DIRECT_EXTRACTION_TIMEOUT_SECONDS = max(1.0, float(os.getenv("DIRECT_EXTRACTION_TIMEOUT_SECONDS", "20")))
CRAWL4AI_TOTAL_TIMEOUT_SECONDS = max(1.0, float(os.getenv("CRAWL4AI_TOTAL_TIMEOUT_SECONDS", "180")))
DIRECT_FIRST_HEDGE_SECONDS = max(
    0.0, float(os.getenv("DIRECT_FIRST_HEDGE_SECONDS", "1.5"))
)
DIRECT_FIRST_MIN_CONTENT_CHARS = max(
    200, int(os.getenv("DIRECT_FIRST_MIN_CONTENT_CHARS", "2000"))
)
DIRECT_LOW_CONFIDENCE_MARKERS = (
    "access denied",
    "checking your browser",
    "enable javascript",
    "just a moment",
    "please verify you are human",
    "security check required",
)
DIRECT_PRIMARY_EXCLUDED_TAGS = {
    "aside",
    "footer",
    "head",
    "header",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
    "title",
}
DIRECT_PRIMARY_EXCLUDED_HINTS = {
    "banner",
    "breadcrumb",
    "consent",
    "cookie",
    "dialog",
    "footer",
    "header",
    "menu",
    "modal",
    "nav",
    "navigation",
    "sidebar",
    "social",
}
DIRECT_HTML_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
PDF_MAX_RESPONSE_BYTES = max(1024, int(os.getenv("PDF_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))))
PDF_MAX_PAGES = max(1, int(os.getenv("PDF_MAX_PAGES", "200")))
PDF_MAX_EXTRACTED_CHARS = max(1000, int(os.getenv("PDF_MAX_EXTRACTED_CHARS", "1000000")))
PDF_SANDBOX_MEMORY_BYTES = max(
    128 * 1024 * 1024,
    int(os.getenv("PDF_SANDBOX_MEMORY_BYTES", str(512 * 1024 * 1024))),
)
PDF_SANDBOX_CPU_SECONDS = max(1, int(os.getenv("PDF_SANDBOX_CPU_SECONDS", "15")))
PDF_SANDBOX_OUTPUT_BYTES = max(
    65_536,
    int(os.getenv("PDF_SANDBOX_OUTPUT_BYTES", str(PDF_MAX_EXTRACTED_CHARS * 6 + 65_536))),
)
PDF_RUNNER_SOCKET = os.getenv("PDF_RUNNER_SOCKET", "").strip()
CRAWL4AI_NETWORK_ISOLATED = os.getenv("CRAWL4AI_NETWORK_ISOLATED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


UnsafeURLError = DestinationPolicyError


class _DirectPrimaryContentParser(HTMLParser):
    """Count visible page text after excluding common navigation chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._excluded_depth = 0
        self._element_stack: list[tuple[str, bool]] = []

    @staticmethod
    def _is_excluded(tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in DIRECT_PRIMARY_EXCLUDED_TAGS:
            return True
        attrs_map = {str(key).lower(): value or "" for key, value in attrs}
        if (
            "hidden" in attrs_map
            or "inert" in attrs_map
            or attrs_map.get("aria-hidden", "").strip().lower() == "true"
        ):
            return True
        style = attrs_map.get("style", "").lower()
        if re.search(
            r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*hidden|content-visibility\s*:\s*hidden)",
            style,
        ):
            return True
        role = attrs_map.get("role", "").strip().lower()
        if role in {"banner", "complementary", "contentinfo", "dialog", "navigation"}:
            return True
        hints = re.findall(
            r"[a-z0-9]+",
            f"{attrs_map.get('id', '')} {attrs_map.get('class', '')}".lower(),
        )
        return any(hint in DIRECT_PRIMARY_EXCLUDED_HINTS for hint in hints)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        excluded = self._is_excluded(tag, attrs)
        if tag not in DIRECT_HTML_VOID_TAGS:
            self._element_stack.append((tag, excluded))
            if excluded:
                self._excluded_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self._element_stack) - 1, -1, -1):
            if self._element_stack[index][0] != tag:
                continue
            popped = self._element_stack[index:]
            self._element_stack = self._element_stack[:index]
            self._excluded_depth = max(
                0,
                self._excluded_depth - sum(1 for _, excluded in popped if excluded),
            )
            break

    def handle_data(self, data: str) -> None:
        if not self._excluded_depth:
            self.parts.append(data)

    def content_chars(self) -> int:
        return len(re.sub(r"\s+", " ", " ".join(self.parts)).strip())


def _direct_html_primary_content_chars(raw_html: str) -> int:
    try:
        parser = _DirectPrimaryContentParser()
        parser.feed(raw_html or "")
        parser.close()
        return parser.content_chars()
    except Exception:
        return 0


def _safe_error_detail(exc: Exception, max_chars: int = 1000) -> str:
    redacted, _ = redact_sensitive_text(str(exc))
    return redacted[:max_chars]


def _normalized_url_host(url: str) -> tuple[str, int]:
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL must be a non-empty string")

    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError(f"Invalid URL: {_safe_error_detail(exc)}") from exc

    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        raise UnsafeURLError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise UnsafeURLError("URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URLs containing credentials are not allowed")

    host = normalize_destination_host(parsed.hostname, SAFE_EGRESS_DENY_NETWORKS)
    destination_port = port if port is not None else (443 if scheme == "https" else 80)
    validate_destination_port(destination_port, SAFE_EGRESS_ALLOWED_PORTS)
    return host, destination_port


def _validate_public_ip(address: str) -> None:
    validate_public_address(address, SAFE_EGRESS_DENY_NETWORKS)


def _canonical_ip(address: str) -> str:
    try:
        return str(ipaddress.ip_address(address))
    except ValueError as exc:
        raise UnsafeURLError("Connected response peer was not a valid IP address") from exc


def _validate_response_peer(response: Any, expected_addresses: set[str] | None = None) -> str:
    extensions = getattr(response, "extensions", None) or {}
    stream = extensions.get("network_stream")
    if stream is None or not hasattr(stream, "get_extra_info"):
        raise UnsafeURLError("Unable to verify the connected response peer")

    peer = stream.get_extra_info("server_addr")
    if not isinstance(peer, (tuple, list)) or not peer:
        raise UnsafeURLError("Unable to determine the connected response peer")

    peer_address = _canonical_ip(str(peer[0]))
    _validate_public_ip(peer_address)
    if expected_addresses is not None:
        canonical_expected = {_canonical_ip(address) for address in expected_addresses}
        if peer_address not in canonical_expected:
            raise UnsafeURLError("Connected peer was not one of the validated addresses")
    return peer_address


async def _resolve_public_addresses(url: str) -> tuple[str, int, tuple[str, ...]]:
    host, port = _normalized_url_host(url)
    addresses = await resolve_public_addresses(
        host,
        port,
        allowed_ports=SAFE_EGRESS_ALLOWED_PORTS,
        denied_networks=SAFE_EGRESS_DENY_NETWORKS,
        dns_timeout_seconds=SAFE_EGRESS_DNS_TIMEOUT_SECONDS,
    )
    return host, port, addresses


async def validate_url_safety(url: str) -> str:
    """Validate scheme, hostname, and every resolved address for an outbound URL."""
    await _resolve_public_addresses(url)
    return url


def _host_header(host: str, port: int, scheme: str, explicit_port: int | None) -> str:
    try:
        is_ipv6 = ipaddress.ip_address(host).version == 6
    except ValueError:
        is_ipv6 = False
    value = f"[{host}]" if is_ipv6 else host
    default_port = 443 if scheme == "https" else 80
    if explicit_port is not None or port != default_port:
        value += f":{port}"
    return value


def _pinned_request(url: str, address: str) -> tuple[str, dict[str, str], dict[str, str]]:
    parsed = urlsplit(url)
    host, port = _normalized_url_host(url)
    ip = ipaddress.ip_address(address)
    pinned_host = f"[{ip}]" if ip.version == 6 else str(ip)
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    netloc = pinned_host
    if parsed.port is not None or port != default_port:
        netloc += f":{port}"

    pinned_url = urlunsplit(
        (
            parsed.scheme,
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )
    headers = dict(BROWSER_HEADERS)
    headers["Host"] = _host_header(host, port, parsed.scheme.lower(), parsed.port)
    return pinned_url, headers, {"sni_hostname": host}


async def _read_limited_response(response: Any, max_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise RuntimeError("Response has an invalid Content-Length header") from exc
        if declared_size < 0 or declared_size > max_bytes:
            raise RuntimeError(f"Response exceeds the {max_bytes}-byte limit")

    chunks = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise RuntimeError(f"Response exceeds the {max_bytes}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _require_crawl4ai_isolation() -> None:
    if not CRAWL4AI_NETWORK_ISOLATED:
        raise RuntimeError(
            "Crawl4AI is disabled until CRAWL4AI_NETWORK_ISOLATED=true is set after placing "
            "the service on an egress-only network with no access to private/backend services"
        )


async def _crawl4ai_post(path: str, payload: dict, timeout: float) -> tuple[bytes, str | None]:
    _require_crawl4ai_isolation()
    total_timeout = min(max(1.0, timeout), CRAWL4AI_TOTAL_TIMEOUT_SECONDS)
    runner_socket = os.getenv("WEB_RUNNER_SOCKET", "").strip()
    if runner_socket:
        if path not in {"/crawl", "/md"}:
            raise ValueError("Unsupported Crawl4AI operation")
        transport = httpx.AsyncHTTPTransport(uds=runner_socket)
        async with asyncio.timeout(total_timeout + 5.0):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://web-runner",
                timeout=httpx.Timeout(total_timeout + 5.0),
                trust_env=False,
            ) as client:
                async with client.stream(
                    "POST",
                    f"/v1/crawl4ai{path}",
                    params={"timeout_seconds": total_timeout},
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    body = await _read_limited_response(resp, CRAWL4AI_MAX_RESPONSE_BYTES)
                    return body, resp.headers.get("x-upstream-encoding") or resp.encoding

    if os.getenv("RESEARCH_REQUIRE_WEB_ISOLATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError("Crawl4AI requires the isolated web-runner socket")
    async with asyncio.timeout(total_timeout):
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), trust_env=False) as client:
            async with client.stream("POST", f"{CRAWL4AI_URL}{path}", json=payload) as resp:
                resp.raise_for_status()
                body = await _read_limited_response(resp, CRAWL4AI_MAX_RESPONSE_BYTES)
                return body, resp.encoding


def _decode_response_body(raw_body: bytes, encoding: str | None = None) -> str:
    for candidate in (encoding, "utf-8"):
        if not candidate:
            continue
        try:
            return raw_body.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw_body.decode("utf-8", errors="replace")


def _extract_pdf_text(raw_body: bytes) -> tuple[str, str | None, str | None]:
    if len(raw_body) > PDF_MAX_RESPONSE_BYTES:
        return "", None, f"PDF exceeds the {PDF_MAX_RESPONSE_BYTES}-byte limit"
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", None, "Direct PDF extraction requires the optional 'pypdf' package"

    try:
        reader = PdfReader(io.BytesIO(raw_body))
        if len(reader.pages) > PDF_MAX_PAGES:
            return "", None, f"PDF exceeds the {PDF_MAX_PAGES}-page limit"
        parts = []
        remaining = PDF_MAX_EXTRACTED_CHARS
        truncated = False
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            if len(page_text) > remaining:
                parts.append(page_text[:remaining])
                truncated = True
                break
            parts.append(page_text)
            remaining -= len(page_text)
            if remaining <= 0:
                truncated = True
                break
        text = "\n\n".join(parts)
        metadata = reader.metadata
        title = str(metadata.title).strip()[:2000] if metadata and metadata.title else None
        error = (
            f"PDF extracted text was truncated at {PDF_MAX_EXTRACTED_CHARS} characters"
            if truncated
            else None
        )
        return text.strip(), title, error
    except Exception as exc:
        return "", None, f"PDF extraction failed: {_safe_error_detail(exc)}"


def _normalize_pdf_result(result: Any) -> tuple[str, str | None, str | None]:
    if not isinstance(result, dict):
        return "", None, "PDF extraction subprocess returned invalid output"
    content = result.get("content")
    title = result.get("title")
    error = result.get("error")
    content = content if isinstance(content, str) else ""
    title = title[:2000] if isinstance(title, str) and title else None
    error = _safe_error_detail(RuntimeError(error)) if isinstance(error, str) and error else None
    if len(content) > PDF_MAX_EXTRACTED_CHARS:
        content = content[:PDF_MAX_EXTRACTED_CHARS]
        error = f"PDF extracted text was truncated at {PDF_MAX_EXTRACTED_CHARS} characters"
    return content, title, error


async def _extract_pdf_text_subprocess(raw_body: bytes) -> tuple[str, str | None, str | None]:
    if len(raw_body) > PDF_MAX_RESPONSE_BYTES:
        return "", None, f"PDF exceeds the {PDF_MAX_RESPONSE_BYTES}-byte limit"

    child_env = {
        key: os.environ[key]
        for key in ("LANG", "LC_ALL", "PATH")
        if os.environ.get(key)
    }
    child_env.update(
        {
            "PDF_MAX_RESPONSE_BYTES": str(PDF_MAX_RESPONSE_BYTES),
            "PDF_MAX_PAGES": str(PDF_MAX_PAGES),
            "PDF_MAX_EXTRACTED_CHARS": str(PDF_MAX_EXTRACTED_CHARS),
            "PDF_SANDBOX_MEMORY_BYTES": str(PDF_SANDBOX_MEMORY_BYTES),
            "PDF_SANDBOX_CPU_SECONDS": str(PDF_SANDBOX_CPU_SECONDS),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    if os.name == "nt" and os.environ.get("SYSTEMROOT"):
        child_env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    script_path = Path(__file__).with_name("pdf_sandbox.py")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=child_env,
    )
    try:
        async with asyncio.timeout(DIRECT_EXTRACTION_TIMEOUT_SECONDS):
            stdout, _ = await process.communicate(input=raw_body)
    except asyncio.CancelledError:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        raise
    except TimeoutError:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        return "", None, "PDF extraction exceeded its subprocess time limit"

    if process.returncode != 0:
        return "", None, "PDF extraction subprocess exceeded a resource limit or failed"
    if len(stdout) > PDF_SANDBOX_OUTPUT_BYTES:
        return "", None, "PDF extraction subprocess exceeded its output limit"
    try:
        result = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "", None, "PDF extraction subprocess returned invalid output"
    return _normalize_pdf_result(result)


async def _extract_pdf_text_runner(raw_body: bytes) -> tuple[str, str | None, str | None]:
    transport = httpx.AsyncHTTPTransport(uds=PDF_RUNNER_SOCKET)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://pdf-runner",
            timeout=httpx.Timeout(DIRECT_EXTRACTION_TIMEOUT_SECONDS),
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST",
                "/v1/extract",
                content=raw_body,
                headers={"content-type": "application/pdf"},
            ) as response:
                response.raise_for_status()
                encoded = await _read_limited_response(response, PDF_SANDBOX_OUTPUT_BYTES)
    except Exception as exc:
        raise RuntimeError("Isolated PDF extraction failed") from exc
    try:
        result = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Isolated PDF extraction returned invalid output") from exc
    return _normalize_pdf_result(result)


async def _extract_pdf_text_sandboxed(raw_body: bytes) -> tuple[str, str | None, str | None]:
    require_isolation = os.getenv("RESEARCH_REQUIRE_PDF_ISOLATION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if PDF_RUNNER_SOCKET:
        return await _extract_pdf_text_runner(raw_body)
    if require_isolation:
        raise RuntimeError("PDF extraction requires the isolated pdf-runner")
    return await _extract_pdf_text_subprocess(raw_body)


def extract_direct_response(
    raw_body: bytes,
    content_type: str,
    encoding: str | None = None,
) -> tuple[str, str | None, str, str | None]:
    """Return extracted text, title, format label, and an optional extraction error."""
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    stripped = raw_body.lstrip()

    if media_type == "application/pdf" or stripped.startswith(b"%PDF-"):
        text, title, error = _extract_pdf_text(raw_body)
        return text, title, "pdf", error

    decoded = _decode_response_body(raw_body, encoding)
    decoded_stripped = decoded.lstrip()

    if "json" in media_type or decoded_stripped.startswith(("{", "[")):
        return parse_maybe_json_text(decoded), None, "json", None

    looks_html = (
        media_type in {"text/html", "application/xhtml+xml"}
        or decoded_stripped[:100].lower().startswith(("<!doctype html", "<html"))
    )
    if looks_html:
        return html_to_text(decoded), extract_title_from_html(decoded), "html", None

    if media_type.endswith("xml") or media_type in {"application/atom+xml", "application/rss+xml"}:
        return html_to_text(decoded), None, "xml", None

    if media_type.startswith("text/") or not media_type:
        return decoded.strip(), None, "text", None

    if b"\x00" not in raw_body[:4096]:
        return decoded.strip(), None, "text", None

    return "", None, "unsupported", f"Unsupported response content type: {media_type or 'unknown'}"


def extract_direct_response_details(
    raw_body: bytes,
    content_type: str,
    encoding: str | None = None,
) -> tuple[str, str | None, str, str | None, dict[str, int]]:
    text, title, body_format, extraction_error = extract_direct_response(
        raw_body,
        content_type,
        encoding,
    )
    metrics: dict[str, int] = {"raw_html_chars": 0}
    if body_format in {"html", "json", "text", "xml"}:
        decoded = _decode_response_body(raw_body, encoding)
        metrics["raw_html_chars"] = len(decoded)
        if body_format == "html":
            metrics["primary_content_chars"] = _direct_html_primary_content_chars(
                decoded
            )
    return text, title, body_format, extraction_error, metrics


async def crawl4ai_request(payload: dict, timeout: float = 180.0) -> dict:
    urls = payload.get("urls", []) if isinstance(payload, dict) else []
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        await validate_url_safety(url)

    body, encoding = await _crawl4ai_post("/crawl", payload, timeout)
    decoded = _decode_response_body(body, encoding)
    data = json.loads(decoded)
    if not isinstance(data, dict):
        raise ValueError("Crawl4AI returned a non-object JSON response")
    return data


async def crawl4ai_markdown_request(url: str, timeout: float = 120.0) -> dict:
    await validate_url_safety(url)
    body, encoding = await _crawl4ai_post("/md", {"url": url, "f": "fit", "c": "0"}, timeout)
    decoded = _decode_response_body(body, encoding)
    try:
        data = json.loads(decoded)
    except Exception:
        data = {"url": url, "markdown": decoded, "success": True}
    if not isinstance(data, dict):
        raise ValueError("Crawl4AI /md returned a non-object response")

    markdown = data.get("markdown") or ""
    result_url = urljoin(url, str(data.get("url") or url))
    await validate_url_safety(result_url)
    return {
        "url": result_url,
        "markdown": markdown,
        "content": markdown,
        "title": None,
        "success": bool(data.get("success", True)),
        "extraction_method": "crawl4ai_md",
    }


def first_crawl4ai_result(data: dict) -> dict:
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for item in data["results"]:
            if isinstance(item, dict):
                item = dict(item)
                item["_crawl4ai_success"] = data.get("success")
                item["_crawl4ai_server_processing_time_s"] = data.get("server_processing_time_s")
                return item
        return {}

    return data if isinstance(data, dict) else {}


def extract_markdown(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        for key in ("fit_markdown", "raw_markdown", "markdown_with_citations", "markdown"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text

    return ""


def extract_content(crawl_data: dict) -> str:
    content = crawl_data.get("content") or crawl_data.get("cleaned_text") or ""

    if not content:
        content = extract_markdown(crawl_data.get("markdown"))

    if not content and crawl_data.get("extracted_content"):
        extracted = crawl_data.get("extracted_content")
        content = parse_maybe_json_text(extracted) if isinstance(extracted, str) else json.dumps(extracted)

    if not content and crawl_data.get("html"):
        content = html_to_text(crawl_data.get("html") or "")

    if isinstance(content, (dict, list)):
        content = json.dumps(content)
    elif content is not None and not isinstance(content, str):
        content = str(content)

    return (content or "").strip()


def extract_title(crawl_data: dict, fallback: str | None = None) -> str | None:
    title = crawl_data.get("title")
    if title:
        return title

    metadata = crawl_data.get("metadata")
    if isinstance(metadata, dict):
        title = metadata.get("title")
        if title:
            return title

    return fallback


async def direct_fetch_url(url: str) -> dict:
    async with asyncio.timeout(DIRECT_TOTAL_TIMEOUT_SECONDS):
        current_url = url
        redirect_chain = []
        raw_body = b""
        response_headers: dict[str, str] = {}
        response_encoding = None
        status_code = 0

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                _, _, addresses = await _resolve_public_addresses(current_url)
                last_connect_error = None
                response_received = False

                for address in addresses:
                    pinned_url, headers, extensions = _pinned_request(current_url, address)
                    try:
                        async with client.stream(
                            "GET",
                            pinned_url,
                            headers=headers,
                            extensions=extensions,
                        ) as resp:
                            _validate_response_peer(resp, {address})
                            response_headers = dict(resp.headers)
                            response_encoding = resp.encoding
                            status_code = resp.status_code

                            location = resp.headers.get("location")
                            if status_code in REDIRECT_STATUS_CODES and location:
                                raw_body = b""
                            else:
                                resp.raise_for_status()
                                response_media_type = (
                                    resp.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                                )
                                response_limit = (
                                    min(DIRECT_MAX_RESPONSE_BYTES, PDF_MAX_RESPONSE_BYTES)
                                    if response_media_type == "application/pdf"
                                    else DIRECT_MAX_RESPONSE_BYTES
                                )
                                raw_body = await _read_limited_response(resp, response_limit)
                            response_received = True
                            break
                    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                        last_connect_error = exc
                        continue

                if not response_received:
                    if last_connect_error:
                        raise last_connect_error
                    raise RuntimeError("No validated address accepted the request")

                location = response_headers.get("location")
                if status_code in REDIRECT_STATUS_CODES and location:
                    if redirect_count >= MAX_REDIRECTS:
                        raise RuntimeError(f"Too many redirects (maximum {MAX_REDIRECTS})")

                    next_url = urljoin(current_url, location)
                    await validate_url_safety(next_url)
                    redirect_chain.append(next_url)
                    current_url = next_url
                    continue
                break
            else:
                raise RuntimeError(f"Too many redirects (maximum {MAX_REDIRECTS})")

        content_type = response_headers.get("content-type", "")
        direct_metrics: dict[str, int] = {"raw_html_chars": 0}
        try:
            async with asyncio.timeout(DIRECT_EXTRACTION_TIMEOUT_SECONDS):
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type == "application/pdf" or raw_body.lstrip().startswith(b"%PDF-"):
                    text, title, extraction_error = await _extract_pdf_text_sandboxed(raw_body)
                    body_format = "pdf"
                else:
                    (
                        text,
                        title,
                        body_format,
                        extraction_error,
                        direct_metrics,
                    ) = await asyncio.to_thread(
                        extract_direct_response_details,
                        raw_body,
                        content_type,
                        response_encoding,
                    )
        except TimeoutError as exc:
            raise RuntimeError("Direct response extraction exceeded its time limit") from exc

    result = {
        "url": current_url,
        "status_code": status_code,
        "content_type": content_type,
        "title": title,
        "content": text,
        "markdown": text,
        "raw_html_chars": direct_metrics.get("raw_html_chars", 0),
        "raw_body_bytes": len(raw_body),
        "body_format": body_format,
        "redirect_chain": redirect_chain,
        "extraction_method": "direct_http_fallback",
    }

    if extraction_error:
        result["extraction_error"] = extraction_error
    if body_format == "html":
        result["_direct_primary_content_chars"] = direct_metrics.get(
            "primary_content_chars",
            0,
        )

    return result


def crawl4ai_payload(url: str, crawler_config: dict | None = None) -> dict:
    # Current Crawl4AI Docker API expects urls/browser_config/crawler_config.
    return {
        "urls": [url],
        "browser_config": {},
        "crawler_config": crawler_config or {},
    }


def _direct_result_is_sufficient(result: dict) -> bool:
    content = extract_content(result)
    if result.get("extraction_error") or len(content) < DIRECT_FIRST_MIN_CONTENT_CHARS:
        return False
    preview = content[:20_000].lower()
    if any(marker in preview for marker in DIRECT_LOW_CONFIDENCE_MARKERS):
        return False
    if result.get("body_format") == "html":
        primary_chars = result.get("_direct_primary_content_chars")
        if not isinstance(primary_chars, int):
            return False
        minimum_primary_chars = max(200, DIRECT_FIRST_MIN_CONTENT_CHARS // 2)
        if primary_chars < minimum_primary_chars:
            return False
    return True


async def _cancel_and_drain(task: asyncio.Task | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def crawl_url_impl(url: str, config: dict | None = None) -> dict:
    await validate_url_safety(url)
    payload = crawl4ai_payload(url, config or {})
    crawl4ai_errors: list[str] = []
    direct_error: str | None = None
    direct_result: dict | None = None
    direct_task: asyncio.Task | None = asyncio.create_task(direct_fetch_url(url))
    crawl4ai_task: asyncio.Task | None = None

    try:
        done, _ = await asyncio.wait(
            {direct_task},
            timeout=DIRECT_FIRST_HEDGE_SECONDS,
        )
        if direct_task in done:
            try:
                direct_result = await direct_task
            except Exception as exc:
                direct_error = f"direct fetch failed: {_safe_error_detail(exc)}"
            direct_task = None
            if direct_result is not None:
                if _direct_result_is_sufficient(direct_result):
                    return direct_result
                direct_result["_direct_low_confidence"] = True

        crawl4ai_task = asyncio.create_task(crawl4ai_request(payload))
        while direct_task is not None or crawl4ai_task is not None:
            active_tasks = {
                task for task in (direct_task, crawl4ai_task) if task is not None
            }
            done, _ = await asyncio.wait(
                active_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Prefer the richer Crawl4AI result when both hedged requests finish
            # in the same event-loop turn.
            if crawl4ai_task is not None and crawl4ai_task in done:
                try:
                    data = first_crawl4ai_result(await crawl4ai_task)
                    if isinstance(data, dict):
                        for result_url in (
                            data.get("url"),
                            data.get("final_url"),
                            data.get("redirected_url"),
                        ):
                            if result_url:
                                await validate_url_safety(urljoin(url, str(result_url)))
                    content = extract_content(data)
                    if content and len(content) >= 200:
                        data["extraction_method"] = "crawl4ai"
                        return data
                    crawl4ai_errors.append("Crawl4AI returned too little content")
                except Exception as exc:
                    crawl4ai_errors.append(_safe_error_detail(exc))
                crawl4ai_task = None

            if direct_task is not None and direct_task in done:
                try:
                    direct_result = await direct_task
                except Exception as exc:
                    direct_error = f"direct fetch failed: {_safe_error_detail(exc)}"
                direct_task = None
                if direct_result is not None:
                    if _direct_result_is_sufficient(direct_result):
                        return direct_result
                    direct_result["_direct_low_confidence"] = True

        # The isolated web runner deliberately rejects /md because that endpoint
        # cannot be forced through its pinned public-only proxy.
        if not os.getenv("WEB_RUNNER_SOCKET", "").strip():
            try:
                data = await crawl4ai_markdown_request(url)
                content = extract_content(data)
                if content and len(content) >= 200:
                    data["crawl4ai_errors"] = crawl4ai_errors
                    return data
                crawl4ai_errors.append("Crawl4AI /md returned too little content")
            except Exception as exc:
                crawl4ai_errors.append(
                    f"crawl4ai /md failed: {_safe_error_detail(exc)}"
                )

        if direct_result is not None:
            direct_result["crawl4ai_errors"] = crawl4ai_errors
            return direct_result

        errors = list(crawl4ai_errors)
        if direct_error:
            errors.append(direct_error)
        raise RuntimeError("; ".join(errors) or "All crawl strategies failed")
    finally:
        await _cancel_and_drain(direct_task)
        await _cancel_and_drain(crawl4ai_task)
