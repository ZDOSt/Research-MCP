import asyncio
import base64
import html
import json
import os
import re
from typing import Any, List, Optional
from urllib.parse import urlparse

import httpx
from playwright.async_api import async_playwright

from crawler import BROWSER_HEADERS, validate_url_safety
from extractors import extract_table_like_rows, html_to_text, parse_maybe_json_text, unique_preserve_order
from redaction import redact_sensitive_text

DEFAULT_MAX_CHARS = 300000
ABSOLUTE_MAX_CHARS = 750000
NETWORK_BODY_LIMIT = 1_000_000
NETWORK_TEXT_LIMIT = 8_000
NETWORK_PREVIEW_LIMIT = 600
NETWORK_COMBINED_TEXT_LIMIT = 4_000
MAX_NETWORK_RESPONSES = 6
MAX_NETWORK_CANDIDATES = 40
NETWORK_MIN_SCORE = 8
MAX_NETWORK_CAPTURE_TASKS = 8
MAX_DOM_ACCUMULATED_CHARS = 1_000_000
DOM_SNAPSHOT_CHAR_LIMIT = 120_000
MAX_CLICK_CANDIDATES = 250
MAX_BROWSER_REQUESTS = 200
BROWSER_TOTAL_TIMEOUT_SECONDS = max(
    15.0,
    float(os.getenv("RESEARCH_BROWSER_TOTAL_TIMEOUT_SECONDS", "90")),
)
WEB_RUNNER_MAX_RESPONSE_BYTES = max(
    1024,
    int(os.getenv("WEB_RUNNER_MAX_RESPONSE_BYTES", str(8 * 1024 * 1024))),
)

STATIC_URL_RE = re.compile(
    r"\.(?:js|mjs|css|png|jpe?g|gif|webp|svg|ico|woff2?|ttf|otf|mp4|webm|m3u8|ts|map)(?:[?#]|$)",
    re.I,
)
HARD_NOISY_URL_MARKERS = [
    "adtech", "advertis", "analytics", "beacon", "brightline.tv", "cookielaw",
    "comscore", "consent", "doubleclick", "fave", "googletag", "gtm.js",
    "hotjar", "onetrust", "optimizely", "prebid", "scorecardresearch",
    "segment.io", "sentry", "sourcepoint", "tinypass", "tracking",
    "web-vitals", "widgetapi", "youtube.com/iframe",
]
SOFT_NOISY_URL_MARKERS = [
    "/assets/", "/bundles/", "/dist/", "/static/", "bootstrap", "chunk-",
    "feature-flag", "font", "metrics", "player", "polyfill", "runtime-config",
    "session-context", "telemetry", "token", "vendor", "webpack",
]
STATIC_RESOURCE_TYPES = {"script", "stylesheet", "image", "media", "font", "manifest"}
CAPTURABLE_RESOURCE_TYPES = {"xhr", "fetch"}
STATIC_CONTENT_TYPE_MARKERS = [
    "javascript", "ecmascript", "text/css", "font/", "image/", "video/", "audio/",
    "mpegurl", "dash+xml", "octet-stream",
]
DATA_PATH_SEGMENTS = {
    "api", "graphql", "gql", "content", "contents", "article", "articles",
    "story", "stories", "live", "search", "product", "products", "commerce",
    "connect", "aura", "apexremote", "sfsites", "webruntime", "services",
    "catalog", "reference", "references", "spec", "specs", "specification",
    "specifications", "equipment", "maintenance", "docs", "documentation",
    "download", "downloads", "extension", "extensions", "plugin", "plugins",
    "manifest", "package", "packages", "readme", "repo", "repos", "repository",
    "raw", "registry", "release", "releases", "changelog", "schema", "schemas",
    "config", "configs", "settings", "source", "sources", "metadata", "version",
    "versions", "module", "modules", "compose", "list", "lists", "table",
    "tables", "data", "dataset", "datasets", "feed", "feeds",
}
DATA_FILE_RE = re.compile(r"\.(?:json|ndjson|graphql|xml)(?:[?#]|$)", re.I)
NETWORK_RELEVANCE_STOP_WORDS = {
    "about", "after", "also", "answer", "before", "compare", "details", "events",
    "extract", "find", "from", "guide", "help", "information", "into", "learn",
    "made", "more", "news", "overview", "page", "please", "random", "reference",
    "search", "show", "summarize", "that", "this", "using", "verify", "what",
    "when", "where", "with",
}

_browser_semaphore = asyncio.Semaphore(1)
_resolved_chromium_sandbox: Optional[bool] = None


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def chromium_sandbox_mode() -> str:
    if env_flag("RESEARCH_BROWSER_DISABLE_SANDBOX"):
        return "disabled"
    mode = os.getenv("RESEARCH_BROWSER_SANDBOX_MODE", "required").strip().lower()
    if mode not in {"auto", "required", "disabled"}:
        raise ValueError(
            "RESEARCH_BROWSER_SANDBOX_MODE must be auto, required, or disabled"
        )
    return mode


def set_resolved_chromium_sandbox(enabled: Optional[bool]) -> None:
    global _resolved_chromium_sandbox
    _resolved_chromium_sandbox = enabled


def chromium_sandbox_enabled() -> bool:
    if _resolved_chromium_sandbox is not None:
        return _resolved_chromium_sandbox
    return chromium_sandbox_mode() != "disabled"


def chromium_launch_options(*, sandbox_enabled: Optional[bool] = None) -> dict[str, Any]:
    proxy_url = os.getenv("RESEARCH_BROWSER_PROXY", "").strip()
    if sandbox_enabled is None:
        sandbox_enabled = chromium_sandbox_enabled()
    options: dict[str, Any] = {
        "headless": True,
        "chromium_sandbox": sandbox_enabled,
        "args": ["--disable-dev-shm-usage", "--disable-gpu"],
    }
    if proxy_url:
        options["proxy"] = {"server": proxy_url, "bypass": ""}
        options["args"].extend(
            [
                "--disable-quic",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--proxy-bypass-list=<-loopback>",
            ]
        )
    return options


def safe_exception_detail(exc: Exception, max_chars: int = 1000) -> str:
    redacted, _ = redact_sensitive_text(str(exc))
    return redacted[:max_chars]


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def normalize_heading(text: str) -> str:
    text = html.unescape(text or "")
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def root_domain(domain: str) -> str:
    parts = (domain or "").lower().split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    return ".".join(parts[-2:])


def response_path_query(response_url: str) -> str:
    parsed = urlparse(response_url)
    return f"{parsed.path}?{parsed.query}".lower()


def response_path_tokens(response_url: str) -> set:
    return set(re.findall(r"[a-z0-9]{2,}", response_path_query(response_url)))


def has_noisy_network_signal(response_url: str) -> bool:
    lower_url = response_url.lower()
    return bool(STATIC_URL_RE.search(lower_url)) or any(marker in lower_url for marker in HARD_NOISY_URL_MARKERS)


def has_soft_noisy_network_signal(response_url: str) -> bool:
    lower_url = response_url.lower()
    return any(marker in lower_url for marker in SOFT_NOISY_URL_MARKERS)


def has_data_endpoint_signal(response_url: str) -> bool:
    path_query = response_path_query(response_url)
    if DATA_FILE_RE.search(path_query):
        return True

    tokens = response_path_tokens(response_url)
    return bool(tokens & DATA_PATH_SEGMENTS)


def network_relevance_terms(task: Optional[str], labels: Optional[List[str]]) -> List[str]:
    label_text = " ".join(labels or []) if labels and len(labels) <= 12 else ""
    text = " ".join([task or "", label_text])
    terms = []

    for term in re.findall(r"[a-z0-9][a-z0-9-]{2,}", text.lower()):
        if len(term) < 4 or term in NETWORK_RELEVANCE_STOP_WORDS:
            continue
        terms.append(term)

    return unique_preserve_order(terms)[:25]


def should_capture_network_response(
    response_url: str,
    content_type: str,
    resource_type: str,
    start_domain: str,
) -> bool:
    lower_url = response_url.lower()
    content_type = (content_type or "").lower()
    resource_type = (resource_type or "").lower()

    if resource_type in STATIC_RESOURCE_TYPES:
        return False
    if has_noisy_network_signal(lower_url):
        return False
    if any(marker in content_type for marker in STATIC_CONTENT_TYPE_MARKERS):
        return False
    if resource_type and resource_type not in CAPTURABLE_RESOURCE_TYPES:
        return False

    is_json = "json" in content_type or "graphql" in content_type
    is_xml = "xml" in content_type
    is_text = "text/plain" in content_type
    if not (is_json or is_xml or is_text):
        return False

    # DOM extraction already captures rendered HTML. Treat network capture as a
    # data-channel only, otherwise browser/framework assets become "evidence".
    if "text/html" in content_type:
        return False

    parsed = urlparse(response_url)
    response_domain = parsed.netloc.lower()
    same_site = root_domain(response_domain) == root_domain(start_domain)
    data_endpoint = has_data_endpoint_signal(response_url)

    if same_site:
        return data_endpoint or is_json or is_xml

    return data_endpoint and (is_json or is_xml)


def looks_like_script_or_config(text: str) -> bool:
    sample = (text or "").lstrip()[:4000]
    lower = sample.lower()
    script_markers = [
        "webpack", "function(", "=>", "window.", "document.", "createscript",
        "sourcemappingurl", "__webpack_require__", "define(", "var ", "const ",
    ]
    if any(marker in lower for marker in script_markers) and sample.count(";") >= 8:
        return True

    return False


def score_network_response(
    item: dict,
    start_domain: str,
    task: Optional[str],
    labels: Optional[List[str]],
) -> int:
    response_url = item.get("url") or ""
    content_type = (item.get("content_type") or "").lower()
    parsed = urlparse(response_url)
    response_domain = parsed.netloc.lower()
    same_site = root_domain(response_domain) == root_domain(start_domain)
    data_endpoint = has_data_endpoint_signal(response_url)
    text = item.get("text") or item.get("preview") or ""
    lower_text = text.lower()
    lower_url = response_url.lower()
    status = item.get("status") or 0

    score = 0
    if 200 <= int(status) < 300:
        score += 1
    if "json" in content_type or "graphql" in content_type:
        score += 4
    elif "xml" in content_type:
        score += 3
    elif "text/plain" in content_type:
        score += 1
    if same_site:
        score += 2
    if data_endpoint:
        score += 3

    term_hits = 0
    for term in network_relevance_terms(task, labels):
        if term in lower_text or term in lower_url:
            term_hits += 1
    score += min(term_hits, 5) * 2

    if len(text.strip()) < 80 and term_hits == 0:
        score -= 2
    if has_noisy_network_signal(response_url) or looks_like_script_or_config(text):
        score -= 20
    elif has_soft_noisy_network_signal(response_url) and term_hits == 0:
        score -= 8

    item["_network_score"] = score
    item["_network_term_hits"] = term_hits
    item["_network_data_endpoint"] = data_endpoint
    item["_network_same_site"] = same_site
    return score


def select_network_responses(
    responses: List[dict],
    start_domain: str,
    task: Optional[str],
    labels: Optional[List[str]],
) -> List[dict]:
    deduped = []
    seen_urls = set()

    for item in responses:
        response_url = item.get("url")
        if not response_url or response_url in seen_urls:
            continue
        seen_urls.add(response_url)

        score = score_network_response(item, start_domain, task, labels)
        if score < NETWORK_MIN_SCORE:
            continue
        if not item.get("_network_same_site") and not item.get("_network_term_hits"):
            continue
        if not item.get("_network_data_endpoint") and not item.get("_network_term_hits"):
            continue

        deduped.append(item)

    deduped.sort(key=lambda item: item.get("_network_score", 0), reverse=True)
    return deduped[:MAX_NETWORK_RESPONSES]


def bounded_network_response_length(headers: dict) -> Optional[int]:
    """Return a safe capture size, or None when Playwright would need an unbounded read."""
    content_encoding = (headers.get("content-encoding") or "").strip().lower()
    if content_encoding not in {"", "identity"}:
        return None

    raw_length = (headers.get("content-length") or "").strip()
    if not raw_length:
        return None
    try:
        length = int(raw_length)
    except ValueError:
        return None
    if length <= 0 or length > NETWORK_BODY_LIMIT:
        return None
    return length


def decode_bounded_cdp_body(
    payload: dict,
    declared_length: int,
    encoded_data_length: float,
) -> Optional[str]:
    """Decode a CDP body only after Chrome reports a bounded transfer."""
    if encoded_data_length <= 0 or encoded_data_length > NETWORK_BODY_LIMIT:
        return None
    raw_body = payload.get("body")
    if not isinstance(raw_body, str) or not raw_body:
        return None
    try:
        body_bytes = (
            base64.b64decode(raw_body, validate=True)
            if payload.get("base64Encoded")
            else raw_body.encode("utf-8", errors="replace")
        )
    except (ValueError, TypeError):
        return None
    if not body_bytes or len(body_bytes) > NETWORK_BODY_LIMIT or len(body_bytes) > declared_length:
        return None
    return body_bytes.decode("utf-8", errors="replace")


def safe_diagnostic_url(value: str) -> str:
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.hostname:
        return parsed.scheme or "unknown"
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        host += f":{port}"
    return f"{parsed.scheme}://{host}"



def build_click_script(
    labels: List[str],
    max_clicks: int = 8,
    max_candidates: int = MAX_CLICK_CANDIDATES,
    text_limit: int = DOM_SNAPSHOT_CHAR_LIMIT,
) -> str:
    labels_json = json.dumps(
        [normalize_heading(label) for label in labels[:50] if normalize_heading(label)]
    )
    max_clicks = clamp_int(max_clicks, 1, 20)
    max_candidates = clamp_int(max_candidates, 20, MAX_CLICK_CANDIDATES)
    text_limit = clamp_int(text_limit, 1000, DOM_SNAPSHOT_CHAR_LIMIT)

    return f"""
    async () => {{
      const labels = {labels_json};
      const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
      const normalize = (text) => String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\\s+/g, ' ').trim();
      const destructive = /\\b(delete|remove|destroy|purchase|buy|checkout|pay|submit|sign out|log out|unsubscribe|cancel order|confirm)\\b/i;
      const visibleText = (el) => (
        el.innerText ||
        el.textContent ||
        el.getAttribute('aria-label') ||
        el.getAttribute('title') ||
        el.getAttribute('data-label') ||
        el.getAttribute('name') ||
        ''
      ).trim();

      const selector = [
        'button[type="button"]',
        'button:not([type])',
        '[role="tab"]',
        '[role="button"]',
        'summary',
        '[aria-expanded]',
        '[aria-controls]',
        '[data-toggle]',
        '[data-bs-toggle]',
        '[data-testid]',
        '[data-tab]',
        '.tab',
        '.accordion-button',
        '[class*="tab"]',
        '[class*="accordion"]',
        '[class*="load"]',
        '[class*="more"]',
        'a[href^="#"][role="button"]',
        'a[href^="#"][aria-controls]'
      ].join(',');

      const candidates = Array.from(document.querySelectorAll(selector)).slice(0, {max_candidates});
      const clicked = [];
      const startUrl = location.href;

      const safeToClick = (el, text) => {{
        if (!el || destructive.test(text)) return false;
        if (el.matches(':disabled, [disabled], [aria-disabled="true"]')) return false;
        if (el.closest('form')) return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
        if (el.tagName === 'A') {{
          const href = el.getAttribute('href') || '';
          if (!href.startsWith('#')) return false;
        }}
        return true;
      }};

      for (const label of labels) {{
        for (const el of candidates) {{
          if (clicked.length >= {max_clicks}) break;
          const text = normalize(visibleText(el));
          if (!text) continue;
          if (clicked.includes(text)) continue;

          const matches = text === label || text.startsWith(label + ' ') || label.startsWith(text + ' ');

          if (matches && safeToClick(el, text)) {{
            try {{
              el.scrollIntoView({{block: 'center', inline: 'center'}});
              await sleep(100);
              el.click();
              clicked.push(text);
              await sleep(500);
              if (location.href !== startUrl) break;
            }} catch (e) {{}}
          }}
        }}
        if (clicked.length >= {max_clicks} || location.href !== startUrl) break;
      }}

      return {{
        clicked,
        title: document.title,
        url: location.href,
        navigation_changed: location.href !== startUrl,
        text: document.body ? document.body.innerText.slice(0, {text_limit}) : ''
      }};
    }}
    """


def build_scrollable_capture_script(
    labels: List[str],
    profile: str,
    max_chars: int = MAX_DOM_ACCUMULATED_CHARS,
) -> str:
    labels_json = json.dumps(
        [normalize_heading(label) for label in labels[:50] if normalize_heading(label)]
    )
    page_steps = 4 if profile == "targeted" else 6 if profile == "balanced" else 8
    container_steps = 4 if profile == "targeted" else 6 if profile == "balanced" else 8
    max_elements = 6 if profile == "targeted" else 10 if profile == "balanced" else 14
    max_chars = clamp_int(max_chars, 10_000, MAX_DOM_ACCUMULATED_CHARS)

    return f"""
    async () => {{
      const labels = {labels_json};
      const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));
      const normalize = (text) => String(text || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\\s+/g, ' ').trim();
      const maxChars = {max_chars};
      const perSnapshotChars = {DOM_SNAPSHOT_CHAR_LIMIT};
      let capturedChars = 0;
      let lastSnapshot = '';

      const isScrollable = (el) => {{
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const overflowY = style.overflowY;
        const overflowX = style.overflowX;
        const canScrollY = el.scrollHeight > el.clientHeight + 20 && ['auto', 'scroll', 'overlay'].includes(overflowY);
        const canScrollX = el.scrollWidth > el.clientWidth + 20 && ['auto', 'scroll', 'overlay'].includes(overflowX);
        return canScrollY || canScrollX;
      }};

      const textNearLabels = (el) => {{
        const text = normalize((el.innerText || el.textContent || '').slice(0, perSnapshotChars));
        if (!labels.length) return true;
        return labels.some(label => text.includes(label));
      }};

      const pushSnapshot = (value, snapshots) => {{
        if (capturedChars >= maxChars) return false;
        const raw = String(value || '').trim();
        if (!raw || raw === lastSnapshot) return true;
        const remaining = maxChars - capturedChars;
        const part = raw.slice(0, Math.min(perSnapshotChars, remaining));
        if (!part) return false;
        snapshots.push(part);
        capturedChars += part.length;
        lastSnapshot = raw.slice(0, perSnapshotChars);
        return capturedChars < maxChars;
      }};

      const all = Array.from(document.querySelectorAll('body, main, section, article, div, table, tbody, [role="table"], [role="grid"], [class*="table"], [class*="grid"], [class*="scroll"], [class*="list"], [class*="result"], [class*="data"]'));
      const scrollables = all.filter(isScrollable).slice(0, {max_elements});
      const snapshots = [];
      const used = [];

      for (const el of scrollables) {{
        if (capturedChars >= maxChars) break;
        try {{
          const text = normalize((el.innerText || el.textContent || '').slice(0, perSnapshotChars));
          const relevant = textNearLabels(el) || text.length > 200;
          if (!relevant) continue;

          used.push({{
            tag: el.tagName,
            className: el.className ? String(el.className).slice(0, 200) : '',
            id: el.id || '',
            scrollHeight: el.scrollHeight,
            clientHeight: el.clientHeight,
            scrollWidth: el.scrollWidth,
            clientWidth: el.clientWidth
          }});

          const maxY = Math.max(0, el.scrollHeight - el.clientHeight);
          const maxX = Math.max(0, el.scrollWidth - el.clientWidth);
          const ySteps = {container_steps};
          const xSteps = maxX > 100 ? 2 : 1;

          for (let yi = 0; yi <= ySteps; yi++) {{
            if (capturedChars >= maxChars) break;
            el.scrollTop = Math.floor(maxY * yi / ySteps);

            for (let xi = 0; xi <= xSteps; xi++) {{
              if (capturedChars >= maxChars) break;
              el.scrollLeft = Math.floor(maxX * xi / Math.max(1, xSteps));
              await sleep(125);
              const part = el.innerText || el.textContent || '';
              pushSnapshot(part, snapshots);
            }}
          }}
        }} catch (e) {{}}
      }}

      const pageSnapshots = [];
      const pageSteps = {page_steps};
      for (let i = 0; i <= pageSteps; i++) {{
        if (capturedChars >= maxChars) break;
        window.scrollTo(0, Math.floor(document.body.scrollHeight * i / pageSteps));
        await sleep(150);
        pushSnapshot(document.body ? document.body.innerText : '', pageSnapshots);
      }}

      window.scrollTo(0, 0);
      await sleep(150);

      return {{
        title: document.title,
        url: location.href,
        scrollable_elements: used,
        budget_exhausted: capturedChars >= maxChars,
        text: [...pageSnapshots, ...snapshots].join('\\n\\n').slice(0, maxChars)
      }};
    }}
    """


async def _read_runner_response(response: httpx.Response) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise RuntimeError("web-runner returned an invalid Content-Length header") from exc
        if declared_size < 0 or declared_size > WEB_RUNNER_MAX_RESPONSE_BYTES:
            raise RuntimeError("web-runner response exceeds the byte limit")

    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > WEB_RUNNER_MAX_RESPONSE_BYTES:
            raise RuntimeError("web-runner response exceeds the byte limit")
        body.extend(chunk)
    return bytes(body)


async def _remote_playwright_explore_page(
    socket_path: str,
    url: str,
    labels: Optional[List[str]],
    task: Optional[str],
    max_chars: int,
    profile: str,
    timeout_ms: int,
) -> dict:
    labels = unique_preserve_order(str(label)[:120] for label in (labels or [])[:50])
    task = str(task or "")[:4000]
    max_chars = clamp_int(max_chars, 10_000, ABSOLUTE_MAX_CHARS)
    profile = profile if profile in {"targeted", "balanced", "exhaustive"} else "targeted"
    timeout_ms = clamp_int(timeout_ms, 5_000, 60_000)
    transport = httpx.AsyncHTTPTransport(uds=socket_path)
    timeout_seconds = BROWSER_TOTAL_TIMEOUT_SECONDS + 15.0
    async with asyncio.timeout(timeout_seconds):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://web-runner",
            timeout=httpx.Timeout(timeout_seconds),
            trust_env=False,
        ) as client:
            async with client.stream(
                "POST",
                "/v1/explore",
                json={
                    "url": url,
                    "labels": labels,
                    "task": task,
                    "max_chars": max_chars,
                    "profile": profile,
                    "timeout_ms": timeout_ms,
                },
            ) as response:
                response.raise_for_status()
                body = await _read_runner_response(response)
    try:
        result = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("web-runner returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("web-runner returned a non-object response")
    return result


async def playwright_explore_page(
    url: str,
    labels: Optional[List[str]] = None,
    task: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    profile: str = "targeted",
    timeout_ms: int = 60000,
) -> dict:
    socket_path = os.getenv("WEB_RUNNER_SOCKET", "").strip()
    if socket_path:
        return await _remote_playwright_explore_page(
            socket_path=socket_path,
            url=url,
            labels=labels,
            task=task,
            max_chars=max_chars,
            profile=profile,
            timeout_ms=timeout_ms,
        )
    proxy_url = os.getenv("RESEARCH_BROWSER_PROXY", "").strip()
    if (
        not env_flag("RESEARCH_REQUIRE_WEB_ISOLATION")
        or not env_flag("RESEARCH_WEB_NETWORK_ISOLATED")
        or not proxy_url.lower().startswith("socks5://")
    ):
        raise RuntimeError("Browser execution requires the isolated web-runner socket")
    return await playwright_explore_page_local(
        url=url,
        labels=labels,
        task=task,
        max_chars=max_chars,
        profile=profile,
        timeout_ms=timeout_ms,
    )


async def playwright_explore_page_local(
    url: str,
    labels: Optional[List[str]] = None,
    task: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    profile: str = "targeted",
    timeout_ms: int = 60000,
) -> dict:
    async with _browser_semaphore:
        try:
            async with asyncio.timeout(BROWSER_TOTAL_TIMEOUT_SECONDS):
                return await _playwright_explore_page_inner(
                    url=url,
                    labels=labels,
                    task=task,
                    max_chars=max_chars,
                    profile=profile,
                    timeout_ms=timeout_ms,
                )
        except TimeoutError as exc:
            raise RuntimeError(
                f"Browser exploration exceeded the {BROWSER_TOTAL_TIMEOUT_SECONDS:g}-second limit"
            ) from exc


async def _playwright_explore_page_inner(
    url: str,
    labels: Optional[List[str]] = None,
    task: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    profile: str = "targeted",
    timeout_ms: int = 60000,
) -> dict:
    await validate_url_safety(url)

    labels = unique_preserve_order(str(label)[:120] for label in (labels or [])[:50])
    task = str(task or "")[:4000]
    max_chars = clamp_int(max_chars, 10000, ABSOLUTE_MAX_CHARS)
    profile = profile if profile in {"targeted", "balanced", "exhaustive"} else "targeted"
    timeout_ms = clamp_int(timeout_ms, 5000, 60000)

    captured_responses = []
    capture_tasks = set()
    cdp_candidates: dict[str, dict[str, Any]] = {}
    popup_tasks = set()
    clicked = []
    errors = []
    dom_text_parts = []
    scrollable_elements = []
    title = None
    final_url = url
    dom_char_budget = min(MAX_DOM_ACCUMULATED_CHARS, max(20_000, max_chars * 2))

    def append_dom_text(value: object) -> None:
        nonlocal dom_char_budget
        text = str(value or "").strip()
        if not text or dom_char_budget <= 0:
            return
        part = text[:dom_char_budget]
        if dom_text_parts and part == dom_text_parts[-1]:
            return
        dom_text_parts.append(part)
        dom_char_budget -= len(part)

    parsed_start = urlparse(url)
    start_domain = parsed_start.netloc.lower()

    async with async_playwright() as p:
        proxy_url = os.getenv("RESEARCH_BROWSER_PROXY", "").strip()
        require_isolation = env_flag("RESEARCH_REQUIRE_WEB_ISOLATION")
        if not require_isolation:
            raise RuntimeError("Browser runner requires web isolation")
        if not env_flag("RESEARCH_WEB_NETWORK_ISOLATED"):
            raise RuntimeError("Browser runner is not marked as network-isolated")
        if not proxy_url.lower().startswith("socks5://"):
            raise RuntimeError("Browser runner requires a SOCKS5 egress proxy")

        browser = await p.chromium.launch(**chromium_launch_options())

        context = await browser.new_context(
            accept_downloads=False,
            user_agent=BROWSER_HEADERS["User-Agent"],
            locale="en-US",
            viewport={"width": 1600, "height": 1400},
            ignore_https_errors=env_flag("RESEARCH_BROWSER_IGNORE_HTTPS_ERRORS"),
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            permissions=[],
            service_workers="block",
        )

        page = await context.new_page()
        blocked_request_errors = []
        request_count = 0

        async def guard_outbound_request(route, request):
            nonlocal request_count
            request_url = request.url
            parsed_request = urlparse(request_url)

            if parsed_request.scheme in {"about", "blob", "data"}:
                await route.continue_()
                return

            request_count += 1
            if request_count > MAX_BROWSER_REQUESTS:
                if len(blocked_request_errors) < 100:
                    blocked_request_errors.append(
                        f"Blocked browser request after the {MAX_BROWSER_REQUESTS}-request limit"
                    )
                await route.abort("blockedbyclient")
                return

            try:
                await asyncio.wait_for(validate_url_safety(request_url), timeout=5.0)
            except Exception as exc:
                message = (
                    f"Blocked unsafe browser request to {safe_diagnostic_url(request_url)}: "
                    f"{safe_exception_detail(exc)}"
                )
                if len(blocked_request_errors) < 100:
                    blocked_request_errors.append(message)
                await route.abort("blockedbyclient")
                return

            await route.continue_()

        # Guard every browser request, not only navigations. Otherwise a public
        # page could use images, scripts, or XHR as a blind request into a private
        # network even though its response is excluded from evidence capture.
        await context.route("**/*", guard_outbound_request)

        async def block_websocket(web_socket):
            message = f"Blocked browser WebSocket to {safe_diagnostic_url(web_socket.url)}"
            if len(blocked_request_errors) < 100:
                blocked_request_errors.append(message)
            await web_socket.close(code=1008, reason="WebSockets are disabled during research")

        await context.route_web_socket("**/*", block_websocket)

        def close_popup(popup):
            task = asyncio.create_task(popup.close())
            popup_tasks.add(task)
            task.add_done_callback(popup_tasks.discard)

        context.on("page", close_popup)

        cdp = await context.new_cdp_session(page)
        await cdp.send("Network.enable")

        async def capture_response(request_id: str, encoded_data_length: float):
            try:
                if len(captured_responses) >= MAX_NETWORK_CANDIDATES:
                    return
                candidate = cdp_candidates.pop(request_id, None)
                if candidate is None:
                    return
                body_payload = await cdp.send(
                    "Network.getResponseBody",
                    {"requestId": request_id},
                )
                body = decode_bounded_cdp_body(
                    body_payload,
                    candidate["declared_length"],
                    encoded_data_length,
                )
                if body is None:
                    return

                text = (
                    parse_maybe_json_text(body)
                    if "json" in candidate["content_type"]
                    or body.strip().startswith(("{", "["))
                    else html_to_text(body)
                )

                if text:
                    if len(captured_responses) >= MAX_NETWORK_CANDIDATES:
                        return
                    if looks_like_script_or_config(text):
                        return
                    captured_responses.append(
                        {
                            "url": candidate["url"],
                            "status": candidate["status"],
                            "content_type": candidate["content_type"],
                            "resource_type": candidate["resource_type"],
                            "text": text[:NETWORK_TEXT_LIMIT],
                            "text_chars": len(text),
                        }
                    )

            except Exception:
                cdp_candidates.pop(request_id, None)
                return

        def register_cdp_response(event: dict[str, Any]):
            if (
                len(cdp_candidates) >= MAX_NETWORK_CANDIDATES
                or len(captured_responses) >= MAX_NETWORK_CANDIDATES
            ):
                return
            try:
                response = event["response"]
                response_url = str(response["url"])
                headers = {
                    str(key).lower(): str(value)
                    for key, value in (response.get("headers") or {}).items()
                }
                content_type = (headers.get("content-type") or response.get("mimeType") or "").lower()
                resource_type = str(event.get("type") or "").lower()
                declared_length = bounded_network_response_length(headers)
                if declared_length is None or not should_capture_network_response(
                    response_url,
                    content_type,
                    resource_type,
                    start_domain,
                ):
                    return
            except Exception:
                return
            cdp_candidates[str(event["requestId"])] = {
                "url": response_url,
                "status": int(response.get("status") or 0),
                "content_type": content_type,
                "resource_type": resource_type,
                "declared_length": declared_length,
            }

        def schedule_cdp_capture(event: dict[str, Any]):
            request_id = str(event.get("requestId") or "")
            if not request_id or request_id not in cdp_candidates:
                return
            if (
                len(capture_tasks) >= MAX_NETWORK_CAPTURE_TASKS
                or len(captured_responses) >= MAX_NETWORK_CANDIDATES
            ):
                cdp_candidates.pop(request_id, None)
                return
            try:
                encoded_data_length = float(event.get("encodedDataLength") or 0)
            except (TypeError, ValueError):
                cdp_candidates.pop(request_id, None)
                return
            if encoded_data_length <= 0 or encoded_data_length > NETWORK_BODY_LIMIT:
                cdp_candidates.pop(request_id, None)
                return
            task = asyncio.create_task(capture_response(request_id, encoded_data_length))
            capture_tasks.add(task)
            task.add_done_callback(capture_tasks.discard)

        def discard_cdp_candidate(event: dict[str, Any]):
            cdp_candidates.pop(str(event.get("requestId") or ""), None)

        cdp.on("Network.responseReceived", register_cdp_response)
        cdp.on("Network.loadingFinished", schedule_cdp_capture)
        cdp.on("Network.loadingFailed", discard_cdp_candidate)

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                await page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            await page.wait_for_timeout(1000)

            title = await page.title()
            final_url = page.url
            await validate_url_safety(final_url)

            try:
                initial_text = await page.evaluate(
                    "limit => document.body ? document.body.innerText.slice(0, limit) : ''",
                    min(DOM_SNAPSHOT_CHAR_LIMIT, dom_char_budget),
                )
                append_dom_text(initial_text)
            except Exception:
                pass

            page_steps = 4 if profile == "targeted" else 6 if profile == "balanced" else 8
            for i in range(0, page_steps + 1):
                pct = i / page_steps
                await page.evaluate(f"window.scrollTo(0, Math.floor(document.body.scrollHeight * {pct}))")
                await page.wait_for_timeout(150)

            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(200)

            click_labels = unique_preserve_order(
                list(labels[:50])
                + ["show more", "view more", "load more", "more", "details", "learn more", "expand"]
            )
            click_limit = 6 if profile == "targeted" else 10 if profile == "balanced" else 14
            result = await page.evaluate(
                build_click_script(
                    click_labels,
                    max_clicks=click_limit,
                    text_limit=min(DOM_SNAPSHOT_CHAR_LIMIT, max(1000, dom_char_budget)),
                )
            )
            if isinstance(result, dict):
                clicked.extend(result.get("clicked") or [])
                if result.get("title"):
                    title = result["title"]
                if result.get("url"):
                    final_url = result["url"]
                if result.get("navigation_changed"):
                    await validate_url_safety(final_url)
                    raise RuntimeError("Automated reveal action changed page navigation; stopped exploration")
                append_dom_text(result.get("text"))

            await page.wait_for_timeout(500)

            scroll_result = await page.evaluate(
                build_scrollable_capture_script(
                    labels,
                    profile,
                    max_chars=max(10_000, dom_char_budget),
                )
            )
            if isinstance(scroll_result, dict):
                if scroll_result.get("title"):
                    title = scroll_result["title"]
                if scroll_result.get("url"):
                    final_url = scroll_result["url"]
                append_dom_text(scroll_result.get("text"))
                if scroll_result.get("scrollable_elements"):
                    scrollable_elements.extend(scroll_result["scrollable_elements"])

            await page.wait_for_timeout(500)

            try:
                final_text = await page.evaluate(
                    "limit => document.body ? document.body.innerText.slice(0, limit) : ''",
                    min(DOM_SNAPSHOT_CHAR_LIMIT, max(1000, dom_char_budget)),
                )
                append_dom_text(final_text)
            except Exception:
                pass

            final_url = page.url
            await validate_url_safety(final_url)

            if capture_tasks:
                done, pending = await asyncio.wait(capture_tasks, timeout=5)
                for task in pending:
                    task.cancel()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            errors.append(safe_exception_detail(exc))
        finally:
            pending_tasks = [task for task in capture_tasks | popup_tasks if not task.done()]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            cdp_candidates.clear()
            errors.extend(unique_preserve_order(blocked_request_errors))
            await context.close()
            await browser.close()

    captured_responses = select_network_responses(captured_responses, start_domain, task, labels)
    network_text = "\n\n".join(
        f"Network response: {item['url']}\n{item['text'][:NETWORK_COMBINED_TEXT_LIMIT]}"
        for item in captured_responses
        if item.get("text")
    )
    dom_text = "\n\n".join(dom_text_parts)
    combined = "\n\n".join(part for part in [dom_text, network_text] if part)
    combined = html.unescape(combined)
    combined = re.sub(r"\n{4,}", "\n\n\n", combined)
    combined = combined.strip()

    table_like_rows = extract_table_like_rows(combined, max_rows=20000)

    return {
        "url": url,
        "final_url": final_url,
        "title": title,
        "profile": profile,
        "clicked": unique_preserve_order(clicked),
        "scrollable_elements": scrollable_elements[:50],
        "scrollable_element_count": len(scrollable_elements),
        "content": combined[:max_chars],
        "content_chars": len(combined),
        "truncated": len(combined) > max_chars,
        "table_like_rows": table_like_rows[:10000],
        "table_like_row_count": len(table_like_rows),
        "network_responses": [
            {
                "url": item["url"],
                "status": item["status"],
                "content_type": item["content_type"],
                "resource_type": item.get("resource_type"),
                "text_chars": item["text_chars"],
                "preview": item["text"][:NETWORK_PREVIEW_LIMIT],
            }
            for item in captured_responses[:MAX_NETWORK_RESPONSES]
        ],
        "network_response_count": len(captured_responses),
        "errors": errors,
        "extraction_method": f"playwright_{profile}_browser_network_capture_scrollable_tables",
    }
