import asyncio
import calendar
import copy
import hashlib
import math
import ipaddress
import json
import os
import re
import threading
import time
import unicodedata
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

try:
    import redis.asyncio as redis_async
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis is installed in production.
    redis_async = None

    class RedisError(Exception):
        pass

from shared import SEARXNG_URL, get_domain, runtime_retrieval_context
from url_identity import canonicalize_web_url


SEARXNG_MAX_RESPONSE_BYTES = max(
    1024,
    int(os.getenv("SEARXNG_MAX_RESPONSE_BYTES", "4194304")),
)
SEARXNG_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("SEARXNG_TIMEOUT_SECONDS", "12")),
)
SEARCH_CACHE_TTL_SECONDS = max(
    1.0,
    min(3600.0, float(os.getenv("SEARCH_CACHE_TTL_SECONDS", "600"))),
)
SEARCH_CACHE_STALE_TTL_SECONDS = max(
    SEARCH_CACHE_TTL_SECONDS,
    min(86400.0, float(os.getenv("SEARCH_CACHE_STALE_TTL_SECONDS", "3600"))),
)
SEARCH_CACHE_MAX_ENTRIES = max(
    16,
    min(4096, int(os.getenv("SEARCH_CACHE_MAX_ENTRIES", "256"))),
)
SEARCH_CACHE_REDIS_ENABLED = os.getenv(
    "SEARCH_CACHE_REDIS_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
SEARCH_CACHE_REDIS_TIMEOUT_SECONDS = max(
    0.05,
    min(1.0, float(os.getenv("SEARCH_CACHE_REDIS_TIMEOUT_SECONDS", "0.2"))),
)
SEARCH_CACHE_MAX_PAYLOAD_BYTES = max(
    4096,
    min(4194304, int(os.getenv("SEARCH_CACHE_MAX_PAYLOAD_BYTES", "262144"))),
)
SEARCH_ENGINE_FAILURE_THRESHOLD = max(
    1,
    min(10, int(os.getenv("SEARCH_ENGINE_FAILURE_THRESHOLD", "2"))),
)
SEARCH_ENGINE_TRANSIENT_COOLDOWN_SECONDS = max(
    5.0,
    min(
        3600.0,
        float(
            os.getenv(
                "SEARCH_ENGINE_TRANSIENT_COOLDOWN_SECONDS",
                os.getenv("SEARCH_ENGINE_COOLDOWN_SECONDS", "120"),
            )
        ),
    ),
)
SEARCH_ENGINE_RATE_LIMIT_COOLDOWN_SECONDS = max(
    SEARCH_ENGINE_TRANSIENT_COOLDOWN_SECONDS,
    min(
        86400.0,
        float(os.getenv("SEARCH_ENGINE_RATE_LIMIT_COOLDOWN_SECONDS", "900")),
    ),
)
SEARCH_ENGINE_MAX_COOLDOWN_SECONDS = max(
    SEARCH_ENGINE_RATE_LIMIT_COOLDOWN_SECONDS,
    min(86400.0, float(os.getenv("SEARCH_ENGINE_MAX_COOLDOWN_SECONDS", "3600"))),
)
SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED = os.getenv(
    "SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}
SEARCH_ENGINE_REDIS_TIMEOUT_SECONDS = max(
    0.05,
    min(1.0, float(os.getenv("SEARCH_ENGINE_REDIS_TIMEOUT_SECONDS", "0.2"))),
)
SEARCH_MAX_CONCURRENT_REQUESTS = max(
    1,
    min(8, int(os.getenv("SEARCH_MAX_CONCURRENT_REQUESTS", "2"))),
)
SEARCH_STAGE_MIN_RESULTS = max(
    1,
    min(20, int(os.getenv("SEARCH_STAGE_MIN_RESULTS", "4"))),
)
SEARCH_MAX_ENGINE_STAGES = max(
    1,
    min(3, int(os.getenv("SEARCH_MAX_ENGINE_STAGES", "2"))),
)
SEARCH_DEEP_MAX_ENGINE_STAGES = max(
    SEARCH_MAX_ENGINE_STAGES,
    min(5, int(os.getenv("SEARCH_DEEP_MAX_ENGINE_STAGES", "4"))),
)
SEARCH_RELEVANCE_MIN_SCORE = max(
    0.05,
    min(0.95, float(os.getenv("SEARCH_RELEVANCE_MIN_SCORE", "0.28"))),
)
SEARCH_STAGE_MIN_RELEVANT_RATIO = max(
    0.25,
    min(1.0, float(os.getenv("SEARCH_STAGE_MIN_RELEVANT_RATIO", "0.75"))),
)

_SEARCH_CACHE_VERSION = 2
_SEARCH_CACHE_PREFIX = "research:search-cache:v2:"
_SEARCH_ENGINE_CIRCUIT_PREFIX = "research:search-engine-circuit:v1:"
_SEARCH_ENGINE_COOLDOWN_ZSET = f"{_SEARCH_ENGINE_CIRCUIT_PREFIX}cooldowns"
_SEARX_SERVICE_CIRCUIT = "__searxng_service__"
_SEARCH_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SEARCH_CACHE_LOCK = threading.RLock()
_SEARCH_REDIS_URL = os.getenv("REDIS_URL", "").strip()
_SEARCH_REDIS_CLIENT = None
_SEARCH_REDIS_DISABLED_UNTIL = 0.0
_SEARCH_REDIS_LOCK = threading.RLock()


@dataclass
class _EngineHealth:
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    reason: str = ""


_ENGINE_HEALTH: dict[str, _EngineHealth] = {}
_ENGINE_HEALTH_LOCK = threading.RLock()
_LOOP_LIMITERS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_LOOP_LIMITERS_LOCK = threading.RLock()

RESEARCH_MODE_CONFIG = {
    # Interactive modes have an end-to-end latency target in addition to their
    # crawl budget. Deep mode is deliberately the durable/background option.
    "quick": {
        "max_urls": 2,
        "search_results": 6,
        "top_k": 4,
        "planner_budget": 1,
        "search_budget": 5,
        "crawl_budget": 10,
        "total_budget": 12,
    },
    "balanced": {
        "max_urls": 4,
        "search_results": 10,
        "top_k": 6,
        "planner_budget": 2,
        "search_budget": 8,
        "crawl_budget": 22,
        "total_budget": 30,
    },
    "deep": {
        "max_urls": 8,
        "search_results": 16,
        "top_k": 10,
        "planner_budget": 15,
        "search_budget": 25,
        "crawl_budget": 150,
        "total_budget": 180,
    },
    "technical": {
        "max_urls": 6,
        "search_results": 14,
        "top_k": 8,
        "planner_budget": 3,
        "search_budget": 10,
        "crawl_budget": 34,
        "total_budget": 45,
    },
    "academic": {
        "max_urls": 6,
        "search_results": 14,
        "top_k": 8,
        "planner_budget": 4,
        "search_budget": 12,
        "crawl_budget": 38,
        "total_budget": 50,
    },
    "local_only": {
        "max_urls": 0,
        "search_results": 0,
        "top_k": 8,
        "planner_budget": 0,
        "search_budget": 0,
        "crawl_budget": 0,
        "total_budget": 15,
    },
    "web_only": {
        "max_urls": 5,
        "search_results": 12,
        "top_k": 0,
        "planner_budget": 2,
        "search_budget": 7,
        "crawl_budget": 18,
        "total_budget": 25,
    },
}


@dataclass(frozen=True)
class SearchPolicy:
    """Structured SearX policy derived from the research intent."""

    categories: tuple[str, ...] = ("general",)
    time_range: str | None = None
    language: str = "auto"
    timezone: str = "UTC"
    temporal_intent: str = "none"
    reference_date: date | None = None
    start_date: date | None = None
    target_date: date | None = None
    cutoff_date: date | None = None
    event_start_date: date | None = None
    event_end_date: date | None = None
    strict_date: bool = False
    news_intent: bool = False
    freshness_max_age_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        for key in (
            "reference_date",
            "start_date",
            "target_date",
            "cutoff_date",
            "event_start_date",
            "event_end_date",
        ):
            value = getattr(self, key)
            if value is not None:
                output[key] = value.isoformat()
        return output


class SearchResults(list):
    """List-compatible search results carrying non-fatal search diagnostics."""

    def __init__(
        self,
        values=(),
        *,
        diagnostics: dict[str, Any] | None = None,
        policy: SearchPolicy | None = None,
    ) -> None:
        super().__init__(values)
        self.diagnostics = diagnostics or {}
        self.search_policy = policy.to_dict() if policy else None


class _SearxHTTPError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(f"SearXNG returned HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


_ENGINE_STAGES = {
    "general": (
        ("bing", "qwant"),
        ("startpage", "mojeek"),
    ),
    "news": (
        ("reuters", "bing news"),
        ("qwant news", "mojeek news"),
        ("bing", "qwant"),
        ("startpage", "mojeek"),
    ),
    "technical": (
        ("bing", "github"),
        ("stackoverflow", "mdn"),
        ("docker hub", "askubuntu"),
        ("qwant", "mojeek"),
        ("startpage",),
    ),
    "academic": (
        ("semantic scholar", "pubmed"),
        ("arxiv", "openalex"),
        ("crossref", "bing"),
        ("qwant", "mojeek"),
    ),
}
_ALLOWED_SEARX_ENGINES = frozenset(
    engine
    for stages in _ENGINE_STAGES.values()
    for stage in stages
    for engine in stage
)

_RATE_LIMIT_REASON_RE = re.compile(
    r"(?:rate[ -]?limit|too many requests|http\s*429|captcha|robot check)",
    re.I,
)
_ACCESS_BLOCK_REASON_RE = re.compile(
    r"(?:access denied|forbidden|http\s*403|(?:^|\s)blocked(?:\s|$))",
    re.I,
)
_RETRY_AFTER_REASON_RE = re.compile(
    r"retry(?:ing)?(?:\s+|-)after(?:\s+|:)?(?P<seconds>\d+(?:\.\d+)?)",
    re.I,
)


def _normalize_search_query(query: str) -> str:
    normalized = unicodedata.normalize("NFC", str(query or ""))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _escape_searx_control_tokens(query: str) -> str:
    """Keep user text from overriding explicit SearX request controls."""

    def escape(part: str) -> str:
        lowered = part.casefold()
        is_bang = (
            lowered.startswith("!!")
            or (lowered.startswith("!") and lowered != "!important")
        )
        is_language = bool(re.fullmatch(r":[a-z][a-z0-9_-]*", part, re.I))
        is_timeout = bool(re.fullmatch(r"<\d+", part))
        return f"\\{part}" if is_bang or is_language or is_timeout else part

    return "".join(
        part if part.isspace() else escape(part)
        for part in re.split(r"(\s+)", str(query or ""))
    )


def _search_cache_key(
    query: str,
    outbound_query: str,
    max_results: int,
    mode: str,
    policy: SearchPolicy,
    base_url: str,
    cache_scope: str | None,
) -> str:
    normalized_scope = unicodedata.normalize(
        "NFC",
        str(cache_scope or "anonymous").strip(),
    )
    identity = {
        "version": _SEARCH_CACHE_VERSION,
        "base_url": base_url,
        "query": _normalize_search_query(query),
        "outbound_query": _normalize_search_query(outbound_query),
        "cache_scope_hash": hashlib.sha256(
            normalized_scope.encode("utf-8")
        ).hexdigest(),
        "max_results": max(0, int(max_results)),
        "mode": str(mode or "balanced").strip().casefold(),
        "policy": policy.to_dict(),
        "engine_stages": _engine_stages(policy, mode),
        "stage_min_results": SEARCH_STAGE_MIN_RESULTS,
        "relevance_min_score": SEARCH_RELEVANCE_MIN_SCORE,
        "stage_min_relevant_ratio": SEARCH_STAGE_MIN_RELEVANT_RATIO,
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_cache_payload(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    cached_at = value.get("cached_at")
    results = value.get("results")
    diagnostics = value.get("diagnostics")
    search_policy = value.get("search_policy")
    if (
        not isinstance(cached_at, (int, float))
        or isinstance(cached_at, bool)
        or not math.isfinite(float(cached_at))
        or float(cached_at) > time.time() + 60
        or not isinstance(results, list)
        or len(results) > 100
        or any(not isinstance(item, dict) for item in results)
        or not isinstance(diagnostics, dict)
        or (search_policy is not None and not isinstance(search_policy, dict))
    ):
        return None
    return value


def _local_cache_get(key: str) -> dict[str, Any] | None:
    now = time.time()
    with _SEARCH_CACHE_LOCK:
        payload = _SEARCH_CACHE.get(key)
        if payload is None:
            return None
        if now - float(payload["cached_at"]) > SEARCH_CACHE_STALE_TTL_SECONDS:
            _SEARCH_CACHE.pop(key, None)
            return None
        _SEARCH_CACHE.move_to_end(key)
        return copy.deepcopy(payload)


def _local_cache_set(key: str, payload: dict[str, Any]) -> None:
    with _SEARCH_CACHE_LOCK:
        _SEARCH_CACHE[key] = copy.deepcopy(payload)
        _SEARCH_CACHE.move_to_end(key)
        while len(_SEARCH_CACHE) > SEARCH_CACHE_MAX_ENTRIES:
            _SEARCH_CACHE.popitem(last=False)


def _search_redis_client():
    global _SEARCH_REDIS_CLIENT
    if (
        not _SEARCH_REDIS_URL
        or redis_async is None
        or time.monotonic() < _SEARCH_REDIS_DISABLED_UNTIL
    ):
        return None
    with _SEARCH_REDIS_LOCK:
        if _SEARCH_REDIS_CLIENT is None:
            _SEARCH_REDIS_CLIENT = redis_async.from_url(
                _SEARCH_REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=SEARCH_CACHE_REDIS_TIMEOUT_SECONDS,
                socket_timeout=SEARCH_CACHE_REDIS_TIMEOUT_SECONDS,
            )
        return _SEARCH_REDIS_CLIENT


def _disable_redis_cache_temporarily() -> None:
    global _SEARCH_REDIS_DISABLED_UNTIL
    with _SEARCH_REDIS_LOCK:
        _SEARCH_REDIS_DISABLED_UNTIL = time.monotonic() + 30.0


async def _cache_get(key: str) -> dict[str, Any] | None:
    payload = _local_cache_get(key)
    if payload is not None:
        return payload
    client = _search_redis_client() if SEARCH_CACHE_REDIS_ENABLED else None
    if client is None:
        return None
    try:
        raw = await asyncio.wait_for(
            client.get(f"{_SEARCH_CACHE_PREFIX}{key}"),
            timeout=SEARCH_CACHE_REDIS_TIMEOUT_SECONDS,
        )
    except (RedisError, OSError, TimeoutError):
        _disable_redis_cache_temporarily()
        return None
    if not raw:
        return None
    if isinstance(raw, bytes):
        if len(raw) > SEARCH_CACHE_MAX_PAYLOAD_BYTES:
            return None
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(raw, str):
        return None
    if len(raw.encode("utf-8")) > SEARCH_CACHE_MAX_PAYLOAD_BYTES:
        return None
    try:
        payload = _valid_cache_payload(json.loads(raw))
    except (TypeError, json.JSONDecodeError, RecursionError):
        return None
    if payload is None:
        return None
    if time.time() - float(payload["cached_at"]) > SEARCH_CACHE_STALE_TTL_SECONDS:
        return None
    try:
        _local_cache_set(key, payload)
    except RecursionError:
        return None
    return copy.deepcopy(payload)


async def _cache_set(key: str, results: SearchResults) -> None:
    diagnostics = copy.deepcopy(results.diagnostics)
    diagnostics.pop("cache", None)
    payload = {
        "cached_at": time.time(),
        "results": list(results),
        "diagnostics": diagnostics,
        "search_policy": copy.deepcopy(results.search_policy),
    }
    try:
        encoded = json.dumps(
            payload,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return
    if len(encoded.encode("utf-8")) > SEARCH_CACHE_MAX_PAYLOAD_BYTES:
        return
    _local_cache_set(key, payload)
    client = _search_redis_client() if SEARCH_CACHE_REDIS_ENABLED else None
    if client is None:
        return
    try:
        await asyncio.wait_for(
            client.set(
                f"{_SEARCH_CACHE_PREFIX}{key}",
                encoded,
                ex=max(1, math.ceil(SEARCH_CACHE_STALE_TTL_SECONDS)),
            ),
            timeout=SEARCH_CACHE_REDIS_TIMEOUT_SECONDS,
        )
    except (RedisError, OSError, TimeoutError):
        _disable_redis_cache_temporarily()


def _cached_results(
    payload: dict[str, Any],
    status: str,
    *,
    attempt_diagnostics: dict[str, Any] | None = None,
) -> SearchResults:
    diagnostics = copy.deepcopy(payload["diagnostics"])
    age_seconds = max(0.0, time.time() - float(payload["cached_at"]))
    cached_at_utc = datetime.fromtimestamp(
        float(payload["cached_at"]),
        tz=timezone.utc,
    ).isoformat()
    diagnostics["cache"] = {
        "status": status,
        "age_seconds": round(age_seconds, 3),
    }
    if status == "stale_fallback":
        policy = payload.get("search_policy") or {}
        temporal = str(policy.get("temporal_intent") or "none")
        diagnostics["cache"].update(
            {
                "freshness_unverified": True,
                "warning": (
                    "Stale cached evidence was returned after a transient search "
                    "failure; publication freshness must be reverified."
                    if temporal != "none" or policy.get("strict_date")
                    else "Stale cached evidence was returned after a transient search failure."
                ),
            }
        )
    if attempt_diagnostics:
        diagnostics["stale_fallback_attempt"] = copy.deepcopy(attempt_diagnostics)
    cached_items = []
    for item in payload["results"]:
        cached_item = copy.deepcopy(item)
        cached_item["search_cache_status"] = status
        cached_item["search_cached_at_utc"] = cached_at_utc
        cached_item["retrieved_at_utc"] = cached_at_utc
        if status == "stale_fallback":
            cached_item["freshness"] = "stale_cache_unverified"
            cached_item["freshness_unverified"] = True
        else:
            cached_item["freshness"] = "cached_runtime_retrieval"
        cached_items.append(cached_item)
    results = SearchResults(cached_items, diagnostics=diagnostics)
    results.search_policy = copy.deepcopy(payload.get("search_policy"))
    return results


def _engine_policy_name(policy: SearchPolicy) -> str:
    if "science" in policy.categories:
        return "academic"
    if "it" in policy.categories:
        return "technical"
    if "news" in policy.categories:
        return "news"
    return "general"


def _engine_stages(
    policy: SearchPolicy,
    mode: str = "balanced",
) -> tuple[tuple[str, ...], ...]:
    stages = _ENGINE_STAGES[_engine_policy_name(policy)]
    limit = (
        SEARCH_DEEP_MAX_ENGINE_STAGES
        if mode == "deep"
        else SEARCH_MAX_ENGINE_STAGES
    )
    return stages[:limit]


def _validated_stage_engines(engines: tuple[str, ...]) -> tuple[str, ...]:
    if not engines or any(engine not in _ALLOWED_SEARX_ENGINES for engine in engines):
        return ()
    return tuple(dict.fromkeys(engines))


def _parse_retry_after(value: object) -> float | None:
    text = strip_text(str(value or ""))
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, OverflowError):
        parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_after_from_reason(reason: str) -> float | None:
    match = _RETRY_AFTER_REASON_RE.search(reason)
    return float(match.group("seconds")) if match else None


def _failure_reason_code(reason: str) -> str:
    if reason in {
        "rate_limited",
        "access_blocked",
        "timeout",
        "service_error",
        "service_unavailable",
        "upstream_unresponsive",
    }:
        return reason
    if _RATE_LIMIT_REASON_RE.search(reason):
        return "rate_limited"
    if _ACCESS_BLOCK_REASON_RE.search(reason):
        return "access_blocked"
    lowered = reason.casefold()
    if "timeout" in lowered:
        return "timeout"
    if re.search(r"http\s*5\d\d", reason, re.I):
        return "service_error"
    if "connect" in lowered or "unavailable" in lowered:
        return "service_unavailable"
    return "upstream_unresponsive"


async def _publish_shared_cooldown(
    engine: str,
    cooldown_seconds: float,
) -> None:
    if not SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED:
        return
    client = _search_redis_client()
    if client is None:
        return
    member = hashlib.sha256(engine.casefold().encode("utf-8")).hexdigest()[:24]
    proposed_until = time.time() + cooldown_seconds
    try:
        await asyncio.wait_for(
            client.zadd(
                _SEARCH_ENGINE_COOLDOWN_ZSET,
                {member: proposed_until},
                gt=True,
            ),
            timeout=SEARCH_ENGINE_REDIS_TIMEOUT_SECONDS,
        )
    except (RedisError, OSError, TimeoutError):
        _disable_redis_cache_temporarily()


async def _record_engine_failure(
    engine: str,
    reason: str,
    *,
    retry_after: float | None = None,
) -> None:
    normalized = strip_text(engine).casefold()
    if not normalized:
        return
    now = time.monotonic()
    cooldown = None
    reason_code = _failure_reason_code(reason)
    with _ENGINE_HEALTH_LOCK:
        health = _ENGINE_HEALTH.setdefault(normalized, _EngineHealth())
        health.consecutive_failures += 1
        health.reason = reason_code
        immediate = reason_code in {"rate_limited", "access_blocked"}
        if not immediate and health.consecutive_failures < SEARCH_ENGINE_FAILURE_THRESHOLD:
            return
        exponent = max(0, health.consecutive_failures - SEARCH_ENGINE_FAILURE_THRESHOLD)
        base_cooldown = (
            SEARCH_ENGINE_RATE_LIMIT_COOLDOWN_SECONDS
            if immediate
            else SEARCH_ENGINE_TRANSIENT_COOLDOWN_SECONDS
        )
        cooldown = min(
            SEARCH_ENGINE_MAX_COOLDOWN_SECONDS,
            base_cooldown * (2**exponent),
        )
        if retry_after is not None:
            cooldown = max(cooldown, min(SEARCH_ENGINE_MAX_COOLDOWN_SECONDS, retry_after))
        health.cooldown_until = max(health.cooldown_until, now + cooldown)
    await _publish_shared_cooldown(normalized, cooldown)


async def _record_engine_success(engine: str) -> None:
    normalized = strip_text(engine).casefold()
    with _ENGINE_HEALTH_LOCK:
        health = _ENGINE_HEALTH.get(normalized)
        if health is not None and health.cooldown_until > time.monotonic():
            return
        _ENGINE_HEALTH.pop(normalized, None)


async def _eligible_engines(
    engines: tuple[str, ...],
) -> tuple[list[str], list[dict[str, Any]]]:
    now = time.monotonic()
    eligible = []
    skipped = []
    with _ENGINE_HEALTH_LOCK:
        for engine in engines:
            health = _ENGINE_HEALTH.get(engine.casefold())
            if health is None or health.cooldown_until <= now:
                eligible.append(engine)
                continue
            skipped.append(
                {
                    "engine": engine,
                    "reason": health.reason or "circuit open",
                    "retry_after_seconds": round(health.cooldown_until - now, 3),
                }
            )
    if (
        not eligible
        or not SEARCH_ENGINE_CIRCUIT_REDIS_ENABLED
        or (client := _search_redis_client()) is None
    ):
        return eligible, skipped
    members = [
        hashlib.sha256(engine.casefold().encode("utf-8")).hexdigest()[:24]
        for engine in eligible
    ]
    try:
        pipeline = client.pipeline(transaction=False)
        for member in members:
            pipeline.zscore(_SEARCH_ENGINE_COOLDOWN_ZSET, member)
        shared_values = await asyncio.wait_for(
            pipeline.execute(),
            timeout=SEARCH_ENGINE_REDIS_TIMEOUT_SECONDS,
        )
    except (RedisError, OSError, TimeoutError):
        _disable_redis_cache_temporarily()
        return eligible, skipped
    shared_eligible = []
    epoch_now = time.time()
    for engine, raw in zip(eligible, shared_values, strict=True):
        if raw is None:
            shared_eligible.append(engine)
            continue
        try:
            cooldown_until = float(raw)
        except (TypeError, ValueError):
            shared_eligible.append(engine)
            continue
        remaining = cooldown_until - epoch_now
        if remaining <= 0 or remaining > SEARCH_ENGINE_MAX_COOLDOWN_SECONDS + 60:
            shared_eligible.append(engine)
            continue
        skipped.append(
            {
                "engine": engine,
                "reason": "shared_circuit_open",
                "retry_after_seconds": round(remaining, 3),
                "shared": True,
            }
        )
    return shared_eligible, skipped


def _loop_limiters() -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    with _LOOP_LIMITERS_LOCK:
        limiters = _LOOP_LIMITERS.get(loop)
        if limiters is None:
            limiters = {
                "global": asyncio.Semaphore(SEARCH_MAX_CONCURRENT_REQUESTS),
                "engines": {},
                "cache_keys": OrderedDict(),
            }
            _LOOP_LIMITERS[loop] = limiters
        return limiters


def _cache_key_lock(key: str) -> asyncio.Lock:
    locks = _loop_limiters()["cache_keys"]
    lock = locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        locks[key] = lock
        while len(locks) > SEARCH_CACHE_MAX_ENTRIES:
            locks.popitem(last=False)
    else:
        locks.move_to_end(key)
    return lock


@asynccontextmanager
async def _engine_request_slots(engines: list[str]):
    limiters = _loop_limiters()
    engine_limiters = limiters["engines"]
    acquired = []
    try:
        for engine in sorted(set(engines)):
            limiter = engine_limiters.setdefault(engine, asyncio.Semaphore(1))
            await limiter.acquire()
            acquired.append(limiter)
        await limiters["global"].acquire()
        acquired.append(limiters["global"])
        yield
    finally:
        for limiter in reversed(acquired):
            limiter.release()


def _coverage_sufficient(results: SearchResults, max_results: int) -> bool:
    needed = min(max(0, max_results), SEARCH_STAGE_MIN_RESULTS)
    if needed <= 0 or len(results) < needed:
        return needed <= 0
    relevance = results.diagnostics.get("topical_relevance")
    if isinstance(relevance, dict):
        relevant_needed = max(1, math.ceil(needed * SEARCH_STAGE_MIN_RELEVANT_RATIO))
        relevant_count = int(relevance.get("relevant_count") or 0)
        relevant_owners = int(relevance.get("distinct_relevant_owners") or 0)
        return (
            relevant_count >= relevant_needed
            and relevant_owners >= min(2, relevant_needed)
        )
    owners = {
        estimate_source_owner_domain(str(item.get("domain") or ""))
        for item in results
        if item.get("domain")
    }
    return len(owners) >= min(2, needed)


def _cacheable_results(results: SearchResults, max_results: int) -> bool:
    if not _coverage_sufficient(results, max_results):
        return False
    if results.diagnostics.get("unresponsive_engines"):
        return False
    stages = results.diagnostics.get("search_stages") or []
    return bool(stages) and all(stage.get("status") == "ok" for stage in stages)


_NEWS_INTENT_RE = re.compile(
    r"\b(?:news|headlines?|breaking|current\s+events?|newsworthy|news\s+articles?|"
    r"press\s+coverage|media\s+coverage)\b",
    re.I,
)
_HACKER_NEWS_TECHNICAL_RE = re.compile(
    r"\bhacker\s+news\b[^.!?\r\n]{0,120}\b(?:api|documentation|docs?|sdk|cli|"
    r"source\s+code)\b|"
    r"\b(?:api|documentation|docs?|sdk|cli|source\s+code)\b[^.!?\r\n]{0,120}"
    r"\bhacker\s+news\b",
    re.I,
)
_EVENT_INTENT_RE = re.compile(
    r"\b(?:happened|announced|announcement|event|release(?:d|s)?|launch(?:ed)?|reported|"
    r"developments?|stories|articles?)\b",
    re.I,
)
_TECHNICAL_INTENT_RE = re.compile(
    r"\b(?:install(?:ation)?|setup|set\s+up|configure|configuration|deploy(?:ment)?|"
    r"upgrade|migrate|troubleshoot|debug|fix|repair|error|exception|documentation|"
    r"docs?|sdk|cli|source\s+code|github|release\s+notes?|breaking\s+changes?|"
    r"api\s+(?:documentation|docs?|reference|integration|guide|endpoint|schema|usage)|"
    r"integration\s+guide)\b",
    re.I,
)
_ACADEMIC_INTENT_RE = re.compile(
    r"\b(?:academic|scholarly|peer[ -]reviewed|research\s+papers?|journal\s+articles?|"
    r"clinical\s+trials?|meta-analysis|systematic\s+reviews?|arxiv|doi)\b|"
    r"\b(?:stud(?:y|ies)|papers?)\s+(?:about|on|of|examining|investigating)\b|"
    r"\b(?:latest|recent|new|published|scientific|research)\b"
    r"(?:\W+\w+){0,6}\W+(?:studies|papers)\b",
    re.I,
)
_STRICT_TODAY_RE = re.compile(r"\btoday(?:'s)?\b", re.I)
_STRICT_YESTERDAY_RE = re.compile(r"\byesterday(?:'s)?\b", re.I)
_CURRENT_RE = re.compile(r"\b(?:current(?:ly)?|latest|newest|as\s+of)\b", re.I)
_RECENT_RE = re.compile(
    r"\b(?:recent(?:ly)?|this\s+(?:week|month)|(?:past|last)\s+"
    r"(?:(?:\d+)\s+)?(?:hours?|days?|weeks?|months?))\b",
    re.I,
)
_RELATIVE_WINDOW_RE = re.compile(
    r"\b(?:past|last)\s+(?:(\d+)\s+)?(hours?|days?|weeks?|months?)\b",
    re.I,
)
_EXPLICIT_LANGUAGE_RE = re.compile(
    r"\b(?:lang(?:uage)?):([a-z]{2,3}(?:-[a-z]{2})?)\b",
    re.I,
)
_ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "about",
        "and",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "into",
        "is",
        "my",
        "not",
        "out",
        "should",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "without",
        "would",
        "your",
    }
)
_ENGLISH_SEARCH_WORDS = frozenset(
    {
        "alternative",
        "alternatives",
        "benchmark",
        "benchmarks",
        "best",
        "compare",
        "comparison",
        "configure",
        "current",
        "debug",
        "deploy",
        "documentation",
        "docs",
        "error",
        "fix",
        "guide",
        "install",
        "installation",
        "latest",
        "news",
        "newest",
        "powerful",
        "release",
        "releases",
        "repository",
        "review",
        "reviews",
        "setup",
        "today",
        "troubleshoot",
        "versus",
    }
)
_STRONG_ENGLISH_SEARCH_WORDS = frozenset(
    {
        "debug",
        "docs",
        "fix",
        "install",
        "latest",
        "news",
        "newest",
        "powerful",
        "repository",
        "today",
        "troubleshoot",
        "weather",
    }
)
_NON_ENGLISH_FUNCTION_WORD_GROUPS = (
    frozenset({"der", "die", "das", "den", "dem", "des", "ein", "eine", "ist", "und", "oder", "wie"}),
    frozenset({"le", "la", "les", "des", "une", "est", "et", "ou", "avec", "comment", "quel"}),
    frozenset({"el", "la", "los", "las", "una", "es", "y", "con", "como", "cual", "que"}),
    frozenset({"il", "lo", "la", "gli", "una", "con", "come", "quale", "che", "per"}),
    frozenset({"o", "a", "os", "as", "uma", "com", "como", "qual", "que", "para"}),
)
_VALID_SEARCH_LANGUAGE_RE = re.compile(r"[a-z]{2,3}(?:-[a-z]{2})?", re.I)
_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
_DATE_EXPRESSION_PATTERN = (
    r"(?:19|20)\d{2}-\d{1,2}-\d{1,2}|"
    r"\d{1,2}/\d{1,2}/(?:19|20)\d{2}|"
    rf"(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+)(?:19|20)\d{{2}}|"
    rf"(?:{_MONTH_NAMES})\s+(?:19|20)\d{{2}}|"
    r"(?:19|20)\d{2}"
)
_MONTH_DAY_EXPRESSION_PATTERN = (
    rf"(?:{_MONTH_NAMES})\s+\d{{1,2}}(?:st|nd|rd|th)?"
)
_AS_OF_DATE_RE = re.compile(
    rf"\bas\s+of\s+(?P<date>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_ON_DATE_RE = re.compile(
    rf"\bon\s+(?P<date>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_SINCE_DATE_RE = re.compile(
    rf"\bsince\s+(?P<date>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_AFTER_DATE_RE = re.compile(
    rf"\bafter\s+(?P<date>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_BEFORE_DATE_RE = re.compile(
    rf"\bbefore\s+(?P<date>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_RANGE_DATE_RE = re.compile(
    rf"\bfrom\s+(?P<start>{_DATE_EXPRESSION_PATTERN})\s+"
    rf"(?:to|through|until|-)\s+(?P<end>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_BETWEEN_DATE_RE = re.compile(
    rf"\bbetween\s+(?P<start>{_DATE_EXPRESSION_PATTERN})\s+and\s+"
    rf"(?P<end>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_SHARED_YEAR_RANGE_RE = re.compile(
    rf"\bfrom\s+(?P<start>{_MONTH_DAY_EXPRESSION_PATTERN})\s+"
    rf"(?:to|through|until|-)\s+(?P<end>{_MONTH_DAY_EXPRESSION_PATTERN})"
    rf"(?:,?\s+)(?P<year>(?:19|20)\d{{2}})(?![\w/-])",
    re.I,
)
_IN_DATE_RE = re.compile(
    rf"\b(?:in|during)\s+(?P<date>{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_PUBLICATION_DATE_RE = re.compile(
    r"\b(?:articles?|papers?|reports?|posts?|sources?)\s+(?:published|posted|dated)\b|"
    r"\b(?:articles?|papers?|reports?|posts?|sources?)\s+"
    r"(?:on|since|after|before|from|between|in|during)\b|"
    r"\b(?:published|publication\s+date|posted|dated)\s+(?:on|since|after|before|"
    r"from|between|in|during)\b|"
    r"\b(?:published|posted|dated)\s+(?:today|yesterday)\b",
    re.I,
)
_EVENT_ON_DATE_RE = re.compile(
    rf"\b(?:announce(?:d|ment)?|happen(?:ed)?|launch(?:ed)?|occur(?:red|rence)?|"
    rf"release(?:d|s)?|event)\s+on\s+(?:{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_NEWS_ABOUT_EVENT_ON_DATE_RE = re.compile(
    rf"\b(?:news|headlines?|press\s+coverage|media\s+coverage)\s+"
    rf"(?:about|of|regarding|concerning)\s+[^.!?\r\n]{{1,200}}?\s+on\s+"
    rf"(?:{_DATE_EXPRESSION_PATTERN})(?![\w/-])",
    re.I,
)
_RELATIVE_PUBLISHED_RE = re.compile(
    r"^(?:about\s+)?(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?)\s+ago$",
    re.I,
)


def _coerce_date(value: date | str | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value:
        try:
            return date.fromisoformat(str(value).strip())
        except ValueError:
            pass
    try:
        configured = runtime_retrieval_context().get("current_date_local")
        if configured:
            return date.fromisoformat(str(configured))
    except (TypeError, ValueError):
        pass
    return datetime.now(timezone.utc).date()


def _parse_date_span(value: str | None) -> tuple[date, date] | None:
    """Parse a bounded natural date expression without substituting runtime state."""
    if not value:
        return None
    candidate = strip_text(str(value)).strip(" ,")
    if not candidate:
        return None

    if re.fullmatch(r"(?:19|20)\d{2}", candidate):
        year = int(candidate)
        return date(year, 1, 1), date(year, 12, 31)

    normalized = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", candidate, flags=re.I)
    normalized = re.sub(r"^Sept\b", "Sep", normalized, flags=re.I)
    normalized = normalized.replace(",", "")
    for date_format in ("%Y-%m-%d", "%m/%d/%Y", "%B %d %Y", "%b %d %Y"):
        try:
            parsed = datetime.strptime(normalized, date_format).date()
            return parsed, parsed
        except ValueError:
            continue

    for month_format in ("%B %Y", "%b %Y"):
        try:
            parsed = datetime.strptime(normalized, month_format).date()
        except ValueError:
            continue
        last_day = calendar.monthrange(parsed.year, parsed.month)[1]
        return date(parsed.year, parsed.month, 1), date(
            parsed.year,
            parsed.month,
            last_day,
        )
    return None


def _matched_date_span(match: re.Match[str] | None, group: str = "date") -> tuple[date, date] | None:
    if match is None:
        return None
    return _parse_date_span(match.group(group))


def _matched_date_range_span(query: str) -> tuple[date, date] | None:
    match = _RANGE_DATE_RE.search(query) or _BETWEEN_DATE_RE.search(query)
    if match is not None:
        start = _matched_date_span(match, "start")
        end = _matched_date_span(match, "end")
    else:
        match = _SHARED_YEAR_RANGE_RE.search(query)
        if match is None:
            return None
        year = match.group("year")
        start = _parse_date_span(f'{match.group("start")} {year}')
        end = _parse_date_span(f'{match.group("end")} {year}')
    if start and end and start[0] <= end[1]:
        return start[0], end[1]
    return None


def _publication_date_scope(query: str, news_intent: bool) -> bool:
    return news_intent or bool(_PUBLICATION_DATE_RE.search(query))


def _coerce_timezone_name(value: str | None) -> str:
    candidate = str(value or "").strip()
    if not candidate:
        try:
            candidate = str(runtime_retrieval_context().get("timezone") or "").strip()
        except (TypeError, ValueError):
            candidate = ""
    candidate = candidate or "UTC"
    try:
        ZoneInfo(candidate)
    except (ValueError, ZoneInfoNotFoundError):
        return "UTC"
    return candidate


def _requested_max_age_days(query: str) -> int | None:
    windows = []
    for amount_value, unit_value in _RELATIVE_WINDOW_RE.findall(query):
        amount = max(1, int(amount_value or "1"))
        unit = unit_value.lower()
        if unit.startswith("hour"):
            windows.append(max(1, math.ceil(amount / 24)))
        elif unit.startswith("day"):
            windows.append(amount)
        elif unit.startswith("week"):
            windows.append(amount * 7)
        else:
            windows.append(amount * 31)
    if re.search(r"\bthis\s+week\b", query, re.I):
        windows.append(7)
    if re.search(r"\bthis\s+month\b", query, re.I):
        windows.append(31)
    if windows:
        return min(windows)
    if re.search(r"\brecent(?:ly)?\b", query, re.I):
        return 31
    return None


def _normalized_search_language(language: object) -> str:
    value = str(language or "auto").strip().replace("_", "-")
    if value.casefold() in {"auto", "all"}:
        return value.casefold()
    if not _VALID_SEARCH_LANGUAGE_RE.fullmatch(value):
        return "auto"
    parts = value.split("-", 1)
    return parts[0].lower() + (f"-{parts[1].upper()}" if len(parts) == 2 else "")


def _query_is_clearly_english(query: str) -> bool:
    """Conservatively identify English search prose without a language model."""

    words = [
        word.casefold()
        for word in re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKC", query))
        if len(word) > 1
    ]
    if not words:
        return False

    letters = [character for character in query if character.isalpha()]
    latin_letters = sum(
        1
        for character in letters
        if "LATIN" in unicodedata.name(character, "")
    )
    if letters and latin_letters / len(letters) < 0.9:
        return False

    word_set = set(words)
    if any(len(word_set & group) >= 2 for group in _NON_ENGLISH_FUNCTION_WORD_GROUPS):
        return False

    function_matches = len(word_set & _ENGLISH_FUNCTION_WORDS)
    search_matches = len(word_set & _ENGLISH_SEARCH_WORDS)
    strong_search_matches = len(word_set & _STRONG_ENGLISH_SEARCH_WORDS)
    return (
        function_matches >= 2
        or (function_matches >= 1 and search_matches >= 1)
        or strong_search_matches >= 1
        or ("current" in word_set and search_matches >= 2)
        or bool(re.search(r"\bmost\s+(?:powerful|recent|reliable|popular)\b", query, re.I))
    )


def _infer_query_language(query: str) -> str:
    return "en" if _query_is_clearly_english(query) else "auto"


def _accept_language_header(language: object) -> str:
    normalized = _normalized_search_language(language)
    if normalized in {"auto", "all"}:
        return "*"
    base = normalized.split("-", 1)[0]
    if normalized == "en":
        return "en-US,en;q=0.9,*;q=0.5"
    if "-" in normalized:
        return f"{normalized},{base};q=0.9,*;q=0.5"
    return f"{normalized},*;q=0.5"


def infer_search_policy(
    query: str,
    mode: str = "balanced",
    *,
    current_date: date | str | None = None,
    timezone_name: str | None = None,
) -> SearchPolicy:
    """Infer broad search controls without constraining ordinary web research."""
    query = str(query or "")
    reference_date = _coerce_date(current_date)
    language_match = _EXPLICIT_LANGUAGE_RE.search(query)
    language = _normalized_search_language(
        language_match.group(1) if language_match else _infer_query_language(query)
    )
    policy_timezone = _coerce_timezone_name(timezone_name)

    today = bool(_STRICT_TODAY_RE.search(query))
    yesterday = bool(_STRICT_YESTERDAY_RE.search(query))
    as_of_match = _AS_OF_DATE_RE.search(query)
    on_date_match = _ON_DATE_RE.search(query)
    since_match = _SINCE_DATE_RE.search(query)
    after_match = _AFTER_DATE_RE.search(query)
    before_match = _BEFORE_DATE_RE.search(query)
    in_date_match = _IN_DATE_RE.search(query)
    current = bool(_CURRENT_RE.search(query)) and as_of_match is None
    recent = bool(_RECENT_RE.search(query))
    requested_max_age_days = _requested_max_age_days(query)
    academic_intent = mode == "academic" or bool(_ACADEMIC_INTENT_RE.search(query))
    explicit_news_intent = bool(_NEWS_INTENT_RE.search(query))
    hacker_news_technical = bool(_HACKER_NEWS_TECHNICAL_RE.search(query))
    technical_intent = mode == "technical" or bool(
        not academic_intent
        and _TECHNICAL_INTENT_RE.search(query)
        and (not explicit_news_intent or hacker_news_technical)
    )
    news_intent = not technical_intent and (
        explicit_news_intent or bool(
            (today or yesterday or current or recent)
            and _EVENT_INTENT_RE.search(query)
        )
    )
    explicit_publication_scope = bool(_PUBLICATION_DATE_RE.search(query))
    publication_scope = _publication_date_scope(query, news_intent)
    event_on_date_scope = bool(
        _EVENT_ON_DATE_RE.search(query)
        or (
            _NEWS_ABOUT_EVENT_ON_DATE_RE.search(query)
            and not explicit_publication_scope
        )
    )
    publication_on_date_scope = publication_scope and not event_on_date_scope

    if academic_intent:
        categories = ("science", "general")
    elif technical_intent:
        categories = ("it", "general")
    elif news_intent:
        categories = ("news", "general")
    else:
        categories = ("general",)

    start_date = None
    target_date = None
    cutoff_date = None
    event_start_date = None
    event_end_date = None
    strict_date = False

    as_of_span = _matched_date_span(as_of_match)
    on_date_span = _matched_date_span(on_date_match)
    since_span = _matched_date_span(since_match)
    after_span = _matched_date_span(after_match)
    before_span = _matched_date_span(before_match)
    in_date_span = _matched_date_span(in_date_match)
    range_span = _matched_date_range_span(query)

    this_year = bool(re.search(r"\bthis\s+year\b", query, re.I))
    last_year = bool(re.search(r"\blast\s+year\b", query, re.I))
    if this_year:
        in_date_span = (date(reference_date.year, 1, 1), reference_date)
    elif last_year:
        previous_year = reference_date.year - 1
        in_date_span = (date(previous_year, 1, 1), date(previous_year, 12, 31))

    if today or yesterday:
        relative_date = reference_date - timedelta(days=1) if yesterday else reference_date
        if publication_scope:
            target_date = relative_date
            strict_date = True
            temporal_intent = "yesterday" if yesterday else "today"
            # A rolling one-day filter can exclude the first part of yesterday.
            time_range = "week" if yesterday else "day"
        else:
            event_start_date = relative_date
            event_end_date = relative_date
            temporal_intent = "event_date"
            time_range = None
    elif as_of_span is not None:
        # "As of" is a knowledge cutoff, not a recency window. Sources from any
        # earlier date remain eligible.
        cutoff_date = as_of_span[1]
        temporal_intent = "as_of"
        time_range = None
    elif range_span is not None:
        if publication_scope:
            start_date, cutoff_date = range_span
            temporal_intent = "publication_range"
        else:
            event_start_date, event_end_date = range_span
            temporal_intent = "event_range"
        time_range = None
    elif since_span is not None or after_span is not None:
        span = since_span or after_span
        assert span is not None
        boundary = span[0] if since_span is not None else span[1] + timedelta(days=1)
        if publication_scope:
            start_date = boundary
            temporal_intent = "publication_since"
        else:
            event_start_date = boundary
            temporal_intent = "event_since"
        time_range = None
    elif before_span is not None:
        boundary = before_span[0] - timedelta(days=1)
        if publication_scope:
            cutoff_date = boundary
            temporal_intent = "publication_before"
        else:
            event_end_date = boundary
            temporal_intent = "event_before"
        time_range = None
    elif on_date_span is not None:
        if publication_on_date_scope:
            if on_date_span[0] == on_date_span[1]:
                target_date = on_date_span[0]
                strict_date = True
                temporal_intent = "exact_date"
            else:
                start_date, cutoff_date = on_date_span
                temporal_intent = "publication_range"
        else:
            event_start_date, event_end_date = on_date_span
            temporal_intent = "event_date" if event_start_date == event_end_date else "event_range"
        time_range = None
    elif in_date_span is not None:
        if publication_scope:
            start_date, cutoff_date = in_date_span
            temporal_intent = "publication_range"
        else:
            event_start_date, event_end_date = in_date_span
            temporal_intent = "event_range"
        time_range = None
    elif requested_max_age_days is not None:
        temporal_intent = "recent"
        time_range = (
            "day"
            if requested_max_age_days <= 1
            else "week"
            if requested_max_age_days <= 7
            else "month"
            if requested_max_age_days <= 31
            else None
        )
    elif news_intent and current:
        temporal_intent = "current"
        time_range = "day"
    elif current:
        temporal_intent = "current"
        time_range = None
    elif recent:
        temporal_intent = "recent"
        time_range = None
    else:
        temporal_intent = "none"
        time_range = None

    return SearchPolicy(
        categories=categories,
        time_range=time_range,
        language=language,
        timezone=policy_timezone,
        temporal_intent=temporal_intent,
        reference_date=reference_date,
        start_date=start_date,
        target_date=target_date,
        cutoff_date=cutoff_date,
        event_start_date=event_start_date,
        event_end_date=event_end_date,
        strict_date=strict_date,
        news_intent=news_intent,
        freshness_max_age_days=(
            0
            if strict_date
            else None
            if temporal_intent.startswith("event_")
            else requested_max_age_days
            if requested_max_age_days is not None
            else 1
            if news_intent and current
            else 90
            if current
            else None
        ),
    )

BALANCED_DOMAIN_BOOSTS = {
    "docs.python.org": 0.7,
    "developer.mozilla.org": 0.7,
    "kubernetes.io": 0.7,
    "docs.docker.com": 0.7,
    "docs.github.com": 0.7,
    "learn.microsoft.com": 0.6,
    "cloud.google.com": 0.6,
    "docs.aws.amazon.com": 0.6,
    "github.com": 0.35,
}

TECHNICAL_DOMAIN_BOOSTS = {
    "github.com": 2.0,
    "docs.python.org": 2.2,
    "developer.mozilla.org": 2.2,
    "kubernetes.io": 2.2,
    "docs.docker.com": 2.2,
    "docs.github.com": 2.2,
    "learn.microsoft.com": 2.0,
    "cloud.google.com": 1.8,
    "docs.aws.amazon.com": 1.8,
    "stackoverflow.com": 1.6,
    "serverfault.com": 1.5,
    "superuser.com": 1.4,
    "unix.stackexchange.com": 1.6,
    "askubuntu.com": 1.4,
    "wiki.archlinux.org": 2.0,
    "man7.org": 1.8,
    "mankier.com": 1.5,
}

ACADEMIC_DOMAIN_BOOSTS = {
    "arxiv.org": 2.0,
    "semanticscholar.org": 1.8,
    "pubmed.ncbi.nlm.nih.gov": 2.0,
    "doi.org": 1.5,
    "crossref.org": 1.5,
}

# Backward-compatible public aggregate. Ranking selects a mode-specific subset.
DOMAIN_BOOSTS = {
    **BALANCED_DOMAIN_BOOSTS,
    **TECHNICAL_DOMAIN_BOOSTS,
    **ACADEMIC_DOMAIN_BOOSTS,
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


_TOPIC_TOKEN_RE = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)
_TOPIC_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "answer",
        "any",
        "are",
        "article",
        "articles",
        "as",
        "at",
        "authoritative",
        "be",
        "before",
        "best",
        "better",
        "between",
        "but",
        "by",
        "can",
        "clear",
        "concise",
        "could",
        "current",
        "currently",
        "did",
        "do",
        "does",
        "each",
        "find",
        "for",
        "from",
        "give",
        "good",
        "how",
        "i",
        "identify",
        "important",
        "in",
        "information",
        "into",
        "is",
        "it",
        "its",
        "lang",
        "language",
        "latest",
        "less",
        "look",
        "looking",
        "me",
        "more",
        "most",
        "my",
        "need",
        "new",
        "newest",
        "not",
        "of",
        "official",
        "on",
        "one",
        "or",
        "out",
        "please",
        "powerful",
        "provide",
        "rank",
        "recent",
        "recently",
        "research",
        "search",
        "should",
        "source",
        "sources",
        "summary",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "thing",
        "things",
        "this",
        "those",
        "through",
        "to",
        "today",
        "top",
        "up",
        "us",
        "using",
        "want",
        "was",
        "way",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "without",
        "would",
        "you",
        "your",
    }
)
_TOPIC_SYNONYM_ROOTS = {
    "advanced": "advanced",
    "applications": "app",
    "application": "app",
    "cats": "cat",
    "canine": "dog",
    "cordless": "wireless",
    "corded": "wired",
    "configure": "guidance",
    "configured": "guidance",
    "configuration": "guidance",
    "configuring": "guidance",
    "deploy": "guidance",
    "deployment": "guidance",
    "docs": "guidance",
    "documentation": "guidance",
    "guide": "guidance",
    "install": "guidance",
    "installation": "guidance",
    "installed": "guidance",
    "installing": "guidance",
    "manual": "guidance",
    "setup": "guidance",
    "tutorial": "guidance",
    "dogs": "dog",
    "expert": "advanced",
    "feline": "cat",
    "gratis": "free",
    "introductory": "beginner",
    "novice": "beginner",
    "no-cost": "free",
    "plant-based": "vegan",
    "plantbased": "vegan",
    "premium": "paid",
    "alternative": "comparison",
    "alternatives": "comparison",
    "benchmark": "comparison",
    "benchmarks": "comparison",
    "compare": "comparison",
    "compared": "comparison",
    "comparing": "comparison",
    "review": "comparison",
    "reviews": "comparison",
    "versus": "comparison",
}

# A broad lexical match is not sufficient when the result explicitly substitutes
# an opposing audience, product type, or hard qualifier. These groups are kept
# deliberately small: they reject observed query drift without trying to model
# arbitrary semantic entailment.
_TOPIC_CONTRAST_GROUPS = (
    (
        frozenset(),
        (frozenset({"vegan"}), frozenset({"chicken", "meat"})),
    ),
    (frozenset(), (frozenset({"wireless"}), frozenset({"wired"}))),
    (frozenset(), (frozenset({"free"}), frozenset({"paid"}))),
    (frozenset(), (frozenset({"indoor"}), frozenset({"outdoor"}))),
    (frozenset(), (frozenset({"beginner"}), frozenset({"advanced"}))),
    (frozenset(), (frozenset({"cat"}), frozenset({"dog"}))),
    (
        frozenset({"android", "tv"}),
        (
            frozenset({"box", "device", "hardware", "streamer"}),
            frozenset({"app", "remote"}),
        ),
    ),
)
_GENERIC_TOPIC_ROOTS = frozenset(
    {
        "comparison",
        "guidance",
        "happening",
        "headline",
        "new",  # ``news`` after the deliberately small inflection normalizer.
        "note",
        "paper",
        "post",
        "publication",
        "publish",
        "release",
        "report",
        "source",
        "study",
    }
)


def _normalized_topic_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).casefold().strip("-.")


def _topic_script_group(character: str) -> str:
    if not character.isalpha():
        return ""
    if character.isascii():
        return "latin"
    name = unicodedata.name(character, "")
    if name == "KATAKANA-HIRAGANA PROLONGED SOUND MARK":
        return ""
    for marker, group in (
        ("LATIN", "latin"),
        ("HIRAGANA", "hiragana"),
        ("KATAKANA", "katakana"),
        ("CJK", "han"),
        ("IDEOGRAPH", "han"),
        ("HANGUL", "hangul"),
    ):
        if marker in name:
            return group
    return "other"


def _split_topic_token(token: str) -> list[str]:
    """Split mixed-script words while preserving model identifiers and punctuation."""

    parts: list[str] = []
    current: list[str] = []
    current_group = ""
    for character in token:
        group = _topic_script_group(character)
        if group and current_group and group != current_group:
            parts.append("".join(current))
            current = []
        current.append(character)
        if group:
            current_group = group
    if current:
        parts.append("".join(current))
    return parts


_CJK_TOPIC_GROUPS = frozenset({"han", "hiragana", "katakana", "hangul"})
_CJK_BIGRAM_MIN_COVERAGE = 0.30


def _cjk_topic_bigrams(text: object, *, limit: int) -> list[str]:
    """Extract bounded bigrams without crossing CJK script or text boundaries."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    output: list[str] = []
    seen: set[str] = set()
    current: list[str] = []
    current_group = ""

    def flush() -> None:
        nonlocal current
        for index in range(max(0, len(current) - 1)):
            bigram = "".join(current[index : index + 2])
            if bigram in seen:
                continue
            seen.add(bigram)
            output.append(bigram)
            if len(output) >= limit:
                break
        current = []

    for character in normalized:
        group = _topic_script_group(character)
        if group not in _CJK_TOPIC_GROUPS:
            flush()
            current_group = ""
            if len(output) >= limit:
                break
            continue
        if current_group and group != current_group:
            flush()
            if len(output) >= limit:
                break
        current.append(character)
        current_group = group
    if len(output) < limit:
        flush()
    return output[:limit]


def _topic_term_root(term: str) -> str:
    synonym = _TOPIC_SYNONYM_ROOTS.get(term)
    if synonym:
        return synonym
    if len(term) < 4 or (any(char.isdigit() for char in term) and any(char.isalpha() for char in term)):
        return term
    if term.endswith("ies") and len(term) > 5:
        return f"{term[:-3]}y"
    if term.endswith(("ches", "shes", "sses", "xes", "zes")) and len(term) >= 5:
        return term[:-2]
    if term.endswith("ed") and len(term) > 5:
        return term[:-1] if term[-3] == "e" else term[:-2]
    if term.endswith("s") and not term.endswith(("ss", "us", "is")):
        root = term[:-1]
        return _TOPIC_SYNONYM_ROOTS.get(root, root)
    return _TOPIC_SYNONYM_ROOTS.get(term, term)


def _conflicting_topic_qualifiers(
    query_roots: set[str],
    document_roots: set[str],
) -> list[dict[str, list[str]]]:
    """Identify explicit substitutions while allowing comparisons and mixed uses."""

    conflicts = []
    for required_context, variants in _TOPIC_CONTRAST_GROUPS:
        if not required_context <= query_roots:
            continue
        requested_variants = [variant for variant in variants if query_roots & variant]
        # A query mentioning both sides is normally a comparison or a compound
        # concept (for example, "vegan chicken"), so neither side is exclusive.
        if len(requested_variants) != 1:
            continue
        requested = requested_variants[0]
        if document_roots & requested:
            continue
        opposing = set().union(*(variant for variant in variants if variant != requested))
        substituted = document_roots & opposing
        if substituted:
            conflicts.append(
                {
                    "requested": sorted(query_roots & requested),
                    "substituted": sorted(substituted),
                }
            )
    return conflicts


def _semantically_matched_query_roots(
    query_roots: set[str],
    document_roots: set[str],
) -> set[str]:
    """Include same-variant qualifier aliases without equating opposing variants."""

    matched = query_roots & document_roots
    for required_context, variants in _TOPIC_CONTRAST_GROUPS:
        if not required_context <= query_roots or not required_context <= document_roots:
            continue
        for variant in variants:
            if query_roots & variant and document_roots & variant:
                matched.update(query_roots & variant)
    return matched


def _topic_terms(text: object, *, query: bool = False) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    output: list[str] = []
    seen: set[str] = set()
    raw_terms = (
        part
        for raw in _TOPIC_TOKEN_RE.findall(normalized)
        for part in _split_topic_token(raw)
    )
    for raw in raw_terms:
        term = _normalized_topic_token(raw)
        if (
            not term
            or term in _TOPIC_STOP_WORDS
            or term.isdigit()
            or (
                len(term) == 1
                and (
                    not term.isalpha()
                    or "LATIN" in unicodedata.name(term, "")
                )
            )
        ):
            continue
        root = _topic_term_root(term)
        if root in seen:
            continue
        seen.add(root)
        output.append(term)
    if query and len(output) > 32:
        ranked = sorted(
            enumerate(output),
            key=lambda item: (-_topic_term_weight(item[1]), item[0]),
        )[:32]
        output = [term for _, term in sorted(ranked)]
    return output


def _topic_term_weight(term: str) -> float:
    has_alpha = any(character.isalpha() for character in term)
    has_digit = any(character.isdigit() for character in term)
    if has_alpha and has_digit:
        return 2.2
    if len(term) >= 10:
        return 1.6
    if len(term) >= 7:
        return 1.35
    if len(term) <= 2:
        return 1.25
    return 1.0


def _distinctive_topic_terms(terms: list[str]) -> list[str]:
    """Exclude research-form and intent words that cannot identify a subject."""

    return [
        term
        for term in terms
        if _topic_term_root(term) not in _GENERIC_TOPIC_ROOTS
    ]


def _result_topic_text(result: dict[str, Any]) -> str:
    title = result.get("title") or ""
    snippet = result.get("snippet") or result.get("content") or ""
    raw_url = str(result.get("url") or "")
    parsed = urlsplit(raw_url)
    url_text = f"{parsed.hostname or ''} {parsed.path.replace('/', ' ')}"
    return f"{title} {snippet} {url_text}"


def search_result_relevance(
    result: dict[str, Any],
    query: str,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Return bounded lexical topical relevance for a raw or compact result."""

    resolved_threshold = (
        SEARCH_RELEVANCE_MIN_SCORE if threshold is None else max(0.0, min(1.0, float(threshold)))
    )
    query_terms = _topic_terms(query, query=True)
    if not query_terms:
        return {
            "score": 1.0,
            "is_relevant": True,
            "matched_terms": [],
            "query_terms": [],
            "distinctive_query_terms": [],
            "matched_distinctive_terms": [],
            "matched_phrases": [],
            "threshold": round(resolved_threshold, 3),
            "reason": "no_usable_query_terms",
        }

    document_terms = _topic_terms(_result_topic_text(result))
    document_roots = {_topic_term_root(term) for term in document_terms}
    query_roots = [_topic_term_root(term) for term in query_terms]
    matched_terms = [
        term
        for term, root in zip(query_terms, query_roots, strict=True)
        if root in document_roots
    ]
    matched_roots = {_topic_term_root(term) for term in matched_terms}
    semantic_matched_roots = _semantically_matched_query_roots(
        set(query_roots),
        document_roots,
    )
    distinctive_query_terms = _distinctive_topic_terms(query_terms)
    matched_distinctive_terms = [
        term
        for term in distinctive_query_terms
        if _topic_term_root(term) in semantic_matched_roots
    ]

    query_phrases = [
        (query_roots[index], query_roots[index + 1])
        for index in range(len(query_roots) - 1)
    ]
    document_root_sequence = [_topic_term_root(term) for term in document_terms]
    document_phrases = set(zip(document_root_sequence, document_root_sequence[1:]))
    matched_phrase_roots = [phrase for phrase in query_phrases if phrase in document_phrases]
    matched_phrases = [" ".join(phrase) for phrase in matched_phrase_roots[:10]]

    total_weight = sum(_topic_term_weight(term) for term in query_terms)
    matched_weight = sum(
        _topic_term_weight(term)
        for term in query_terms
        if _topic_term_root(term) in matched_roots
    )
    coverage = matched_weight / total_weight if total_weight else 0.0
    breadth = len(matched_terms) / min(3, len(query_terms))
    phrase_score = min(1.0, len(matched_phrase_roots) / min(2, len(query_phrases))) if query_phrases else 0.0
    identifier_match = any(
        any(character.isalpha() for character in term)
        and any(character.isdigit() for character in term)
        for term in matched_terms
    )
    score = min(
        1.0,
        (coverage * 0.6)
        + (breadth * 0.3)
        + (phrase_score * 0.1)
        + (0.12 if identifier_match else 0.0),
    )
    minimum_matches = 2 if len(query_terms) >= 4 and not identifier_match else 1
    generic_only_overlap = bool(
        matched_terms
        and distinctive_query_terms
        and not matched_distinctive_terms
    )
    topic_conflicts = _conflicting_topic_qualifiers(
        set(query_roots),
        document_roots,
    )
    is_relevant = (
        score >= resolved_threshold
        and len(matched_terms) >= minimum_matches
        and not generic_only_overlap
        and not topic_conflicts
    )
    cjk_overlap = None
    cjk_rescue = False
    if not is_relevant and not topic_conflicts:
        query_cjk_bigrams = _cjk_topic_bigrams(query, limit=64)
        if query_cjk_bigrams:
            document_cjk_bigrams = set(
                _cjk_topic_bigrams(_result_topic_text(result), limit=256)
            )
            matched_cjk_bigrams = [
                bigram
                for bigram in query_cjk_bigrams
                if bigram in document_cjk_bigrams
            ]
            required_matches = min(2, len(query_cjk_bigrams))
            cjk_coverage = len(matched_cjk_bigrams) / len(query_cjk_bigrams)
            cjk_score = min(
                1.0,
                (cjk_coverage * 0.75)
                + (min(1.0, len(matched_cjk_bigrams) / 4) * 0.25),
            )
            cjk_rescue = (
                len(matched_cjk_bigrams) >= required_matches
                and cjk_coverage >= _CJK_BIGRAM_MIN_COVERAGE
                and cjk_score >= resolved_threshold
            )
            cjk_overlap = {
                "query_bigram_count": len(query_cjk_bigrams),
                "matched_bigram_count": len(matched_cjk_bigrams),
                "matched_bigrams": matched_cjk_bigrams[:16],
                "coverage": round(cjk_coverage, 3),
                "score": round(cjk_score, 3),
                "minimum_matches": required_matches,
                "minimum_coverage": _CJK_BIGRAM_MIN_COVERAGE,
                "used": cjk_rescue,
            }
            if cjk_rescue:
                score = max(score, cjk_score)
                is_relevant = True
    reason = (
        "conflicting_topic_qualifier"
        if topic_conflicts
        else "relevant_cjk_bigram_overlap"
        if cjk_rescue
        else "relevant_topic_overlap"
        if is_relevant
        else "generic_only_topic_overlap"
        if generic_only_overlap
        else "insufficient_topic_overlap"
    )
    analysis = {
        "score": round(score, 3),
        "is_relevant": is_relevant,
        "matched_terms": matched_terms,
        "query_terms": query_terms,
        "distinctive_query_terms": distinctive_query_terms,
        "matched_distinctive_terms": matched_distinctive_terms,
        "matched_phrases": matched_phrases,
        "threshold": round(resolved_threshold, 3),
        "reason": reason,
    }
    if cjk_overlap is not None:
        analysis["cjk_overlap"] = cjk_overlap
    if topic_conflicts:
        analysis["topic_conflicts"] = topic_conflicts
    return analysis


def topical_relevance_summary(
    results,
    query: str,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Summarize relevance and owner diversity without discarding diagnostics."""

    indexed_analyses = [
        (index, search_result_relevance(result, query, threshold=threshold))
        for index, result in enumerate(results)
        if isinstance(result, dict)
    ]
    analyses = [analysis for _, analysis in indexed_analyses]
    relevant_indexes = [
        index for index, analysis in indexed_analyses if analysis["is_relevant"]
    ]
    relevant_owners = {
        estimate_source_owner_domain(
            str(results[index].get("domain") or get_domain(results[index].get("url") or ""))
        )
        for index in relevant_indexes
    }
    relevant_owners.discard("")
    relevant_count = len(relevant_indexes)
    result_count = len(analyses)
    return {
        "result_count": result_count,
        "relevant_count": relevant_count,
        "relevant_ratio": round(relevant_count / result_count, 3) if result_count else 0.0,
        "distinct_relevant_owners": len(relevant_owners),
        "relevant_indexes": relevant_indexes,
        "scores": [analysis["score"] for analysis in analyses],
        "threshold": round(
            SEARCH_RELEVANCE_MIN_SCORE if threshold is None else max(0.0, min(1.0, float(threshold))),
            3,
        ),
        "status": (
            "no_results"
            if not analyses
            else "relevant"
            if relevant_count
            else "low_relevance"
        ),
    }


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


def parse_published_datetime(
    value: object,
    *,
    now: datetime | None = None,
    timezone_name: str | None = None,
) -> datetime | None:
    """Parse common SearX publication date representations into UTC."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, (int, float)):
        timestamp = float(value)
        if not math.isfinite(timestamp):
            return None
        if abs(timestamp) >= 100_000_000_000:
            timestamp /= 1000
        try:
            parsed = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        text = strip_text(str(value))
        if not text:
            return None
        relative = _RELATIVE_PUBLISHED_RE.fullmatch(text)
        if relative:
            amount = int(relative.group(1))
            unit = relative.group(2).lower()
            delta_key = (
                "seconds"
                if unit.startswith("second")
                else "minutes"
                if unit.startswith("minute")
                else "hours"
                if unit.startswith("hour")
                else "days"
                if unit.startswith("day")
                else "weeks"
            )
            reference = now or datetime.now(timezone.utc)
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=timezone.utc)
            parsed = reference - timedelta(**{delta_key: amount})
        else:
            iso_candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
            try:
                parsed = datetime.fromisoformat(iso_candidate)
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(text)
                except (TypeError, ValueError, OverflowError):
                    parsed = None
                if parsed is None:
                    for date_format in (
                        "%Y/%m/%d",
                        "%B %d, %Y",
                        "%b %d, %Y",
                        "%d %B %Y",
                        "%d %b %Y",
                    ):
                        try:
                            parsed = datetime.strptime(text, date_format)
                            break
                        except ValueError:
                            continue
                if parsed is None:
                    return None

    if parsed.tzinfo is None:
        default_timezone = timezone.utc
        if timezone_name:
            try:
                default_timezone = ZoneInfo(timezone_name)
            except (ValueError, ZoneInfoNotFoundError):
                pass
        parsed = parsed.replace(tzinfo=default_timezone)
    return parsed.astimezone(timezone.utc)


def normalize_published_at(value: object) -> str | None:
    parsed = parse_published_datetime(value)
    return parsed.isoformat() if parsed is not None else None


def _publication_freshness_status(
    published: datetime | None,
    policy: SearchPolicy,
    reference_date: date,
) -> str:
    if published is None:
        return "undated"
    try:
        published_date = published.astimezone(ZoneInfo(policy.timezone)).date()
    except (ValueError, ZoneInfoNotFoundError):
        published_date = published.date()
    if policy.strict_date and policy.target_date is not None:
        return "exact_match" if published_date == policy.target_date else "outside_window"
    if policy.start_date is not None and published_date < policy.start_date:
        return "outside_window"
    if policy.cutoff_date is not None and published_date > policy.cutoff_date:
        return "outside_window"
    if policy.start_date is not None or policy.cutoff_date is not None:
        return "within_window"
    max_age_days = policy.freshness_max_age_days
    if max_age_days is None:
        return "not_evaluated"
    age_days = (reference_date - published_date).days
    return "within_window" if 0 <= age_days <= max_age_days else "outside_window"


def _domain_boosts_for(mode: str, policy: SearchPolicy) -> Dict[str, float]:
    if policy.news_intent:
        return {}
    if mode == "technical" or "it" in policy.categories:
        return TECHNICAL_DOMAIN_BOOSTS
    if mode == "academic" or "science" in policy.categories:
        return ACADEMIC_DOMAIN_BOOSTS
    return BALANCED_DOMAIN_BOOSTS


def score_search_result(
    result: Dict[str, Any],
    query: str,
    mode: str = "balanced",
    *,
    policy: SearchPolicy | None = None,
) -> Dict[str, Any]:
    policy = policy or infer_search_policy(query, mode)
    title = result.get("title") or ""
    url = result.get("url") or ""
    snippet = result.get("content") or result.get("snippet") or ""
    engine = result.get("engine")
    domain = normalize_domain(get_domain(url))

    score = 1.0
    reasons = []

    search_rank = result.get("search_rank")
    if isinstance(search_rank, int) and search_rank > 0:
        rank_bonus = 2.5 / math.sqrt(search_rank)
        score += rank_bonus
        reasons.append(f"SearX rank: {search_rank}")

    searxng_score = result.get("searxng_score")
    if (
        isinstance(searxng_score, (int, float))
        and not isinstance(searxng_score, bool)
        and math.isfinite(float(searxng_score))
        and searxng_score > 0
    ):
        score_bonus = min(1.5, math.log1p(float(searxng_score)) * 0.35)
        score += score_bonus
        reasons.append("SearX aggregate score")

    boost, boost_domain = domain_adjustment(domain, _domain_boosts_for(mode, policy))
    if boost_domain:
        score += boost
        reasons.append(f"domain boost: {boost_domain}")

    penalty, penalty_domain = domain_adjustment(domain, DOMAIN_PENALTIES)
    if penalty_domain:
        score += penalty
        reasons.append(f"domain penalty: {penalty_domain}")

    if domain.endswith(".gov"):
        score += 0.75 if policy.news_intent else 1.5
        reasons.append("government primary source")
    elif domain.endswith(".edu"):
        score += 1.2 if mode == "academic" or "science" in policy.categories else 0.5
        reasons.append("academic institution source")

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

    if (
        engine in {"github", "stackoverflow"}
        and (mode == "technical" or "it" in policy.categories)
        and not policy.news_intent
    ):
        score += 0.8
        reasons.append(f"engine boost: {engine}")
    elif (
        engine == "arxiv"
        and (mode == "academic" or "science" in policy.categories)
        and not policy.news_intent
    ):
        score += 0.8
        reasons.append("engine boost: arxiv")

    freshness_status = result.get("freshness_status")
    if freshness_status == "exact_match":
        score += 4.0
        reasons.append("publication date exact match")
    elif freshness_status == "within_window" and policy.temporal_intent != "none":
        score += 2.0
        reasons.append("publication date within requested window")
    elif freshness_status == "outside_window":
        score -= 2.5
        reasons.append("publication date outside requested window")
    elif freshness_status == "undated" and policy.strict_date:
        score -= 1.5
        reasons.append("undated fallback for exact-date search")
    elif freshness_status == "undated" and policy.temporal_intent != "none":
        score -= 0.75
        reasons.append("undated fallback for temporal search")

    result["score"] = round(score, 3)
    result["score_reasons"] = reasons
    return result


def compact_search_results(
    data: dict,
    query: str,
    max_results: int = 10,
    mode: str = "balanced",
    *,
    policy: SearchPolicy | None = None,
) -> SearchResults:
    policy = policy or infer_search_policy(query, mode)
    reference_date = policy.reference_date or _coerce_date(None)
    results_by_url: dict[str, dict[str, Any]] = {}
    counts = {
        "raw_results": 0,
        "accepted_results": 0,
        "exact_match_results": 0,
        "within_window_results": 0,
        "not_evaluated_results": 0,
        "undated_results": 0,
        "outside_window_results": 0,
        "outside_window_dropped": 0,
    }

    raw_results = data.get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    for search_rank, item in enumerate(raw_results, start=1):
        counts["raw_results"] += 1
        if not isinstance(item, dict):
            continue
        raw_url = item.get("url")
        if not isinstance(raw_url, str) or len(raw_url) > 8192:
            continue
        url = normalize_search_url(raw_url)
        title = item.get("title")
        content = item.get("content") or ""

        if not url or not title:
            continue

        domain = normalize_domain(get_domain(url))
        if any(domain_matches(domain, item) for item in BLOCKED_DOMAINS):
            continue

        raw_published_at = item.get("publishedDate") or item.get("published_at")
        published = parse_published_datetime(
            raw_published_at,
            timezone_name=policy.timezone,
        )
        freshness_status = _publication_freshness_status(
            published,
            policy,
            reference_date,
        )
        counts[f"{freshness_status}_results"] += 1
        if (
            policy.strict_date
            or policy.start_date is not None
            or policy.cutoff_date is not None
        ) and freshness_status == "outside_window":
            counts["outside_window_dropped"] += 1
            continue

        raw_engine = item.get("engine")
        engine = strip_text(str(raw_engine))[:100] if raw_engine is not None else None
        raw_engines = item.get("engines")
        engines = (
            [strip_text(str(value))[:100] for value in raw_engines[:10]]
            if isinstance(raw_engines, list)
            else None
        )
        result = {
            "title": strip_text(str(title))[:500],
            "url": url,
            "domain": domain,
            "snippet": strip_text(str(content))[:900],
            "engine": engine,
            "engines": engines,
            "search_rank": search_rank,
            "searxng_score": item.get("score"),
            "published_at": published.isoformat() if published is not None else None,
            "freshness_status": freshness_status,
            "content_trust": "untrusted_external_content",
        }
        if raw_published_at is not None and published is None:
            result["published_at_raw"] = strip_text(str(raw_published_at))[:200]

        result["topical_relevance"] = search_result_relevance(result, query)
        result = score_search_result(result, query=query, mode=mode, policy=policy)
        existing = results_by_url.get(url)
        freshness_priority = {
            "exact_match": 3,
            "within_window": 2,
            "not_evaluated": 1,
            "undated": 1,
            "outside_window": 0,
        }
        relevance = result["topical_relevance"]
        result_key = (
            bool(relevance.get("is_relevant")),
            freshness_priority.get(result.get("freshness_status"), 1),
            result.get("score", 0),
            float(relevance.get("score") or 0.0),
        )
        if existing:
            existing_relevance = existing.get("topical_relevance") or {}
            existing_key = (
                bool(existing_relevance.get("is_relevant")),
                freshness_priority.get(existing.get("freshness_status"), 1),
                existing.get("score", 0),
                float(existing_relevance.get("score") or 0.0),
            )
        else:
            existing_key = (False, -1.0, -1, float("-inf"))
        if result_key > existing_key:
            results_by_url[url] = result

    results = list(results_by_url.values())
    results.sort(
        key=lambda item: (
            bool((item.get("topical_relevance") or {}).get("is_relevant")),
            float(item.get("score", 0) or 0.0),
            float((item.get("topical_relevance") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    limited = results[: max(0, max_results)]
    counts["eligible_results"] = len(results)
    counts["returned_results"] = len(limited)
    counts["accepted_results"] = len(limited)
    relevance = topical_relevance_summary(limited, query)
    diagnostics = {
        "search_policy": policy.to_dict(),
        "counts": counts,
        "topical_relevance": relevance,
    }
    return SearchResults(limited, diagnostics=diagnostics, policy=policy)


def _normalize_unresponsive_engines(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output = []
    for item in value[:100]:
        if isinstance(item, dict):
            engine = strip_text(str(item.get("engine") or item.get("name") or ""))
            reason = strip_text(str(item.get("reason") or item.get("error") or ""))
        elif isinstance(item, (list, tuple)):
            engine = strip_text(str(item[0])) if item else ""
            reason = strip_text(str(item[1])) if len(item) > 1 else ""
        else:
            engine = strip_text(str(item))
            reason = ""
        if not engine:
            continue
        diagnostic = {"engine": engine[:200]}
        if reason:
            diagnostic["reason_code"] = _failure_reason_code(reason)
            retry_after = _retry_after_from_reason(reason)
            if retry_after is not None:
                diagnostic["retry_after_seconds"] = min(
                    SEARCH_ENGINE_MAX_COOLDOWN_SECONDS,
                    retry_after,
                )
        output.append(diagnostic)
    return output


async def searxng_search(
    query: str,
    max_results: int = 10,
    mode: str = "balanced",
    *,
    policy: SearchPolicy | None = None,
    current_date: date | str | None = None,
    timezone_name: str | None = None,
    cache_scope: str | None = None,
) -> SearchResults:
    policy = policy or infer_search_policy(
        query,
        mode,
        current_date=current_date,
        timezone_name=timezone_name,
    )
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
    outbound_query = _escape_searx_control_tokens(query)
    cache_key = _search_cache_key(
        query,
        outbound_query,
        max_results,
        mode,
        policy,
        base_url,
        cache_scope,
    )
    cached = await _cache_get(cache_key)
    stale = cached
    if cached is not None:
        age = time.time() - float(cached["cached_at"])
        if age <= SEARCH_CACHE_TTL_SECONDS:
            return _cached_results(cached, "fresh")

    try:
        async with asyncio.timeout(SEARXNG_TIMEOUT_SECONDS):
            async with _cache_key_lock(cache_key):
                # Coalesce normalized duplicate requests in this process. The
                # first waiter populates Redis/local cache for every follower.
                cached_after_wait = await _cache_get(cache_key)
                if cached_after_wait is not None:
                    age = time.time() - float(cached_after_wait["cached_at"])
                    if age <= SEARCH_CACHE_TTL_SECONDS:
                        return _cached_results(cached_after_wait, "fresh_coalesced")
                    stale = cached_after_wait

                results = await _staged_searxng_search(
                    query,
                    outbound_query=outbound_query,
                    max_results=max_results,
                    mode=mode,
                    policy=policy,
                    base_url=base_url,
                )
                if results and _cacheable_results(results, max_results):
                    results.diagnostics["cache"] = {"status": "miss"}
                    await _cache_set(cache_key, results)
                    return results
                if results:
                    results.diagnostics["cache"] = {"status": "bypassed_partial"}
                    return results
                if (
                    stale is not None
                    and stale.get("results")
                    and results.diagnostics.get("acquisition_status") == "failed"
                    and results.diagnostics.get("failure_class") == "transient"
                ):
                    return _cached_results(
                        stale,
                        "stale_fallback",
                        attempt_diagnostics=results.diagnostics,
                    )
                results.diagnostics["cache"] = {"status": "miss"}
                return results
    except TimeoutError:
        if stale is not None and stale.get("results"):
            return _cached_results(
                stale,
                "stale_fallback",
                attempt_diagnostics={"error": "search deadline exceeded"},
            )
        raise


async def _staged_searxng_search(
    query: str,
    *,
    outbound_query: str,
    max_results: int,
    mode: str,
    policy: SearchPolicy,
    base_url: str,
) -> SearchResults:
    aggregate_data: dict[str, Any] = {"results": []}
    stage_diagnostics: list[dict[str, Any]] = []
    all_unresponsive: dict[tuple[str, str], dict[str, str]] = {}
    seen_result_urls: set[str] = set()
    successful_responses = 0
    responsive_engines = 0
    transient_failures = 0
    configuration_failures = 0

    async with httpx.AsyncClient(timeout=SEARXNG_TIMEOUT_SECONDS) as client:
        for index, configured_engines in enumerate(
            _engine_stages(policy, mode), start=1
        ):
            service_eligible, service_skipped = await _eligible_engines(
                (_SEARX_SERVICE_CIRCUIT,)
            )
            if not service_eligible:
                transient_failures += 1
                stage_diagnostics.append(
                    {
                        "stage": index,
                        "configured_engines": list(configured_engines),
                        "engines": [],
                        "skipped_cooldowns": [],
                        "status": "service_circuit_open",
                        "retry_after_seconds": service_skipped[0].get(
                            "retry_after_seconds"
                        ),
                    }
                )
                break
            validated_engines = _validated_stage_engines(configured_engines)
            if not validated_engines:
                configuration_failures += 1
                stage_diagnostics.append(
                    {
                        "stage": index,
                        "configured_engines": list(configured_engines),
                        "engines": [],
                        "skipped_cooldowns": [],
                        "status": "invalid_engine_configuration",
                    }
                )
                continue
            eligible, skipped = await _eligible_engines(validated_engines)
            stage_diagnostic: dict[str, Any] = {
                "stage": index,
                "configured_engines": list(configured_engines),
                "engines": list(eligible),
                "skipped_cooldowns": skipped,
            }
            if not eligible:
                transient_failures += 1
                stage_diagnostic["status"] = "circuit_open"
                stage_diagnostics.append(stage_diagnostic)
                continue

            started = time.monotonic()
            try:
                async with _engine_request_slots(eligible):
                    # An overlapping request may have opened a circuit while
                    # this request waited for the per-engine slot.
                    service_after_wait, service_skipped_after_wait = (
                        await _eligible_engines((_SEARX_SERVICE_CIRCUIT,))
                    )
                    if not service_after_wait:
                        transient_failures += 1
                        stage_diagnostic.update(
                            {
                                "status": "service_circuit_open",
                                "retry_after_seconds": (
                                    service_skipped_after_wait[0].get(
                                        "retry_after_seconds"
                                    )
                                ),
                            }
                        )
                        stage_diagnostics.append(stage_diagnostic)
                        break
                    eligible_after_wait, newly_skipped = await _eligible_engines(
                        tuple(eligible)
                    )
                    stage_diagnostic["skipped_cooldowns"].extend(newly_skipped)
                    stage_diagnostic["engines"] = list(eligible_after_wait)
                    if not eligible_after_wait:
                        stage_diagnostic["status"] = "circuit_open"
                        stage_diagnostics.append(stage_diagnostic)
                        continue
                    data = await _searxng_stage_request(
                        client,
                        base_url=base_url,
                        query=outbound_query,
                        policy=policy,
                        engines=eligible_after_wait,
                    )
            except _SearxHTTPError as exc:
                reason = f"HTTP {exc.status_code}"
                is_transient = exc.status_code in {408, 425, 429} or (
                    500 <= exc.status_code <= 599
                )
                if is_transient:
                    transient_failures += 1
                    await _record_engine_failure(
                        _SEARX_SERVICE_CIRCUIT,
                        reason,
                        retry_after=exc.retry_after,
                    )
                else:
                    configuration_failures += 1
                stage_diagnostic.update(
                    {
                        "status": "service_error",
                        "http_status": exc.status_code,
                        "retry_after_seconds": exc.retry_after,
                    }
                )
                stage_diagnostics.append(stage_diagnostic)
                break
            except httpx.TimeoutException:
                transient_failures += 1
                await _record_engine_failure(_SEARX_SERVICE_CIRCUIT, "timeout")
                stage_diagnostic.update({"status": "timeout"})
                stage_diagnostics.append(stage_diagnostic)
                continue
            except httpx.RequestError as exc:
                transient_failures += 1
                await _record_engine_failure(
                    _SEARX_SERVICE_CIRCUIT,
                    type(exc).__name__,
                )
                stage_diagnostic.update(
                    {"status": "service_unavailable", "error": type(exc).__name__}
                )
                stage_diagnostics.append(stage_diagnostic)
                break

            stage_diagnostic["duration_seconds"] = round(
                max(0.0, time.monotonic() - started),
                3,
            )
            raw_results = data.get("results")
            if isinstance(raw_results, list):
                for item in raw_results:
                    raw_url = item.get("url") if isinstance(item, dict) else None
                    normalized_url = (
                        normalize_search_url(raw_url)
                        if isinstance(raw_url, str)
                        else ""
                    )
                    if normalized_url and normalized_url in seen_result_urls:
                        continue
                    if normalized_url:
                        seen_result_urls.add(normalized_url)
                    aggregate_data["results"].append(item)

            unresponsive = _normalize_unresponsive_engines(
                data.get("unresponsive_engines")
            )
            failures = {
                item["engine"].casefold(): item
                for item in unresponsive
            }
            for item in unresponsive:
                key = (
                    item["engine"].casefold(),
                    item.get("reason_code", "").casefold(),
                )
                all_unresponsive[key] = item
            provider_updates = [_record_engine_success(_SEARX_SERVICE_CIRCUIT)]
            requested_failures = 0
            for engine in stage_diagnostic["engines"]:
                failure = failures.get(engine.casefold())
                if failure is None:
                    provider_updates.append(_record_engine_success(engine))
                    continue
                requested_failures += 1
                provider_updates.append(
                    _record_engine_failure(
                        engine,
                        failure.get("reason_code", "upstream_unresponsive"),
                        retry_after=failure.get("retry_after_seconds"),
                    )
                )
            await asyncio.gather(*provider_updates)
            successful_responses += 1
            responsive_engines += max(
                0,
                len(stage_diagnostic["engines"]) - requested_failures,
            )
            if requested_failures:
                transient_failures += 1

            stage_diagnostic.update(
                {
                    "status": "partial" if unresponsive else "ok",
                    "raw_results": len(raw_results)
                    if isinstance(raw_results, list)
                    else 0,
                    "unresponsive_engines": unresponsive,
                }
            )
            stage_diagnostics.append(stage_diagnostic)
            interim = compact_search_results(
                aggregate_data,
                query=query,
                max_results=max_results,
                mode=mode,
                policy=policy,
            )
            stage_diagnostic["topical_relevance"] = copy.deepcopy(
                interim.diagnostics["topical_relevance"]
            )
            stage_diagnostic["coverage_sufficient"] = _coverage_sufficient(
                interim,
                max_results,
            )
            if stage_diagnostic["coverage_sufficient"]:
                break

    results = compact_search_results(
        aggregate_data,
        query=query,
        max_results=max_results,
        mode=mode,
        policy=policy,
    )
    results.diagnostics["unresponsive_engines"] = list(all_unresponsive.values())
    results.diagnostics["counts"]["unresponsive_engines"] = len(all_unresponsive)
    results.diagnostics["search_stages"] = stage_diagnostics
    results.diagnostics["engine_policy"] = _engine_policy_name(policy)
    if successful_responses and responsive_engines:
        acquisition_status = "partial" if transient_failures else "succeeded"
        failure_class = "transient" if transient_failures else None
    elif configuration_failures and not transient_failures:
        acquisition_status = "failed"
        failure_class = "configuration"
    elif transient_failures:
        acquisition_status = "failed"
        failure_class = "transient"
    else:
        acquisition_status = "failed"
        failure_class = "configuration"
    results.diagnostics["acquisition_status"] = acquisition_status
    if failure_class:
        results.diagnostics["failure_class"] = failure_class
    results.diagnostics["acquisition_error"] = (
        None
        if acquisition_status == "succeeded"
        else {
            "code": (
                "search_backend_unavailable"
                if failure_class == "transient"
                else "search_configuration_invalid"
            ),
            "successful_responses": successful_responses,
            "responsive_engines": responsive_engines,
        }
    )
    return results


async def _searxng_stage_request(
    client,
    *,
    base_url: str,
    query: str,
    policy: SearchPolicy,
    engines: list[str],
) -> dict[str, Any]:
    validated_engines = _validated_stage_engines(tuple(engines))
    if not validated_engines or len(validated_engines) != len(engines):
        raise ValueError("SearXNG engine stage is empty or contains an unknown engine")
    params = {
        "q": query,
        "format": "json",
        "language": _normalized_search_language(policy.language),
        "engines": ",".join(validated_engines),
    }
    if policy.time_range:
        params["time_range"] = policy.time_range

    async with client.stream(
        "GET",
        f"{base_url}/search",
        params=params,
        headers={"Accept-Language": _accept_language_header(policy.language)},
    ) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            retry_after = _parse_retry_after(
                exc.response.headers.get("retry-after")
            )
            raise _SearxHTTPError(
                status_code,
                retry_after=retry_after,
            ) from exc
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise ValueError("SearXNG returned an invalid Content-Length") from exc
            if declared_length > SEARXNG_MAX_RESPONSE_BYTES:
                raise ValueError(
                    "SearXNG response exceeded SEARXNG_MAX_RESPONSE_BYTES"
                )
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > SEARXNG_MAX_RESPONSE_BYTES:
                raise ValueError("SearXNG response exceeded SEARXNG_MAX_RESPONSE_BYTES")
            body.extend(chunk)

    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SearXNG returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("SearXNG returned a non-object JSON response")
    return data
