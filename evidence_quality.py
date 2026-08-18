"""Deterministic source, freshness, and evidence-quality metadata helpers."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


CURRENT_QUERY_RE = re.compile(
    r"\b(?:current|currently|latest|newest|news|recent|today|tonight|this\s+"
    r"(?:week|month|year)|updated?|release(?:d|s)?|changelog|security\s+advisory|cve)\b",
    re.I,
)
VERSION_SENSITIVE_RE = re.compile(
    r"\b(?:api|compatib(?:ility|le)|configure|configuration|docker|driver|firmware|"
    r"install(?:ation)?|kubernetes|library|manual|package|patch|release|sdk|setup|"
    r"software|upgrade|version)\b",
    re.I,
)
VERSION_RE = re.compile(
    r"(?<![\w.])v?(\d{1,4}\.\d{1,3}(?:\.\d{1,4}){0,2}"
    r"(?:[-+][0-9A-Za-z][0-9A-Za-z.-]{0,30})?)(?![\w.])"
)
NAMED_VERSION_RE = re.compile(
    r"\b(Android|Chrome|Debian|Docker|Firefox|iOS|Java|macOS|Node(?:\.js)?|"
    r"Python|Ubuntu|Windows)\s+(?:version\s+)?(\d{1,4}(?:\.\d{1,3}){0,3})\b",
    re.I,
)
DOCUMENTATION_PATH_RE = re.compile(
    r"/(?:api|docs?|documentation|guide|guides|kb|manual|reference|support)(?:/|$)",
    re.I,
)
RELEASE_PATH_RE = re.compile(r"/(?:changelog|releases?|tags?)(?:/|$)", re.I)

STANDARD_DOMAINS = {
    "datatracker.ietf.org",
    "ietf.org",
    "iso.org",
    "oasis-open.org",
    "rfc-editor.org",
    "w3.org",
}
ACADEMIC_DOMAINS = {
    "acm.org",
    "arxiv.org",
    "doi.org",
    "ieee.org",
    "nature.com",
    "ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "science.org",
    "sciencedirect.com",
    "springer.com",
}
KNOWN_DOCUMENTATION_DOMAINS = {
    "developer.android.com",
    "developer.apple.com",
    "docs.docker.com",
    "docs.github.com",
    "docs.kernel.org",
    "docs.microsoft.com",
    "docs.python.org",
    "kubernetes.io",
    "learn.microsoft.com",
    "man7.org",
    "developer.mozilla.org",
    "support.apple.com",
    "support.mozilla.org",
}
TECHNICAL_REFERENCE_DOMAINS = {
    "archlinux.org",
    "askubuntu.com",
    "digitalocean.com",
    "serverfault.com",
    "stackoverflow.com",
    "superuser.com",
}
COMMUNITY_DOMAINS = {
    "discourse.org",
    "reddit.com",
    "stackexchange.com",
}
NEWS_DOMAINS = {
    "apnews.com",
    "arstechnica.com",
    "bbc.com",
    "reuters.com",
    "theguardian.com",
}
LOW_QUALITY_DOMAINS = {
    "facebook.com",
    "fandom.com",
    "instagram.com",
    "pinterest.com",
    "tiktok.com",
}
PRIMARY_SOURCE_TYPES = {
    "academic_primary",
    "documentation_candidate",
    "government",
    "release_notes",
    "source_repository",
    "standard",
}


def _domain_matches(domain: str, candidates: Iterable[str]) -> bool:
    return any(domain == item or domain.endswith("." + item) for item in candidates)


def root_domain(domain: str) -> str:
    parts = [part for part in domain.rstrip(".").split(".") if part]
    if len(parts) < 2:
        return domain
    common_second_level = {"ac", "co", "com", "edu", "gov", "net", "org"}
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in common_second_level:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def canonical_citation_url(value: object) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower().rstrip(".")
        if scheme not in {"http", "https"} or not host:
            return ""
        port = parsed.port
        authority = f"[{host}]" if ":" in host else host
        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            authority = f"{authority}:{port}"
        return urlunsplit((scheme, authority, parsed.path or "/", parsed.query, ""))
    except (TypeError, UnicodeError, ValueError):
        return ""


def stable_evidence_id(url: object) -> str:
    canonical = canonical_citation_url(url)
    if not canonical:
        return ""
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"ev-{digest}"


def normalize_date(value: object) -> str | None:
    """Normalize common page/search date values to UTC ISO 8601."""

    if value is None or isinstance(value, bool):
        return None
    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    elif isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            if number > 10_000_000_000:
                number /= 1000
            try:
                parsed = datetime.fromtimestamp(number, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                parsed = None
    else:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if not text or len(text) > 200:
            return None
        if re.fullmatch(r"\d{10,13}(?:\.\d+)?", text):
            return normalize_date(float(text))
        candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(text)
            except (TypeError, ValueError, OverflowError):
                match = re.search(r"\b(19\d{2}|20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
                if match:
                    try:
                        parsed = datetime(
                            int(match.group(1)),
                            int(match.group(2)),
                            int(match.group(3)),
                            tzinfo=timezone.utc,
                        )
                    except ValueError:
                        parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if parsed.hour == parsed.minute == parsed.second == parsed.microsecond == 0:
        return parsed.date().isoformat()
    return parsed.isoformat(timespec="seconds").replace("+00:00", "Z")


def temporal_requirement(query: str, time_range: str | None = None) -> str:
    if time_range or CURRENT_QUERY_RE.search(query or ""):
        return "high"
    if VERSION_SENSITIVE_RE.search(query or ""):
        return "moderate"
    return "none"


def freshness_score(value: object, requirement: str, *, now: datetime | None = None) -> float:
    normalized = normalize_date(value)
    if requirement == "none" or not normalized:
        return 0.0
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    age_days = (current - parsed.astimezone(timezone.utc)).total_seconds() / 86400
    if age_days < -2:
        return 0.0
    if age_days <= 7:
        score = 1.0
    elif age_days <= 30:
        score = 0.9
    elif age_days <= 90:
        score = 0.72
    elif age_days <= 365:
        score = 0.48
    elif age_days <= 1095:
        score = 0.2
    else:
        score = 0.05
    return round(score if requirement == "high" else score * 0.5, 4)


def extract_version_markers(*values: object, limit: int = 8) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    text = " ".join(str(value or "")[:20_000] for value in values)
    for match in NAMED_VERSION_RE.finditer(text):
        marker = f"{match.group(1)} {match.group(2)}"
        key = marker.casefold()
        if key not in seen:
            seen.add(key)
            seen.add(match.group(2).casefold())
            markers.append(marker)
    for match in VERSION_RE.finditer(text):
        marker = match.group(1)
        parts = marker.split(".")
        if len(parts) == 2 and parts[0].isdigit() and 1900 <= int(parts[0]) <= 2100:
            continue
        key = marker.casefold()
        if key not in seen:
            seen.add(key)
            markers.append(marker)
        if len(markers) >= limit:
            break
    return markers[:limit]


def source_profile(url: object, *, title: str = "", snippet: str = "", query: str = "") -> dict[str, Any]:
    """Classify a source using bounded heuristics, never a claim of ownership."""

    try:
        parsed = urlsplit(str(url or ""))
        domain = (parsed.hostname or "").lower().removeprefix("www.")
        path = parsed.path or "/"
    except ValueError:
        domain, path = "", "/"
    source_type = "general_web"
    tier = 4
    authority = 0.45
    primary_candidate = False
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]{3,}", (query or "").casefold())
        if term
        not in {
            "com",
            "documentation",
            "guide",
            "install",
            "official",
            "org",
            "the",
        }
    }
    candidate_terms = set(
        re.findall(
            r"[a-z0-9]{3,}",
            f"{root_domain(domain)} {path[:500]} {title[:500]}".casefold(),
        )
    )
    query_affinity = bool(query_terms & candidate_terms)

    if not domain:
        authority = 0.0
        tier = 5
    elif _domain_matches(domain, LOW_QUALITY_DOMAINS):
        source_type, tier, authority = "low_quality_or_social", 5, 0.08
    elif domain.endswith(".gov") or ".gov." in domain or domain == "europa.eu" or domain.endswith(".europa.eu"):
        source_type, tier, authority, primary_candidate = "government", 1, 0.96, True
    elif _domain_matches(domain, STANDARD_DOMAINS):
        source_type, tier, authority, primary_candidate = "standard", 1, 0.95, True
    elif _domain_matches(domain, ACADEMIC_DOMAINS) or domain.endswith(".edu") or ".edu." in domain:
        source_type, tier, authority, primary_candidate = "academic_primary", 1, 0.91, True
    elif domain == "github.com" or domain.endswith(".github.com"):
        if re.search(r"/(?:discussions|issues)(?:/|$)", path, re.I):
            source_type, tier, authority = "technical_community", 3, 0.66
        elif RELEASE_PATH_RE.search(path):
            source_type, tier, authority = "release_notes", 1, 0.9
        else:
            source_type, tier, authority = "source_repository", 2, 0.84
        primary_candidate = query_affinity and source_type != "technical_community"
    elif _domain_matches(domain, KNOWN_DOCUMENTATION_DOMAINS):
        source_type, tier, authority, primary_candidate = "documentation_candidate", 1, 0.93, True
    elif domain.startswith(("docs.", "developer.", "support.")) or DOCUMENTATION_PATH_RE.search(path):
        source_type, tier, authority = "documentation_candidate", 2, 0.81
        primary_candidate = query_affinity
    elif _domain_matches(domain, TECHNICAL_REFERENCE_DOMAINS):
        source_type, tier, authority = "technical_reference", 2, 0.78
    elif _domain_matches(domain, NEWS_DOMAINS):
        source_type, tier, authority = "news", 2, 0.79
    elif _domain_matches(domain, COMMUNITY_DOMAINS) or "forum" in domain:
        source_type, tier, authority = "technical_community", 3, 0.61

    if query_affinity and source_type in {"documentation_candidate", "release_notes", "source_repository"}:
        authority = min(1.0, authority + 0.04)

    ranking_adjustment = {1: 1.35, 2: 0.7, 3: 0.2, 4: 0.0, 5: -2.0}[tier]
    return {
        "source_type": source_type,
        "source_tier": tier,
        "authority_score": round(authority, 4),
        "authority_adjustment": ranking_adjustment,
        "primary_source_candidate": primary_candidate,
        "query_domain_affinity": query_affinity,
        "classification_method": "domain-path-query heuristic",
    }


def normalize_page_metadata(metadata: object, *, last_modified: object = None) -> dict[str, Any]:
    raw = metadata if isinstance(metadata, dict) else {}
    casefolded = {str(key).casefold(): value for key, value in raw.items()}

    def first(*keys: str) -> object:
        for key in keys:
            value = raw.get(key)
            if value in (None, "", [], {}):
                value = casefolded.get(key.casefold())
            if value not in (None, "", [], {}):
                return value
        return None

    published = normalize_date(
        first("publishedDate", "publishedTime", "datePublished", "article:published_time")
    )
    modified = normalize_date(
        first("modifiedDate", "modifiedTime", "dateModified", "article:modified_time", "lastModified")
        or last_modified
    )
    declared = normalize_date(first("declaredDate", "pageDate", "date", "dc.date"))
    language = re.sub(r"\s+", " ", str(first("language", "inLanguage", "og:locale") or "")).strip()[:64]
    author_value = first("author", "byline")
    if isinstance(author_value, dict):
        author_value = author_value.get("name")
    if isinstance(author_value, list):
        author_value = ", ".join(
            str(item.get("name") if isinstance(item, dict) else item) for item in author_value[:5]
        )
    author = re.sub(r"\s+", " ", str(author_value or "")).strip()[:500]
    description = re.sub(r"\s+", " ", str(first("description", "og:description") or "")).strip()[:1500]
    version_value = first("version", "softwareVersion", "software_version")
    versions = extract_version_markers(version_value, raw.get("title"), description)
    if version_value and not versions:
        cleaned = re.sub(r"\s+", " ", str(version_value)).strip()[:100]
        if cleaned:
            versions = [cleaned]
    normalized: dict[str, Any] = {}
    for key, value in (
        ("publishedDate", published),
        ("modifiedDate", modified),
        ("declaredDate", declared),
        ("language", language),
        ("author", author),
        ("description", description),
    ):
        if value:
            normalized[key] = value
    if versions:
        normalized["version_context"] = versions[:8]
    return normalized


def evidence_summary(results: list[dict[str, Any]], query: str, *, time_range: str | None = None) -> dict[str, Any]:
    owners: set[str] = set()
    for item in results:
        try:
            domain = (urlsplit(str(item.get("url") or "")).hostname or "").lower()
        except ValueError:
            continue
        owners.add(root_domain(domain))
    owners.discard("")
    primary_count = sum(bool(item.get("primary_source_candidate")) for item in results)
    dated_count = sum(
        bool(
            item.get("publishedDate")
            or item.get("modifiedDate")
            or item.get("declaredDate")
        )
        for item in results
    )
    extracted_count = sum(bool(item.get("extraction_method")) for item in results)
    citation_count = sum(bool(item.get("citation_url")) for item in results)
    versions: list[str] = []
    for item in results:
        for marker in item.get("version_context") or []:
            if marker not in versions:
                versions.append(marker)
    requirement = temporal_requirement(query, time_range)
    warnings: list[str] = []
    if len(owners) < min(2, len(results)) and len(results) > 1:
        warnings.append("Limited independent-source diversity.")
    if results and primary_count == 0:
        warnings.append("No likely primary or authoritative source was identified.")
    if requirement == "high" and dated_count == 0:
        warnings.append("The request is time-sensitive, but sources exposed no usable publication or modification dates.")
    if requirement != "none" and not versions and VERSION_SENSITIVE_RE.search(query or ""):
        warnings.append("No explicit software or product version context was extracted.")
    if extracted_count < len(results):
        warnings.append("Some results rely on search snippets because page extraction was unavailable.")
    return {
        "status": "sufficient" if results and (len(owners) >= 2 or primary_count > 0) else "limited",
        "result_count": len(results),
        "independent_source_count": len(owners),
        "primary_source_candidate_count": primary_count,
        "dated_source_count": dated_count,
        "extracted_source_count": extracted_count,
        "citation_url_count": citation_count,
        "temporal_requirement": requirement,
        "version_contexts": versions[:12],
        "warnings": warnings,
        "limitations": (
            "Coverage metadata is heuristic. It does not prove source ownership, independence, "
            "claim entailment, or factual agreement."
        ),
    }
