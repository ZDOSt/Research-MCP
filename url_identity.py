"""Dependency-free canonical identities for web sources."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit


TRACKING_QUERY_KEYS = frozenset(
    {
        "_hsenc",
        "_hsmi",
        "dclid",
        "fbclid",
        "gclid",
        "igshid",
        "mc_cid",
        "mc_eid",
        "msclkid",
        "oly_anon_id",
        "oly_enc_id",
        "ref_src",
        "s_cid",
        "spm",
        "vero_conv",
        "vero_id",
    }
)

_PERCENT_ESCAPE_RE = re.compile(r"%([0-9a-fA-F]{2})")
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)


def _normalize_percent_escape(match: re.Match[str]) -> str:
    value = int(match.group(1), 16)
    character = chr(value)
    return character if character in _UNRESERVED else f"%{value:02X}"


def _normalize_path(path: str) -> str:
    """Normalize only RFC dot segments and unreserved percent escapes."""
    path = _PERCENT_ESCAPE_RE.sub(_normalize_percent_escape, path or "/")
    path = quote(path, safe="/%:@!$&'()*+,;=-._~")

    normalized: list[str] = []
    for segment in path.split("/"):
        if segment == ".":
            continue
        if segment == "..":
            if normalized and normalized[-1] not in {"", ".."}:
                normalized.pop()
            continue
        normalized.append(segment)

    result = "/".join(normalized)
    if not result:
        result = "/"
    return result


def _normalize_host(hostname: str) -> str:
    host = hostname.rstrip(".").lower()
    if not host:
        raise ValueError("URL host is empty")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        host = host.encode("idna").decode("ascii")
        if not host or len(host) > 253:
            raise ValueError("URL host is invalid")
        return host

    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def canonicalize_web_url(value: object) -> str:
    """Return a stable HTTP(S) URL identity, or an empty string if invalid."""
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""

    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        if parsed.username is not None or parsed.password is not None:
            return ""

        host = _normalize_host(parsed.hostname)
        port = parsed.port
        default_port = (scheme == "http" and port == 80) or (
            scheme == "https" and port == 443
        )
        netloc = host if port is None or default_port else f"{host}:{port}"

        query_items = []
        for key, item_value in parse_qsl(parsed.query, keep_blank_values=True):
            lowered = key.casefold()
            if lowered.startswith("utm_") or lowered in TRACKING_QUERY_KEYS:
                continue
            query_items.append((key, item_value))
        query_items.sort(key=lambda item: (item[0].casefold(), item[0], item[1]))

        return urlunsplit(
            (
                scheme,
                netloc,
                _normalize_path(parsed.path),
                urlencode(query_items, doseq=True),
                "",
            )
        )
    except (UnicodeError, ValueError):
        return ""


def canonicalize_source_identity(value: object) -> str:
    """Canonicalize web URLs while preserving non-URL source identifiers."""
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""

    canonical = canonicalize_web_url(raw)
    if canonical:
        return canonical

    # Inputs that claim to be URLs fail closed rather than persisting malformed
    # or credential-bearing variants as separate source identities.
    if "://" in raw:
        return ""
    return raw
