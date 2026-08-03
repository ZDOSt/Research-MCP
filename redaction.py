import ipaddress
import re
from typing import List, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


SECRET_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (
        re.compile(
            r"(?im)^(\s*(?:export\s+)?[A-Z0-9_.-]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|ACCESS_KEY)[A-Z0-9_.-]*\s*[=:]\s*)(?!\$\{|<|example|changeme)([^\s#]{6,})"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r'''(?i)(["'](?:password|passwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|(?:access|auth|refresh|id)[_-]?token|(?:j|php)?session[_-]?id|phpsessid|session(?:[_-]?(?:id|key|token))?|sid|[a-z0-9.-]+[_-](?:password|passwd|secret|token|api[_-]?key|access[_-]?key|session(?:[_-]?id)?)|x[_-]?amz[_-]?signature|x[_-]?goog[_-]?signature|signature|sig)["']\s*:\s*["'])(?!\$\{|<|example|changeme)([^"']{4,})(["'])'''
        ),
        r"\1[REDACTED]\3",
    ),
    (
        re.compile(
            r"(?i)([?&](?:access[_-]?token|auth[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|client[_-]?secret|(?:j|php)?session[_-]?id|phpsessid|session(?:[_-]?(?:id|key|token))?|sid|[a-z0-9.-]+[_-](?:password|passwd|secret|token|api[_-]?key|access[_-]?key|session(?:[_-]?id)?)|x[_-]?amz[_-]?signature|x[_-]?goog[_-]?signature|signature|sig|password|passwd|secret|token)=)(?!\$\{|%24%7B|<|example|changeme)([^\s&#\"'<>]{4,})"
        ),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r"(?im)^(\s*(?:x-api-key|api-key)\s*:\s*)(?!\$\{|<|example|changeme)([^\s,;]{6,})"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=]{8,}"), "Basic [REDACTED]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_ACCESS_KEY]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "[REDACTED_GOOGLE_API_KEY]"),
    (
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}|npm_[A-Za-z0-9]{20,}|pypi-[A-Za-z0-9_-]{20,})\b"
        ),
        "[REDACTED_PROVIDER_TOKEN]",
    ),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        "[REDACTED_JWT]",
    ),
    (
        re.compile(r"((?:https?|postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss|amqp|amqps)://)([^/@\s:]+):([^/@\s]+)@", re.I),
        r"\1[REDACTED]:[REDACTED]@",
    ),
]

SENSITIVE_URL_QUERY_KEYS = frozenset(
    {
        "access_key",
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "auth_token",
        "authorization",
        "client_secret",
        "code",
        "credential",
        "id_token",
        "key",
        "password",
        "passwd",
        "oauth_token",
        "refresh_token",
        "secret",
        "session",
        "session_id",
        "session_key",
        "session_token",
        "sessionid",
        "sig",
        "signature",
        "sid",
        "token",
        "jsessionid",
        "phpsessionid",
        "phpsessid",
        "x_amz_credential",
        "x_amz_security_token",
        "x_amz_signature",
        "x_goog_credential",
        "x_goog_signature",
    }
)

_SENSITIVE_URL_QUERY_KEYS_COMPACT = frozenset(
    re.sub(r"[^a-z0-9]", "", value.casefold())
    for value in SENSITIVE_URL_QUERY_KEYS
)


def _is_sensitive_url_query_key(value: str) -> bool:
    normalized = value.casefold().replace("-", "_")
    compact = re.sub(r"[^a-z0-9]", "", value.casefold())
    return (
        normalized in SENSITIVE_URL_QUERY_KEYS
        or compact in _SENSITIVE_URL_QUERY_KEYS_COMPACT
        or normalized.startswith("session_")
        or compact.startswith("session")
        or normalized.endswith(
            (
                "_token",
                "_secret",
                "_password",
                "_signature",
                "_credential",
                "_session",
                "_session_id",
                "_sessionid",
            )
        )
        or compact.endswith(
            ("token", "secret", "password", "signature", "credential", "sessionid")
        )
    )


_URL_PARAMETER_ASSIGNMENT_RE = re.compile(
    r"(?i)(?P<prefix>^|[;&])(?P<key>[A-Z0-9_.\[\]-]+)=(?P<value>[^;&]*)"
)

PUBLIC_QUERY_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"), "[REDACTED_ID]"),
    (
        re.compile(
            r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}"
            r"(?::\d{1,5})?(?:/[^\s<>\"']*)?(?![\d.])"
        ),
        "[REDACTED_ADDRESS]",
    ),
    (
        re.compile(
            r"(?i)\b\d{1,6}\s+(?:[A-Z0-9][A-Z0-9.'-]*\s+){0,5}"
            r"(?:avenue|ave|boulevard|blvd|court|ct|drive|dr|highway|hwy|"
            r"lane|ln|parkway|pkwy|place|pl|road|rd|street|st|terrace|ter|way)"
            r"(?:\s+(?:apartment|apt|suite|unit)\s*[A-Z0-9-]+)?\b"
        ),
        "[REDACTED_ADDRESS]",
    ),
    (
        re.compile(
            r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)|\d{2,4})[ .-]\d{3,4}[ .-]\d{4}(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(
            r"(?i)(?<![\w.-])(?:"
            r"localhost|"
            r"(?:[a-z0-9](?:[a-z0-9-]{0,62})\.)+"
            r"(?:internal|corp|home|lan|local|localdomain|localhost)|"
            r"(?:[a-z0-9](?:[a-z0-9-]{0,62})\.)*home\.arpa"
            r")(?:\.)?(?::\d{1,5})?"
            r"(?:/[^\s<>\"']*)?(?![\w.-])"
        ),
        "[REDACTED_PRIVATE_HOST]",
    ),
    (
        re.compile(
            r"(?i)\b(?:internal|private|confidential)\s+"
            r"(?:project|customer|client|tenant|host|server|system|repository|"
            r"repo|codebase)\s+"
            r"[^\W_][\w .'-]{1,80}?"
            r"(?=\s+(?:and|at|but|for|from|on|to|with)\b|[,;:!?]|\s*$)"
        ),
        "[REDACTED_PRIVATE_CONTEXT]",
    ),
    (
        re.compile(
            r"(?i)\b(?:codename|customer|client|tenant|account|ticket|incident|case)"
            r"\s+(?:(?:id|identifier|name|number)\s+)?"
            r"[^\W_][\w .'-]{1,80}?"
            r"(?=\s+(?:and|at|but|for|from|on|to|with)\b|[,;:!?]|\s*$)"
        ),
        "[REDACTED_PRIVATE_CONTEXT]",
    ),
    (
        re.compile(r"(?i)(?<![A-Z0-9])(?:[A-Z]:\\|/(?:home|root|srv|var|opt|etc|users)/)[^\s\"']+"),
        "[REDACTED_PATH]",
    ),
    (
        re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{8}(?:-[A-Fa-f0-9]{4}){3}-[A-Fa-f0-9]{12}(?![A-Fa-f0-9])"),
        "[REDACTED_ID]",
    ),
]

_HTTP_URL_IN_TEXT_RE = re.compile(r"https?://[^\s<>\]\[(){}\"']+", re.I)
_MODEL_INLINE_SECRET_RE = re.compile(
    r"(?i)(?<![A-Z0-9_.-])"
    r"((?:[A-Z0-9_.\[\]-]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API[_-]?KEY|ACCESS[_-]?KEY)"
    r"[A-Z0-9_.\[\]-]*|CLIENT[_-]?SECRET|SESSION(?:[_-]?(?:ID|KEY|TOKEN))?|SID)"
    r"\s*[=:]\s*)"
    r"(?!\$\{|<|example|changeme|\[REDACTED)([^\s,;#\"'<>]{6,})"
)
_BRACKETED_IPV6_RE = re.compile(
    r"(?<![\w:])\[(?P<address>[0-9A-Fa-f:.]+(?:%[A-Za-z0-9_.-]+)?)\]"
    r"(?::\d{1,5})?(?:/[^\s<>\"']*)?(?![\w:])"
)
_BARE_IPV6_RE = re.compile(
    r"(?<![\w:])(?P<address>"
    r"(?=[0-9A-Fa-f:]*:[0-9A-Fa-f:]*:)"
    r"[0-9A-Fa-f:]{2,}(?:%[A-Za-z0-9_.-]+)?"
    r")(?:/[^\s<>\"']*)?(?![\w:])"
)


def redact_sensitive_text(text: str) -> tuple[str, int]:
    output = text or ""
    count = 0
    for pattern, replacement in SECRET_PATTERNS:
        output, replacements = pattern.subn(replacement, output)
        count += replacements
    return output, count


def redact_public_query_text(text: str) -> tuple[str, int]:
    """Remove credentials and common private identifiers before public search."""
    output, count = redact_sensitive_text(text)
    for pattern, replacement in PUBLIC_QUERY_PATTERNS:
        output, replacements = pattern.subn(replacement, output)
        count += replacements

    def redact_ipv6(match: re.Match[str]) -> str:
        nonlocal count
        candidate = match.group("address").split("%", 1)[0]
        try:
            parsed = ipaddress.ip_address(candidate)
        except ValueError:
            return match.group(0)
        if parsed.version != 6:
            return match.group(0)
        count += 1
        return "[REDACTED_ADDRESS]"

    output = _BRACKETED_IPV6_RE.sub(redact_ipv6, output)
    output = _BARE_IPV6_RE.sub(redact_ipv6, output)
    return output, count


def redact_url_credentials(value: str) -> tuple[str, int]:
    """Return an HTTP(S) URL with credential-like query values removed."""
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw, 0
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return raw, 0
    path, path_redactions = re.subn(
        r"(?i)(;(?:asp\.net[_-]?)?"
        r"(?:j?sessionid|phpsessid|phpsessionid|"
        r"session(?:[_-]?(?:id|key|token))?|sid)=)"
        r"[^;/]+",
        r"\1[REDACTED]",
        parsed.path,
    )
    redactions = path_redactions

    def redact_parameter_assignment(match: re.Match[str]) -> str:
        nonlocal redactions
        item_value = match.group("value")
        if (
            _is_sensitive_url_query_key(match.group("key"))
            and item_value
            and item_value != "[REDACTED]"
        ):
            item_value = "[REDACTED]"
            redactions += 1
        return f"{match.group('prefix')}{match.group('key')}={item_value}"

    query = _URL_PARAMETER_ASSIGNMENT_RE.sub(
        redact_parameter_assignment,
        parsed.query,
    )
    items = []
    for key, item_value in parse_qsl(query, keep_blank_values=True):
        if _is_sensitive_url_query_key(key) and item_value != "[REDACTED]":
            item_value = "[REDACTED]"
            redactions += 1
        item_value = _URL_PARAMETER_ASSIGNMENT_RE.sub(
            redact_parameter_assignment,
            item_value,
        )
        items.append((key, item_value))
    return (
        urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc,
                path,
                urlencode(items, doseq=True),
                "",
            )
        ),
        redactions,
    )


def redact_model_input_text(text: str) -> tuple[str, int]:
    """Remove secrets and common private identifiers at the model boundary."""
    output = text or ""
    url_redactions = 0

    def redact_embedded_url(match: re.Match[str]) -> str:
        nonlocal url_redactions
        redacted, count = redact_url_credentials(match.group(0))
        url_redactions += count
        return redacted

    output = _HTTP_URL_IN_TEXT_RE.sub(redact_embedded_url, output)
    output, text_redactions = redact_public_query_text(output)
    output, inline_redactions = _MODEL_INLINE_SECRET_RE.subn(
        r"\1[REDACTED]",
        output,
    )
    return output, url_redactions + text_redactions + inline_redactions
