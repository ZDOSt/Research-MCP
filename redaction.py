import re
from typing import List, Tuple


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
            r'''(?i)(["'](?:password|passwd|secret|token|api[_-]?key|access[_-]?key|client[_-]?secret|(?:access|auth|refresh|id)[_-]?token|[a-z0-9.-]+[_-](?:password|passwd|secret|token|api[_-]?key|access[_-]?key)|x[_-]?amz[_-]?signature|x[_-]?goog[_-]?signature|signature|sig)["']\s*:\s*["'])(?!\$\{|<|example|changeme)([^"']{4,})(["'])'''
        ),
        r"\1[REDACTED]\3",
    ),
    (
        re.compile(
            r"(?i)([?&](?:access[_-]?token|auth[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|client[_-]?secret|[a-z0-9.-]+[_-](?:password|passwd|secret|token|api[_-]?key|access[_-]?key)|x[_-]?amz[_-]?signature|x[_-]?goog[_-]?signature|signature|sig|password|passwd|secret|token)=)(?!\$\{|%24%7B|<|example|changeme)([^\s&#\"'<>]{4,})"
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


def redact_sensitive_text(text: str) -> tuple[str, int]:
    output = text or ""
    count = 0
    for pattern, replacement in SECRET_PATTERNS:
        output, replacements = pattern.subn(replacement, output)
        count += replacements
    return output, count
