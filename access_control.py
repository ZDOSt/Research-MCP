"""Static MCP token policy parsing and authorization helpers."""

from __future__ import annotations

import fnmatch
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional


_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")


class AccessPolicyError(ValueError):
    """Raised when an access-control configuration is invalid."""


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: Optional[str] = None


def _string_list(value: Any, field: str, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        values = [str(item).strip() for item in value]
    else:
        raise AccessPolicyError(f"{field} must be a list or comma-separated string")
    return list(dict.fromkeys(item for item in values if item))


def _legacy_token_policy(token: str, environ: Mapping[str, str]) -> dict[str, Any]:
    default_github_repositories = "" if environ.get("GITHUB_TOKEN", "").strip() else "*"
    return {
        "client_id": environ.get("MCP_AUTH_CLIENT_ID", "research-mcp-client"),
        "scopes": [
            "research",
            "memory:write",
            "memory:delete",
            "artifacts:read",
            "github:read",
        ],
        "namespaces": _string_list(
            environ.get("MCP_ALLOWED_NAMESPACES", "*"),
            "MCP_ALLOWED_NAMESPACES",
            default=["*"],
        ),
        "github_repositories": _string_list(
            environ.get("GITHUB_ALLOWED_REPOSITORIES", default_github_repositories),
            "GITHUB_ALLOWED_REPOSITORIES",
            default=[],
        ),
    }


def load_token_policies(environ: Optional[Mapping[str, str]] = None) -> dict[str, dict[str, Any]]:
    """Load legacy or multi-token static policies from environment variables."""
    env = os.environ if environ is None else environ
    raw = str(env.get("MCP_AUTH_TOKENS_JSON", "") or "").strip()
    policies: dict[str, Any]
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AccessPolicyError(f"MCP_AUTH_TOKENS_JSON is invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise AccessPolicyError("MCP_AUTH_TOKENS_JSON must be an object keyed by token")
        policies = parsed
    else:
        legacy_token = str(env.get("MCP_AUTH_TOKEN", "") or "").strip()
        policies = {legacy_token: _legacy_token_policy(legacy_token, env)} if legacy_token else {}

    normalized: dict[str, dict[str, Any]] = {}
    for token, raw_policy in policies.items():
        token_value = str(token or "").strip()
        if not token_value or len(token_value) > 4096:
            raise AccessPolicyError("static tokens must be non-empty and at most 4096 characters")
        if not isinstance(raw_policy, dict):
            raise AccessPolicyError("each static token policy must be an object")
        client_id = str(raw_policy.get("client_id") or "").strip()
        if not _CLIENT_ID_RE.fullmatch(client_id):
            raise AccessPolicyError("token client_id contains unsupported characters")
        scopes = _string_list(raw_policy.get("scopes"), "scopes", default=["research"])
        if "research" not in scopes:
            raise AccessPolicyError("every MCP token requires the research scope")
        normalized[token_value] = {
            "client_id": client_id,
            "scopes": scopes,
            "namespaces": _string_list(raw_policy.get("namespaces"), "namespaces", default=[]),
            "github_repositories": _string_list(
                raw_policy.get("github_repositories"),
                "github_repositories",
                default=[],
            ),
        }
    return normalized


def _matches(value: str, patterns: list[str], *, case_sensitive: bool = True) -> bool:
    candidate = value if case_sensitive else value.lower()
    for pattern in patterns:
        match_pattern = pattern if case_sensitive else pattern.lower()
        if match_pattern == "*" or fnmatch.fnmatchcase(candidate, match_pattern):
            return True
    return False


def authorize_claims(
    claims: Mapping[str, Any],
    *,
    scope: str = "research",
    namespace: Optional[str] = None,
    repository: Optional[str] = None,
    require_global_repository_access: bool = False,
) -> AuthorizationDecision:
    scopes = _string_list(claims.get("scopes"), "scopes", default=[])
    if scope not in scopes:
        return AuthorizationDecision(False, f"missing required scope: {scope}")

    if namespace is not None:
        namespaces = _string_list(claims.get("namespaces"), "namespaces", default=[])
        if not _matches(namespace, namespaces):
            return AuthorizationDecision(False, "namespace is not allowed for this client")

    if repository is not None or require_global_repository_access:
        repositories = _string_list(
            claims.get("github_repositories"),
            "github_repositories",
            default=[],
        )
        if require_global_repository_access and "*" not in repositories:
            return AuthorizationDecision(
                False,
                "global repository search is not allowed for this client",
            )
        if repository is not None and not _matches(
            repository,
            repositories,
            case_sensitive=False,
        ):
            return AuthorizationDecision(False, "repository is not allowed for this client")

    return AuthorizationDecision(True)
