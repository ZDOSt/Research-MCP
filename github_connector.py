import asyncio
import base64
import binascii
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlsplit

import httpx

from redaction import redact_sensitive_text


GITHUB_API_URL = os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_API_VERSION = os.getenv("GITHUB_API_VERSION", "2022-11-28")
GITHUB_MAX_FILE_CHARS = int(os.getenv("GITHUB_MAX_FILE_CHARS", "250000"))
GITHUB_MAX_RESPONSE_BYTES = max(
    65536,
    int(os.getenv("GITHUB_MAX_RESPONSE_BYTES", "8388608")),
)
GITHUB_TIMEOUT_SECONDS = max(1.0, float(os.getenv("GITHUB_TIMEOUT_SECONDS", "45")))
GITHUB_ALLOW_INSECURE_HTTP = os.getenv("GITHUB_ALLOW_INSECURE_HTTP", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GITHUB_REDACT_SECRETS = os.getenv("GITHUB_REDACT_SECRETS", "true").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
REPOSITORY_QUALIFIER_RE = re.compile(r"\brepo\s*:", re.I)


def _redact_external_value(value: Any) -> tuple[Any, int]:
    if not GITHUB_REDACT_SECRETS:
        return value, 0
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        count = 0
        for key, item in value.items():
            redacted_item, item_count = _redact_external_value(item)
            redacted[key] = redacted_item
            count += item_count
        return redacted, count
    if isinstance(value, list):
        redacted_items = []
        count = 0
        for item in value:
            redacted_item, item_count = _redact_external_value(item)
            redacted_items.append(redacted_item)
            count += item_count
        return redacted_items, count
    if isinstance(value, tuple):
        redacted_items = []
        count = 0
        for item in value:
            redacted_item, item_count = _redact_external_value(item)
            redacted_items.append(redacted_item)
            count += item_count
        return tuple(redacted_items), count
    return value, 0


def normalize_repository(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    value = re.sub(r"^https?://(?:www\.)?github\.com/", "", value, flags=re.I)
    value = value.removesuffix(".git")
    parts = value.split("/")
    if len(parts) != 2 or any(part in {".", ".."} for part in parts):
        raise ValueError("repository must be an owner/name pair or a github.com repository URL")
    repository = "/".join(parts)
    if not REPOSITORY_RE.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair or a github.com repository URL")
    return repository


def _search_result_repository(item: dict[str, Any]) -> Optional[str]:
    repository = item.get("repository") or {}
    if isinstance(repository, dict) and repository.get("full_name"):
        return str(repository["full_name"])
    if item.get("full_name"):
        return str(item["full_name"])
    repository_url = str(item.get("repository_url") or "")
    marker = "/repos/"
    if marker in repository_url:
        candidate = repository_url.split(marker, 1)[1].strip("/")
        try:
            return normalize_repository(candidate)
        except ValueError:
            return None
    return None


def _validated_api_base() -> str:
    parsed = urlsplit(GITHUB_API_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("GITHUB_API_URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("GITHUB_API_URL must not contain credentials, query, or fragment")
    if parsed.scheme != "https" and not GITHUB_ALLOW_INSECURE_HTTP:
        raise ValueError("GITHUB_API_URL must use HTTPS unless GITHUB_ALLOW_INSECURE_HTTP=true")
    return GITHUB_API_URL


def _safe_api_path(path: str) -> str:
    if not path.startswith("/") or "\\" in path or any(ord(char) < 32 for char in path):
        raise ValueError("invalid GitHub API path")
    decoded_parts = [unquote(part) for part in path.split("/")]
    if any(part in {".", ".."} for part in decoded_parts):
        raise ValueError("GitHub API path may not contain dot segments")
    return path


def _path_segment(value: str, field: str, *, max_length: int = 255) -> str:
    normalized = str(value or "").strip()
    if (
        not normalized
        or len(normalized) > max_length
        or any(ord(char) < 32 for char in normalized)
        or normalized in {".", ".."}
    ):
        raise ValueError(f"{field} is invalid")
    return quote(normalized, safe="")


def _headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "research-mcp",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


async def _github_get(path: str, params: Optional[dict] = None) -> Any:
    url = f"{_validated_api_base()}{_safe_api_path(path)}"
    async with asyncio.timeout(GITHUB_TIMEOUT_SECONDS):
        async with httpx.AsyncClient(
            timeout=GITHUB_TIMEOUT_SECONDS,
            headers=_headers(),
            follow_redirects=False,
        ) as client:
            async with client.stream("GET", url, params=params) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length and int(length) > GITHUB_MAX_RESPONSE_BYTES:
                    raise ValueError("GitHub response exceeded GITHUB_MAX_RESPONSE_BYTES")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > GITHUB_MAX_RESPONSE_BYTES:
                        raise ValueError("GitHub response exceeded GITHUB_MAX_RESPONSE_BYTES")
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("GitHub returned invalid JSON") from exc


async def search_github(
    query: str,
    kind: str = "issues",
    repository: Optional[str] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    kind = kind.strip().lower()
    if kind not in {"issues", "code", "repositories"}:
        raise ValueError("kind must be issues, code, or repositories")
    max_results = max(1, min(max_results, 30))

    search_query = query.strip()
    normalized_repository = None
    if repository:
        normalized_repository = normalize_repository(repository)
        # GitHub unions repeated repo: qualifiers. Without rejecting a caller's
        # qualifier, appending the authorized repository can broaden a scoped
        # search to another private repository accessible to the server token.
        if REPOSITORY_QUALIFIER_RE.search(search_query):
            raise ValueError("query must not contain repo: qualifiers when repository is set")
        search_query = f"{search_query} repo:{normalized_repository}"

    payload = await _github_get(
        f"/search/{kind}",
        params={"q": search_query, "per_page": max_results},
    )

    results: List[dict] = []
    for item in payload.get("items", [])[:max_results]:
        item_repository = _search_result_repository(item)
        if normalized_repository and (
            not item_repository
            or item_repository.casefold() != normalized_repository.casefold()
        ):
            continue
        text_match = str(item.get("body") or item.get("description") or "")
        if GITHUB_REDACT_SECRETS:
            text_match = redact_sensitive_text(text_match)[0]
        results.append(
            {
                "name": item.get("name") or item.get("title") or item.get("full_name"),
                "url": item.get("html_url"),
                "api_url": item.get("url"),
                "repository": item_repository or normalized_repository,
                "path": item.get("path"),
                "state": item.get("state"),
                "updated_at": item.get("updated_at"),
                "score": item.get("score"),
                "text_match": text_match[:1600],
            }
        )

    response = {
        "query": query,
        "kind": kind,
        "repository": normalized_repository,
        # A scoped response must not expose a count from any result the API may
        # have returned outside the post-filtered authorization boundary.
        "total_count": len(results) if normalized_repository else payload.get("total_count", len(results)),
        "results": results,
        "authentication": "token" if GITHUB_TOKEN else "anonymous_rate_limited",
    }
    return _redact_external_value(response)[0]


async def inspect_github_repository(
    repository: str,
    ref: Optional[str] = None,
    max_files: int = 200,
) -> Dict[str, Any]:
    repository = normalize_repository(repository)
    max_files = max(1, min(max_files, 1000))
    owner, name = repository.split("/", 1)
    repository_path = f"{_path_segment(owner, 'owner')}/{_path_segment(name, 'repository')}"
    repo = await _github_get(f"/repos/{repository_path}")
    selected_ref = ref or repo.get("default_branch") or "HEAD"
    encoded_ref = _path_segment(selected_ref, "ref", max_length=1024)
    tree = await _github_get(
        f"/repos/{repository_path}/git/trees/{encoded_ref}",
        params={"recursive": "1"},
    )

    files = []
    priority_names = {
        "readme.md",
        "package.json",
        "pyproject.toml",
        "dockerfile",
        "docker-compose.yml",
        "compose.yml",
        "manifest.json",
        "extension.json",
        "requirements.txt",
    }
    for item in tree.get("tree", []):
        if item.get("type") != "blob":
            continue
        path = str(item.get("path") or "")
        lower = path.lower()
        priority = lower.rsplit("/", 1)[-1] in priority_names or lower.startswith(
            ("docs/", ".github/", "src/", "examples/")
        )
        files.append(
            {
                "path": path,
                "size": item.get("size"),
                "url": f"https://github.com/{repository}/blob/{selected_ref}/{path}",
                "priority": priority,
            }
        )

    files.sort(key=lambda item: (not item["priority"], item["path"].lower()))
    response = {
        "repository": repository,
        "description": repo.get("description"),
        "default_branch": repo.get("default_branch"),
        "ref": selected_ref,
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "license": (repo.get("license") or {}).get("spdx_id"),
        "homepage": repo.get("homepage"),
        "tree_truncated": bool(tree.get("truncated")),
        "files": files[:max_files],
        "file_count_returned": min(len(files), max_files),
        "file_count_seen": len(files),
    }
    return _redact_external_value(response)[0]


async def get_github_file(
    repository: str,
    path: str,
    ref: Optional[str] = None,
    max_chars: int = GITHUB_MAX_FILE_CHARS,
) -> Dict[str, Any]:
    repository = normalize_repository(repository)
    path = (path or "").strip().lstrip("/")
    path_parts = path.split("/")
    if (
        not path
        or len(path) > 4096
        or any(part in {"", ".", ".."} for part in path_parts)
        or "\\" in path
        or any(ord(char) < 32 for char in path)
    ):
        raise ValueError("path must identify a repository file")
    max_chars = max(1000, min(max_chars, GITHUB_MAX_FILE_CHARS))

    if ref is not None:
        ref = str(ref).strip()
        if not ref or len(ref) > 1024 or any(ord(char) < 32 for char in ref):
            raise ValueError("ref is invalid")
    params = {"ref": ref} if ref else None
    owner, name = repository.split("/", 1)
    repository_path = f"{_path_segment(owner, 'owner')}/{_path_segment(name, 'repository')}"
    encoded_path = "/".join(_path_segment(part, "path component", max_length=1024) for part in path_parts)
    payload = await _github_get(f"/repos/{repository_path}/contents/{encoded_path}", params=params)
    if isinstance(payload, list):
        response = {
            "repository": repository,
            "path": path,
            "ref": ref,
            "type": "directory",
            "entries": [
                {"name": item.get("name"), "path": item.get("path"), "type": item.get("type")}
                for item in payload[:500]
            ],
        }
        return _redact_external_value(response)[0]

    encoded = payload.get("content") or ""
    if payload.get("encoding") != "base64":
        raise ValueError("GitHub did not return base64 file content")
    try:
        compact_encoded = re.sub(r"\s+", "", str(encoded))
        content = base64.b64decode(compact_encoded, validate=True).decode(
            "utf-8", errors="replace"
        )
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError("GitHub returned invalid base64 file content") from exc
    redaction_count = 0
    if GITHUB_REDACT_SECRETS:
        content, redaction_count = redact_sensitive_text(content)
    response = {
        "repository": repository,
        "path": path,
        "ref": ref,
        "type": "file",
        "sha": payload.get("sha"),
        "size": payload.get("size"),
        "url": payload.get("html_url"),
        "content": content[:max_chars],
        "truncated": len(content) > max_chars,
    }
    response, metadata_redaction_count = _redact_external_value(response)
    response["redactions_applied"] = redaction_count + metadata_redaction_count
    return response
