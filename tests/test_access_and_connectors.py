import json
import unittest
from unittest.mock import AsyncMock, patch

import api
from access_control import AccessPolicyError, authorize_claims, load_token_policies
from fastapi import HTTPException
from github_connector import (
    _safe_api_path,
    get_github_file,
    inspect_github_repository,
    normalize_repository,
    search_github,
)
from redaction import redact_sensitive_text


class AccessPolicyTests(unittest.TestCase):
    def test_multi_token_policy_enforces_scope_namespace_and_repository(self):
        policies = load_token_policies(
            {
                "MCP_AUTH_TOKENS_JSON": (
                    '{"token-a":{"client_id":"alice","scopes":["research","github:read"],'
                    '"namespaces":["alice-*"],"github_repositories":["owner/repo"]}}'
                )
            }
        )
        claims = policies["token-a"]
        self.assertTrue(authorize_claims(claims, namespace="alice-notes").allowed)
        self.assertFalse(authorize_claims(claims, namespace="bob-notes").allowed)
        self.assertTrue(
            authorize_claims(
                claims,
                scope="github:read",
                repository="OWNER/REPO",
            ).allowed
        )
        self.assertFalse(authorize_claims(claims, scope="memory:delete").allowed)
        self.assertFalse(
            authorize_claims(
                claims,
                scope="github:read",
                require_global_repository_access=True,
            ).allowed
        )

        global_claims = {**claims, "github_repositories": ["*"]}
        self.assertTrue(
            authorize_claims(
                global_claims,
                scope="github:read",
                require_global_repository_access=True,
            ).allowed
        )
        wildcard_claims = {**claims, "github_repositories": ["*/*", "?"]}
        self.assertFalse(
            authorize_claims(
                wildcard_claims,
                scope="github:read",
                require_global_repository_access=True,
            ).allowed
        )

    def test_credentialed_legacy_github_policy_is_fail_closed(self):
        policies = load_token_policies(
            {"MCP_AUTH_TOKEN": "token-a", "GITHUB_TOKEN": "github-secret"}
        )
        self.assertEqual(policies["token-a"]["github_repositories"], [])

    def test_invalid_policy_is_rejected(self):
        with self.assertRaises(AccessPolicyError):
            load_token_policies({"MCP_AUTH_TOKENS_JSON": "[]"})


class RedactionTests(unittest.TestCase):
    def test_redacts_json_basic_database_and_provider_credentials(self):
        text = "\n".join(
            [
                '"password": "correct-horse-battery-staple"',
                "Authorization: Basic dXNlcjpwYXNzd29yZA==",
                "DATABASE_URL=postgresql://admin:swordfish@db/app",
                "sk-abcdefghijklmnopqrstuvwxyz123456",
                "eyJabcdefghijk.abcdefghijkl.abcdefghijkl",
            ]
        )
        redacted, count = redact_sensitive_text(text)
        self.assertGreaterEqual(count, 5)
        for secret in [
            "correct-horse-battery-staple",
            "dXNlcjpwYXNzd29yZA==",
            "swordfish",
            "sk-abcdefghijklmnopqrstuvwxyz123456",
            "eyJabcdefghijk.abcdefghijkl.abcdefghijkl",
        ]:
            self.assertNotIn(secret, redacted)

    def test_redacts_query_header_and_google_api_credentials(self):
        secrets = [
            "super-secret-token-123456789",
            "abcdefghijklmnopqrstuvwxyz123456",
            "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567",
        ]
        text = "\n".join(
            [
                f"GET https://example.test/reset?access_token={secrets[0]}",
                f"X-API-Key: {secrets[1]}",
                f"Google key: {secrets[2]}",
            ]
        )

        redacted, count = redact_sensitive_text(text)

        self.assertEqual(count, 3)
        for secret in secrets:
            self.assertNotIn(secret, redacted)

    def test_redacts_prefixed_json_credentials_and_cloud_signatures(self):
        secrets = [
            "client-secret-abcdef",
            "refresh-token-abcdef",
            "auth-token-abcdef",
            "aws-signature-abcdef",
            "azure-signature-abcdef",
            "google-signature-abcdef",
            "session-token-abcdef",
            "aws-session-token-abcdef",
        ]
        text = "\n".join(
            [
                f'{{"client_secret":"{secrets[0]}"}}',
                f'{{"refresh_token":"{secrets[1]}"}}',
                f'{{"auth_token":"{secrets[2]}"}}',
                f'{{"session_token":"{secrets[6]}"}}',
                f"https://s3.example.test/object?X-Amz-Signature={secrets[3]}",
                f"https://s3.example.test/object?X-Amz-Security-Token={secrets[7]}",
                f"https://blob.example.test/object?sig={secrets[4]}",
                f"https://storage.example.test/object?X-Goog-Signature={secrets[5]}",
            ]
        )

        redacted, count = redact_sensitive_text(text)

        self.assertEqual(count, len(secrets))
        for secret in secrets:
            self.assertNotIn(secret, redacted)


class GitHubValidationTests(unittest.TestCase):
    def test_repository_and_api_paths_reject_dot_segments(self):
        for value in ["../rate_limit", "owner/..", "owner/repo/extra"]:
            with self.assertRaises(ValueError):
                normalize_repository(value)
        with self.assertRaises(ValueError):
            _safe_api_path("/repos/owner/repo/git/trees/%2e%2e")
        self.assertEqual(
            normalize_repository("https://github.com/Owner/repo.git"), "Owner/repo"
        )


class GitHubRedactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_scoped_search_rejects_caller_repository_qualifiers(self):
        with patch("github_connector._github_get") as github_get:
            with self.assertRaisesRegex(ValueError, "repo: qualifiers"):
                await search_github(
                    "error repo:other/private",
                    repository="allowed/repo",
                )
        github_get.assert_not_called()

    async def test_scoped_search_filters_unexpected_repository_results(self):
        payload = {
            "total_count": 2,
            "items": [
                {
                    "title": "allowed",
                    "repository_url": "https://api.github.com/repos/allowed/repo",
                },
                {
                    "title": "private",
                    "repository_url": "https://api.github.com/repos/other/private",
                },
            ],
        }
        with patch("github_connector._github_get", return_value=payload):
            result = await search_github("error", repository="allowed/repo")

        self.assertEqual(result["total_count"], 1)
        self.assertEqual([item["name"] for item in result["results"]], ["allowed"])
        self.assertEqual(result["results"][0]["repository"], "allowed/repo")

    async def test_search_redacts_before_truncating_preview(self):
        google_key = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
        query_secret = "search-result-secret-123456789"
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("A" * 1800)
            + "\n-----END PRIVATE KEY-----"
        )
        payload = {
            "total_count": 1,
            "items": [
                {
                    "title": f"Leaked key {google_key}",
                    "html_url": f"https://github.com/example/repo?access_token={query_secret}",
                    "body": private_key,
                }
            ],
        }

        with (
            patch("github_connector.GITHUB_REDACT_SECRETS", True),
            patch("github_connector._github_get", return_value=payload),
        ):
            result = await search_github("query", max_results=1)

        preview = result["results"][0]["text_match"]
        self.assertEqual(preview, "[REDACTED_PRIVATE_KEY]")
        self.assertNotIn("BEGIN PRIVATE KEY", preview)
        serialized = json.dumps(result)
        self.assertNotIn(google_key, serialized)
        self.assertNotIn(query_secret, serialized)

    async def test_inspect_and_file_results_redact_all_string_metadata(self):
        google_key = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
        header_secret = "abcdefghijklmnopqrstuvwxyz123456"
        query_secret = "metadata-secret-123456789"
        repository_payload = {
            "description": f"X-API-Key: {header_secret}",
            "default_branch": "main",
            "homepage": f"https://example.test/?access_token={query_secret}",
        }
        tree_payload = {
            "tree": [
                {
                    "type": "blob",
                    "path": f"docs/{google_key}.md",
                    "size": 12,
                }
            ]
        }
        file_payload = {
            "encoding": "base64",
            "content": "U2FmZSBjb250ZW50",
            "html_url": f"https://github.com/example/repo?access_token={query_secret}",
            "sha": "abc",
            "size": 12,
        }

        with (
            patch("github_connector.GITHUB_REDACT_SECRETS", True),
            patch(
                "github_connector._github_get",
                side_effect=[repository_payload, tree_payload, file_payload],
            ),
        ):
            inspected = await inspect_github_repository("example/repo")
            file_result = await get_github_file("example/repo", "README.md")

        serialized = json.dumps([inspected, file_result])
        for secret in [google_key, header_secret, query_secret]:
            self.assertNotIn(secret, serialized)
        self.assertGreaterEqual(file_result["redactions_applied"], 1)

    async def test_directory_results_redact_entry_strings(self):
        google_key = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ1234567"
        directory_payload = [
            {
                "name": google_key,
                "path": f"docs/{google_key}",
                "type": "file",
            }
        ]

        with (
            patch("github_connector.GITHUB_REDACT_SECRETS", True),
            patch(
                "github_connector._github_get",
                return_value=directory_payload,
            ),
        ):
            result = await get_github_file("example/repo", "docs")

        self.assertNotIn(google_key, json.dumps(result))


class APIAuthorizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_management_api_fails_closed_without_token(self):
        with (
            patch.object(api, "RESEARCH_API_TOKEN", ""),
            patch.object(api, "RESEARCH_API_ALLOW_UNAUTHENTICATED", False),
        ):
            with self.assertRaises(HTTPException) as raised:
                await api.require_api_authorization("")
        self.assertEqual(raised.exception.status_code, 503)

    async def test_management_api_uses_constant_time_bearer_check(self):
        with patch.object(api, "RESEARCH_API_TOKEN", "expected"):
            await api.require_api_authorization("Bearer expected")
            with self.assertRaises(HTTPException) as raised:
                await api.require_api_authorization("Bearer wrong")
        self.assertEqual(raised.exception.status_code, 401)

    async def test_attempt_invalidation_route_delegates_after_authentication(self):
        body = api.IngestionAttemptInvalidationRequest(
            ingestion_attempt_id="opaque-attempt-id",
            reason="worker_lease_lost",
        )
        with patch.object(
            api,
            "invalidate_ingestion_attempt_async",
            AsyncMock(return_value={"invalidated": 2}),
        ) as invalidate:
            result = await api.invalidate_ingestion_attempt_route(body)

        self.assertEqual(result, {"invalidated": 2})
        invalidate.assert_awaited_once_with(
            "opaque-attempt-id",
            reason="worker_lease_lost",
        )


if __name__ == "__main__":
    unittest.main()
