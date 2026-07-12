import asyncio
import os
import socket
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_original_shared = sys.modules.get("shared")
if _original_shared is None:
    shared_stub = types.ModuleType("shared")
    shared_stub.CRAWL4AI_URL = "http://crawl4ai:11235"
    sys.modules["shared"] = shared_stub

from crawler import (  # noqa: E402
    DIRECT_MAX_RESPONSE_BYTES,
    UnsafeURLError,
    crawl4ai_request,
    direct_fetch_url,
    extract_direct_response,
    validate_url_safety,
)
from browser import (  # noqa: E402
    NETWORK_BODY_LIMIT,
    bounded_network_response_length,
    build_click_script,
    build_scrollable_capture_script,
    decode_bounded_cdp_body,
    safe_diagnostic_url,
)
import browser  # noqa: E402

if _original_shared is None:
    sys.modules.pop("shared", None)


PUBLIC_DNS_RECORDS = [
    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
]


class FakeNetworkStream:
    def __init__(self, peer_address="93.184.216.34"):
        self.peer_address = peer_address

    def get_extra_info(self, name):
        if name == "server_addr":
            return (self.peer_address, 443)
        return None


class FakeResponse:
    def __init__(
        self,
        url,
        status_code=200,
        headers=None,
        content=b"",
        peer_address="93.184.216.34",
        chunk_delay=0,
    ):
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.encoding = "utf-8"
        self.extensions = {"network_stream": FakeNetworkStream(peer_address)}
        self.chunk_delay = chunk_delay
        self.body_read = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        self.body_read = True
        if self.chunk_delay:
            await asyncio.sleep(self.chunk_delay)
        midpoint = max(1, len(self.content) // 2)
        for start in range(0, len(self.content), midpoint):
            yield self.content[start : start + midpoint]


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeAsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requested_urls = []
        self.requests = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method, url, **kwargs):
        self.requested_urls.append(url)
        self.requests.append({"method": method, "url": url, **kwargs})
        return FakeStreamContext(self.responses.pop(0))


class URLSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_non_http_credentials_and_local_targets(self):
        blocked = [
            "file:///etc/passwd",
            "ftp://example.com/file",
            "http://user:secret@example.com/",
            "http://localhost/",
            "http://service.internal/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
        ]

        for url in blocked:
            with self.subTest(url=url), self.assertRaises(UnsafeURLError):
                await validate_url_safety(url)

    async def test_accepts_public_literal_addresses(self):
        self.assertEqual(
            await validate_url_safety("https://93.184.216.34/resource"),
            "https://93.184.216.34/resource",
        )
        self.assertEqual(
            await validate_url_safety("https://[2001:4860:4860::8888]/"),
            "https://[2001:4860:4860::8888]/",
        )

    async def test_rejects_hostname_if_any_dns_answer_is_private(self):
        records = PUBLIC_DNS_RECORDS + [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.10.10.10", 443)),
        ]
        with patch("socket.getaddrinfo", return_value=records):
            with self.assertRaises(UnsafeURLError):
                await validate_url_safety("https://mixed.example/path")

    async def test_checked_redirect_blocks_private_target_before_request(self):
        client = FakeAsyncClient(
            [
                FakeResponse(
                    "https://public.example/start",
                    status_code=302,
                    headers={"location": "http://127.0.0.1/admin"},
                )
            ]
        )

        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.httpx.AsyncClient", return_value=client):
                with self.assertRaises(UnsafeURLError):
                    await direct_fetch_url("https://public.example/start")

        self.assertEqual(client.requested_urls, ["https://93.184.216.34/start"])

    async def test_checked_public_redirect_is_followed(self):
        client = FakeAsyncClient(
            [
                FakeResponse(
                    "https://public.example/start",
                    status_code=302,
                    headers={"location": "https://cdn.example/final"},
                ),
                FakeResponse(
                    "https://cdn.example/final",
                    headers={"content-type": "text/html; charset=utf-8"},
                    content=b"<html><title>Safe</title><body>Public content</body></html>",
                ),
            ]
        )

        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.httpx.AsyncClient", return_value=client):
                result = await direct_fetch_url("https://public.example/start")

        self.assertEqual(result["url"], "https://cdn.example/final")
        self.assertEqual(result["redirect_chain"], ["https://cdn.example/final"])
        self.assertIn("Public content", result["content"])
        self.assertEqual(client.requests[0]["headers"]["Host"], "public.example")
        self.assertEqual(client.requests[0]["extensions"]["sni_hostname"], "public.example")

    async def test_peer_mismatch_fails_before_response_body_is_read(self):
        response = FakeResponse(
            "https://public.example/start",
            headers={"content-type": "text/plain"},
            content=b"private data",
            peer_address="10.0.0.8",
        )
        client = FakeAsyncClient([response])

        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.httpx.AsyncClient", return_value=client):
                with self.assertRaises(UnsafeURLError):
                    await direct_fetch_url("https://public.example/start")

        self.assertFalse(response.body_read)

    async def test_direct_fetch_rejects_declared_and_streamed_oversize_bodies(self):
        declared = FakeResponse(
            "https://public.example/large",
            headers={"content-type": "text/plain", "content-length": str(DIRECT_MAX_RESPONSE_BYTES + 1)},
            content=b"ignored",
        )
        streamed = FakeResponse(
            "https://public.example/chunked",
            headers={"content-type": "text/plain"},
            content=b"123456",
        )

        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.httpx.AsyncClient", return_value=FakeAsyncClient([declared])):
                with self.assertRaisesRegex(RuntimeError, "byte limit"):
                    await direct_fetch_url("https://public.example/large")
            with patch("crawler.DIRECT_MAX_RESPONSE_BYTES", 5):
                with patch("crawler.httpx.AsyncClient", return_value=FakeAsyncClient([streamed])):
                    with self.assertRaisesRegex(RuntimeError, "byte limit"):
                        await direct_fetch_url("https://public.example/chunked")

    async def test_direct_fetch_has_total_deadline(self):
        response = FakeResponse(
            "https://public.example/slow",
            headers={"content-type": "text/plain"},
            content=b"slow",
            chunk_delay=0.05,
        )
        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.DIRECT_TOTAL_TIMEOUT_SECONDS", 0.01):
                with patch("crawler.httpx.AsyncClient", return_value=FakeAsyncClient([response])):
                    with self.assertRaises(TimeoutError):
                        await direct_fetch_url("https://public.example/slow")

    async def test_crawl4ai_requires_explicit_network_isolation(self):
        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.CRAWL4AI_NETWORK_ISOLATED", False):
                with self.assertRaisesRegex(RuntimeError, "egress-only network"):
                    await crawl4ai_request({"urls": ["https://public.example/"]})

    async def test_crawl4ai_response_is_stream_bounded(self):
        response = FakeResponse(
            "http://crawl4ai:11235/crawl",
            headers={"content-type": "application/json"},
            content=b'{"too":"large"}',
        )
        with patch("socket.getaddrinfo", return_value=PUBLIC_DNS_RECORDS):
            with patch("crawler.CRAWL4AI_NETWORK_ISOLATED", True):
                with patch("crawler.CRAWL4AI_MAX_RESPONSE_BYTES", 5):
                    with patch("crawler.httpx.AsyncClient", return_value=FakeAsyncClient([response])):
                        with self.assertRaisesRegex(RuntimeError, "byte limit"):
                            await crawl4ai_request({"urls": ["https://public.example/"]})

    async def test_browser_wrapper_has_total_deadline(self):
        async def slow_exploration(**kwargs):
            await asyncio.sleep(0.05)
            return {}

        with patch("browser.BROWSER_TOTAL_TIMEOUT_SECONDS", 0.01), patch.dict(
            os.environ,
            {
                "RESEARCH_REQUIRE_WEB_ISOLATION": "true",
                "RESEARCH_WEB_NETWORK_ISOLATED": "true",
                "RESEARCH_BROWSER_PROXY": "socks5://safe-egress:1080",
            },
            clear=False,
        ):
            with patch("browser._playwright_explore_page_inner", slow_exploration):
                with self.assertRaisesRegex(RuntimeError, "Browser exploration exceeded"):
                    await browser.playwright_explore_page("https://public.example/")

    async def test_browser_wrapper_rejects_unisolated_local_execution(self):
        with patch.dict(
            os.environ,
            {
                "WEB_RUNNER_SOCKET": "",
                "RESEARCH_REQUIRE_WEB_ISOLATION": "false",
                "RESEARCH_WEB_NETWORK_ISOLATED": "false",
                "RESEARCH_BROWSER_PROXY": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "isolated web-runner"):
                await browser.playwright_explore_page("https://public.example/")


class ContentDispatchTests(unittest.TestCase):
    def test_json_html_and_plain_text_are_dispatched_by_content_type(self):
        json_text, _, json_format, json_error = extract_direct_response(
            b'{"name":"widget","count":2}', "application/json"
        )
        html_text, html_title, html_format, html_error = extract_direct_response(
            b"<html><title>Widget</title><body><p>Details</p></body></html>", "text/html"
        )
        plain_text, _, plain_format, plain_error = extract_direct_response(
            b"plain response", "text/plain; charset=utf-8"
        )

        self.assertIn("name: widget", json_text)
        self.assertEqual((json_format, json_error), ("json", None))
        self.assertIn("Details", html_text)
        self.assertEqual((html_title, html_format, html_error), ("Widget", "html", None))
        self.assertEqual((plain_text, plain_format, plain_error), ("plain response", "text", None))

    def test_browser_source_uses_secure_defaults_and_url_guard(self):
        source = (Path(__file__).resolve().parents[1] / "browser.py").read_text(encoding="utf-8")
        self.assertNotIn("--disable-web-security", source)
        self.assertNotIn("ignore_https_errors=True", source)
        self.assertIn('"chromium_sandbox": not env_flag("RESEARCH_BROWSER_DISABLE_SANDBOX")', source)
        self.assertIn('service_workers="block"', source)
        self.assertIn("await validate_url_safety(url)", source)
        self.assertIn('await context.route("**/*", guard_outbound_request)', source)
        self.assertIn('await context.route_web_socket("**/*", block_websocket)', source)
        self.assertNotIn("validated_resource_origins", source)

    def test_browser_capture_requires_small_uncompressed_declared_body(self):
        self.assertEqual(
            bounded_network_response_length({"content-length": "512"}),
            512,
        )
        self.assertIsNone(bounded_network_response_length({}))
        self.assertIsNone(
            bounded_network_response_length(
                {"content-length": "512", "content-encoding": "gzip"}
            )
        )
        self.assertIsNone(
            bounded_network_response_length(
                {"content-length": str(NETWORK_BODY_LIMIT + 1)}
            )
        )

    def test_browser_capture_checks_transferred_and_decoded_body_size(self):
        self.assertEqual(
            decode_bounded_cdp_body(
                {"body": "hello", "base64Encoded": False},
                declared_length=5,
                encoded_data_length=5,
            ),
            "hello",
        )
        self.assertIsNone(
            decode_bounded_cdp_body(
                {"body": "oversized", "base64Encoded": False},
                declared_length=1,
                encoded_data_length=9,
            )
        )
        self.assertIsNone(
            decode_bounded_cdp_body(
                {"body": "ignored", "base64Encoded": False},
                declared_length=7,
                encoded_data_length=NETWORK_BODY_LIMIT + 1,
            )
        )

    def test_browser_scripts_have_action_and_scroll_budgets(self):
        click_script = build_click_script(["Details"], max_clicks=3)
        scroll_script = build_scrollable_capture_script(["Details"], "exhaustive", max_chars=20000)
        self.assertIn("clicked.length >= 3", click_script)
        self.assertIn("el.closest('form')", click_script)
        self.assertIn("destructive.test(text)", click_script)
        self.assertNotIn("'a',", click_script)
        self.assertIn("const pageSteps = 8", scroll_script)
        self.assertIn("const maxChars = 20000", scroll_script)

    def test_browser_diagnostics_strip_query_secrets(self):
        self.assertEqual(
            safe_diagnostic_url("http://127.0.0.1/admin?token=secret#fragment"),
            "http://127.0.0.1",
        )


if __name__ == "__main__":
    unittest.main()
