import asyncio
import ipaddress
import json
import os
import socket
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

import browser
import crawler
import safe_egress
import web_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeStreamingResponse:
    def __init__(self, chunks, headers=None):
        self._chunks = chunks
        self.headers = headers or {}

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class FakePlaywrightManager:
    def __init__(self, launch):
        self.playwright = types.SimpleNamespace(
            chromium=types.SimpleNamespace(launch=launch)
        )

    async def __aenter__(self):
        return self.playwright

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class EgressPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_rejects_every_non_global_address_family(self):
        addresses = [
            "0.0.0.0",
            "10.0.0.1",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "172.16.0.1",
            "192.168.0.1",
            "198.18.0.1",
            "224.0.0.1",
            "::",
            "::1",
            "fc00::1",
            "fe80::1",
        ]
        for address in addresses:
            with self.subTest(address=address), self.assertRaises(safe_egress.EgressPolicyError):
                safe_egress.validate_public_address(address)

    def test_optional_deny_cidrs_can_block_the_vps_public_address(self):
        with self.assertRaises(safe_egress.EgressPolicyError):
            safe_egress.validate_public_address(
                "93.184.216.34",
                [ipaddress.ip_network("93.184.216.34/32")],
            )

    def test_rejects_single_label_destinations(self):
        with self.assertRaises(safe_egress.EgressPolicyError):
            safe_egress.normalize_destination_host("redis")

    async def test_rejects_non_web_destination_ports_before_dns(self):
        with patch("socket.getaddrinfo") as getaddrinfo:
            with self.assertRaises(safe_egress.EgressPolicyError):
                await safe_egress.resolve_public_addresses("example.com", 22)
        getaddrinfo.assert_not_called()

    def test_allowed_port_override_is_bounded(self):
        with self.assertRaisesRegex(ValueError, "more than 32"):
            safe_egress._parse_allowed_ports(",".join(str(port) for port in range(1, 34)))

    async def test_rejects_hostname_if_any_dns_answer_is_private(self):
        records = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.8", 443)),
        ]
        with patch("socket.getaddrinfo", return_value=records):
            with self.assertRaises(safe_egress.EgressPolicyError):
                await safe_egress.resolve_public_addresses("mixed.example", 443)

    async def test_connection_uses_validated_ip_without_second_dns_lookup(self):
        reader = object()
        writer = object()
        with patch(
            "safe_egress.resolve_public_addresses",
            new=AsyncMock(return_value=("93.184.216.34",)),
        ), patch(
            "safe_egress.asyncio.open_connection",
            new=AsyncMock(return_value=(reader, writer)),
        ) as open_connection:
            result = await safe_egress.open_pinned_connection("rebind.example", 443)

        self.assertEqual(result, (reader, writer))
        open_connection.assert_awaited_once_with(
            "93.184.216.34",
            443,
            family=socket.AF_INET,
        )

    async def _request_reply(self, command: int, address: bytes, port: int = 443) -> int:
        handler = safe_egress.SocksEgressServer(max_connections=2)
        server = await asyncio.start_server(handler.handle_client, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"\x05\x01\x00")
            await writer.drain()
            self.assertEqual(await reader.readexactly(2), b"\x05\x00")
            writer.write(bytes((5, command, 0, 1)) + address + port.to_bytes(2, "big"))
            await writer.drain()
            response = await reader.readexactly(10)
            writer.close()
            await writer.wait_closed()
            return response[1]
        finally:
            server.close()
            await server.wait_closed()

    async def test_socks_server_rejects_private_connect(self):
        reply = await self._request_reply(1, ipaddress.IPv4Address("127.0.0.1").packed)
        self.assertEqual(reply, safe_egress.REPLY_NOT_ALLOWED)

    async def test_socks_server_rejects_udp_associate(self):
        reply = await self._request_reply(3, ipaddress.IPv4Address("93.184.216.34").packed)
        self.assertEqual(reply, safe_egress.REPLY_COMMAND_UNSUPPORTED)

    async def test_socks_server_rejects_public_non_web_port(self):
        reply = await self._request_reply(
            1,
            ipaddress.IPv4Address("93.184.216.34").packed,
            port=22,
        )
        self.assertEqual(reply, safe_egress.REPLY_NOT_ALLOWED)


class RunnerBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_fails_closed_when_runner_socket_is_missing(self):
        with patch.dict(
            os.environ,
            {"RESEARCH_REQUIRE_WEB_ISOLATION": "true", "WEB_RUNNER_SOCKET": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "isolated web-runner"):
                await browser.playwright_explore_page("https://example.com")

    async def test_crawl4ai_fails_closed_when_runner_socket_is_missing(self):
        with patch.object(crawler, "CRAWL4AI_NETWORK_ISOLATED", True), patch.dict(
            os.environ,
            {"RESEARCH_REQUIRE_WEB_ISOLATION": "true", "WEB_RUNNER_SOCKET": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "isolated web-runner"):
                await crawler._crawl4ai_post(
                    "/crawl",
                    {"urls": ["https://example.com"]},
                    30.0,
                )


    async def test_runner_response_is_stream_bounded(self):
        response = FakeStreamingResponse([b"123", b"456"])
        with patch.object(browser, "WEB_RUNNER_MAX_RESPONSE_BYTES", 5):
            with self.assertRaisesRegex(RuntimeError, "byte limit"):
                await browser._read_runner_response(response)

    async def test_runner_strips_crawl4ai_proxy_fields_without_mutating_input(self):
        payload = {
            "urls": ["https://example.com"],
            "browser_config": {
                "headless": True,
                "proxy": "http://attacker.invalid",
                "proxy_config": {"server": "http://attacker.invalid"},
            },
            "crawler_config": {
                "word_count_threshold": 10,
                "proxy_config": {"server": "http://attacker.invalid"},
                "proxy_rotation_strategy": "custom",
                "proxy_session_id": "attacker-session",
                "proxy_session_ttl": 3600,
                "proxy_session_auto_release": False,
            },
        }
        with patch(
            "web_runner.validate_url_safety",
            new=AsyncMock(return_value="https://example.com"),
        ):
            forwarded = await web_runner.prepare_crawl_payload(payload)

        self.assertEqual(forwarded["browser_config"], {"headless": True})
        self.assertEqual(forwarded["crawler_config"], {"word_count_threshold": 10})
        self.assertEqual(
            payload["browser_config"]["proxy_config"],
            {"server": "http://attacker.invalid"},
        )
        self.assertEqual(
            payload["crawler_config"]["proxy_config"],
            {"server": "http://attacker.invalid"},
        )

    def test_runner_sends_dedicated_crawl4ai_bearer_token(self):
        with patch.object(web_runner, "WEB_RUNNER_CRAWL4AI_API_TOKEN", "crawler-secret"):
            self.assertEqual(
                web_runner.crawl4ai_auth_headers(),
                {"Authorization": "Bearer crawler-secret"},
            )
        with patch.object(web_runner, "WEB_RUNNER_CRAWL4AI_API_TOKEN", ""):
            self.assertEqual(web_runner.crawl4ai_auth_headers(), {})

    def test_runner_requires_both_network_marker_and_socks_proxy(self):
        with patch.dict(
            os.environ,
            {
                "RESEARCH_REQUIRE_WEB_ISOLATION": "true",
                "RESEARCH_WEB_NETWORK_ISOLATED": "false",
            },
            clear=False,
        ), patch.object(web_runner, "WEB_RUNNER_PROXY_URL", "socks5://safe-egress:1080"):
            with self.assertRaisesRegex(RuntimeError, "internal Docker network"):
                web_runner.require_isolated_runtime()

    def test_browser_and_crawler_error_details_are_redacted(self):
        error = RuntimeError("request failed?token=super-secret-value")
        self.assertNotIn("super-secret-value", browser.safe_exception_detail(error))
        self.assertNotIn("super-secret-value", crawler._safe_error_detail(error))

    def test_browser_context_denies_downloads_and_permissions(self):
        source = (PROJECT_ROOT / "browser.py").read_text("utf-8")
        self.assertIn("accept_downloads=False", source)
        self.assertIn("permissions=[]", source)
        self.assertIn('service_workers="block"', source)


class ChromiumSandboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        browser.set_resolved_chromium_sandbox(None)

    def tearDown(self):
        browser.set_resolved_chromium_sandbox(None)

    @staticmethod
    def _browser():
        return types.SimpleNamespace(close=AsyncMock())

    async def test_auto_mode_keeps_native_sandbox_when_available(self):
        launched_browser = self._browser()
        launch = AsyncMock(return_value=launched_browser)
        manager = FakePlaywrightManager(launch)

        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "auto",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ), patch.object(web_runner, "async_playwright", return_value=manager):
            await web_runner.verify_chromium_runtime()

        launch.assert_awaited_once()
        self.assertTrue(launch.await_args.kwargs["chromium_sandbox"])
        self.assertTrue(browser.chromium_launch_options()["chromium_sandbox"])

    async def test_auto_mode_falls_back_only_for_known_host_denial(self):
        denial = RuntimeError(
            "FATAL:sandbox/linux/services/credentials.cc:131 "
            "Check failed: Permission denied (13)"
        )
        launched_browser = self._browser()
        launch = AsyncMock(side_effect=[denial, launched_browser])
        manager = FakePlaywrightManager(launch)

        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "auto",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ), patch.object(
            web_runner, "async_playwright", return_value=manager
        ), patch.object(web_runner.LOGGER, "warning") as warning:
            await web_runner.verify_chromium_runtime()

        self.assertEqual(launch.await_count, 2)
        self.assertTrue(launch.await_args_list[0].kwargs["chromium_sandbox"])
        self.assertFalse(launch.await_args_list[1].kwargs["chromium_sandbox"])
        self.assertFalse(browser.chromium_launch_options()["chromium_sandbox"])
        warning.assert_called_once()

    async def test_auto_mode_fails_closed_for_unrelated_launch_error(self):
        launch = AsyncMock(side_effect=RuntimeError("browser executable missing"))
        manager = FakePlaywrightManager(launch)

        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "auto",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ), patch.object(web_runner, "async_playwright", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "sandbox preflight failed"):
                await web_runner.verify_chromium_runtime()

        self.assertEqual(launch.await_count, 1)

    def test_host_denial_match_requires_both_specific_markers(self):
        self.assertFalse(
            web_runner.chromium_sandbox_denied_by_host(
                RuntimeError("sandbox/linux/services/credentials.cc:131 failed")
            )
        )
        self.assertFalse(
            web_runner.chromium_sandbox_denied_by_host(
                RuntimeError("unrelated operation: Permission denied (13)")
            )
        )

    async def test_auto_mode_fails_if_compatibility_launch_fails(self):
        denial = RuntimeError(
            "FATAL:sandbox/linux/services/credentials.cc:131 "
            "Check failed: Permission denied (13)"
        )
        launch = AsyncMock(
            side_effect=[denial, RuntimeError("compatibility launch failed")]
        )
        manager = FakePlaywrightManager(launch)

        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "auto",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ), patch.object(web_runner, "async_playwright", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "compatibility preflight failed"):
                await web_runner.verify_chromium_runtime()

        self.assertEqual(launch.await_count, 2)
        self.assertIsNone(browser._resolved_chromium_sandbox)

    async def test_required_mode_never_falls_back(self):
        denial = RuntimeError(
            "FATAL:sandbox/linux/services/credentials.cc:131 "
            "Check failed: Permission denied (13)"
        )
        launch = AsyncMock(side_effect=denial)
        manager = FakePlaywrightManager(launch)

        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "required",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ), patch.object(web_runner, "async_playwright", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "sandbox preflight failed"):
                await web_runner.verify_chromium_runtime()

        self.assertEqual(launch.await_count, 1)
        self.assertTrue(launch.await_args.kwargs["chromium_sandbox"])

    async def test_disabled_mode_never_attempts_native_sandbox(self):
        launched_browser = self._browser()
        launch = AsyncMock(return_value=launched_browser)
        manager = FakePlaywrightManager(launch)

        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "disabled",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ), patch.object(web_runner, "async_playwright", return_value=manager):
            await web_runner.verify_chromium_runtime()

        launch.assert_awaited_once()
        self.assertFalse(launch.await_args.kwargs["chromium_sandbox"])

    def test_legacy_disable_switch_overrides_required_mode(self):
        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "required",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "true",
            },
            clear=False,
        ):
            self.assertEqual(browser.chromium_sandbox_mode(), "disabled")

    def test_invalid_mode_is_rejected(self):
        with patch.dict(
            os.environ,
            {
                "RESEARCH_BROWSER_SANDBOX_MODE": "sometimes",
                "RESEARCH_BROWSER_DISABLE_SANDBOX": "false",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must be auto, required, or disabled"):
                browser.chromium_sandbox_mode()


class PdfSandboxTests(unittest.IsolatedAsyncioTestCase):
    def _fake_pypdf(self, page_texts, title="Example"):
        class FakePage:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class FakeReader:
            def __init__(self, _stream):
                self.pages = [FakePage(text) for text in page_texts]
                self.metadata = types.SimpleNamespace(title=title)

        module = types.ModuleType("pypdf")
        module.PdfReader = FakeReader
        return module

    def test_pdf_input_page_and_text_limits(self):
        with patch.object(crawler, "PDF_MAX_RESPONSE_BYTES", 4):
            _, _, error = crawler._extract_pdf_text(b"12345")
            self.assertIn("byte limit", error)

        fake_pypdf = self._fake_pypdf(["one", "two"])
        with patch.dict(sys.modules, {"pypdf": fake_pypdf}), patch.object(
            crawler, "PDF_MAX_PAGES", 1
        ):
            _, _, error = crawler._extract_pdf_text(b"%PDF")
            self.assertIn("page limit", error)

        fake_pypdf = self._fake_pypdf(["abcdefghij"])
        with patch.dict(sys.modules, {"pypdf": fake_pypdf}), patch.object(
            crawler, "PDF_MAX_EXTRACTED_CHARS", 5
        ):
            content, title, error = crawler._extract_pdf_text(b"%PDF")
            self.assertEqual(content, "abcde")
            self.assertEqual(title, "Example")
            self.assertIn("truncated", error)

    def test_pdf_subprocess_has_linux_resource_limits(self):
        source = (PROJECT_ROOT / "pdf_sandbox.py").read_text("utf-8")
        self.assertIn("resource.RLIMIT_AS", source)
        self.assertIn("resource.RLIMIT_CPU", source)
        self.assertIn("resource.RLIMIT_NOFILE", source)

    async def test_pdf_subprocess_receives_no_application_secrets(self):
        class FakeProcess:
            returncode = 0

            async def communicate(self, input):
                self.input = input
                return b'{"content":"ok","title":null,"error":null}', b""

        process = FakeProcess()
        create_process = AsyncMock(return_value=process)
        with patch(
            "crawler.asyncio.create_subprocess_exec",
            new=create_process,
        ), patch.dict(
            os.environ,
            {
                "MCP_AUTH_TOKEN": "do-not-copy",
                "GITHUB_TOKEN": "do-not-copy",
                "PLANNER_API_KEY": "do-not-copy",
            },
            clear=False,
        ):
            content, title, error = await crawler._extract_pdf_text_sandboxed(b"%PDF")

        self.assertEqual((content, title, error), ("ok", None, None))
        child_environment = create_process.await_args.kwargs["env"]
        self.assertNotIn("MCP_AUTH_TOKEN", child_environment)
        self.assertNotIn("GITHUB_TOKEN", child_environment)
        self.assertNotIn("PLANNER_API_KEY", child_environment)

    async def test_pdf_subprocess_is_killed_on_timeout(self):
        class SlowProcess:
            returncode = None

            def __init__(self):
                self.kill = MagicMock(side_effect=self._mark_killed)
                self.wait = AsyncMock(return_value=-9)

            def _mark_killed(self):
                self.returncode = -9

            async def communicate(self, input):
                await asyncio.sleep(0.05)
                return b"", b""

        process = SlowProcess()
        with patch(
            "crawler.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=process),
        ), patch.object(crawler, "DIRECT_EXTRACTION_TIMEOUT_SECONDS", 0.01):
            _, _, error = await crawler._extract_pdf_text_sandboxed(b"%PDF")

        process.kill.assert_called_once_with()
        process.wait.assert_awaited_once_with()
        self.assertIn("time limit", error)


class ComposeIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text("utf-8"))
        cls.services = cls.compose["services"]

    def test_executable_web_services_have_no_backend_or_direct_egress(self):
        self.assertEqual(self.services["web-runner"]["networks"], ["web-sandbox"])
        self.assertEqual(self.services["crawl4ai"]["networks"], ["web-sandbox"])
        self.assertTrue(self.compose["networks"]["web-sandbox"]["internal"])
        self.assertEqual(
            set(self.services["safe-egress"]["networks"]),
            {"web-sandbox", "egress"},
        )

    def test_no_web_facing_engine_can_reach_backend(self):
        for service_name in ("searxng", "web-runner", "safe-egress", "crawl4ai"):
            with self.subTest(service=service_name):
                self.assertNotIn("backend", self.services[service_name]["networks"])
        self.assertEqual(
            set(self.services["searxng"]["networks"]),
            {"search-control", "egress"},
        )
        self.assertTrue(self.compose["networks"]["search-control"]["internal"])
        self.assertIn("search-control", self.services["research-worker"]["networks"])

    def test_searxng_uses_a_file_backed_config_with_runtime_secret(self):
        self.assertEqual(
            self.compose["configs"]["searxng-settings"],
            {"file": "./searxng-settings.yml"},
        )
        searxng = self.services["searxng"]
        self.assertTrue(searxng["read_only"])
        self.assertEqual(searxng["user"], "977:977")
        self.assertIn(
            {
                "source": "searxng-settings",
                "target": "/etc/searxng/settings.yml",
            },
            searxng["configs"],
        )
        self.assertEqual(
            searxng["environment"]["SEARXNG_SECRET"],
            "${SEARXNG_SECRET:-replace-this-private-instance-secret}",
        )
        settings = yaml.safe_load(
            (PROJECT_ROOT / "searxng-settings.yml").read_text("utf-8")
        )
        self.assertIn("json", settings["search"]["formats"])
        self.assertEqual(
            settings["server"]["secret_key"],
            "ultrasecretkey",
        )

    def test_worker_uses_socket_and_is_not_on_sandbox_network(self):
        self.assertNotIn("web-sandbox", self.services["research-worker"]["networks"])
        volumes = self.services["research-worker"]["volumes"]
        self.assertIn("web-runner-control:/run/research-web:ro", volumes)
        self.assertEqual(
            self.services["research-worker"]["environment"]["WEB_RUNNER_SOCKET"],
            "/run/research-web/runner.sock",
        )

    def test_performance_settings_reach_the_services_that_use_them(self):
        worker_environment = self.services["research-worker"]["environment"]
        gateway_environment = self.services["mcp-gateway"]["environment"]

        self.assertEqual(
            worker_environment["DIRECT_FIRST_HEDGE_SECONDS"],
            "${DIRECT_FIRST_HEDGE_SECONDS:-1.5}",
        )
        self.assertEqual(
            worker_environment["DIRECT_FIRST_MIN_CONTENT_CHARS"],
            "${DIRECT_FIRST_MIN_CONTENT_CHARS:-2000}",
        )
        self.assertEqual(
            worker_environment["RESEARCH_SOURCE_CONCURRENCY"],
            "${RESEARCH_SOURCE_CONCURRENCY:-2}",
        )
        self.assertEqual(
            gateway_environment["MCP_JOB_LONG_POLL_SECONDS"],
            "${MCP_JOB_LONG_POLL_SECONDS:-15}",
        )
        self.assertEqual(
            gateway_environment["MCP_SYNC_JOB_WAIT_SECONDS"],
            "${MCP_SYNC_JOB_WAIT_SECONDS:-60}",
        )

    def test_runner_has_no_backend_or_application_credentials(self):
        runner_environment = self.services["web-runner"]["environment"]
        forbidden_fragments = (
            "GITHUB",
            "MCP_",
            "PLANNER",
            "QDRANT",
            "REDIS",
            "TOKEN",
            "API_KEY",
        )
        self.assertFalse(
            [
                key
                for key in runner_environment
                if key != "CRAWL4AI_API_TOKEN"
                if any(fragment in key for fragment in forbidden_fragments)
            ]
        )
        self.assertEqual(
            runner_environment["CRAWL4AI_API_TOKEN"],
            self.services["crawl4ai"]["environment"]["CRAWL4AI_API_TOKEN"],
        )
        self.assertEqual(
            self.services["crawl4ai"]["environment"]["GUNICORN_BIND"],
            "0.0.0.0:11235",
        )

    def test_runner_defaults_to_automatic_sandbox_compatibility(self):
        self.assertEqual(
            self.services["web-runner"]["environment"][
                "RESEARCH_BROWSER_SANDBOX_MODE"
            ],
            "${RESEARCH_BROWSER_SANDBOX_MODE:-auto}",
        )

    def test_crawl4ai_is_derived_and_chained_through_healthy_broker(self):
        service = self.services["crawl4ai"]
        self.assertEqual(service["build"]["target"], "crawl4ai-runtime")
        self.assertIn("CRAWL4AI_IMAGE", service["build"]["args"])
        self.assertIn("CRAWL4AI_DERIVED_IMAGE", service["image"])
        self.assertEqual(
            service["environment"]["CRAWL4AI_EGRESS_SOCKS_HOST"],
            "safe-egress",
        )
        self.assertEqual(service["environment"]["CRAWL4AI_EGRESS_SOCKS_PORT"], "1080")
        self.assertEqual(
            service["depends_on"]["safe-egress"]["condition"],
            "service_healthy",
        )

        dockerfile = (PROJECT_ROOT / "Dockerfile").read_text("utf-8")
        self.assertIn("AS crawl4ai-runtime", dockerfile)
        self.assertIn("crawl4ai_egress_proxy.py /app/egress_proxy.py", dockerfile)
        self.assertIn("socks5_client.py /app/socks5_client.py", dockerfile)

    def test_worker_does_not_receive_mcp_or_github_credentials(self):
        worker_environment = self.services["research-worker"]["environment"]
        self.assertNotIn("MCP_AUTH_TOKEN", worker_environment)
        self.assertNotIn("MCP_AUTH_TOKENS_JSON", worker_environment)
        self.assertNotIn("GITHUB_TOKEN", worker_environment)
        self.assertNotIn("GITHUB_ALLOWED_REPOSITORIES", worker_environment)

    def test_only_broker_bridges_sandbox_to_external_egress(self):
        bridging_services = []
        for name, service in self.services.items():
            networks = set(service.get("networks", []))
            if "web-sandbox" in networks and "egress" in networks:
                bridging_services.append(name)
        self.assertEqual(bridging_services, ["safe-egress"])

    def test_browser_services_use_namespace_seccomp_profile(self):
        for service_name in ("web-runner", "crawl4ai"):
            self.assertIn(
                "seccomp=./seccomp_profile.json",
                self.services[service_name]["security_opt"],
            )
        profile = json.loads((PROJECT_ROOT / "seccomp_profile.json").read_text("utf-8"))
        namespace_rules = [
            rule
            for rule in profile["syscalls"]
            if set(rule.get("names", [])) == {"clone", "setns", "unshare"}
            and rule.get("action") == "SCMP_ACT_ALLOW"
            and not rule.get("includes")
            and not rule.get("excludes")
        ]
        self.assertEqual(len(namespace_rules), 1)

    def test_broker_is_not_published_to_the_host(self):
        self.assertNotIn("ports", self.services["safe-egress"])
        self.assertNotIn("expose", self.services["safe-egress"])


if __name__ == "__main__":
    unittest.main()
