import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import yaml

import crawler
import pdf_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PdfRunnerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_required_isolation_fails_closed_without_socket(self):
        with patch.object(crawler, "PDF_RUNNER_SOCKET", ""), patch.dict(
            os.environ,
            {"RESEARCH_REQUIRE_PDF_ISOLATION": "true"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "isolated pdf-runner"):
                await crawler._extract_pdf_text_sandboxed(b"%PDF")

    async def test_configured_socket_uses_isolated_runner(self):
        expected = ("content", "title", None)
        runner = AsyncMock(return_value=expected)
        with patch.object(crawler, "PDF_RUNNER_SOCKET", "/run/research-pdf/runner.sock"), patch(
            "crawler._extract_pdf_text_runner",
            new=runner,
        ):
            result = await crawler._extract_pdf_text_sandboxed(b"%PDF")
        self.assertEqual(result, expected)
        runner.assert_awaited_once_with(b"%PDF")


class PdfRunnerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def _post(self, content: bytes) -> httpx.Response:
        transport = httpx.ASGITransport(app=pdf_runner.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://pdf-runner",
        ) as client:
            return await client.post("/v1/extract", content=content)

    async def test_rejects_oversized_pdf_before_parser(self):
        parser = AsyncMock()
        with patch.object(pdf_runner, "PDF_MAX_RESPONSE_BYTES", 4), patch(
            "pdf_runner._extract_pdf_text_subprocess",
            new=parser,
        ):
            response = await self._post(b"12345")
        self.assertEqual(response.status_code, 413)
        parser.assert_not_awaited()

    async def test_returns_bounded_parser_result(self):
        parser = AsyncMock(return_value=("content", "title", None))
        with patch(
            "pdf_runner._extract_pdf_text_subprocess",
            new=parser,
        ):
            response = await self._post(b"%PDF")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"content": "content", "title": "title", "error": None},
        )

    def test_service_requires_network_isolation_marker(self):
        with patch.dict(
            os.environ,
            {"RESEARCH_PDF_NETWORK_ISOLATED": "false"},
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "isolated network namespace"):
                pdf_runner.require_isolated_runtime()


class PdfRunnerComposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text("utf-8"))
        cls.services = compose["services"]

    def test_runner_has_no_network_or_application_data_mounts(self):
        service = self.services["pdf-runner"]
        self.assertEqual(service["network_mode"], "none")
        self.assertNotIn("networks", service)
        self.assertNotIn("ports", service)
        self.assertNotIn("artifacts:/data/artifacts", service["volumes"])
        self.assertNotIn("model-cache:/data/models", service["volumes"])
        self.assertIn("pdf-runner-control:/run/research-pdf", service["volumes"])

    def test_worker_uses_read_only_socket_and_fails_closed(self):
        worker = self.services["research-worker"]
        self.assertIn(
            "pdf-runner-control:/run/research-pdf:ro",
            worker["volumes"],
        )
        self.assertEqual(
            worker["environment"]["PDF_RUNNER_SOCKET"],
            "/run/research-pdf/runner.sock",
        )
        self.assertEqual(worker["environment"]["RESEARCH_REQUIRE_PDF_ISOLATION"], "true")
        self.assertEqual(
            worker["depends_on"]["pdf-runner"]["condition"],
            "service_healthy",
        )


if __name__ == "__main__":
    unittest.main()
