import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

import crawler


class DirectFirstCrawlerStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_sufficient_direct_result_skips_crawl4ai(self):
        direct_result = {
            "content": "d" * 300,
            "body_format": "html",
            "extraction_method": "direct",
            "_direct_primary_content_chars": 300,
        }

        with patch.object(
            crawler, "DIRECT_FIRST_MIN_CONTENT_CHARS", 200
        ), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 10.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(return_value=direct_result)
        ) as direct, patch.object(
            crawler, "crawl4ai_request", new=AsyncMock()
        ) as crawl4ai, patch.object(
            crawler, "crawl4ai_markdown_request", new=AsyncMock()
        ) as markdown:
            result = await crawler.crawl_url_impl("https://example.com")

        self.assertIs(result, direct_result)
        direct.assert_awaited_once_with("https://example.com")
        crawl4ai.assert_not_awaited()
        markdown.assert_not_awaited()

    async def test_direct_challenge_page_escalates_to_crawl4ai(self):
        direct_result = {
            "content": "Checking your browser. " + ("placeholder " * 100),
            "extraction_method": "direct",
        }
        crawl_result = {"results": [{"content": "authoritative article " * 30}]}

        with patch.object(
            crawler, "DIRECT_FIRST_MIN_CONTENT_CHARS", 200
        ), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 10.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(return_value=direct_result)
        ), patch.object(
            crawler, "crawl4ai_request", new=AsyncMock(return_value=crawl_result)
        ) as crawl4ai:
            result = await crawler.crawl_url_impl("https://example.com")

        crawl4ai.assert_awaited_once()
        self.assertEqual(result["extraction_method"], "crawl4ai")

    async def test_long_html_chrome_shell_escalates_to_crawl4ai(self):
        shell_html = (
            "<html><body>"
            f"<nav>{'Documentation menu ' * 180}</nav>"
            '<div id="root"></div>'
            f"<footer>{'Privacy terms and links ' * 120}</footer>"
            "</body></html>"
        )
        direct_result = {
            "content": "Documentation menu " * 180,
            "body_format": "html",
            "_direct_primary_content_chars": crawler._direct_html_primary_content_chars(
                shell_html
            ),
        }
        crawl_result = {"results": [{"content": "rendered article " * 30}]}

        with patch.object(
            crawler, "DIRECT_FIRST_MIN_CONTENT_CHARS", 200
        ), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 10.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(return_value=direct_result)
        ), patch.object(
            crawler, "crawl4ai_request", new=AsyncMock(return_value=crawl_result)
        ) as crawl4ai:
            result = await crawler.crawl_url_impl("https://example.com")

        self.assertEqual(direct_result["_direct_primary_content_chars"], 0)
        crawl4ai.assert_awaited_once()
        self.assertEqual(result["extraction_method"], "crawl4ai")

    def test_primary_content_metric_keeps_real_article_content(self):
        article_html = (
            "<html><body>"
            f"<nav>{'Menu ' * 300}</nav>"
            f"<main><article><p>{'Detailed evidence and explanation. ' * 80}</p>"
            f"<pre>{'configuration = true\n' * 40}</pre></article></main>"
            f"<footer>{'Privacy ' * 200}</footer>"
            "</body></html>"
        )

        self.assertGreater(
            crawler._direct_html_primary_content_chars(article_html),
            2000,
        )

    async def test_crawl4ai_winner_cancels_and_drains_direct_task(self):
        direct_started = asyncio.Event()
        direct_cancelled = asyncio.Event()

        async def slow_direct(_url):
            direct_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                direct_cancelled.set()
                raise

        crawl_result = {"results": [{"content": "c" * 300}]}
        with patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", side_effect=slow_direct
        ), patch.object(
            crawler, "crawl4ai_request", new=AsyncMock(return_value=crawl_result)
        ):
            result = await crawler.crawl_url_impl("https://example.com")

        self.assertTrue(direct_started.is_set())
        self.assertTrue(direct_cancelled.is_set())
        self.assertEqual(result["extraction_method"], "crawl4ai")

    async def test_direct_winner_cancels_and_drains_crawl4ai_task(self):
        crawl_started = asyncio.Event()
        crawl_cancelled = asyncio.Event()

        async def delayed_direct(_url):
            await crawl_started.wait()
            return {"content": "d" * 300}

        async def slow_crawl(_payload):
            crawl_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                crawl_cancelled.set()
                raise

        with patch.object(
            crawler, "DIRECT_FIRST_MIN_CONTENT_CHARS", 200
        ), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", side_effect=delayed_direct
        ), patch.object(
            crawler, "crawl4ai_request", side_effect=slow_crawl
        ):
            result = await crawler.crawl_url_impl("https://example.com")

        self.assertTrue(crawl_started.is_set())
        self.assertTrue(crawl_cancelled.is_set())
        self.assertEqual(result["content"], "d" * 300)

    async def test_isolated_runner_skips_guaranteed_markdown_failure(self):
        direct_result = {"content": "short"}
        with patch.dict(os.environ, {"WEB_RUNNER_SOCKET": "/run/web.sock"}, clear=False), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(return_value=direct_result)
        ), patch.object(
            crawler,
            "crawl4ai_request",
            new=AsyncMock(side_effect=RuntimeError("crawl unavailable")),
        ), patch.object(
            crawler, "crawl4ai_markdown_request", new=AsyncMock()
        ) as markdown:
            result = await crawler.crawl_url_impl("https://example.com")

        markdown.assert_not_awaited()
        self.assertIs(result, direct_result)
        self.assertEqual(result["crawl4ai_errors"], ["crawl unavailable"])

    async def test_nonisolated_runner_preserves_markdown_quality_fallback(self):
        with patch.dict(os.environ, {"WEB_RUNNER_SOCKET": ""}, clear=False), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(return_value={"content": "short"})
        ), patch.object(
            crawler, "crawl4ai_request", new=AsyncMock(return_value={"results": []})
        ), patch.object(
            crawler,
            "crawl4ai_markdown_request",
            new=AsyncMock(return_value={"content": "m" * 300}),
        ) as markdown:
            result = await crawler.crawl_url_impl("https://example.com")

        markdown.assert_awaited_once_with("https://example.com")
        self.assertEqual(result["content"], "m" * 300)
        self.assertEqual(
            result["crawl4ai_errors"],
            ["Crawl4AI returned too little content"],
        )

    async def test_failed_richer_crawl_marks_direct_fallback_low_confidence(self):
        direct_result = {
            "content": "Navigation and legal links " * 150,
            "body_format": "html",
            "_direct_primary_content_chars": 0,
        }
        with patch.dict(
            os.environ,
            {"WEB_RUNNER_SOCKET": "/run/web.sock"},
            clear=False,
        ), patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(return_value=direct_result)
        ), patch.object(
            crawler,
            "crawl4ai_request",
            new=AsyncMock(side_effect=RuntimeError("crawl unavailable")),
        ):
            result = await crawler.crawl_url_impl("https://example.com")

        self.assertIs(result, direct_result)
        self.assertTrue(result["_direct_low_confidence"])

    async def test_crawl4ai_result_url_is_revalidated(self):
        validate = AsyncMock(return_value="https://example.com")
        crawl_result = {
            "results": [
                {
                    "content": "c" * 300,
                    "final_url": "https://www.example.com/final",
                }
            ]
        }
        with patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=validate
        ), patch.object(
            crawler, "direct_fetch_url", new=AsyncMock(side_effect=RuntimeError("no direct"))
        ), patch.object(
            crawler, "crawl4ai_request", new=AsyncMock(return_value=crawl_result)
        ):
            await crawler.crawl_url_impl("https://example.com/start")

        self.assertEqual(
            [call.args[0] for call in validate.await_args_list],
            ["https://example.com/start", "https://www.example.com/final"],
        )

    async def test_caller_cancellation_drains_both_hedged_tasks(self):
        direct_started = asyncio.Event()
        crawl_started = asyncio.Event()
        direct_cancelled = asyncio.Event()
        crawl_cancelled = asyncio.Event()

        async def blocked(started, cancelled):
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def blocked_direct(_url):
            return await blocked(direct_started, direct_cancelled)

        async def blocked_crawl(_payload):
            return await blocked(crawl_started, crawl_cancelled)

        with patch.object(
            crawler, "DIRECT_FIRST_HEDGE_SECONDS", 0.0
        ), patch.object(
            crawler, "validate_url_safety", new=AsyncMock(return_value="https://example.com")
        ), patch.object(
            crawler,
            "direct_fetch_url",
            side_effect=blocked_direct,
        ), patch.object(
            crawler,
            "crawl4ai_request",
            side_effect=blocked_crawl,
        ):
            task = asyncio.create_task(crawler.crawl_url_impl("https://example.com"))
            await asyncio.gather(direct_started.wait(), crawl_started.wait())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertTrue(direct_cancelled.is_set())
        self.assertTrue(crawl_cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
