import asyncio
import time
from unittest.mock import AsyncMock

import pytest

import pipelines


def _candidate(
    name: str,
    *,
    host: str | None = None,
    score: float = 1.0,
) -> dict:
    host = host or f"{name}.example"
    return {
        "title": f"Source {name}",
        "url": f"https://{host}/{name}",
        "domain": host,
        "snippet": f"Discovery evidence from {name}",
        "score": score,
        "score_reasons": [],
    }


def _successful_crawl(candidate: dict) -> dict:
    return {
        "ok": True,
        "title": candidate["title"],
        "url": candidate["url"],
        "requested_url": candidate["url"],
        "domain": candidate["domain"],
        "evidence_text": f"Extracted evidence from {candidate['title']}",
    }


@pytest.mark.asyncio
async def test_bounded_cleanup_consumes_tasks_that_already_finished(monkeypatch):
    consumed = []
    original_consumer = pipelines._consume_task_result

    def consume(task):
        consumed.append(task)
        original_consumer(task)

    async def fail():
        raise RuntimeError("finished before cleanup")

    monkeypatch.setattr(pipelines, "_consume_task_result", consume)
    task = asyncio.create_task(fail())
    await asyncio.sleep(0)

    await pipelines._cancel_tasks_bounded([task], timeout_seconds=0)

    assert consumed == [task]


def _configure_pipeline(
    monkeypatch,
    *,
    candidates: list[dict],
    crawl,
    crawl_budget: float = 1.0,
    max_urls: int = 4,
) -> None:
    async def plan(query: str, mode: str) -> dict:
        return {"query": query, "mode": mode, "queries": [query]}

    async def search(**_kwargs) -> list[dict]:
        return candidates

    monkeypatch.setitem(
        pipelines.RESEARCH_MODE_CONFIG,
        "balanced",
        {
            "max_urls": max_urls,
            "search_results": max(len(candidates), 1),
            "top_k": 0,
            "crawl_budget": crawl_budget,
        },
    )
    monkeypatch.setattr(pipelines, "RESEARCH_SOURCE_CONCURRENCY", 2)
    monkeypatch.setattr(pipelines, "build_research_plan", plan)
    monkeypatch.setattr(pipelines, "searxng_search", search)
    monkeypatch.setattr(pipelines, "crawl_and_ingest_limited", crawl)

    async def persist_crawled_source(result, *_args, **_kwargs):
        return result

    # Keep scheduler tests compatible with the extraction/persistence split:
    # timed work extracts only, and the parent persists accepted results.
    monkeypatch.setattr(
        pipelines,
        "crawl_source_limited",
        crawl,
        raising=False,
    )
    monkeypatch.setattr(
        pipelines,
        "persist_crawled_source",
        persist_crawled_source,
        raising=False,
    )
    monkeypatch.setattr(
        pipelines,
        "rag_query_impl",
        AsyncMock(return_value={"results": []}),
    )


@pytest.mark.asyncio
async def test_hung_source_does_not_block_fast_results_and_replacements(monkeypatch):
    candidates = [
        _candidate("hung", score=10),
        _candidate("failed", score=9),
        _candidate("replacement-one", score=8),
        _candidate("replacement-two", score=7),
    ]
    attempts = []
    hung_started = asyncio.Event()
    hung_cancelled = asyncio.Event()

    async def crawl(_semaphore, candidate, **_kwargs):
        name = candidate["url"].rsplit("/", 1)[-1]
        attempts.append(name)
        if name == "hung":
            hung_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                hung_cancelled.set()
        if name == "failed":
            return {
                "ok": False,
                "url": candidate["url"],
                "domain": candidate["domain"],
                "reason": "publisher rejected extraction",
            }
        await asyncio.sleep(0)
        return _successful_crawl(candidate)

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=1.0,
        max_urls=2,
    )
    monkeypatch.setattr(
        pipelines, "_source_crawl_timeout_seconds", lambda _budget: 0.05
    )
    monkeypatch.setattr(pipelines, "CRAWL_CANCEL_GRACE_SECONDS", 0.01)

    result = await asyncio.wait_for(
        pipelines.research_pipeline(
            "compare reliable source coverage",
            mode="balanced",
            max_sources=2,
            verify=False,
            persist_source_artifacts=False,
        ),
        timeout=0.5,
    )

    assert hung_started.is_set()
    assert hung_cancelled.is_set()
    assert attempts == ["hung", "failed", "replacement-one", "replacement-two"]
    assert [item["url"] for item in result["crawled_sources"]] == [
        candidates[2]["url"],
        candidates[3]["url"],
    ]
    assert {item["url"] for item in result["failed_sources"]} == {
        candidates[0]["url"],
        candidates[1]["url"],
    }
    assert result["crawl_budget"]["exhausted"] is False


@pytest.mark.asyncio
async def test_total_crawl_budget_does_not_wait_for_slow_cancellation_cleanup(
    monkeypatch,
):
    candidates = [_candidate("slow-one", score=2), _candidate("slow-two")]
    cleanup_finished = [asyncio.Event(), asyncio.Event()]

    async def crawl(_semaphore, candidate, **_kwargs):
        index = 0 if candidate["url"].endswith("slow-one") else 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.25)
            raise
        finally:
            cleanup_finished[index].set()

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=0.03,
        max_urls=2,
    )
    monkeypatch.setattr(pipelines, "_source_crawl_timeout_seconds", lambda _budget: 1.0)
    monkeypatch.setattr(pipelines, "CRAWL_CANCEL_GRACE_SECONDS", 0.01)

    started = time.monotonic()
    result = await pipelines.research_pipeline(
        "compare reliable source coverage",
        mode="balanced",
        max_sources=2,
        verify=False,
        persist_source_artifacts=False,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert result["crawl_budget"]["exhausted"] is True
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in cleanup_finished)),
        timeout=0.5,
    )


@pytest.mark.asyncio
async def test_timed_out_extraction_cannot_persist_after_pipeline_returns(monkeypatch):
    candidate = _candidate("late-write")
    release_extraction = asyncio.Event()
    extraction_finished = asyncio.Event()
    simulated_writes = []
    artifact_persist = AsyncMock(return_value={"artifact_id": "unexpected"})
    rag_persist = AsyncMock(return_value={"stored": 1})

    async def extract(_semaphore, candidate_item, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release_extraction.wait()
            extraction_finished.set()
            return {
                **_successful_crawl(candidate_item),
                "content": "content produced after the crawl deadline",
            }

    async def persist(result, *_args, **_kwargs):
        simulated_writes.append(result["url"])
        await artifact_persist(result)
        await rag_persist(result)
        return result

    async def legacy_crawl_and_ingest(semaphore, candidate_item, **kwargs):
        extracted = await pipelines.crawl_source_limited(
            semaphore,
            candidate_item,
            **kwargs,
        )
        return await pipelines.persist_crawled_source(extracted)

    _configure_pipeline(
        monkeypatch,
        candidates=[candidate],
        crawl=legacy_crawl_and_ingest,
        crawl_budget=0.02,
        max_urls=1,
    )
    monkeypatch.setattr(pipelines, "crawl_source_limited", extract, raising=False)
    monkeypatch.setattr(
        pipelines,
        "persist_crawled_source",
        persist,
        raising=False,
    )
    monkeypatch.setattr(pipelines, "_source_crawl_timeout_seconds", lambda _budget: 1.0)
    monkeypatch.setattr(pipelines, "CRAWL_CANCEL_GRACE_SECONDS", 0.005)

    result = await pipelines.research_pipeline(
        "research without post-timeout writes",
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=True,
    )

    assert result["crawled_sources"] == []
    assert result["crawl_budget"]["exhausted"] is True

    release_extraction.set()
    await asyncio.wait_for(extraction_finished.wait(), timeout=0.2)
    await asyncio.sleep(0)

    assert simulated_writes == []
    artifact_persist.assert_not_awaited()
    rag_persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_completed_during_other_source_cleanup_is_harvested(monkeypatch):
    candidates = [
        _candidate("initial-failure", score=3),
        _candidate("slow-cleanup", score=2),
        _candidate("replacement", score=1),
    ]
    slow_cleanup_started = asyncio.Event()
    release_slow_cleanup = asyncio.Event()
    slow_cleanup_finished = asyncio.Event()

    async def crawl(_semaphore, candidate, **_kwargs):
        name = candidate["url"].rsplit("/", 1)[-1]
        if name == "initial-failure":
            await asyncio.sleep(0.05)
            return {
                "ok": False,
                "url": candidate["url"],
                "title": candidate["title"],
                "domain": candidate["domain"],
                "reason": "initial extraction failed",
            }
        if name == "slow-cleanup":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                slow_cleanup_started.set()
                await release_slow_cleanup.wait()
                raise
            finally:
                slow_cleanup_finished.set()

        await slow_cleanup_started.wait()
        return _successful_crawl(candidate)

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=0.30,
        max_urls=2,
    )
    monkeypatch.setattr(
        pipelines, "_source_crawl_timeout_seconds", lambda _budget: 0.18
    )
    monkeypatch.setattr(pipelines, "CRAWL_CANCEL_GRACE_SECONDS", 1.0)

    result = await pipelines.research_pipeline(
        "retain work completed while another source cleans up",
        mode="balanced",
        max_sources=2,
        verify=False,
        persist_source_artifacts=False,
    )

    release_slow_cleanup.set()
    await asyncio.wait_for(slow_cleanup_finished.wait(), timeout=0.2)

    assert [item["url"] for item in result["crawled_sources"]] == [candidates[2]["url"]]
    assert candidates[2]["url"] not in {
        item["url"] for item in result["failed_sources"]
    }


@pytest.mark.asyncio
async def test_replacement_selection_preserves_source_owner_diversity(monkeypatch):
    candidates = [
        _candidate("primary", host="news.example.com", score=10),
        _candidate("same-owner", host="blog.example.com", score=9),
        _candidate("second-owner", host="independent.test", score=8),
        _candidate("third-owner", host="another.test", score=7),
    ]
    attempts = []

    async def crawl(_semaphore, candidate, **_kwargs):
        attempts.append(candidate["url"])
        if candidate["url"].endswith("/primary"):
            return {
                "ok": False,
                "url": candidate["url"],
                "domain": candidate["domain"],
                "reason": "publisher rejected extraction",
            }
        return _successful_crawl(candidate)

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=1.0,
        max_urls=2,
    )

    result = await pipelines.research_pipeline(
        "compare reliable source coverage",
        mode="balanced",
        max_sources=2,
        verify=True,
        persist_source_artifacts=False,
    )

    assert set(attempts[:2]) == {candidates[0]["url"], candidates[2]["url"]}
    assert attempts[2] == candidates[3]["url"]
    assert candidates[1]["url"] not in attempts
    assert {item["domain"] for item in result["crawled_sources"]} == {
        "independent.test",
        "another.test",
    }


@pytest.mark.asyncio
async def test_redirected_canonical_candidate_is_skipped_before_scheduling(monkeypatch):
    candidates = [
        _candidate("alias", score=3),
        _candidate("canonical", score=2),
        _candidate("replacement", score=1),
    ]
    attempts = []

    async def crawl(_semaphore, candidate, **_kwargs):
        name = candidate["url"].rsplit("/", 1)[-1]
        attempts.append(name)
        if name == "alias":
            return {
                **_successful_crawl(candidate),
                "url": candidates[1]["url"],
                "final_url": candidates[1]["url"],
                "domain": candidates[1]["domain"],
            }
        return _successful_crawl(candidate)

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=1.0,
        max_urls=2,
    )
    monkeypatch.setattr(pipelines, "RESEARCH_SOURCE_CONCURRENCY", 1)

    result = await pipelines.research_pipeline(
        "compare redirected sources",
        mode="balanced",
        max_sources=2,
        verify=False,
        persist_source_artifacts=False,
    )

    assert attempts == ["alias", "replacement"]
    assert [item["url"] for item in result["crawled_sources"]] == [
        candidates[1]["url"],
        candidates[2]["url"],
    ]


@pytest.mark.asyncio
async def test_concurrent_redirect_duplicate_does_not_consume_replacement_quota(
    monkeypatch,
):
    candidates = [
        _candidate("alias", score=5),
        _candidate("canonical", score=4),
        _candidate("failure-one", score=3),
        _candidate("failure-two", score=2),
        _candidate("replacement", score=1),
    ]
    attempts = []
    alias_finished = asyncio.Event()

    async def crawl(_semaphore, candidate, **_kwargs):
        name = candidate["url"].rsplit("/", 1)[-1]
        attempts.append(name)
        if name == "alias":
            alias_finished.set()
            return {
                **_successful_crawl(candidate),
                "url": candidates[1]["url"],
                "final_url": candidates[1]["url"],
                "domain": candidates[1]["domain"],
            }
        if name == "canonical":
            await alias_finished.wait()
            await asyncio.sleep(0.01)
            return _successful_crawl(candidate)
        if name.startswith("failure"):
            return {
                "ok": False,
                "url": candidate["url"],
                "domain": candidate["domain"],
                "reason": "publisher rejected extraction",
            }
        return _successful_crawl(candidate)

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=1.0,
        max_urls=2,
    )

    result = await pipelines.research_pipeline(
        "compare redirected sources",
        mode="balanced",
        max_sources=2,
        verify=False,
        persist_source_artifacts=False,
    )

    assert attempts == [
        "alias",
        "canonical",
        "failure-one",
        "failure-two",
        "replacement",
    ]
    assert {item["url"] for item in result["crawled_sources"]} == {
        candidates[1]["url"],
        candidates[4]["url"],
    }


@pytest.mark.asyncio
async def test_crawl_budget_exhaustion_is_reported_with_attempt_count(monkeypatch):
    candidates = [_candidate("never-finishes")]
    cancelled = asyncio.Event()

    async def crawl(_semaphore, _candidate_item, **_kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    _configure_pipeline(
        monkeypatch,
        candidates=candidates,
        crawl=crawl,
        crawl_budget=0.02,
        max_urls=1,
    )
    monkeypatch.setattr(pipelines, "_source_crawl_timeout_seconds", lambda _budget: 1.0)
    monkeypatch.setattr(pipelines, "CRAWL_CANCEL_GRACE_SECONDS", 0.01)

    result = await pipelines.research_pipeline(
        "compare reliable source coverage",
        mode="balanced",
        max_sources=1,
        verify=False,
        persist_source_artifacts=False,
    )

    await asyncio.wait_for(cancelled.wait(), timeout=0.1)
    assert result["crawl_budget"] == {
        "seconds": 0.02,
        "exhausted": True,
        "attempted_sources": 1,
    }
    assert [item["url"] for item in result["selected_for_crawl"]] == [
        candidates[0]["url"]
    ]
    assert result["crawled_sources"] == []
