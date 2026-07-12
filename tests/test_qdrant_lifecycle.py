import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

import shared
from shared import IngestRequest, QueryRequest
from url_identity import canonicalize_source_identity


def _client(monkeypatch, name="lifecycle"):
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE),
    )
    monkeypatch.setattr(shared, "COLLECTION_NAME", name)
    monkeypatch.setattr(shared, "get_qdrant_client", lambda: client)
    monkeypatch.setattr(shared, "init_qdrant", lambda: None)
    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", False)
    return client


def _point(
    point_id,
    *,
    source="https://example.com/source",
    ingestion_id=None,
    order=None,
    version=None,
    committed=False,
    latest=False,
    status="pending",
    extra=None,
):
    payload = {
        "namespace": "ns",
        "source": source,
        "url": source,
        "text": str(point_id),
        "chunk_index": 0,
    }
    if ingestion_id is not None:
        payload.update(
            {
                "ingestion_id": ingestion_id,
                "ingestion_order_ns": order,
                "source_version": version,
                "ingestion_chunk_count": 1,
                "ingestion_committed": committed,
                "is_latest_version": latest,
                "lifecycle_status": status,
            }
        )
    if extra:
        payload.update(extra)
    return PointStruct(id=point_id, vector=[0.1], payload=payload)


def test_source_identity_canonicalization_is_stable_and_dependency_free():
    assert (
        canonicalize_source_identity(
            "HTTPS://BÜCHER.Example:443/a/./b/../c/%7e?q=2&utm_source=x&A=1#fragment"
        )
        == "https://xn--bcher-kva.example/a/c/~?A=1&q=2"
    )
    assert canonicalize_source_identity("http://[2001:0db8::1]:80/a") == (
        "http://[2001:db8::1]/a"
    )
    assert canonicalize_source_identity("manual-source") == "manual-source"
    assert canonicalize_source_identity("https://user:secret@example.com/") == ""


def test_attempt_identity_prevents_retry_point_overwrites():
    first = shared.point_id_for(
        "https://example.com/", 0, "ns", "v1", "run", ingestion_id="attempt-a"
    )
    second = shared.point_id_for(
        "https://example.com/", 0, "ns", "v1", "run", ingestion_id="attempt-b"
    )
    assert first != second


@pytest.mark.asyncio
async def test_incomplete_ingestion_cannot_commit_or_displace_legacy_active(
    monkeypatch,
):
    client = _client(monkeypatch, "incomplete")
    pending = _point(
        2,
        ingestion_id="partial",
        order=200,
        version="v2",
        extra={"ingestion_chunk_count": 2},
    )
    client.upsert(
        collection_name="incomplete",
        points=[_point(1, extra={"is_latest_version": True}), pending],
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        await shared.supersede_source_versions_async(
            namespace="ns",
            source="https://example.com/source",
            active_ingestion_id="partial",
            active_source_version="v2",
            active_ingestion_order_ns=200,
            superseded_at="2026-01-01T00:00:00+00:00",
        )

    records, _ = client.scroll(
        collection_name="incomplete", limit=10, with_payload=True
    )
    payloads = {record.id: record.payload for record in records}
    assert payloads[1]["is_latest_version"] is True
    assert payloads[2]["ingestion_committed"] is False
    assert payloads[2]["is_latest_version"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guard_results,expected_status",
    [
        ([False], "pending"),
        ([True, False], "invalid"),
        ([True, True, False], "invalid"),
    ],
)
async def test_lease_guard_prevents_or_revokes_commit(
    monkeypatch, guard_results, expected_status
):
    client = _client(monkeypatch, f"guard-{expected_status}")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)

    async def embed(_texts):
        return [[0.1]]

    calls = 0

    async def guard():
        nonlocal calls
        result = guard_results[min(calls, len(guard_results) - 1)]
        calls += 1
        return result

    monkeypatch.setattr(shared, "embed_texts_async", embed)
    token = shared.set_ingestion_commit_guard(guard)
    try:
        with pytest.raises(HTTPException, match="ingestion lease"):
            await shared.rag_ingest_impl(
                IngestRequest(
                    text="attempt-scoped content",
                    metadata={
                        "source": "https://example.com/guarded",
                        "namespace": "ns",
                        "research_run_id": "run",
                        "ingestion_attempt_id": "attempt-a",
                        "ingestion_order_ns": 100,
                    },
                )
            )
    finally:
        shared.reset_ingestion_commit_guard(token)

    records, _ = client.scroll(
        collection_name=f"guard-{expected_status}", limit=10, with_payload=True
    )
    assert len(records) == 1
    assert records[0].payload["ingestion_attempt_id"] == "attempt-a"
    assert records[0].payload["ingestion_committed"] is False
    assert records[0].payload["is_latest_version"] is False
    assert records[0].payload["lifecycle_status"] == expected_status


@pytest.mark.asyncio
async def test_reconciliation_failure_invalidates_attempt_and_restores_previous(
    monkeypatch,
):
    client = _client(monkeypatch, "reconcile-failure")
    client.upsert(
        collection_name="reconcile-failure",
        points=[
            _point(1, extra={"is_latest_version": True}),
            _point(2, ingestion_id="new", order=200, version="v2"),
        ],
    )
    original_batch = client.batch_update_points
    calls = 0

    def fail_first_batch(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated reconciliation failure")
        return original_batch(**kwargs)

    monkeypatch.setattr(client, "batch_update_points", fail_first_batch)
    with pytest.raises(RuntimeError, match="simulated reconciliation failure"):
        await shared.supersede_source_versions_async(
            namespace="ns",
            source="https://example.com/source",
            active_ingestion_id="new",
            active_source_version="v2",
            active_ingestion_order_ns=200,
            superseded_at="2026-01-01T00:00:00+00:00",
        )

    records, _ = client.scroll(
        collection_name="reconcile-failure", limit=10, with_payload=True
    )
    payloads = {record.id: record.payload for record in records}
    assert payloads[1]["is_latest_version"] is True
    assert payloads[2]["ingestion_committed"] is False
    assert payloads[2]["lifecycle_status"] == "invalid"


@pytest.mark.asyncio
async def test_current_lease_attempt_commits_and_becomes_active(monkeypatch):
    client = _client(monkeypatch, "guard-current")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)

    async def embed(_texts):
        return [[0.1]]

    guard = AsyncMock(return_value=True)
    monkeypatch.setattr(shared, "embed_texts_async", embed)
    token = shared.set_ingestion_commit_guard(guard)
    try:
        result = await shared.rag_ingest_impl(
            IngestRequest(
                text="current attempt content",
                metadata={
                    "source": "https://example.com/current",
                    "namespace": "ns",
                    "research_run_id": "run",
                    "ingestion_attempt_id": "attempt-current",
                    "ingestion_order_ns": 200,
                },
            )
        )
    finally:
        shared.reset_ingestion_commit_guard(token)

    assert guard.await_count == 3
    assert result["ingestion_attempt_id"] == "attempt-current"
    records, _ = client.scroll(
        collection_name="guard-current", limit=10, with_payload=True
    )
    assert records[0].payload["ingestion_committed"] is True
    assert records[0].payload["is_latest_version"] is True


@pytest.mark.asyncio
async def test_remote_worker_compensates_when_post_request_lease_check_fails(
    monkeypatch,
):
    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", True)
    remote_request = AsyncMock(return_value={"stored": 1})
    monkeypatch.setattr(shared, "_remote_rag_request", remote_request)
    token = shared.set_ingestion_commit_guard(AsyncMock(side_effect=[True, False]))
    try:
        with pytest.raises(RuntimeError, match="lease"):
            await shared.rag_ingest_impl(
                IngestRequest(
                    text="content",
                    metadata={"ingestion_attempt_id": "remote-attempt"},
                )
            )
    finally:
        shared.reset_ingestion_commit_guard(token)
    assert [call.args[1] for call in remote_request.await_args_list] == [
        "/rag/ingest",
        "/rag/invalidate-attempt",
    ]
    assert remote_request.await_args_list[1].kwargs["json_body"] == {
        "ingestion_attempt_id": "remote-attempt",
        "reason": "worker_lease_lost",
    }


@pytest.mark.asyncio
async def test_remote_worker_compensates_when_request_is_cancelled(monkeypatch):
    monkeypatch.setattr(shared, "USE_RESEARCH_API_RAG", True)
    remote_request = AsyncMock(
        side_effect=[asyncio.CancelledError(), {"invalidated": 0}]
    )
    monkeypatch.setattr(shared, "_remote_rag_request", remote_request)
    token = shared.set_ingestion_commit_guard(AsyncMock(return_value=True))
    try:
        with pytest.raises(asyncio.CancelledError):
            await shared.rag_ingest_impl(
                IngestRequest(
                    text="content",
                    metadata={"ingestion_attempt_id": "cancelled-remote-attempt"},
                )
            )
    finally:
        shared.reset_ingestion_commit_guard(token)
    assert [call.args[1] for call in remote_request.await_args_list] == [
        "/rag/ingest",
        "/rag/invalidate-attempt",
    ]
    assert remote_request.await_args_list[1].kwargs["json_body"]["reason"] == (
        "remote_request_cancelled"
    )


@pytest.mark.asyncio
async def test_equal_orders_use_ingestion_id_as_cross_process_tiebreak(monkeypatch):
    client = _client(monkeypatch, "tie")
    client.upsert(
        collection_name="tie",
        points=[
            _point(1, ingestion_id="attempt-a", order=100, version="a"),
            _point(2, ingestion_id="attempt-z", order=100, version="z"),
        ],
    )

    for ingestion_id, version in [
        ("attempt-a", "a"),
        ("attempt-z", "z"),
        ("attempt-a", "a"),
    ]:
        result = await shared.supersede_source_versions_async(
            namespace="ns",
            source="https://example.com/source",
            active_ingestion_id=ingestion_id,
            active_source_version=version,
            active_ingestion_order_ns=100,
            superseded_at="2026-01-01T00:00:00+00:00",
        )

    records, _ = client.scroll(collection_name="tie", limit=10, with_payload=True)
    payloads = {record.payload["ingestion_id"]: record.payload for record in records}
    assert result["winner_ingestion_id"] == "attempt-z"
    assert result["is_latest_version"] is False
    assert payloads["attempt-z"]["is_latest_version"] is True
    assert payloads["attempt-a"]["is_latest_version"] is False
    assert sum(payload["is_latest_version"] for payload in payloads.values()) == 1


@pytest.mark.asyncio
async def test_stale_reconciler_batch_cannot_leave_concurrent_winner_active(
    monkeypatch,
):
    client = _client(monkeypatch, "stale-reconciler")
    client.upsert(
        collection_name="stale-reconciler",
        points=[_point(1, ingestion_id="older", order=100, version="old")],
    )
    original_batch = client.batch_update_points
    active_counts = []
    injected = False

    def batch_with_concurrent_winner(**kwargs):
        nonlocal injected
        if not injected:
            injected = True
            client.upsert(
                collection_name="stale-reconciler",
                points=[
                    _point(
                        2,
                        ingestion_id="newer",
                        order=200,
                        version="new",
                        committed=True,
                        latest=True,
                        status="active",
                    )
                ],
            )
        result = original_batch(**kwargs)
        records, _ = client.scroll(
            collection_name="stale-reconciler", limit=10, with_payload=True
        )
        active_counts.append(
            sum(shared._payload_is_active(record.payload or {}) for record in records)
        )
        return result

    monkeypatch.setattr(client, "batch_update_points", batch_with_concurrent_winner)
    result = await shared.supersede_source_versions_async(
        namespace="ns",
        source="https://example.com/source",
        active_ingestion_id="older",
        active_source_version="old",
        active_ingestion_order_ns=100,
        superseded_at="2026-01-01T00:00:00+00:00",
    )

    records, _ = client.scroll(
        collection_name="stale-reconciler", limit=10, with_payload=True
    )
    payloads = {record.payload["ingestion_id"]: record.payload for record in records}
    assert active_counts and max(active_counts) == 1
    assert result["winner_ingestion_id"] == "newer"
    assert payloads["newer"]["is_latest_version"] is True
    assert payloads["older"]["is_latest_version"] is False


def test_retrieval_filter_keeps_legacy_active_but_rejects_modern_non_active(
    monkeypatch,
):
    client = _client(monkeypatch, "retrieval")
    client.upsert(
        collection_name="retrieval",
        points=[
            _point(1, extra={"is_latest_version": True}),
            _point(
                2,
                ingestion_id="active",
                order=3,
                version="v3",
                committed=True,
                latest=True,
                status="active",
            ),
            _point(3, ingestion_id="pending", order=4, version="v4"),
            _point(
                4,
                ingestion_id="old",
                order=2,
                version="v2",
                committed=True,
                latest=False,
                status="superseded",
            ),
            _point(5, extra={"lifecycle_status": "pending"}),
            _point(
                6,
                extra={
                    "ingestion_id": "modern-with-missing-commit-marker",
                    "lifecycle_status": "committed_pending_reconciliation",
                },
            ),
        ],
    )
    query_filter = Filter(
        must=[
            FieldCondition(key="namespace", match=MatchValue(value="ns")),
            shared._retrievable_lifecycle_filter(),
        ]
    )
    records, _ = client.scroll(
        collection_name="retrieval",
        limit=10,
        scroll_filter=query_filter,
        with_payload=True,
    )
    assert {record.id for record in records} == {1, 2}


@pytest.mark.asyncio
async def test_attempt_invalidation_revokes_all_sources_and_restores_previous(
    monkeypatch,
):
    client = _client(monkeypatch, "attempt-invalidation")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)
    points = []
    for point_id, source in enumerate(
        ["https://example.com/a", "https://example.com/b"],
        start=1,
    ):
        points.extend(
            [
                _point(
                    point_id,
                    source=source,
                    ingestion_id=f"previous-{point_id}",
                    order=100,
                    version="previous",
                    committed=True,
                    latest=False,
                    status="superseded",
                    extra={"superseded_at_unix": int(time.time())},
                ),
                _point(
                    point_id + 10,
                    source=source,
                    ingestion_id=f"current-{point_id}",
                    order=200,
                    version="current",
                    committed=True,
                    latest=True,
                    status="active",
                    extra={"ingestion_attempt_id": "attempt-to-revoke"},
                ),
            ]
        )
    client.upsert(collection_name="attempt-invalidation", points=points)

    result = await shared.invalidate_ingestion_attempt_async(
        "attempt-to-revoke",
        reason="job_cancelled",
    )

    records, _ = client.scroll(
        collection_name="attempt-invalidation", limit=10, with_payload=True
    )
    payloads = {record.id: record.payload for record in records}
    assert result["invalidated"] == 2
    assert result["sources_reconciled"] == 2
    assert payloads[1]["is_latest_version"] is True
    assert payloads[2]["is_latest_version"] is True
    assert payloads[11]["ingestion_committed"] is False
    assert payloads[12]["ingestion_committed"] is False
    assert payloads[11]["lifecycle_status"] == "invalid"
    assert payloads[12]["lifecycle_status"] == "invalid"


@pytest.mark.asyncio
async def test_attempt_invalidated_before_first_write_can_never_commit(monkeypatch):
    client = _client(monkeypatch, "attempt-preinvalidated")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)
    await shared.invalidate_ingestion_attempt_async(
        "preinvalidated-attempt",
        reason="worker_lease_lost",
    )

    async def embed(_texts):
        return [[0.1]]

    monkeypatch.setattr(shared, "embed_texts_async", embed)
    with pytest.raises(HTTPException, match="attempt has been invalidated"):
        await shared.rag_ingest_impl(
            IngestRequest(
                text="late content",
                metadata={
                    "source": "https://example.com/late",
                    "ingestion_attempt_id": "preinvalidated-attempt",
                },
            )
        )

    records, _ = client.scroll(
        collection_name="attempt-preinvalidated", limit=10, with_payload=True
    )
    assert len(records) == 1
    assert records[0].payload["record_type"] == "ingestion_attempt_tombstone"


@pytest.mark.asyncio
async def test_invalidation_racing_after_upsert_blocks_late_commit(monkeypatch):
    client = _client(monkeypatch, "attempt-race")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)

    async def embed(_texts):
        return [[0.1]]

    original_upsert = shared.qdrant_upsert_async

    async def upsert_then_invalidate(points):
        await original_upsert(points)
        await shared.invalidate_ingestion_attempt_async(
            "racing-attempt",
            reason="remote_request_cancelled",
        )

    monkeypatch.setattr(shared, "embed_texts_async", embed)
    monkeypatch.setattr(shared, "qdrant_upsert_async", upsert_then_invalidate)
    with pytest.raises(HTTPException, match="attempt has been invalidated"):
        await shared.rag_ingest_impl(
            IngestRequest(
                text="racing content",
                metadata={
                    "source": "https://example.com/race",
                    "ingestion_attempt_id": "racing-attempt",
                },
            )
        )

    records, _ = client.scroll(
        collection_name="attempt-race", limit=10, with_payload=True
    )
    content_records = [
        record
        for record in records
        if record.payload.get("record_type") != "ingestion_attempt_tombstone"
    ]
    assert len(content_records) == 1
    assert content_records[0].payload["ingestion_committed"] is False
    assert content_records[0].payload["lifecycle_status"] == "invalid"


@pytest.mark.asyncio
async def test_tombstone_arriving_after_commit_revokes_before_promotion(monkeypatch):
    client = _client(monkeypatch, "attempt-postcommit-race")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)
    client.upsert(
        collection_name="attempt-postcommit-race",
        points=[
            _point(1, extra={"is_latest_version": True}),
            _point(
                2,
                ingestion_id="late-ingestion",
                order=200,
                version="v2",
                extra={"ingestion_attempt_id": "late-tombstone-attempt"},
            ),
        ],
    )
    original_set_payload = client.set_payload

    def set_payload_then_tombstone(**kwargs):
        result = original_set_payload(**kwargs)
        if kwargs.get("payload", {}).get("ingestion_committed") is True:
            client.upsert(
                collection_name="attempt-postcommit-race",
                points=[
                    PointStruct(
                        id=shared._ingestion_attempt_tombstone_id(
                            "late-tombstone-attempt"
                        ),
                        vector=[0.0],
                        payload={
                            "record_type": "ingestion_attempt_tombstone",
                            "namespace": "__research_mcp_internal__",
                            "ingestion_attempt_id": "late-tombstone-attempt",
                            "ingestion_committed": False,
                            "lifecycle_status": "invalid",
                        },
                    )
                ],
                wait=True,
            )
        return result

    monkeypatch.setattr(client, "set_payload", set_payload_then_tombstone)
    with pytest.raises(RuntimeError, match="attempt has been invalidated"):
        await shared.supersede_source_versions_async(
            namespace="ns",
            source="https://example.com/source",
            active_ingestion_id="late-ingestion",
            active_source_version="v2",
            active_ingestion_order_ns=200,
            superseded_at="2026-01-01T00:00:00+00:00",
            active_ingestion_attempt_id="late-tombstone-attempt",
        )

    records, _ = client.scroll(
        collection_name="attempt-postcommit-race", limit=10, with_payload=True
    )
    payloads = {record.id: record.payload for record in records}
    assert payloads[1]["is_latest_version"] is True
    assert payloads[2]["ingestion_committed"] is False
    assert payloads[2]["lifecycle_status"] == "invalid"


@pytest.mark.asyncio
async def test_run_query_returns_only_the_current_job_attempt(monkeypatch):
    client = _client(monkeypatch, "attempt-query")
    monkeypatch.setattr(shared, "VECTOR_SIZE", 1)

    async def embed(_texts):
        return [[0.1]]

    async def no_rerank(_query, docs, _top_k):
        return docs

    monkeypatch.setattr(shared, "embed_texts_async", embed)
    monkeypatch.setattr(shared, "rerank_docs", no_rerank)
    client.upsert(
        collection_name="attempt-query",
        points=[
            _point(
                1,
                source="https://example.com/a",
                ingestion_id="a",
                order=1,
                version="a",
                committed=True,
                latest=True,
                status="active",
                extra={
                    "research_run_id": "run",
                    "ingestion_attempt_id": "attempt-a",
                },
            ),
            _point(
                2,
                source="https://example.com/b",
                ingestion_id="b",
                order=2,
                version="b",
                committed=True,
                latest=True,
                status="active",
                extra={
                    "research_run_id": "run",
                    "ingestion_attempt_id": "attempt-b",
                },
            ),
        ],
    )

    result = await shared.rag_query_impl(
        QueryRequest(
            query="content",
            namespace="ns",
            research_run_id="run",
            ingestion_attempt_id="attempt-b",
        )
    )
    assert [item["ingestion_attempt_id"] for item in result["results"]] == ["attempt-b"]


@pytest.mark.asyncio
async def test_repair_canonicalizes_and_promotes_committed_ingestion(monkeypatch):
    client = _client(monkeypatch, "repair")
    variant = "HTTPS://Example.COM:443/source?utm_source=test#fragment"
    client.upsert(
        collection_name="repair",
        points=[
            _point(1, source=variant, extra={"is_latest_version": True}),
            _point(
                2,
                source="https://example.com/source",
                ingestion_id="committed",
                order=10,
                version="v2",
                committed=True,
                latest=False,
                status="committed_pending_reconciliation",
            ),
        ],
    )

    result = await shared.repair_qdrant_lifecycle_async(prune_history=False)
    records, _ = client.scroll(collection_name="repair", limit=10, with_payload=True)
    payloads = {record.id: record.payload for record in records}
    assert result["source_identities_repaired"] == 2
    assert payloads[1]["source"] == "https://example.com/source"
    assert payloads[1]["is_latest_version"] is False
    assert payloads[2]["is_latest_version"] is True


@pytest.mark.asyncio
async def test_repair_cursor_reaches_records_beyond_per_pass_limit(monkeypatch):
    client = _client(monkeypatch, "repair-cursor")
    client.upsert(
        collection_name="repair-cursor",
        points=[
            _point(
                point_id,
                source=f"https://example.com/{point_id}",
                ingestion_id=f"ingestion-{point_id}",
                order=point_id,
                version=f"v{point_id}",
                committed=True,
                latest=False,
                status="committed_pending_reconciliation",
            )
            for point_id in range(1, 6)
        ],
    )

    cursor = None
    passes = 0
    while True:
        result = await shared.repair_qdrant_lifecycle_async(
            max_points=2,
            prune_history=False,
            cursor=cursor,
        )
        passes += 1
        cursor = result["next_cursor"]
        if cursor is None:
            break
        assert passes < 10

    records, _ = client.scroll(
        collection_name="repair-cursor", limit=10, with_payload=True
    )
    assert passes == 3
    assert all(record.payload["is_latest_version"] is True for record in records)


@pytest.mark.asyncio
async def test_history_pruning_removes_old_inactive_and_abandoned_pending(monkeypatch):
    client = _client(monkeypatch, "prune")
    old = int(time.time()) - 1000
    client.upsert(
        collection_name="prune",
        points=[
            _point(1, extra={"is_latest_version": True}),
            _point(
                2,
                ingestion_id="old",
                order=1,
                version="v1",
                committed=True,
                latest=False,
                status="superseded",
                extra={"superseded_at_unix": old},
            ),
            _point(
                3,
                ingestion_id="abandoned",
                order=2,
                version="v2",
                extra={"ingested_at_unix": old},
            ),
        ],
    )

    result = await shared.prune_qdrant_history_async(retention_seconds=100)
    records, _ = client.scroll(collection_name="prune", limit=10, with_payload=True)
    assert result["deleted"] == 2
    assert {record.id for record in records} == {1}


@pytest.mark.asyncio
async def test_history_pruning_permanently_retains_attempt_tombstones(monkeypatch):
    client = _client(monkeypatch, "prune-tombstone")
    old = int(time.time()) - 1000
    tombstone_id = shared._ingestion_attempt_tombstone_id("revoked-attempt")
    client.upsert(
        collection_name="prune-tombstone",
        points=[
            _point(
                1,
                ingestion_id="invalid-content",
                order=1,
                version="v1",
                status="invalid",
                extra={"superseded_at_unix": old},
            ),
            PointStruct(
                id=tombstone_id,
                vector=[0.0],
                payload={
                    "record_type": "ingestion_attempt_tombstone",
                    "namespace": "__research_mcp_internal__",
                    "ingestion_attempt_id": "revoked-attempt",
                    "ingestion_committed": False,
                    "is_latest_version": False,
                    "lifecycle_status": "invalid",
                    "superseded_at_unix": old,
                },
            ),
        ],
    )

    result = await shared.prune_qdrant_history_async(retention_seconds=100)
    records, _ = client.scroll(
        collection_name="prune-tombstone", limit=10, with_payload=True
    )

    assert result["deleted"] == 1
    assert [record.id for record in records] == [tombstone_id]


@pytest.mark.asyncio
async def test_history_pruning_cursor_eventually_scans_entire_collection(monkeypatch):
    client = _client(monkeypatch, "prune-cursor")
    old = int(time.time()) - 1000
    client.upsert(
        collection_name="prune-cursor",
        points=[
            _point(
                point_id,
                ingestion_id=f"invalid-{point_id}",
                order=point_id,
                version=f"v{point_id}",
                status="invalid",
                extra={"superseded_at_unix": old},
            )
            for point_id in range(1, 6)
        ],
    )

    cursor = None
    deleted = 0
    passes = 0
    while True:
        result = await shared.prune_qdrant_history_async(
            retention_seconds=100,
            max_points=2,
            cursor=cursor,
        )
        deleted += result["deleted"]
        passes += 1
        cursor = result["next_cursor"]
        if cursor is None:
            break
        assert passes < 10

    records, _ = client.scroll(
        collection_name="prune-cursor", limit=10, with_payload=True
    )
    assert passes == 3
    assert deleted == 5
    assert records == []


@pytest.mark.asyncio
async def test_delete_uses_canonical_source_identity(monkeypatch):
    client = _client(monkeypatch, "delete-canonical")
    variant = "HTTPS://Example.COM:443/source?utm_source=test#fragment"
    canonical = "https://example.com/source"
    client.upsert(
        collection_name="delete-canonical",
        points=[_point(1, source=variant), _point(2, source=canonical)],
    )

    result = await shared.delete_source_impl(variant, namespace="ns")
    records, _ = client.scroll(
        collection_name="delete-canonical", limit=10, with_payload=True
    )
    assert result["source"] == canonical
    assert records == []
