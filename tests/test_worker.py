import asyncio
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from artifact_store import ArtifactStore
import shared
from worker import (
    _INTERNAL_ATTEMPT_ID,
    _INTERNAL_ATTEMPT_ORDER_NS,
    _claimed_attempt_context,
    dispatch_job,
    JobWorker,
)


class FakeWorkerStore:
    def __init__(self, job, cancel_after_checks=None):
        if job is not None:
            job = dict(job)
            job.setdefault("lease_token", uuid.uuid4().hex)
        self.job = job
        self.cancel_after_checks = cancel_after_checks
        self.cancel_checks = 0
        self.completed = None
        self.failed = None
        self.cancelled = None
        self.requeued = None
        self.lease_valid = True
        self.heartbeat_calls = []
        self.stale_recovery_calls = 0
        self.protected_job_ids = set()
        self.result_ttl_seconds = 2_592_000
        self.registered_invalidations = {}
        self.due_invalidations = []
        self.deferred_invalidations = []
        self.acknowledged_invalidations = []
        self.registration_error = None
        self.successful_ingestion_attempt_id = None

    async def claim_job(self, timeout, worker_id):
        job, self.job = self.job, None
        return job

    async def heartbeat_job(self, job_id, worker_id, lease_token):
        self.heartbeat_calls.append((job_id, worker_id, lease_token))
        return self.lease_valid

    async def record_worker_heartbeat(self, worker_id, state="ready", host_id=None):
        return None

    async def requeue_stale_jobs(self, stale_after_seconds):
        self.stale_recovery_calls += 1
        return 0

    async def active_job_ids(self):
        return set(self.protected_job_ids)

    async def is_cancellation_requested(self, job_id):
        self.cancel_checks += 1
        return (
            self.cancel_after_checks is not None
            and self.cancel_checks >= self.cancel_after_checks
        )

    async def complete_job(
        self,
        job_id,
        metadata,
        lease_token=None,
        successful_ingestion_attempt_id=None,
    ):
        self.completed = (job_id, metadata, lease_token)
        self.successful_ingestion_attempt_id = successful_ingestion_attempt_id
        if successful_ingestion_attempt_id is not None:
            self.registered_invalidations.pop(successful_ingestion_attempt_id, None)

    async def fail_job(self, job_id, error, lease_token=None):
        self.failed = (job_id, error, lease_token)

    async def mark_cancelled(self, job_id, reason, lease_token=None):
        self.cancelled = (job_id, reason, lease_token)

    async def requeue_job(self, job_id, reason, lease_token=None):
        self.requeued = (job_id, reason, lease_token)

    async def register_ingestion_invalidation(
        self,
        job_id,
        ingestion_attempt_id,
        lease_token=None,
    ):
        if self.registration_error is not None:
            raise self.registration_error
        self.registered_invalidations[ingestion_attempt_id] = {
            "job_id": job_id,
            "ingestion_attempt_id": ingestion_attempt_id,
            "reason": "worker_attempt_abandoned",
        }

    async def schedule_ingestion_invalidation(
        self,
        job_id,
        ingestion_attempt_id,
        reason,
    ):
        record = self.registered_invalidations.get(ingestion_attempt_id)
        if record is None:
            return False
        record["reason"] = reason
        self.due_invalidations = [
            pending
            for pending in self.due_invalidations
            if pending["ingestion_attempt_id"] != ingestion_attempt_id
        ]
        self.due_invalidations.append(dict(record))
        return True

    async def claim_due_ingestion_invalidations(self, limit, lease_seconds):
        claimed = self.due_invalidations[:limit]
        self.due_invalidations = self.due_invalidations[limit:]
        return claimed

    async def defer_ingestion_invalidation(
        self,
        ingestion_attempt_id,
        delay_seconds,
    ):
        self.deferred_invalidations.append(
            (ingestion_attempt_id, delay_seconds)
        )
        record = self.registered_invalidations.get(ingestion_attempt_id)
        if record is not None:
            self.due_invalidations = [
                pending
                for pending in self.due_invalidations
                if pending["ingestion_attempt_id"] != ingestion_attempt_id
            ]
            self.due_invalidations.append(dict(record))

    async def acknowledge_ingestion_invalidation(self, ingestion_attempt_id):
        self.acknowledged_invalidations.append(ingestion_attempt_id)
        self.registered_invalidations.pop(ingestion_attempt_id, None)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifacts = ArtifactStore(self.temp_dir.name)
        self.invalidate_attempt_patcher = patch(
            "shared.invalidate_ingestion_attempt_async",
            AsyncMock(
                return_value={
                    "invalidated": 0,
                    "sources_reconciled": 0,
                }
            ),
        )
        self.invalidate_attempt = self.invalidate_attempt_patcher.start()

    async def asyncTearDown(self):
        self.invalidate_attempt_patcher.stop()
        self.temp_dir.cleanup()

    async def test_success_writes_full_artifact_and_compact_redis_metadata(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}}
        )

        dispatched_payload = None

        async def dispatch(kind, payload):
            nonlocal dispatched_payload
            dispatched_payload = payload
            return {"query": payload["query"], "results": [{"text": "full evidence"}]}

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        self.assertTrue(await worker.run_once(timeout=0.01))

        _, metadata, lease_token = store.completed
        self.assertEqual(metadata["results_count"], 1)
        full = await self.artifacts.read_json(metadata["artifact"]["relative_path"])
        self.assertEqual(full["results"][0]["text"], "full evidence")
        self.assertEqual(dispatched_payload["research_run_id"], job_id)
        self.assertEqual(lease_token, store.heartbeat_calls[-1][2])
        self.assertIsNotNone(store.successful_ingestion_attempt_id)
        self.assertEqual(store.registered_invalidations, {})
        self.assertIsNone(store.failed)

    async def test_attempt_context_is_authoritative_and_does_not_expose_lease(self):
        job_id = uuid.uuid4().hex
        lease_token = "a" * 32
        store = FakeWorkerStore(
            {
                "job_id": job_id,
                "kind": "research_web",
                "lease_token": lease_token,
                "attempt": "2",
                "attempt_started_at": "2026-01-02T03:04:05.123456+00:00",
                "payload": {
                    "query": "q",
                    "research_run_id": "spoofed-run",
                    "ingestion_attempt_id": "spoofed-attempt",
                    "ingestion_order_ns": 1,
                    _INTERNAL_ATTEMPT_ID: "spoofed-internal-attempt",
                    _INTERNAL_ATTEMPT_ORDER_NS: 2,
                },
            }
        )
        dispatched_payload = None

        async def dispatch(_kind, payload):
            nonlocal dispatched_payload
            dispatched_payload = payload
            await shared._assert_ingestion_commit_allowed()
            return {"query": payload["query"]}

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)

        expected_id, expected_order = _claimed_attempt_context(
            {
                "attempt": "2",
                "attempt_started_at": "2026-01-02T03:04:05.123456+00:00",
            },
            job_id=job_id,
            lease_token=lease_token,
        )
        self.assertEqual(dispatched_payload["research_run_id"], job_id)
        self.assertNotIn("ingestion_attempt_id", dispatched_payload)
        self.assertNotIn("ingestion_order_ns", dispatched_payload)
        self.assertEqual(dispatched_payload[_INTERNAL_ATTEMPT_ID], expected_id)
        self.assertEqual(dispatched_payload[_INTERNAL_ATTEMPT_ORDER_NS], expected_order)
        self.assertNotIn(lease_token, str(dispatched_payload))
        artifact_path = store.completed[1]["artifact"]["relative_path"]
        self.assertIn(expected_id[:16], artifact_path)
        self.assertNotIn(lease_token[:12], artifact_path)

    def test_claimed_retry_order_is_monotonic_and_attempt_ids_are_distinct(self):
        metadata = {"attempt_started_at": "2026-01-02T03:04:05.123456+00:00"}
        first_id, first_order = _claimed_attempt_context(
            {**metadata, "attempt": "1"},
            job_id="a" * 32,
            lease_token="1" * 32,
        )
        second_id, second_order = _claimed_attempt_context(
            {**metadata, "attempt": "2"},
            job_id="a" * 32,
            lease_token="2" * 32,
        )
        self.assertNotEqual(first_id, second_id)
        self.assertGreater(second_order, first_order)

    async def test_dispatch_passes_attempt_scope_to_research_and_manual_ingestion(self):
        attempt_id = "c" * 64
        internal = {
            _INTERNAL_ATTEMPT_ID: attempt_id,
            _INTERNAL_ATTEMPT_ORDER_NS: 123456,
        }
        with patch(
            "pipelines.research_pipeline",
            AsyncMock(return_value={"query": "q"}),
        ) as research:
            await dispatch_job(
                "research_web",
                {"query": "q", "research_run_id": "run", **internal},
            )
        self.assertEqual(research.await_args.kwargs["ingestion_attempt_id"], attempt_id)
        self.assertEqual(research.await_args.kwargs["ingestion_order_ns"], 123456)

        with patch(
            "shared.rag_ingest_impl",
            AsyncMock(return_value={"stored": 1}),
        ) as ingest:
            await dispatch_job(
                "ingest_text",
                {"text": "content", "source": "manual", **internal},
            )
        metadata = ingest.await_args.args[0].metadata
        self.assertEqual(metadata["ingestion_attempt_id"], attempt_id)
        self.assertEqual(metadata["ingestion_order_ns"], 123456)

    async def test_investigation_ingests_redirected_source_identity(self):
        requested_url = "https://start.example/path"
        final_url = "https://docs.example/final"
        with (
            patch(
                "pipelines.explore_url_pipeline",
                AsyncMock(
                    return_value={
                        "full_text_preview": "redirected content",
                        "final_url": final_url,
                        "title": "Final page",
                    }
                ),
            ),
            patch(
                "shared.rag_ingest_impl",
                AsyncMock(return_value={"stored": 1}),
            ) as ingest,
        ):
            result = await dispatch_job(
                "investigate_url",
                {
                    "url": requested_url,
                    "task": "find details",
                    "auto_ingest": True,
                },
            )

        metadata = ingest.await_args.args[0].metadata
        self.assertEqual(metadata["source"], final_url)
        self.assertEqual(metadata["url"], final_url)
        self.assertEqual(metadata["requested_url"], requested_url)
        self.assertEqual(metadata["domain"], "docs.example")
        self.assertEqual(result["stored_chunks"], 1)

    async def test_authenticated_job_binds_artifact_owner_before_dispatch(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {
                "job_id": job_id,
                "kind": "research_web",
                "owner_id": "client-a",
                "payload": {"query": "q"},
            }
        )

        async def dispatch(kind, payload):
            self.assertEqual(await self.artifacts.owner_principal(job_id), "client-a")
            return {"query": payload["query"]}

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)

        self.assertIsNotNone(store.completed)
        self.assertIsNone(store.failed)

    async def test_authenticated_job_fails_closed_when_owner_binding_fails(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {
                "job_id": job_id,
                "kind": "research_web",
                "owner_id": "client-a",
                "payload": {"query": "q"},
            }
        )
        self.artifacts.bind_owner_principal = AsyncMock(
            side_effect=OSError("read only")
        )
        dispatch = AsyncMock()
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)

        dispatch.assert_not_awaited()
        self.assertEqual(store.failed[1]["type"], "ArtifactOwnershipError")

    async def test_dispatch_never_starts_when_compensation_registration_fails(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}}
        )
        store.registration_error = OSError("redis unavailable")
        dispatch = AsyncMock()
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )

        await worker.run_once(timeout=0.01)

        dispatch.assert_not_awaited()
        self.assertEqual(store.failed[1]["type"], "IngestionCompensationError")

    async def test_failed_invalidation_is_replayed_and_acknowledged_later(self):
        job_id = uuid.uuid4().hex
        attempt_id = "a" * 64
        store = FakeWorkerStore(None)
        await store.register_ingestion_invalidation(
            job_id,
            attempt_id,
            lease_token="b" * 32,
        )
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        self.invalidate_attempt.side_effect = RuntimeError("qdrant unavailable")

        self.assertFalse(
            await worker._invalidate_ingestion_attempt(
                job_id,
                attempt_id,
                reason="job_failed",
            )
        )
        self.assertEqual(self.invalidate_attempt.await_count, 3)
        self.assertIn(attempt_id, store.registered_invalidations)
        self.assertEqual(len(store.due_invalidations), 1)

        self.invalidate_attempt.reset_mock(side_effect=True)
        self.invalidate_attempt.return_value = {
            "invalidated": 2,
            "sources_reconciled": 1,
        }
        await worker._maybe_replay_ingestion_invalidations(force=True)

        self.invalidate_attempt.assert_awaited_once()
        self.assertIn(attempt_id, store.acknowledged_invalidations)
        self.assertNotIn(attempt_id, store.registered_invalidations)

    async def test_cancellation_polling_cancels_active_dispatch(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}},
            cancel_after_checks=1,
        )

        async def dispatch(kind, payload):
            await asyncio.sleep(10)

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)
        self.assertEqual(store.cancelled[0], job_id)
        self.assertIsNone(store.completed)
        self.invalidate_attempt.assert_awaited_once()
        self.assertEqual(
            self.invalidate_attempt.await_args.kwargs["reason"],
            "job_cancelled",
        )

    async def test_remote_rag_cancellation_invalidates_entire_attempt_remotely(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}},
            cancel_after_checks=1,
        )

        async def dispatch(_kind, _payload):
            await asyncio.sleep(10)

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        with (
            patch.object(shared, "USE_RESEARCH_API_RAG", True),
            patch.object(
                shared,
                "_remote_rag_request",
                AsyncMock(return_value={"invalidated": 4, "sources_reconciled": 2}),
            ) as remote_request,
        ):
            await worker.run_once(timeout=0.01)

        remote_request.assert_awaited_once()
        self.assertEqual(
            remote_request.await_args.args[:2], ("POST", "/rag/invalidate-attempt")
        )
        body = remote_request.await_args.kwargs["json_body"]
        self.assertEqual(body["reason"], "job_cancelled")
        self.assertEqual(len(body["ingestion_attempt_id"]), 64)
        self.assertEqual(store.cancelled[0], job_id)

    async def test_dispatch_failure_is_recorded_without_crashing_worker(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}}
        )

        async def dispatch(kind, payload):
            raise RuntimeError("provider unavailable")

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)
        self.assertEqual(store.failed[1]["type"], "RuntimeError")
        self.assertIn("provider unavailable", store.failed[1]["message"])
        self.invalidate_attempt.assert_awaited_once()
        self.assertEqual(
            self.invalidate_attempt.await_args.kwargs["reason"],
            "job_failed",
        )

    async def test_cancellation_requested_after_dispatch_failure_wins(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}},
            cancel_after_checks=2,
        )

        async def dispatch(kind, payload):
            raise RuntimeError("late provider failure")

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)

        self.assertEqual(store.cancelled[0], job_id)
        self.assertIsNone(store.failed)

    async def test_dispatch_failure_redacts_before_truncating(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}}
        )
        private_key = (
            "-----BEGIN PRIVATE KEY-----\n"
            + ("A" * 5000)
            + "\n-----END PRIVATE KEY-----"
        )

        async def dispatch(kind, payload):
            raise RuntimeError(private_key)

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)

        self.assertEqual(store.failed[1]["message"], "[REDACTED_PRIVATE_KEY]")

    async def test_artifact_cleanup_runs_on_retention_schedule(self):
        store = FakeWorkerStore(None)
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        worker.artifact_retention_seconds = 123
        worker.artifact_cleanup_interval = 60
        worker.artifacts.prune_older_than = AsyncMock(return_value=0)

        with patch("worker.time.monotonic", side_effect=[100.0, 120.0, 161.0]):
            await worker._maybe_prune_artifacts()
            await worker._maybe_prune_artifacts()
            await worker._maybe_prune_artifacts()

        self.assertEqual(worker.artifacts.prune_older_than.await_count, 2)
        worker.artifacts.prune_older_than.assert_awaited_with(
            123,
            protected_owner_ids=set(),
        )

    async def test_artifact_cleanup_protects_active_job_owners(self):
        protected_job_id = uuid.uuid4().hex
        store = FakeWorkerStore(None)
        store.protected_job_ids = {protected_job_id}
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        worker.artifact_retention_seconds = 123
        worker.artifacts.prune_older_than = AsyncMock(return_value=0)

        await worker._maybe_prune_artifacts()

        worker.artifacts.prune_older_than.assert_awaited_once_with(
            123,
            protected_owner_ids={protected_job_id},
        )

    async def test_result_ttl_must_cover_artifact_retention(self):
        store = FakeWorkerStore(None)
        store.result_ttl_seconds = 119
        with patch.dict("worker.os.environ", {"ARTIFACT_RETENTION_SECONDS": "120"}):
            with self.assertRaisesRegex(ValueError, "JOB_RESULT_TTL_SECONDS"):
                JobWorker(
                    store=store,
                    artifacts=self.artifacts,
                    worker_id="test-worker",
                )

        store.result_ttl_seconds = 0
        with patch.dict("worker.os.environ", {"ARTIFACT_RETENTION_SECONDS": "120"}):
            JobWorker(
                store=store,
                artifacts=self.artifacts,
                worker_id="test-worker",
            )

    async def test_lease_loss_cancels_dispatch_without_terminal_write(self):
        job_id = uuid.uuid4().hex
        store = FakeWorkerStore(
            {"job_id": job_id, "kind": "research_web", "payload": {"query": "q"}}
        )
        store.lease_valid = False
        dispatch_cancelled = asyncio.Event()

        async def dispatch(kind, payload):
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                dispatch_cancelled.set()
                raise

        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            dispatcher=dispatch,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        await worker.run_once(timeout=0.01)

        self.assertTrue(dispatch_cancelled.is_set())
        self.assertIsNone(store.completed)
        self.assertIsNone(store.failed)
        self.assertIsNone(store.cancelled)
        self.invalidate_attempt.assert_awaited_once()
        self.assertEqual(
            self.invalidate_attempt.await_args.kwargs["reason"],
            "worker_lease_lost",
        )

    async def test_stale_recovery_runs_on_periodic_schedule(self):
        store = FakeWorkerStore(None)
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        worker.stale_recovery_interval = 30

        with patch("worker.time.monotonic", side_effect=[100.0, 110.0, 131.0]):
            await worker._maybe_recover_stale_jobs()
            await worker._maybe_recover_stale_jobs()
            await worker._maybe_recover_stale_jobs()

        self.assertEqual(store.stale_recovery_calls, 2)

    async def test_qdrant_lifecycle_repair_runs_periodically_and_failure_is_nonfatal(
        self,
    ):
        store = FakeWorkerStore(None)
        worker = JobWorker(
            store=store,
            artifacts=self.artifacts,
            worker_id="test-worker",
            poll_interval=0.01,
        )
        worker.qdrant_lifecycle_interval = 30

        with patch(
            "shared.repair_qdrant_lifecycle_async",
            AsyncMock(
                side_effect=[
                    RuntimeError("temporarily unavailable"),
                    {"sources_reconciled": 0, "history_cleanup": {"deleted": 0}},
                ]
            ),
        ) as repair:
            with patch("worker.time.monotonic", side_effect=[100.0, 110.0, 131.0]):
                await worker._maybe_repair_qdrant_lifecycle()
                await worker._maybe_repair_qdrant_lifecycle()
                await worker._maybe_repair_qdrant_lifecycle()

        self.assertEqual(repair.await_count, 2)

    async def test_qdrant_lifecycle_repair_resumes_scan_cursors(self):
        worker = JobWorker(
            store=FakeWorkerStore(None),
            artifacts=self.artifacts,
            worker_id="test-worker",
        )
        worker.qdrant_lifecycle_interval = 30
        worker.qdrant_lifecycle_max_points = 7
        first_result = {
            "sources_reconciled": 1,
            "next_cursor": "lifecycle-next",
            "history_cleanup": {"deleted": 0, "next_cursor": 42},
        }
        second_result = {
            "sources_reconciled": 0,
            "next_cursor": None,
            "history_cleanup": {"deleted": 0, "next_cursor": None},
        }

        with patch(
            "shared.repair_qdrant_lifecycle_async",
            AsyncMock(side_effect=[first_result, second_result]),
        ) as repair:
            with patch("worker.time.monotonic", side_effect=[100.0, 131.0]):
                await worker._maybe_repair_qdrant_lifecycle()
                await worker._maybe_repair_qdrant_lifecycle()

        self.assertEqual(repair.await_count, 2)
        self.assertEqual(repair.await_args_list[0].kwargs["max_points"], 7)
        self.assertIsNone(repair.await_args_list[0].kwargs["cursor"])
        self.assertEqual(
            repair.await_args_list[1].kwargs["cursor"],
            "lifecycle-next",
        )
        self.assertEqual(repair.await_args_list[1].kwargs["history_cursor"], 42)
        self.assertIsNone(worker._qdrant_lifecycle_cursor)
        self.assertIsNone(worker._qdrant_history_cursor)

    async def test_forced_startup_lifecycle_repair_fails_closed(self):
        worker = JobWorker(
            store=FakeWorkerStore(None),
            artifacts=self.artifacts,
            worker_id="test-worker",
        )
        with patch(
            "shared.repair_qdrant_lifecycle_async",
            AsyncMock(side_effect=RuntimeError("qdrant unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "qdrant unavailable"):
                await worker._maybe_repair_qdrant_lifecycle(force=True)


if __name__ == "__main__":
    unittest.main()
