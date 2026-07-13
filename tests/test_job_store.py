import asyncio
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis as fakeredis

import job_store
from job_store import (
    CANCELLED,
    FAILED,
    RUNNING,
    SUCCEEDED,
    InvalidJobError,
    JobLeaseLostError,
    JobQueueFullError,
    JobStoreError,
    RedisJobStore,
)


class RedisJobStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.redis = fakeredis.FakeRedis(decode_responses=True)
        self.store = RedisJobStore(
            redis_client=self.redis,
            queue_name="test:jobs",
            result_ttl_seconds=120,
            ingestion_waitaof_timeout_ms=0,
        )

    async def asyncTearDown(self):
        await self.redis.aclose()

    async def test_create_claim_complete_and_fetch_result(self):
        created = await self.store.create_job("research_web", {"query": "evidence"})
        self.assertEqual(created["status"], "queued")
        self.assertNotIn("payload", created)

        claimed = await self.store.claim_job(worker_id="worker-1")
        self.assertEqual(claimed["status"], RUNNING)
        self.assertEqual(claimed["payload"], {"query": "evidence"})
        self.assertRegex(claimed["lease_token"], r"^[a-f0-9]{32}$")
        self.assertNotIn("lease_token", await self.store.get_status(created["job_id"]))

        metadata = {"artifact": {"relative_path": "result.json"}, "results_count": 3}
        await self.store.complete_job(
            created["job_id"],
            metadata,
            lease_token=claimed["lease_token"],
        )
        status = await self.store.get_status(created["job_id"])
        result = await self.store.get_result(created["job_id"])
        self.assertEqual(status["status"], SUCCEEDED)
        self.assertNotIn("result", status)
        self.assertEqual(result["result"], metadata)
        self.assertGreater(
            await self.redis.ttl(self.store._job_key(created["job_id"])), 0
        )
        self.assertEqual(await self.redis.lrange(self.store.processing_key, 0, -1), [])

    async def test_active_coalescing_is_opt_in_and_canonical(self):
        first_default = await self.store.create_job(
            "query_memory",
            {"query": "same"},
            owner_id="client-a",
        )
        second_default = await self.store.create_job(
            "query_memory",
            {"query": "same"},
            owner_id="client-a",
        )
        self.assertNotEqual(first_default["job_id"], second_default["job_id"])

        first, second = await asyncio.gather(
            self.store.create_job(
                "research_web",
                {"query": "evidence", "options": {"mode": "deep", "verify": True}},
                owner_id="client-a",
                coalesce_active=True,
            ),
            self.store.create_job(
                "research_web",
                {"options": {"verify": True, "mode": "deep"}, "query": "evidence"},
                owner_id="client-a",
                coalesce_active=True,
            ),
        )

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertEqual({first["coalesced"], second["coalesced"]}, {False, True})
        queued = await self.redis.lrange(self.store.queue_key, 0, -1)
        self.assertEqual(queued.count(first["job_id"]), 1)

    async def test_active_coalescing_is_scoped_by_owner_kind_and_payload(self):
        jobs = [
            await self.store.create_job(
                "research_web",
                {"query": "same"},
                owner_id="client-a",
                coalesce_active=True,
            ),
            await self.store.create_job(
                "research_web",
                {"query": "same"},
                owner_id="client-b",
                coalesce_active=True,
            ),
            await self.store.create_job(
                "investigate_url",
                {"query": "same"},
                owner_id="client-a",
                coalesce_active=True,
            ),
            await self.store.create_job(
                "research_web",
                {"query": "different"},
                owner_id="client-a",
                coalesce_active=True,
            ),
        ]

        self.assertEqual(len({job["job_id"] for job in jobs}), len(jobs))
        self.assertEqual(await self.redis.llen(self.store.queue_key), len(jobs))

    async def test_queue_full_allows_joining_an_identical_active_job(self):
        self.store.max_queued_jobs = 1
        first = await self.store.create_job(
            "research_web",
            {"query": "same"},
            owner_id="client-a",
            coalesce_active=True,
        )
        joined = await self.store.create_job(
            "research_web",
            {"query": "same"},
            owner_id="client-a",
            coalesce_active=True,
        )

        self.assertEqual(joined["job_id"], first["job_id"])
        self.assertTrue(joined["coalesced"])
        with self.assertRaises(JobQueueFullError):
            await self.store.create_job(
                "research_web",
                {"query": "different"},
                owner_id="client-a",
                coalesce_active=True,
            )

    async def test_cancelling_shared_job_does_not_delete_replacement_index(self):
        payload = {"query": "retry"}
        fingerprint = job_store._coalescing_fingerprint(
            "research_web",
            job_store._json_dumps(payload),
            "client-a",
        )
        active_key = self.store._active_job_key(fingerprint)
        first = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        claimed = await self.store.claim_job(worker_id="worker-1")
        joined = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertEqual(joined["job_id"], first["job_id"])

        await self.store.request_cancellation(first["job_id"])
        replacement = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertNotEqual(replacement["job_id"], first["job_id"])

        await self.store.mark_cancelled(
            first["job_id"],
            lease_token=claimed["lease_token"],
        )
        self.assertEqual(await self.redis.get(active_key), replacement["job_id"])
        self.assertGreater(await self.redis.ttl(active_key), 0)
        replacement_join = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertEqual(replacement_join["job_id"], replacement["job_id"])
        self.assertTrue(replacement_join["coalesced"])

    async def test_terminal_and_stale_jobs_release_the_active_index(self):
        completed = await self.store.create_job(
            "query_memory",
            {"query": "complete"},
            owner_id="client-a",
            coalesce_active=True,
        )
        completed_claim = await self.store.claim_job(worker_id="worker-1")
        await self.store.complete_job(
            completed["job_id"],
            {"count": 1},
            lease_token=completed_claim["lease_token"],
        )
        completed_replacement = await self.store.create_job(
            "query_memory",
            {"query": "complete"},
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertNotEqual(completed_replacement["job_id"], completed["job_id"])
        await self.store.request_cancellation(completed_replacement["job_id"])

        self.store.max_attempts = 1
        stale = await self.store.create_job(
            "research_web",
            {"query": "stale"},
            owner_id="client-a",
            coalesce_active=True,
        )
        await self.store.claim_job(worker_id="dead-worker")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(self.store._job_key(stale["job_id"]), "heartbeat_at", old)
        self.assertEqual(await self.store.requeue_stale_jobs(60), 0)
        stale_replacement = await self.store.create_job(
            "research_web",
            {"query": "stale"},
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertNotEqual(stale_replacement["job_id"], stale["job_id"])

    async def test_stale_or_malformed_active_index_self_heals_without_leaking_metadata(
        self,
    ):
        first = await self.store.create_job(
            "research_web",
            {"query": "private search"},
            owner_id="client-a",
            coalesce_active=True,
        )
        fingerprint = job_store._coalescing_fingerprint(
            "research_web",
            job_store._json_dumps({"query": "private search"}),
            "client-a",
        )
        active_key = self.store._active_job_key(fingerprint)
        await self.redis.set(active_key, "not-a-job-id")

        replacement = await self.store.create_job(
            "research_web",
            {"query": "private search"},
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertNotEqual(replacement["job_id"], first["job_id"])
        self.assertEqual(await self.redis.get(active_key), replacement["job_id"])
        self.assertNotIn("coalesce_fingerprint", replacement)
        self.assertNotIn(
            "coalesce_fingerprint",
            await self.store.get_status(replacement["job_id"]),
        )

    async def test_missing_job_hash_leaves_only_a_bounded_self_healing_active_index(
        self,
    ):
        payload = {"query": "recover missing hash"}
        first = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        fingerprint = job_store._coalescing_fingerprint(
            "research_web",
            job_store._json_dumps(payload),
            "client-a",
        )
        active_key = self.store._active_job_key(fingerprint)
        initial_ttl = await self.redis.ttl(active_key)
        self.assertGreater(initial_ttl, 0)
        self.assertLessEqual(initial_ttl, self.store.active_job_index_ttl_seconds)

        await self.redis.delete(self.store._job_key(first["job_id"]))
        self.assertIsNone(await self.store.claim_job(worker_id="worker-1"))
        self.assertEqual(await self.redis.get(active_key), first["job_id"])
        self.assertGreater(await self.redis.ttl(active_key), 0)

        replacement = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertNotEqual(replacement["job_id"], first["job_id"])
        self.assertEqual(await self.redis.get(active_key), replacement["job_id"])
        self.assertGreater(await self.redis.ttl(active_key), 0)

    async def test_active_index_ttl_refreshes_through_worker_lifecycle(self):
        payload = {"query": "long-running"}
        created = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        fingerprint = job_store._coalescing_fingerprint(
            "research_web",
            job_store._json_dumps(payload),
            "client-a",
        )
        active_key = self.store._active_job_key(fingerprint)

        await self.redis.expire(active_key, 1)
        joined = await self.store.create_job(
            "research_web",
            payload,
            owner_id="client-a",
            coalesce_active=True,
        )
        self.assertEqual(joined["job_id"], created["job_id"])
        self.assertTrue(joined["coalesced"])
        self.assertGreater(await self.redis.ttl(active_key), 1)

        await self.redis.expire(active_key, 1)
        claimed = await self.store.claim_job(worker_id="worker-1")
        self.assertEqual(claimed["job_id"], created["job_id"])
        self.assertGreater(await self.redis.ttl(active_key), 1)

        await self.redis.expire(active_key, 1)
        self.assertTrue(
            await self.store.heartbeat_job(
                created["job_id"],
                "worker-1",
                claimed["lease_token"],
            )
        )
        self.assertGreater(await self.redis.ttl(active_key), 1)

        await self.redis.expire(active_key, 1)
        self.assertTrue(
            await self.store.requeue_job(
                created["job_id"],
                lease_token=claimed["lease_token"],
            )
        )
        self.assertGreater(await self.redis.ttl(active_key), 1)

    async def test_active_index_is_finite_when_result_expiration_is_disabled(self):
        store = RedisJobStore(
            redis_client=self.redis,
            queue_name="test:no-result-expiry",
            result_ttl_seconds=0,
            ingestion_waitaof_timeout_ms=0,
        )
        payload = {"query": "bounded index"}
        created = await store.create_job(
            "research_web",
            payload,
            coalesce_active=True,
        )
        fingerprint = job_store._coalescing_fingerprint(
            "research_web",
            job_store._json_dumps(payload),
            None,
        )
        active_key = store._active_job_key(fingerprint)

        self.assertEqual(await self.redis.get(active_key), created["job_id"])
        self.assertGreater(await self.redis.ttl(active_key), 0)

    async def test_enqueue_job_enables_active_coalescing_by_default(self):
        with patch.object(job_store, "get_job_store", return_value=self.store):
            first = await job_store.enqueue_job(
                "research_web",
                {"query": "queued"},
                owner_id="client-a",
            )
            second = await job_store.enqueue_job(
                "research_web",
                {"query": "queued"},
                owner_id="client-a",
            )

        self.assertEqual(first["job_id"], second["job_id"])
        self.assertFalse(first["coalesced"])
        self.assertTrue(second["coalesced"])

    async def test_cancelling_a_queued_job_removes_it_without_worker_claim(self):
        created = await self.store.create_job("query_memory", {"query": "cached"})
        status = await self.store.request_cancellation(created["job_id"])

        self.assertEqual(status["status"], CANCELLED)
        self.assertTrue(status["cancel_requested"])
        self.assertEqual(await self.redis.lrange(self.store.queue_key, 0, -1), [])

    async def test_cancelling_a_running_job_sets_pollable_flag(self):
        created = await self.store.create_job("research_web", {"query": "slow"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        status = await self.store.request_cancellation(created["job_id"])

        self.assertEqual(status["status"], RUNNING)
        self.assertTrue(await self.store.is_cancellation_requested(created["job_id"]))

        requeued = await self.store.requeue_job(
            created["job_id"],
            reason="worker interrupted",
            lease_token=claimed["lease_token"],
        )
        self.assertFalse(requeued)
        self.assertEqual(
            (await self.store.get_status(created["job_id"]))["status"], CANCELLED
        )
        self.assertEqual(await self.redis.lrange(self.store.queue_key, 0, -1), [])

    async def test_cancellation_wins_atomic_race_with_worker_failure(self):
        created = await self.store.create_job("research_web", {"query": "slow"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        await self.store.request_cancellation(created["job_id"])

        await self.store.fail_job(
            created["job_id"],
            {"type": "RuntimeError", "message": "late failure"},
            lease_token=claimed["lease_token"],
        )

        result = await self.store.get_result(created["job_id"])
        self.assertEqual(result["status"], CANCELLED)
        self.assertNotIn("error", result)

    async def test_ingestion_registration_requires_current_lease_and_is_armed(self):
        created = await self.store.create_job("research_web", {"query": "durable"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        attempt_id = "a" * 64

        with self.assertRaises(JobLeaseLostError):
            await self.store.register_ingestion_invalidation(
                created["job_id"],
                attempt_id,
                lease_token="f" * 32,
            )
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )

        self.assertIsNotNone(
            await self.redis.hget(
                self.store.ingestion_invalidation_payload_key,
                attempt_id,
            )
        )
        self.assertIsNone(
            await self.redis.zscore(
                self.store.ingestion_invalidation_due_key,
                attempt_id,
            )
        )

    async def test_ingestion_registration_waits_for_local_aof_fsync(self):
        created = await self.store.create_job("research_web", {"query": "fsync"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        self.store.ingestion_waitaof_timeout_ms = 123

        with (
            patch.object(
                fakeredis.FakeRedis,
                "eval",
                new=AsyncMock(return_value=1),
            ) as register,
            patch.object(
                fakeredis.FakeRedis,
                "waitaof",
                new=AsyncMock(return_value=[1, 0]),
            ) as waitaof,
        ):
            await self.store.register_ingestion_invalidation(
                created["job_id"],
                "b" * 64,
                lease_token=claimed["lease_token"],
            )

        register.assert_awaited_once()
        waitaof.assert_awaited_once_with(1, 0, 123)

    async def test_success_atomically_clears_ingestion_compensation(self):
        created = await self.store.create_job("research_web", {"query": "success"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        attempt_id = "c" * 64
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )

        await self.store.complete_job(
            created["job_id"],
            {"stored": 1},
            lease_token=claimed["lease_token"],
            successful_ingestion_attempt_id=attempt_id,
        )

        self.assertIsNone(
            await self.redis.hget(
                self.store.ingestion_invalidation_payload_key,
                attempt_id,
            )
        )
        self.assertIsNone(
            await self.redis.zscore(
                self.store.ingestion_invalidation_due_key,
                attempt_id,
            )
        )
        record = await self.redis.hgetall(self.store._job_key(created["job_id"]))
        self.assertNotIn("ingestion_attempt_id", record)

    async def test_cancellation_winning_completion_keeps_compensation_due(self):
        created = await self.store.create_job("research_web", {"query": "cancel"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        attempt_id = "d" * 64
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )
        await self.store.request_cancellation(created["job_id"])

        await self.store.complete_job(
            created["job_id"],
            {"stored": 1},
            lease_token=claimed["lease_token"],
            successful_ingestion_attempt_id=attempt_id,
        )

        self.assertEqual(
            (await self.store.get_result(created["job_id"]))["status"],
            CANCELLED,
        )
        self.assertIsNotNone(
            await self.redis.hget(
                self.store.ingestion_invalidation_payload_key,
                attempt_id,
            )
        )
        self.assertLessEqual(
            await self.redis.zscore(
                self.store.ingestion_invalidation_due_key,
                attempt_id,
            ),
            datetime.now(timezone.utc).timestamp(),
        )

    async def test_active_compensation_claim_defers_instead_of_invalidating_success(
        self,
    ):
        created = await self.store.create_job("research_web", {"query": "race"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        attempt_id = "e" * 64
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )
        await self.store.schedule_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            reason="test_race",
        )

        pending = await self.store.claim_due_ingestion_invalidations(
            lease_seconds=600,
        )
        self.assertEqual(pending, [])
        await self.store.complete_job(
            created["job_id"],
            {"stored": 1},
            lease_token=claimed["lease_token"],
            successful_ingestion_attempt_id=attempt_id,
        )
        self.assertIsNone(
            await self.redis.zscore(
                self.store.ingestion_invalidation_due_key,
                attempt_id,
            )
        )

    async def test_reschedule_after_success_cannot_resurrect_compensation(self):
        created = await self.store.create_job("research_web", {"query": "done"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        attempt_id = "f" * 64
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )
        await self.store.complete_job(
            created["job_id"],
            {"stored": 1},
            lease_token=claimed["lease_token"],
            successful_ingestion_attempt_id=attempt_id,
        )

        self.assertFalse(
            await self.store.schedule_ingestion_invalidation(
                created["job_id"],
                attempt_id,
                reason="late_failure",
            )
        )
        await self.store.defer_ingestion_invalidation(
            attempt_id,
            delay_seconds=1,
        )
        self.assertIsNone(
            await self.redis.zscore(
                self.store.ingestion_invalidation_due_key,
                attempt_id,
            )
        )

    async def test_stale_requeue_makes_old_ingestion_attempt_replayable(self):
        created = await self.store.create_job("research_web", {"query": "stale"})
        claimed = await self.store.claim_job(worker_id="dead-worker")
        attempt_id = "1" * 64
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(
            self.store._job_key(created["job_id"]),
            mapping={"heartbeat_at": old},
        )

        self.assertEqual(await self.store.requeue_stale_jobs(60), 1)
        pending = await self.store.claim_due_ingestion_invalidations()
        self.assertEqual(
            [item["ingestion_attempt_id"] for item in pending], [attempt_id]
        )
        record = await self.redis.hgetall(self.store._job_key(created["job_id"]))
        self.assertNotIn("ingestion_attempt_id", record)

    async def test_outbox_survives_job_expiry_and_concurrent_replay_claims_once(self):
        created = await self.store.create_job("research_web", {"query": "retry"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        attempt_id = "2" * 64
        await self.store.register_ingestion_invalidation(
            created["job_id"],
            attempt_id,
            lease_token=claimed["lease_token"],
        )
        await self.store.fail_job(
            created["job_id"],
            {"message": "failed"},
            lease_token=claimed["lease_token"],
        )
        await self.redis.delete(self.store._job_key(created["job_id"]))

        claims = await asyncio.gather(
            self.store.claim_due_ingestion_invalidations(),
            self.store.claim_due_ingestion_invalidations(),
        )
        self.assertEqual(sum(len(batch) for batch in claims), 1)

    async def test_stale_processing_job_is_requeued(self):
        created = await self.store.create_job("research_web", {"query": "stale"})
        await self.store.claim_job(worker_id="dead-worker")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(
            self.store._job_key(created["job_id"]), mapping={"heartbeat_at": old}
        )

        count = await self.store.requeue_stale_jobs(stale_after_seconds=60)
        self.assertEqual(count, 1)
        self.assertEqual(
            (await self.store.get_status(created["job_id"]))["status"], "queued"
        )

    async def test_stale_job_fails_after_max_attempts(self):
        self.store.max_attempts = 2
        created = await self.store.create_job("research_web", {"query": "poison"})
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

        await self.store.claim_job(worker_id="worker-1")
        await self.redis.hset(
            self.store._job_key(created["job_id"]), "heartbeat_at", old
        )
        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 1)

        await self.store.claim_job(worker_id="worker-2")
        await self.redis.hset(
            self.store._job_key(created["job_id"]), "heartbeat_at", old
        )
        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 0)

        result = await self.store.get_result(created["job_id"])
        self.assertEqual(result["status"], FAILED)
        self.assertEqual(result["error"]["type"], "JobAttemptsExhausted")
        self.assertIn("JOB_MAX_ATTEMPTS=2", result["error"]["message"])
        self.assertLessEqual(len(result["error"]["message"]), 1000)
        self.assertEqual(await self.redis.lrange(self.store.queue_key, 0, -1), [])
        self.assertEqual(await self.redis.lrange(self.store.processing_key, 0, -1), [])
        self.assertGreater(
            await self.redis.ttl(self.store._job_key(created["job_id"])), 0
        )

    async def test_max_attempts_is_loaded_from_environment_with_safe_minimum(self):
        with patch.dict("job_store.os.environ", {"JOB_MAX_ATTEMPTS": "7"}):
            configured = RedisJobStore(
                redis_client=self.redis,
                queue_name="test:configured-attempts",
            )
        self.assertEqual(configured.max_attempts, 7)

        with patch.dict("job_store.os.environ", {"JOB_MAX_ATTEMPTS": "0"}):
            clamped = RedisJobStore(
                redis_client=self.redis,
                queue_name="test:clamped-attempts",
            )
        self.assertEqual(clamped.max_attempts, 1)

    async def test_active_job_ids_snapshots_queue_and_processing_with_safety_bound(
        self,
    ):
        first = await self.store.create_job("research_web", {"query": "running"})
        second = await self.store.create_job("research_web", {"query": "queued"})
        await self.store.claim_job(worker_id="worker-1")

        self.assertEqual(
            await self.store.active_job_ids(),
            {first["job_id"], second["job_id"]},
        )
        with self.assertRaises(JobStoreError):
            await self.store.active_job_ids(limit=1)

    async def test_active_artifact_owner_ids_include_cross_job_payload_dependencies(
        self,
    ):
        parent_id = uuid.uuid4().hex
        artifact_owner_id = uuid.uuid4().hex
        additional_owner_id = uuid.uuid4().hex
        child = await self.store.create_job(
            "persist_research_source",
            {
                "parent_job_id": parent_id,
                "artifact_owner_id": artifact_owner_id,
                "artifact_owner_ids": [additional_owner_id, "not-a-job-id"],
            },
        )

        self.assertEqual(
            await self.store.active_artifact_owner_ids(),
            {
                child["job_id"],
                parent_id,
                artifact_owner_id,
                additional_owner_id,
            },
        )

    async def test_complete_job_with_children_atomically_succeeds(self):
        parent = await self.store.create_job("research_web", {"query": "evidence"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        child_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        child_queue = "test:persistence"

        completion = await self.store.complete_job_with_children(
            parent["job_id"],
            {"artifact_id": f"{parent['job_id']}:result"},
            lease_token=claimed["lease_token"],
            child_queue_name=child_queue,
            child_jobs=[
                {
                    "job_id": child_id,
                    "kind": "persist_research_source",
                    "payload": {
                        "parent_job_id": parent["job_id"],
                        "artifact_owner_id": child_id,
                    },
                    "owner_id": "client-a",
                }
                for child_id in child_ids
            ],
        )

        self.assertEqual(completion["status"], SUCCEEDED)
        self.assertEqual(
            {child["job_id"] for child in completion["children"]},
            set(child_ids),
        )
        self.assertEqual(
            (await self.store.get_result(parent["job_id"]))["result"],
            {"artifact_id": f"{parent['job_id']}:result"},
        )
        self.assertEqual(await self.redis.lrange(self.store.processing_key, 0, -1), [])
        self.assertEqual(
            set(await self.redis.lrange(child_queue, 0, -1)),
            set(child_ids),
        )
        child_store = RedisJobStore(
            redis_client=self.redis,
            queue_name=child_queue,
            ingestion_waitaof_timeout_ms=0,
        )
        child_statuses = [
            await child_store.get_status(child_id) for child_id in child_ids
        ]
        self.assertTrue(all(status["status"] == "queued" for status in child_statuses))
        self.assertTrue(
            all(status["owner_id"] == "client-a" for status in child_statuses)
        )

    async def test_complete_job_with_children_cancellation_wins(self):
        parent = await self.store.create_job("research_web", {"query": "cancel"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        await self.store.request_cancellation(parent["job_id"])
        child_id = uuid.uuid4().hex
        child_queue = "test:persistence"

        completion = await self.store.complete_job_with_children(
            parent["job_id"],
            {"artifact_id": f"{parent['job_id']}:result"},
            lease_token=claimed["lease_token"],
            child_queue_name=child_queue,
            child_jobs=[
                {
                    "job_id": child_id,
                    "kind": "persist_research_source",
                    "payload": {"artifact_owner_id": child_id},
                }
            ],
        )

        self.assertEqual(completion, {"status": CANCELLED, "children": []})
        self.assertEqual(
            (await self.store.get_result(parent["job_id"]))["status"],
            CANCELLED,
        )
        self.assertEqual(await self.redis.lrange(child_queue, 0, -1), [])
        self.assertFalse(await self.redis.exists(f"{child_queue}:job:{child_id}"))

    async def test_complete_job_with_children_rejects_stale_lease(self):
        parent = await self.store.create_job("research_web", {"query": "lease"})
        await self.store.claim_job(worker_id="worker-1")
        child_id = uuid.uuid4().hex
        child_queue = "test:persistence"

        with self.assertRaises(JobLeaseLostError):
            await self.store.complete_job_with_children(
                parent["job_id"],
                {"artifact_id": f"{parent['job_id']}:result"},
                lease_token="f" * 32,
                child_queue_name=child_queue,
                child_jobs=[
                    {
                        "job_id": child_id,
                        "kind": "persist_research_source",
                        "payload": {"artifact_owner_id": child_id},
                    }
                ],
            )

        self.assertEqual(
            (await self.store.get_status(parent["job_id"]))["status"],
            RUNNING,
        )
        self.assertEqual(await self.redis.lrange(child_queue, 0, -1), [])
        self.assertFalse(await self.redis.exists(f"{child_queue}:job:{child_id}"))

    async def test_complete_job_with_children_queue_full_leaves_parent_running(self):
        self.store.max_queued_jobs = 1
        parent = await self.store.create_job("research_web", {"query": "capacity"})
        claimed = await self.store.claim_job(worker_id="worker-1")
        child_id = uuid.uuid4().hex
        child_queue = "test:persistence"
        await self.redis.lpush(child_queue, uuid.uuid4().hex)

        with self.assertRaises(JobQueueFullError):
            await self.store.complete_job_with_children(
                parent["job_id"],
                {"artifact_id": f"{parent['job_id']}:result"},
                lease_token=claimed["lease_token"],
                child_queue_name=child_queue,
                child_jobs=[
                    {
                        "job_id": child_id,
                        "kind": "persist_research_source",
                        "payload": {"artifact_owner_id": child_id},
                    }
                ],
            )

        self.assertEqual(
            (await self.store.get_status(parent["job_id"]))["status"],
            RUNNING,
        )
        self.assertEqual(await self.redis.llen(child_queue), 1)
        self.assertFalse(await self.redis.exists(f"{child_queue}:job:{child_id}"))

    async def test_concurrent_stale_recovery_enqueues_exactly_once(self):
        created = await self.store.create_job("research_web", {"query": "stale"})
        await self.store.claim_job(worker_id="dead-worker")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(
            self.store._job_key(created["job_id"]), mapping={"heartbeat_at": old}
        )

        counts = await asyncio.gather(
            self.store.requeue_stale_jobs(stale_after_seconds=60),
            self.store.requeue_stale_jobs(stale_after_seconds=60),
        )

        self.assertEqual(sum(counts), 1)
        queued = await self.redis.lrange(self.store.queue_key, 0, -1)
        self.assertEqual(len(queued), 1)
        self.assertEqual(await self.redis.lrange(self.store.processing_key, 0, -1), [])

    async def test_restart_before_stale_eventually_recovers_prelease_move(self):
        created = await self.store.create_job("research_web", {"query": "orphaned"})
        await self.redis.brpoplpush(
            self.store.queue_key, self.store.processing_key, timeout=1
        )

        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 0)
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(
            self.store._job_key(created["job_id"]), mapping={"updated_at": old}
        )
        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 1)
        self.assertEqual(
            (await self.store.get_status(created["job_id"]))["status"], "queued"
        )

    async def test_stale_worker_cannot_heartbeat_or_finish_new_attempt(self):
        created = await self.store.create_job("research_web", {"query": "lease"})
        first = await self.store.claim_job(worker_id="worker-1")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(
            self.store._job_key(created["job_id"]), mapping={"heartbeat_at": old}
        )
        await self.store.requeue_stale_jobs(stale_after_seconds=60)
        second = await self.store.claim_job(worker_id="worker-2")

        self.assertNotEqual(first["lease_token"], second["lease_token"])
        self.assertFalse(
            await self.store.heartbeat_job(
                created["job_id"],
                "worker-1",
                first["lease_token"],
            )
        )
        with self.assertRaises(JobLeaseLostError):
            await self.store.complete_job(
                created["job_id"],
                {"attempt": 1},
                lease_token=first["lease_token"],
            )
        await self.store.complete_job(
            created["job_id"],
            {"attempt": 2},
            lease_token=second["lease_token"],
        )
        self.assertEqual(
            (await self.store.get_result(created["job_id"]))["result"], {"attempt": 2}
        )

    async def test_duplicate_queue_entries_produce_only_one_claim(self):
        created = await self.store.create_job("research_web", {"query": "once"})
        await self.redis.lpush(self.store.queue_key, created["job_id"])

        claims = await asyncio.gather(
            self.store.claim_job(worker_id="worker-1"),
            self.store.claim_job(worker_id="worker-2"),
        )
        claimed = [job for job in claims if job is not None]

        self.assertEqual(len(claimed), 1)
        self.assertEqual(await self.redis.lrange(self.store.queue_key, 0, -1), [])
        self.assertEqual(
            len(await self.redis.lrange(self.store.processing_key, 0, -1)), 1
        )

    async def test_owner_id_is_exposed_but_lease_is_not(self):
        created = await self.store.create_job(
            "query_memory",
            {"query": "private"},
            owner_id="client-a",
        )
        claimed = await self.store.claim_job(worker_id="worker")
        status = await self.store.get_status(created["job_id"])

        self.assertEqual(status["owner_id"], "client-a")
        self.assertNotIn("lease_token", status)
        self.assertIn("lease_token", claimed)

    async def test_concurrent_admission_does_not_exceed_queue_limit(self):
        self.store.max_queued_jobs = 1
        results = await asyncio.gather(
            self.store.create_job("query_memory", {"query": "first"}),
            self.store.create_job("query_memory", {"query": "second"}),
            return_exceptions=True,
        )

        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        self.assertEqual(
            sum(isinstance(result, JobQueueFullError) for result in results), 1
        )
        self.assertEqual(await self.redis.llen(self.store.queue_key), 1)

    async def test_worker_heartbeats_are_isolated_by_worker_and_host(self):
        await self.store.record_worker_heartbeat("worker-a", host_id="host-a")
        await self.store.record_worker_heartbeat(
            "worker-b", state="busy", host_id="host-b"
        )

        worker_a = await self.store.get_worker_heartbeat(worker_id="worker-a")
        host_a = await self.store.get_worker_heartbeat(host_id="host-a")
        host_b = await self.store.get_worker_heartbeat(host_id="host-b")
        self.assertEqual(worker_a["worker_id"], "worker-a")
        self.assertEqual(host_a["worker_id"], "worker-a")
        self.assertEqual(host_b["worker_id"], "worker-b")
        self.assertEqual(host_b["state"], "busy")

    async def test_input_validation_rejects_unsafe_or_oversized_values(self):
        with self.assertRaises(InvalidJobError):
            await self.store.create_job("Bad Kind", {})
        with self.assertRaises(InvalidJobError):
            await self.store.get_status("../../escape")


if __name__ == "__main__":
    unittest.main()
