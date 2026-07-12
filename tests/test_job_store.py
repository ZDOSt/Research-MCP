import asyncio
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import fakeredis.aioredis as fakeredis

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
        self.assertGreater(await self.redis.ttl(self.store._job_key(created["job_id"])), 0)
        self.assertEqual(await self.redis.lrange(self.store.processing_key, 0, -1), [])

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
        self.assertEqual((await self.store.get_status(created["job_id"]))["status"], CANCELLED)
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

    async def test_active_compensation_claim_defers_instead_of_invalidating_success(self):
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
        self.assertEqual([item["ingestion_attempt_id"] for item in pending], [attempt_id])
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
        await self.redis.hset(self.store._job_key(created["job_id"]), mapping={"heartbeat_at": old})

        count = await self.store.requeue_stale_jobs(stale_after_seconds=60)
        self.assertEqual(count, 1)
        self.assertEqual((await self.store.get_status(created["job_id"]))["status"], "queued")

    async def test_stale_job_fails_after_max_attempts(self):
        self.store.max_attempts = 2
        created = await self.store.create_job("research_web", {"query": "poison"})
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

        await self.store.claim_job(worker_id="worker-1")
        await self.redis.hset(self.store._job_key(created["job_id"]), "heartbeat_at", old)
        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 1)

        await self.store.claim_job(worker_id="worker-2")
        await self.redis.hset(self.store._job_key(created["job_id"]), "heartbeat_at", old)
        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 0)

        result = await self.store.get_result(created["job_id"])
        self.assertEqual(result["status"], FAILED)
        self.assertEqual(result["error"]["type"], "JobAttemptsExhausted")
        self.assertIn("JOB_MAX_ATTEMPTS=2", result["error"]["message"])
        self.assertLessEqual(len(result["error"]["message"]), 1000)
        self.assertEqual(await self.redis.lrange(self.store.queue_key, 0, -1), [])
        self.assertEqual(await self.redis.lrange(self.store.processing_key, 0, -1), [])
        self.assertGreater(await self.redis.ttl(self.store._job_key(created["job_id"])), 0)

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

    async def test_active_job_ids_snapshots_queue_and_processing_with_safety_bound(self):
        first = await self.store.create_job("research_web", {"query": "running"})
        second = await self.store.create_job("research_web", {"query": "queued"})
        await self.store.claim_job(worker_id="worker-1")

        self.assertEqual(
            await self.store.active_job_ids(),
            {first["job_id"], second["job_id"]},
        )
        with self.assertRaises(JobStoreError):
            await self.store.active_job_ids(limit=1)

    async def test_concurrent_stale_recovery_enqueues_exactly_once(self):
        created = await self.store.create_job("research_web", {"query": "stale"})
        await self.store.claim_job(worker_id="dead-worker")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(self.store._job_key(created["job_id"]), mapping={"heartbeat_at": old})

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
        await self.redis.brpoplpush(self.store.queue_key, self.store.processing_key, timeout=1)

        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 0)
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(self.store._job_key(created["job_id"]), mapping={"updated_at": old})
        self.assertEqual(await self.store.requeue_stale_jobs(stale_after_seconds=60), 1)
        self.assertEqual((await self.store.get_status(created["job_id"]))["status"], "queued")

    async def test_stale_worker_cannot_heartbeat_or_finish_new_attempt(self):
        created = await self.store.create_job("research_web", {"query": "lease"})
        first = await self.store.claim_job(worker_id="worker-1")
        old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        await self.redis.hset(self.store._job_key(created["job_id"]), mapping={"heartbeat_at": old})
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
        self.assertEqual((await self.store.get_result(created["job_id"]))["result"], {"attempt": 2})

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
        self.assertEqual(len(await self.redis.lrange(self.store.processing_key, 0, -1)), 1)

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
        self.assertEqual(sum(isinstance(result, JobQueueFullError) for result in results), 1)
        self.assertEqual(await self.redis.llen(self.store.queue_key), 1)

    async def test_worker_heartbeats_are_isolated_by_worker_and_host(self):
        await self.store.record_worker_heartbeat("worker-a", host_id="host-a")
        await self.store.record_worker_heartbeat("worker-b", state="busy", host_id="host-b")

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
