"""Durable Redis-backed job state and queue primitives.

The public module-level functions use a lazily-created default store so the MCP
gateway can import this module even when it is configured for inline execution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

try:
    import redis.asyncio as redis_async
    from redis.exceptions import WatchError
except ImportError:  # Redis is optional when JOB_BACKEND is not "redis".
    redis_async = None

    class WatchError(Exception):
        """Fallback used only when the optional Redis package is absent."""


QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL_STATUSES = {SUCCEEDED, FAILED, CANCELLED}

_JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
_LEASE_TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")
_INGESTION_ATTEMPT_ID_RE = re.compile(r"^[a-f0-9]{64}$")
_COALESCE_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{64}$")
_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_WATCH_RETRIES = 64
_MAX_ACTIVE_JOB_SCAN = 10_000
_MIN_ACTIVE_JOB_INDEX_TTL_SECONDS = 2_592_000
_REGISTER_INGESTION_LUA = """
if redis.call('HGET', KEYS[1], 'status') ~= ARGV[1]
   or redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
    return 0
end
redis.call('HSET', KEYS[1], 'ingestion_attempt_id', ARGV[3])
redis.call('HSET', KEYS[2], ARGV[3], ARGV[4])
redis.call('ZREM', KEYS[3], ARGV[3])
return 1
"""


class JobStoreError(RuntimeError):
    """Base exception for job store failures."""


class JobNotFoundError(JobStoreError):
    """Raised when an operation targets an unknown job."""


class InvalidJobError(ValueError):
    """Raised when a job identifier, kind, or payload is invalid."""


class JobLeaseLostError(JobStoreError):
    """Raised when a worker tries to mutate a lease it no longer owns."""


class JobQueueFullError(JobStoreError):
    """Raised when queue admission would exceed the configured pending limit."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_job_id(job_id: str) -> str:
    value = str(job_id or "").strip().lower()
    if not _JOB_ID_RE.fullmatch(value):
        raise InvalidJobError("job_id must be a 32-character lowercase hexadecimal UUID")
    return value


def _validate_lease_token(lease_token: Optional[str]) -> str:
    value = str(lease_token or "").strip().lower()
    if not _LEASE_TOKEN_RE.fullmatch(value):
        raise JobLeaseLostError("a valid worker lease token is required")
    return value


def _validate_ingestion_attempt_id(ingestion_attempt_id: str) -> str:
    value = str(ingestion_attempt_id or "").strip().lower()
    if not _INGESTION_ATTEMPT_ID_RE.fullmatch(value):
        raise InvalidJobError(
            "ingestion_attempt_id must be a 64-character lowercase hexadecimal digest"
        )
    return value


def _validate_invalidation_reason(reason: str) -> str:
    value = str(reason or "").strip()
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise InvalidJobError("invalidation reason must contain printable characters")
    return value[:128]


def _registered_ingestion_attempt(record: Mapping[str, Any]) -> Optional[str]:
    raw_attempt_id = record.get("ingestion_attempt_id")
    if not raw_attempt_id:
        return None
    try:
        return _validate_ingestion_attempt_id(str(raw_attempt_id))
    except InvalidJobError as exc:
        raise JobStoreError("job contains an invalid ingestion attempt") from exc


def _validate_owner_id(owner_id: Optional[str]) -> Optional[str]:
    if owner_id is None:
        return None
    value = str(owner_id).strip()
    if not value or len(value) > 128 or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise InvalidJobError("owner_id must be 1-128 printable characters")
    return value


def _coalescing_fingerprint(
    kind: str,
    payload_json: str,
    owner_id: Optional[str],
) -> str:
    """Hash an unambiguous, versioned job identity without exposing its payload."""
    digest = hashlib.sha256()
    components = (
        b"research-mcp-active-job-v1",
        b"owned" if owner_id is not None else b"anonymous",
        (owner_id or "").encode("utf-8"),
        kind.encode("utf-8"),
        payload_json.encode("utf-8"),
    )
    for component in components:
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def _record_coalescing_fingerprint(record: Mapping[str, Any]) -> Optional[str]:
    value = str(record.get("coalesce_fingerprint") or "").strip().lower()
    return value if _COALESCE_FINGERPRINT_RE.fullmatch(value) else None


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value) if value else None
    except (TypeError, ValueError):
        return None
    if parsed is not None and parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise InvalidJobError(f"value is not JSON serializable: {exc}") from exc


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _decode_mapping(values: Mapping[Any, Any]) -> dict[str, str]:
    return {_decode(key): _decode(value) for key, value in values.items()}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


class RedisJobStore:
    """Queue and job state backed by Redis hashes and reliable lists."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        queue_name: Optional[str] = None,
        result_ttl_seconds: Optional[int] = None,
        ingestion_waitaof_timeout_ms: Optional[int] = None,
        redis_client: Any = None,
    ) -> None:
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.queue_key = queue_name or os.getenv("RESEARCH_QUEUE", "research:jobs")
        if not self.queue_key or any(char.isspace() for char in self.queue_key):
            raise ValueError("RESEARCH_QUEUE must be a non-empty Redis key without whitespace")

        self.processing_key = f"{self.queue_key}:processing"
        self.job_key_prefix = f"{self.queue_key}:job:"
        self.active_job_key_prefix = f"{self.queue_key}:active:"
        self.worker_heartbeat_key = f"{self.queue_key}:worker:heartbeat"
        self.worker_heartbeat_prefix = f"{self.worker_heartbeat_key}:id:"
        self.worker_host_heartbeat_prefix = f"{self.worker_heartbeat_key}:host:"
        self.ingestion_invalidation_due_key = (
            f"{self.queue_key}:ingestion-invalidations:due"
        )
        self.ingestion_invalidation_payload_key = (
            f"{self.queue_key}:ingestion-invalidations:payload"
        )
        self.result_ttl_seconds = (
            _env_int("JOB_RESULT_TTL_SECONDS", 2_592_000)
            if result_ttl_seconds is None
            else max(0, int(result_ttl_seconds))
        )
        # Active indexes are disposable coordination metadata. Keep them long
        # enough for queued work, but never let an orphan survive indefinitely.
        self.active_job_index_ttl_seconds = max(
            _MIN_ACTIVE_JOB_INDEX_TTL_SECONDS,
            self.result_ttl_seconds,
        )
        self.max_payload_bytes = _env_int("JOB_MAX_PAYLOAD_BYTES", 1_048_576, minimum=1024)
        self.max_queued_jobs = _env_int("JOB_MAX_QUEUED", 1000, minimum=0)
        self.max_attempts = _env_int("JOB_MAX_ATTEMPTS", 3, minimum=1)
        self.worker_heartbeat_ttl = _env_int("WORKER_HEARTBEAT_TTL_SECONDS", 60, minimum=10)
        self.ingestion_waitaof_timeout_ms = (
            _env_int("JOB_INGESTION_WAITAOF_TIMEOUT_MS", 5000)
            if ingestion_waitaof_timeout_ms is None
            else max(0, int(ingestion_waitaof_timeout_ms))
        )

        if redis_client is not None:
            self.redis = redis_client
        else:
            if redis_async is None:
                raise JobStoreError(
                    "Redis job backend requires the 'redis' package with redis.asyncio support"
                )
            self.redis = redis_async.from_url(self.redis_url, decode_responses=True)

    def _job_key(self, job_id: str) -> str:
        return f"{self.job_key_prefix}{validate_job_id(job_id)}"

    def _active_job_key(self, fingerprint: str) -> str:
        value = str(fingerprint or "").strip().lower()
        if not _COALESCE_FINGERPRINT_RE.fullmatch(value):
            raise InvalidJobError("coalescing fingerprint is invalid")
        return f"{self.active_job_key_prefix}{value}"

    async def _matching_active_job_key(
        self,
        pipe: Any,
        record: Mapping[str, Any],
        job_id: str,
    ) -> Optional[str]:
        """Watch and return this job's index key only when it still owns it."""
        fingerprint = _record_coalescing_fingerprint(record)
        if fingerprint is None:
            return None
        active_key = self._active_job_key(fingerprint)
        await pipe.watch(active_key)
        raw_indexed_job_id = await pipe.get(active_key)
        if raw_indexed_job_id is None:
            return None
        try:
            indexed_job_id = validate_job_id(_decode(raw_indexed_job_id))
        except InvalidJobError:
            return None
        return (
            active_key
            if hmac.compare_digest(indexed_job_id, validate_job_id(job_id))
            else None
        )

    def _worker_key(self, prefix: str, identity: str) -> str:
        digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()
        return f"{prefix}{digest}"

    async def close(self) -> None:
        close = getattr(self.redis, "aclose", None) or getattr(self.redis, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    async def create_job(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        job_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        coalesce_active: bool = False,
    ) -> dict[str, Any]:
        kind_value = str(kind or "").strip().lower()
        if not _KIND_RE.fullmatch(kind_value):
            raise InvalidJobError("kind must use lowercase letters, digits, and underscores")
        if not isinstance(payload, Mapping):
            raise InvalidJobError("payload must be a mapping")

        payload_json = _json_dumps(dict(payload))
        if len(payload_json.encode("utf-8")) > self.max_payload_bytes:
            raise InvalidJobError(
                f"payload exceeds JOB_MAX_PAYLOAD_BYTES ({self.max_payload_bytes} bytes)"
            )

        job_id_value = validate_job_id(job_id) if job_id is not None else uuid.uuid4().hex
        owner_id_value = _validate_owner_id(owner_id)
        if not isinstance(coalesce_active, bool):
            raise InvalidJobError("coalesce_active must be a boolean")
        coalesce_fingerprint = (
            _coalescing_fingerprint(kind_value, payload_json, owner_id_value)
            if coalesce_active
            else None
        )
        active_key = (
            self._active_job_key(coalesce_fingerprint)
            if coalesce_fingerprint is not None
            else None
        )
        now = utc_now_iso()
        record = {
            "job_id": job_id_value,
            "kind": kind_value,
            "payload": payload_json,
            "status": QUEUED,
            "cancel_requested": "0",
            "created_at": now,
            "updated_at": now,
            "enqueued_at": now,
        }
        if owner_id_value is not None:
            record["owner_id"] = owner_id_value
        if coalesce_fingerprint is not None:
            record["coalesce_fingerprint"] = coalesce_fingerprint

        key = self._job_key(job_id_value)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    watched_keys = [key, self.queue_key]
                    if active_key is not None:
                        watched_keys.append(active_key)
                    await pipe.watch(*watched_keys)
                    if await pipe.exists(key):
                        await pipe.unwatch()
                        raise InvalidJobError(f"job_id already exists: {job_id_value}")

                    if active_key is not None:
                        raw_existing_job_id = await pipe.get(active_key)
                        if raw_existing_job_id is not None:
                            try:
                                existing_job_id = validate_job_id(
                                    _decode(raw_existing_job_id)
                                )
                            except InvalidJobError:
                                existing_job_id = None
                            if existing_job_id is not None:
                                existing_key = self._job_key(existing_job_id)
                                await pipe.watch(existing_key)
                                raw_existing = await pipe.hgetall(existing_key)
                                if raw_existing:
                                    existing = _decode_mapping(raw_existing)
                                    exact_match = (
                                        existing.get("coalesce_fingerprint")
                                        == coalesce_fingerprint
                                        and existing.get("owner_id") == owner_id_value
                                        and existing.get("kind") == kind_value
                                        and existing.get("payload") == payload_json
                                    )
                                    reusable = (
                                        existing.get("status") in {QUEUED, RUNNING}
                                        and existing.get("cancel_requested") != "1"
                                    )
                                    if exact_match and reusable:
                                        pipe.multi()
                                        pipe.expire(
                                            active_key,
                                            self.active_job_index_ttl_seconds,
                                        )
                                        refreshed = await pipe.execute()
                                        if not refreshed or not refreshed[0]:
                                            continue
                                        result = self._public_job(
                                            existing,
                                            include_payload=False,
                                        )
                                        result["coalesced"] = True
                                        return result
                    queued_count = int(await pipe.llen(self.queue_key))
                    if self.max_queued_jobs > 0 and queued_count >= self.max_queued_jobs:
                        await pipe.unwatch()
                        raise JobQueueFullError(
                            f"job queue has reached JOB_MAX_QUEUED ({self.max_queued_jobs})"
                        )
                    pipe.multi()
                    pipe.hset(key, mapping=record)
                    pipe.lpush(self.queue_key, job_id_value)
                    if active_key is not None:
                        pipe.set(
                            active_key,
                            job_id_value,
                            ex=self.active_job_index_ttl_seconds,
                        )
                    await pipe.execute()
                    result = self._public_job(record, include_payload=False)
                    if active_key is not None:
                        result["coalesced"] = False
                    return result
                except WatchError:
                    continue
        raise JobStoreError("could not create job because the queue kept changing")

    async def get_job(
        self,
        job_id: str,
        *,
        include_payload: bool = True,
        include_lease: bool = False,
    ) -> Optional[dict[str, Any]]:
        values = await self.redis.hgetall(self._job_key(job_id))
        if not values:
            return None
        return self._public_job(
            _decode_mapping(values),
            include_payload=include_payload,
            include_lease=include_lease,
        )

    async def get_status(self, job_id: str) -> Optional[dict[str, Any]]:
        job = await self.get_job(job_id, include_payload=False)
        if job is not None:
            job.pop("result", None)
            job.pop("error", None)
        return job

    async def get_result(self, job_id: str) -> Optional[dict[str, Any]]:
        job = await self.get_job(job_id, include_payload=False)
        if job is None:
            return None

        response = {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "status": job["status"],
            "completed_at": job.get("completed_at"),
        }
        if job.get("owner_id") is not None:
            response["owner_id"] = job["owner_id"]
        result = _json_loads(job.get("result"))
        error = _json_loads(job.get("error"))
        if result is not None:
            response["result"] = result
        if error is not None:
            response["error"] = error
        if job.get("cancel_reason"):
            response["cancel_reason"] = job["cancel_reason"]
        return response

    async def active_job_ids(self, limit: int = _MAX_ACTIVE_JOB_SCAN) -> set[str]:
        """Return a consistent, bounded snapshot of queued and running job IDs."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("active job scan limit must be a positive integer")

        pipe = self.redis.pipeline(transaction=True)
        pipe.lrange(self.queue_key, 0, limit)
        pipe.lrange(self.processing_key, 0, limit)
        queued, processing = await pipe.execute()
        if len(queued) + len(processing) > limit:
            raise JobStoreError(
                f"active job scan exceeded its safety limit ({limit}); artifact cleanup was skipped"
            )

        active: set[str] = set()
        for raw_job_id in [*queued, *processing]:
            try:
                active.add(validate_job_id(_decode(raw_job_id)))
            except InvalidJobError:
                continue
        return active

    async def active_artifact_owner_ids(
        self,
        limit: int = _MAX_ACTIVE_JOB_SCAN,
    ) -> set[str]:
        """Return active job IDs plus artifact owners referenced by their payloads.

        Deferred jobs may read an artifact owned by another durable job. Artifact
        cleanup is shared by all workers, so the dependency must remain protected
        for as long as the consuming job is queued or running.
        """
        active = await self.active_job_ids(limit=limit)
        if not active:
            return set()

        pipe = self.redis.pipeline(transaction=True)
        ordered_ids = sorted(active)
        for job_id in ordered_ids:
            pipe.hget(self._job_key(job_id), "payload")
        payloads = await pipe.execute()

        protected = set(active)
        for raw_payload in payloads:
            payload = _json_loads(raw_payload, default={})
            if not isinstance(payload, Mapping):
                continue
            candidates = [
                payload.get("artifact_owner_id"),
                payload.get("parent_job_id"),
            ]
            raw_many = payload.get("artifact_owner_ids")
            if isinstance(raw_many, list):
                candidates.extend(raw_many)
            for candidate in candidates:
                if not isinstance(candidate, str):
                    continue
                try:
                    protected.add(validate_job_id(candidate))
                except InvalidJobError:
                    continue
        return protected

    async def request_cancellation(self, job_id: str) -> Optional[dict[str, Any]]:
        job_id_value = validate_job_id(job_id)
        key = self._job_key(job_id_value)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key, self.queue_key, self.processing_key)
                    raw_record = await pipe.hgetall(key)
                    if not raw_record:
                        await pipe.unwatch()
                        return None
                    record = _decode_mapping(raw_record)
                    if record.get("status") in TERMINAL_STATUSES:
                        active_key = await self._matching_active_job_key(
                            pipe,
                            record,
                            job_id_value,
                        )
                        if active_key is not None:
                            pipe.multi()
                            pipe.delete(active_key)
                            await pipe.execute()
                        else:
                            await pipe.unwatch()
                        return self._public_job(record, include_payload=False)

                    now = utc_now_iso()
                    fields = {
                        "cancel_requested": "1",
                        "cancel_requested_at": now,
                        "updated_at": now,
                    }
                    active_key = await self._matching_active_job_key(
                        pipe,
                        record,
                        job_id_value,
                    )
                    pipe.multi()
                    if record.get("status") == QUEUED:
                        fields.update(
                            {
                                "status": CANCELLED,
                                "completed_at": now,
                                "cancel_reason": "cancelled before execution",
                            }
                        )
                        pipe.hset(key, mapping=fields)
                        pipe.hdel(key, "lease_token", "worker_id", "heartbeat_at")
                        pipe.lrem(self.queue_key, 0, job_id_value)
                        pipe.lrem(self.processing_key, 0, job_id_value)
                        if self.result_ttl_seconds > 0:
                            pipe.expire(key, self.result_ttl_seconds)
                    else:
                        pipe.hset(key, mapping=fields)
                    if active_key is not None:
                        pipe.delete(active_key)
                    await pipe.execute()
                    return await self.get_status(job_id_value)
                except WatchError:
                    continue
        raise JobStoreError("could not cancel job because its state kept changing")

    async def is_cancellation_requested(self, job_id: str) -> bool:
        value = await self.redis.hget(self._job_key(job_id), "cancel_requested")
        return _decode(value) == "1" if value is not None else False

    async def claim_job(self, timeout: float = 1.0, worker_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        timeout_seconds = max(1, math.ceil(float(timeout)))
        raw_job_id = await self.redis.brpoplpush(
            self.queue_key,
            self.processing_key,
            timeout=timeout_seconds,
        )
        if raw_job_id is None:
            return None

        try:
            job_id = validate_job_id(_decode(raw_job_id))
        except InvalidJobError:
            await self.redis.lrem(self.processing_key, 0, raw_job_id)
            return None
        worker_id_value = str(worker_id or "unknown")[:256]
        lease_token = uuid.uuid4().hex
        job_id_value = validate_job_id(job_id)
        key = self._job_key(job_id_value)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key, self.queue_key, self.processing_key)
                    raw_record = await pipe.hgetall(key)
                    processing = [
                        _decode(item) for item in await pipe.lrange(self.processing_key, 0, -1)
                    ]
                    if not raw_record:
                        pipe.multi()
                        pipe.lrem(self.processing_key, 0, job_id_value)
                        pipe.lrem(self.queue_key, 0, job_id_value)
                        await pipe.execute()
                        return None

                    record = _decode_mapping(raw_record)
                    status = record.get("status")
                    if status in TERMINAL_STATUSES:
                        active_key = await self._matching_active_job_key(
                            pipe,
                            record,
                            job_id_value,
                        )
                        pipe.multi()
                        pipe.lrem(self.processing_key, 0, job_id_value)
                        pipe.lrem(self.queue_key, 0, job_id_value)
                        if active_key is not None:
                            pipe.delete(active_key)
                        await pipe.execute()
                        return None
                    if status == RUNNING:
                        # A stale duplicate queue entry must never create a second executor.
                        if processing.count(job_id_value) > 1:
                            pipe.multi()
                            pipe.lrem(self.processing_key, 1, job_id_value)
                            await pipe.execute()
                        else:
                            await pipe.unwatch()
                        return None
                    if status != QUEUED or job_id_value not in processing:
                        await pipe.unwatch()
                        return None

                    now = utc_now_iso()
                    if record.get("cancel_requested") == "1":
                        active_key = await self._matching_active_job_key(
                            pipe,
                            record,
                            job_id_value,
                        )
                        pipe.multi()
                        pipe.hset(
                            key,
                            mapping={
                                "status": CANCELLED,
                                "updated_at": now,
                                "completed_at": now,
                                "cancel_reason": "cancelled before execution",
                            },
                        )
                        pipe.hdel(key, "lease_token", "worker_id", "heartbeat_at")
                        pipe.lrem(self.processing_key, 0, job_id_value)
                        pipe.lrem(self.queue_key, 0, job_id_value)
                        if self.result_ttl_seconds > 0:
                            pipe.expire(key, self.result_ttl_seconds)
                        if active_key is not None:
                            pipe.delete(active_key)
                        await pipe.execute()
                        return None

                    try:
                        attempt = max(0, int(record.get("attempt", "0"))) + 1
                    except (TypeError, ValueError):
                        attempt = 1
                    active_key = await self._matching_active_job_key(
                        pipe,
                        record,
                        job_id_value,
                    )
                    pipe.multi()
                    pipe.lrem(self.queue_key, 0, job_id_value)
                    pipe.lrem(self.processing_key, 0, job_id_value)
                    pipe.lpush(self.processing_key, job_id_value)
                    pipe.hset(
                        key,
                        mapping={
                            "status": RUNNING,
                            "worker_id": worker_id_value,
                            "lease_token": lease_token,
                            "attempt": str(attempt),
                            "started_at": record.get("started_at") or now,
                            "attempt_started_at": now,
                            "heartbeat_at": now,
                            "updated_at": now,
                        },
                    )
                    if active_key is not None:
                        pipe.expire(
                            active_key,
                            self.active_job_index_ttl_seconds,
                        )
                    await pipe.execute()
                    return await self.get_job(
                        job_id_value,
                        include_payload=True,
                        include_lease=True,
                    )
                except WatchError:
                    continue
        raise JobStoreError("could not claim job because its state kept changing")

    async def heartbeat_job(self, job_id: str, worker_id: str, lease_token: str) -> bool:
        job_id_value = validate_job_id(job_id)
        lease_token_value = _validate_lease_token(lease_token)
        worker_id_value = str(worker_id)
        key = self._job_key(job_id_value)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    record = _decode_mapping(await pipe.hgetall(key))
                    owns_lease = (
                        record.get("status") == RUNNING
                        and hmac.compare_digest(record.get("lease_token", ""), lease_token_value)
                        and record.get("worker_id") == worker_id_value
                    )
                    if not owns_lease:
                        await pipe.unwatch()
                        return False
                    now = utc_now_iso()
                    active_key = await self._matching_active_job_key(
                        pipe,
                        record,
                        job_id_value,
                    )
                    pipe.multi()
                    pipe.hset(key, mapping={"heartbeat_at": now, "updated_at": now})
                    if active_key is not None:
                        pipe.expire(
                            active_key,
                            self.active_job_index_ttl_seconds,
                        )
                    await pipe.execute()
                    return True
                except WatchError:
                    continue
        raise JobStoreError("could not heartbeat job because its state kept changing")

    async def register_ingestion_invalidation(
        self,
        job_id: str,
        ingestion_attempt_id: str,
        *,
        lease_token: str,
    ) -> None:
        """Durably register compensation before an ingesting attempt can write."""
        job_id_value = validate_job_id(job_id)
        attempt_id = _validate_ingestion_attempt_id(ingestion_attempt_id)
        lease_token_value = _validate_lease_token(lease_token)
        key = self._job_key(job_id_value)
        now = utc_now_iso()
        payload = _json_dumps(
            {
                "ingestion_attempt_id": attempt_id,
                "job_id": job_id_value,
                "reason": "worker_attempt_abandoned",
                "registered_at": now,
                "updated_at": now,
            }
        )
        if self.ingestion_waitaof_timeout_ms > 0:
            async with self.redis.client() as durable_redis:
                try:
                    registered = await durable_redis.eval(
                        _REGISTER_INGESTION_LUA,
                        3,
                        key,
                        self.ingestion_invalidation_payload_key,
                        self.ingestion_invalidation_due_key,
                        RUNNING,
                        lease_token_value,
                        attempt_id,
                        payload,
                    )
                except Exception as exc:
                    raise JobStoreError(
                        "could not durably register ingestion compensation in Redis"
                    ) from exc
                if int(registered) != 1:
                    raise JobLeaseLostError(
                        f"worker lease lost for job: {job_id_value}"
                    )
                try:
                    fsynced = await durable_redis.waitaof(
                        1,
                        0,
                        self.ingestion_waitaof_timeout_ms,
                    )
                    local_fsyncs = int(fsynced[0])
                except Exception as exc:
                    raise JobStoreError(
                        "could not confirm durable ingestion compensation in Redis AOF"
                    ) from exc
                if local_fsyncs < 1:
                    raise JobStoreError(
                        "Redis did not fsync the ingestion compensation before timeout"
                    )
            return

        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    record = _decode_mapping(await pipe.hgetall(key))
                    if (
                        record.get("status") != RUNNING
                        or not hmac.compare_digest(
                            record.get("lease_token", ""), lease_token_value
                        )
                    ):
                        await pipe.unwatch()
                        raise JobLeaseLostError(
                            f"worker lease lost for job: {job_id_value}"
                        )

                    pipe.multi()
                    pipe.hset(key, "ingestion_attempt_id", attempt_id)
                    pipe.hset(
                        self.ingestion_invalidation_payload_key,
                        attempt_id,
                        payload,
                    )
                    # Registration arms cleanup. Only an abandoned lease is due.
                    pipe.zrem(self.ingestion_invalidation_due_key, attempt_id)
                    await pipe.execute()
                    return
                except WatchError:
                    continue
        raise JobStoreError(
            "could not register ingestion compensation because job state kept changing"
        )

    async def schedule_ingestion_invalidation(
        self,
        job_id: str,
        ingestion_attempt_id: str,
        *,
        reason: str,
    ) -> bool:
        """Make an existing compensation record immediately eligible for replay."""
        job_id_value = validate_job_id(job_id)
        attempt_id = _validate_ingestion_attempt_id(ingestion_attempt_id)
        reason_value = _validate_invalidation_reason(reason)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(
                        self.ingestion_invalidation_payload_key,
                        self.ingestion_invalidation_due_key,
                    )
                    existing = _json_loads(
                        await pipe.hget(
                            self.ingestion_invalidation_payload_key,
                            attempt_id,
                        ),
                        {},
                    )
                    if not existing:
                        await pipe.unwatch()
                        return False
                    if (
                        not isinstance(existing, Mapping)
                        or existing.get("job_id") != job_id_value
                        or existing.get("ingestion_attempt_id") != attempt_id
                    ):
                        await pipe.unwatch()
                        raise JobStoreError(
                            "ingestion compensation payload does not match its attempt"
                        )
                    now = utc_now_iso()
                    payload = {
                        "ingestion_attempt_id": attempt_id,
                        "job_id": job_id_value,
                        "reason": reason_value,
                        "registered_at": existing.get("registered_at") or now,
                        "updated_at": now,
                    }
                    pipe.multi()
                    pipe.hset(
                        self.ingestion_invalidation_payload_key,
                        attempt_id,
                        _json_dumps(payload),
                    )
                    pipe.zadd(
                        self.ingestion_invalidation_due_key,
                        {attempt_id: datetime.now(timezone.utc).timestamp()},
                    )
                    await pipe.execute()
                    return True
                except WatchError:
                    continue
        raise JobStoreError(
            "could not schedule ingestion compensation because its state kept changing"
        )

    async def claim_due_ingestion_invalidations(
        self,
        *,
        limit: int = 100,
        lease_seconds: float = 600.0,
    ) -> list[dict[str, str]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("invalidation replay limit must be between 1 and 1000")
        if isinstance(lease_seconds, bool) or float(lease_seconds) < 1:
            raise ValueError("invalidation delivery lease must be at least one second")
        now_timestamp = datetime.now(timezone.utc).timestamp()
        raw_attempt_ids = await self.redis.zrangebyscore(
            self.ingestion_invalidation_due_key,
            "-inf",
            now_timestamp,
            start=0,
            num=limit,
        )
        claimed: list[dict[str, str]] = []
        for raw_attempt_id in raw_attempt_ids:
            try:
                attempt_id = _validate_ingestion_attempt_id(_decode(raw_attempt_id))
            except InvalidJobError:
                await self.redis.zrem(
                    self.ingestion_invalidation_due_key,
                    raw_attempt_id,
                )
                continue

            raw_hint = await self.redis.hget(
                self.ingestion_invalidation_payload_key,
                attempt_id,
            )
            hint = _json_loads(raw_hint, {})
            try:
                job_id = validate_job_id(
                    str(hint.get("job_id") or "")
                    if isinstance(hint, Mapping)
                    else ""
                )
            except InvalidJobError:
                await self.redis.zrem(self.ingestion_invalidation_due_key, attempt_id)
                continue

            key = self._job_key(job_id)
            for _ in range(_MAX_WATCH_RETRIES):
                async with self.redis.pipeline(transaction=True) as pipe:
                    try:
                        await pipe.watch(
                            key,
                            self.ingestion_invalidation_payload_key,
                            self.ingestion_invalidation_due_key,
                        )
                        raw_payload = await pipe.hget(
                            self.ingestion_invalidation_payload_key,
                            attempt_id,
                        )
                        score = await pipe.zscore(
                            self.ingestion_invalidation_due_key,
                            attempt_id,
                        )
                        payload = _json_loads(raw_payload, {})
                        if raw_payload is None or score is None:
                            pipe.multi()
                            pipe.zrem(self.ingestion_invalidation_due_key, attempt_id)
                            await pipe.execute()
                            break
                        if float(score) > now_timestamp:
                            await pipe.unwatch()
                            break
                        if (
                            not isinstance(payload, Mapping)
                            or payload.get("job_id") != job_id
                            or payload.get("ingestion_attempt_id") != attempt_id
                        ):
                            pipe.multi()
                            pipe.zrem(self.ingestion_invalidation_due_key, attempt_id)
                            await pipe.execute()
                            break

                        record = _decode_mapping(await pipe.hgetall(key))
                        lease_until = now_timestamp + float(lease_seconds)
                        active = (
                            record.get("status") == RUNNING
                            and hmac.compare_digest(
                                record.get("ingestion_attempt_id", ""),
                                attempt_id,
                            )
                        )
                        pipe.multi()
                        pipe.zadd(
                            self.ingestion_invalidation_due_key,
                            {attempt_id: lease_until},
                        )
                        await pipe.execute()
                        if not active:
                            claimed.append(
                                {
                                    "ingestion_attempt_id": attempt_id,
                                    "job_id": job_id,
                                    "reason": str(
                                        payload.get("reason")
                                        or "durable_compensation_replay"
                                    )[:128],
                                }
                            )
                        break
                    except WatchError:
                        continue
            else:
                raise JobStoreError(
                    "could not claim ingestion compensation because its state kept changing"
                )
        return claimed

    async def defer_ingestion_invalidation(
        self,
        ingestion_attempt_id: str,
        *,
        delay_seconds: float,
    ) -> None:
        attempt_id = _validate_ingestion_attempt_id(ingestion_attempt_id)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(
                        self.ingestion_invalidation_payload_key,
                        self.ingestion_invalidation_due_key,
                    )
                    if not await pipe.hexists(
                        self.ingestion_invalidation_payload_key,
                        attempt_id,
                    ):
                        await pipe.unwatch()
                        return
                    pipe.multi()
                    pipe.zadd(
                        self.ingestion_invalidation_due_key,
                        {
                            attempt_id: datetime.now(timezone.utc).timestamp()
                            + max(0.1, float(delay_seconds))
                        },
                    )
                    await pipe.execute()
                    return
                except WatchError:
                    continue
        raise JobStoreError(
            "could not defer ingestion compensation because its state kept changing"
        )

    async def acknowledge_ingestion_invalidation(
        self,
        ingestion_attempt_id: str,
    ) -> None:
        attempt_id = _validate_ingestion_attempt_id(ingestion_attempt_id)
        pipe = self.redis.pipeline(transaction=True)
        pipe.hdel(self.ingestion_invalidation_payload_key, attempt_id)
        pipe.zrem(self.ingestion_invalidation_due_key, attempt_id)
        await pipe.execute()

    async def record_worker_heartbeat(
        self,
        worker_id: str,
        state: str = "ready",
        *,
        host_id: Optional[str] = None,
    ) -> None:
        host_id_value = str(host_id or socket.gethostname())
        value = _json_dumps(
            {
                "worker_id": str(worker_id),
                "host_id": host_id_value,
                "state": str(state),
                "heartbeat_at": utc_now_iso(),
            }
        )
        pipe = self.redis.pipeline(transaction=True)
        pipe.set(
            self._worker_key(self.worker_heartbeat_prefix, str(worker_id)),
            value,
            ex=self.worker_heartbeat_ttl,
        )
        pipe.set(
            self._worker_key(self.worker_host_heartbeat_prefix, host_id_value),
            value,
            ex=self.worker_heartbeat_ttl,
        )
        # Retain the aggregate key for compatibility; healthchecks use the host key.
        pipe.set(self.worker_heartbeat_key, value, ex=self.worker_heartbeat_ttl)
        await pipe.execute()

    async def get_worker_heartbeat(
        self,
        *,
        worker_id: Optional[str] = None,
        host_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if worker_id is not None:
            key = self._worker_key(self.worker_heartbeat_prefix, worker_id)
        elif host_id is not None:
            key = self._worker_key(self.worker_host_heartbeat_prefix, host_id)
        else:
            key = self.worker_heartbeat_key
        return _json_loads(await self.redis.get(key))

    async def complete_job(
        self,
        job_id: str,
        result_metadata: Mapping[str, Any],
        *,
        lease_token: Optional[str] = None,
        successful_ingestion_attempt_id: Optional[str] = None,
    ) -> bool:
        return await self._mark_terminal(
            job_id,
            SUCCEEDED,
            result=dict(result_metadata),
            lease_token=lease_token,
            successful_ingestion_attempt_id=successful_ingestion_attempt_id,
        )

    async def complete_job_with_children(
        self,
        job_id: str,
        result_metadata: Mapping[str, Any],
        *,
        lease_token: str,
        child_queue_name: str,
        child_jobs: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Atomically complete an acquisition job and enqueue persistence children.

        This operation is intentionally limited to parents that did not register
        an ingestion attempt. Each child must already have a unique artifact owner
        so a failed source can be retried or invalidated independently.
        """
        if not isinstance(result_metadata, Mapping):
            raise InvalidJobError("result_metadata must be a mapping")
        result_json = _json_dumps(dict(result_metadata))
        queue_value = str(child_queue_name or "").strip()
        if not queue_value or any(char.isspace() for char in queue_value):
            raise InvalidJobError(
                "child_queue_name must be a non-empty Redis key without whitespace"
            )
        if not isinstance(child_jobs, list) or len(child_jobs) > 32:
            raise InvalidJobError("child_jobs must be a list with at most 32 items")

        now = utc_now_iso()
        prepared_children: list[tuple[str, str, dict[str, str]]] = []
        seen_child_ids: set[str] = set()
        for spec in child_jobs:
            if not isinstance(spec, Mapping):
                raise InvalidJobError("each child job must be a mapping")
            child_id = validate_job_id(str(spec.get("job_id") or ""))
            if child_id in seen_child_ids:
                raise InvalidJobError("child job IDs must be unique")
            seen_child_ids.add(child_id)
            kind = str(spec.get("kind") or "").strip().lower()
            if not _KIND_RE.fullmatch(kind):
                raise InvalidJobError(
                    "child kind must use lowercase letters, digits, and underscores"
                )
            payload = spec.get("payload")
            if not isinstance(payload, Mapping):
                raise InvalidJobError("child payload must be a mapping")
            payload_json = _json_dumps(dict(payload))
            if len(payload_json.encode("utf-8")) > self.max_payload_bytes:
                raise InvalidJobError(
                    f"child payload exceeds JOB_MAX_PAYLOAD_BYTES ({self.max_payload_bytes} bytes)"
                )
            owner_id = _validate_owner_id(spec.get("owner_id"))
            record = {
                "job_id": child_id,
                "kind": kind,
                "payload": payload_json,
                "status": QUEUED,
                "cancel_requested": "0",
                "created_at": now,
                "updated_at": now,
                "enqueued_at": now,
            }
            if owner_id is not None:
                record["owner_id"] = owner_id
            child_key = f"{queue_value}:job:{child_id}"
            prepared_children.append((child_id, child_key, record))

        parent_id = validate_job_id(job_id)
        parent_key = self._job_key(parent_id)
        lease_token_value = _validate_lease_token(lease_token)
        child_keys = [key for _child_id, key, _record in prepared_children]

        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    watched_keys = list(
                        dict.fromkeys(
                            [
                                parent_key,
                                self.queue_key,
                                self.processing_key,
                                queue_value,
                                *child_keys,
                            ]
                        )
                    )
                    await pipe.watch(*watched_keys)
                    parent = _decode_mapping(await pipe.hgetall(parent_key))
                    if not parent:
                        await pipe.unwatch()
                        raise JobNotFoundError(f"unknown job: {parent_id}")
                    if parent.get("status") in TERMINAL_STATUSES:
                        await pipe.unwatch()
                        raise JobLeaseLostError(
                            f"worker lease lost for job: {parent_id}"
                        )
                    owns_lease = (
                        parent.get("status") == RUNNING
                        and hmac.compare_digest(
                            parent.get("lease_token", ""), lease_token_value
                        )
                    )
                    if not owns_lease:
                        await pipe.unwatch()
                        raise JobLeaseLostError(
                            f"worker lease lost for job: {parent_id}"
                        )
                    if _registered_ingestion_attempt(parent) is not None:
                        await pipe.unwatch()
                        raise JobStoreError(
                            "atomic child enqueue requires an acquisition-only parent"
                        )

                    cancellation_wins = parent.get("cancel_requested") == "1"
                    if not cancellation_wins and prepared_children:
                        queued_count = int(await pipe.llen(queue_value))
                        if (
                            self.max_queued_jobs > 0
                            and queued_count + len(prepared_children)
                            > self.max_queued_jobs
                        ):
                            await pipe.unwatch()
                            raise JobQueueFullError(
                                f"child queue has reached JOB_MAX_QUEUED ({self.max_queued_jobs})"
                            )
                        for child_id, child_key, _record in prepared_children:
                            if await pipe.exists(child_key):
                                await pipe.unwatch()
                                raise InvalidJobError(
                                    f"child job_id already exists: {child_id}"
                                )

                    active_key = await self._matching_active_job_key(
                        pipe,
                        parent,
                        parent_id,
                    )
                    terminal_status = CANCELLED if cancellation_wins else SUCCEEDED
                    fields = {
                        "status": terminal_status,
                        "updated_at": now,
                        "completed_at": now,
                    }
                    if cancellation_wins:
                        fields["cancel_reason"] = "cancellation requested"
                    else:
                        fields["result"] = result_json

                    pipe.multi()
                    pipe.hset(parent_key, mapping=fields)
                    pipe.hdel(
                        parent_key,
                        "lease_token",
                        "heartbeat_at",
                        "attempt_started_at",
                        "ingestion_attempt_id",
                    )
                    pipe.lrem(self.processing_key, 0, parent_id)
                    pipe.lrem(self.queue_key, 0, parent_id)
                    if active_key is not None:
                        pipe.delete(active_key)
                    if self.result_ttl_seconds > 0:
                        pipe.expire(parent_key, self.result_ttl_seconds)
                    if not cancellation_wins:
                        for child_id, child_key, child_record in prepared_children:
                            pipe.hset(child_key, mapping=child_record)
                            pipe.lpush(queue_value, child_id)
                    await pipe.execute()
                    return {
                        "status": terminal_status,
                        "children": (
                            [
                                self._public_job(record, include_payload=False)
                                for _child_id, _child_key, record in prepared_children
                            ]
                            if not cancellation_wins
                            else []
                        ),
                    }
                except WatchError:
                    continue
        raise JobStoreError(
            "could not complete job with children because state kept changing"
        )

    async def fail_job(
        self,
        job_id: str,
        error: Mapping[str, Any] | str,
        *,
        lease_token: Optional[str] = None,
    ) -> bool:
        error_value = dict(error) if isinstance(error, Mapping) else {"message": str(error)}
        return await self._mark_terminal(
            job_id,
            FAILED,
            error=error_value,
            lease_token=lease_token,
        )

    async def mark_cancelled(
        self,
        job_id: str,
        reason: str = "cancellation requested",
        *,
        lease_token: Optional[str] = None,
    ) -> bool:
        return await self._mark_terminal(
            job_id,
            CANCELLED,
            cancel_reason=reason,
            lease_token=lease_token,
        )

    async def requeue_job(
        self,
        job_id: str,
        reason: str = "worker interrupted",
        *,
        lease_token: Optional[str] = None,
    ) -> bool:
        job_id_value = validate_job_id(job_id)
        lease_token_value = _validate_lease_token(lease_token)
        key = self._job_key(job_id_value)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key, self.queue_key, self.processing_key)
                    record = _decode_mapping(await pipe.hgetall(key))
                    if not record:
                        await pipe.unwatch()
                        raise JobNotFoundError(f"unknown job: {job_id_value}")
                    if (
                        record.get("status") != RUNNING
                        or not hmac.compare_digest(
                            record.get("lease_token", ""), lease_token_value
                        )
                    ):
                        await pipe.unwatch()
                        raise JobLeaseLostError(f"worker lease lost for job: {job_id_value}")

                    now = utc_now_iso()
                    registered_attempt_id = _registered_ingestion_attempt(record)
                    active_key = await self._matching_active_job_key(
                        pipe,
                        record,
                        job_id_value,
                    )
                    pipe.multi()
                    pipe.lrem(self.processing_key, 0, job_id_value)
                    pipe.lrem(self.queue_key, 0, job_id_value)
                    if record.get("cancel_requested") == "1":
                        pipe.hset(
                            key,
                            mapping={
                                "status": CANCELLED,
                                "updated_at": now,
                                "completed_at": now,
                                "cancel_reason": "cancellation requested",
                            },
                        )
                        if self.result_ttl_seconds > 0:
                            pipe.expire(key, self.result_ttl_seconds)
                        if active_key is not None:
                            pipe.delete(active_key)
                    else:
                        pipe.lpush(self.queue_key, job_id_value)
                        pipe.hset(
                            key,
                            mapping={
                                "status": QUEUED,
                                "updated_at": now,
                                "enqueued_at": now,
                                "requeue_reason": str(reason)[:1000],
                            },
                        )
                        if active_key is not None:
                            pipe.expire(
                                active_key,
                                self.active_job_index_ttl_seconds,
                            )
                    if registered_attempt_id is not None:
                        pipe.zadd(
                            self.ingestion_invalidation_due_key,
                            {
                                registered_attempt_id: datetime.now(
                                    timezone.utc
                                ).timestamp()
                            },
                        )
                    pipe.hdel(
                        key,
                        "lease_token",
                        "worker_id",
                        "heartbeat_at",
                        "attempt_started_at",
                        "ingestion_attempt_id",
                    )
                    await pipe.execute()
                    return record.get("cancel_requested") != "1"
                except WatchError:
                    continue
        raise JobStoreError("could not requeue job because its state kept changing")

    async def requeue_stale_jobs(self, stale_after_seconds: Optional[int] = None) -> int:
        stale_seconds = (
            _env_int("JOB_STALE_AFTER_SECONDS", 300, minimum=30)
            if stale_after_seconds is None
            else max(1, int(stale_after_seconds))
        )
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_seconds)
        job_ids = await self.redis.lrange(self.processing_key, 0, -1)
        requeued = 0

        for raw_job_id in job_ids:
            try:
                job_id = validate_job_id(_decode(raw_job_id))
            except InvalidJobError:
                await self.redis.lrem(self.processing_key, 0, raw_job_id)
                continue

            if await self._requeue_stale_job(job_id, cutoff):
                requeued += 1
        return requeued

    async def _requeue_stale_job(self, job_id: str, cutoff: datetime) -> bool:
        job_id_value = validate_job_id(job_id)
        key = self._job_key(job_id_value)
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key, self.queue_key, self.processing_key)
                    processing = [
                        _decode(item) for item in await pipe.lrange(self.processing_key, 0, -1)
                    ]
                    if job_id_value not in processing:
                        await pipe.unwatch()
                        return False
                    raw_record = await pipe.hgetall(key)
                    if not raw_record:
                        pipe.multi()
                        pipe.lrem(self.processing_key, 0, job_id_value)
                        pipe.lrem(self.queue_key, 0, job_id_value)
                        await pipe.execute()
                        return False

                    record = _decode_mapping(raw_record)
                    status = record.get("status")
                    if status in TERMINAL_STATUSES:
                        active_key = await self._matching_active_job_key(
                            pipe,
                            record,
                            job_id_value,
                        )
                        pipe.multi()
                        pipe.lrem(self.processing_key, 0, job_id_value)
                        pipe.lrem(self.queue_key, 0, job_id_value)
                        if active_key is not None:
                            pipe.delete(active_key)
                        await pipe.execute()
                        return False
                    if status not in {QUEUED, RUNNING}:
                        await pipe.unwatch()
                        return False

                    if status == RUNNING:
                        lease_time = _parse_timestamp(
                            record.get("heartbeat_at")
                            or record.get("attempt_started_at")
                            or record.get("started_at")
                        )
                    else:
                        # Covers a crash after BRPOPLPUSH but before the lease CAS.
                        lease_time = _parse_timestamp(
                            record.get("updated_at") or record.get("enqueued_at")
                        )
                    if lease_time is not None and lease_time > cutoff:
                        await pipe.unwatch()
                        return False

                    now = utc_now_iso()
                    cancelled = record.get("cancel_requested") == "1"
                    try:
                        attempt = max(0, int(record.get("attempt", "0")))
                    except (TypeError, ValueError):
                        attempt = self.max_attempts
                    attempts_exhausted = attempt >= self.max_attempts
                    registered_attempt_id = _registered_ingestion_attempt(record)
                    active_key = await self._matching_active_job_key(
                        pipe,
                        record,
                        job_id_value,
                    )
                    pipe.multi()
                    pipe.lrem(self.processing_key, 0, job_id_value)
                    pipe.lrem(self.queue_key, 0, job_id_value)
                    if cancelled:
                        pipe.hset(
                            key,
                            mapping={
                                "status": CANCELLED,
                                "updated_at": now,
                                "completed_at": now,
                                "cancel_reason": "cancellation requested",
                            },
                        )
                        if self.result_ttl_seconds > 0:
                            pipe.expire(key, self.result_ttl_seconds)
                        if active_key is not None:
                            pipe.delete(active_key)
                    elif attempts_exhausted:
                        pipe.hset(
                            key,
                            mapping={
                                "status": FAILED,
                                "updated_at": now,
                                "completed_at": now,
                                "error": _json_dumps(
                                    {
                                        "type": "JobAttemptsExhausted",
                                        "message": (
                                            f"job stopped after {attempt} stale worker attempts; "
                                            f"JOB_MAX_ATTEMPTS={self.max_attempts}"
                                        )[:1000],
                                    }
                                ),
                            },
                        )
                        if self.result_ttl_seconds > 0:
                            pipe.expire(key, self.result_ttl_seconds)
                        if active_key is not None:
                            pipe.delete(active_key)
                    else:
                        pipe.lpush(self.queue_key, job_id_value)
                        pipe.hset(
                            key,
                            mapping={
                                "status": QUEUED,
                                "updated_at": now,
                                "enqueued_at": now,
                                "requeue_reason": "stale worker lease",
                            },
                        )
                        if active_key is not None:
                            pipe.expire(
                                active_key,
                                self.active_job_index_ttl_seconds,
                            )
                    if registered_attempt_id is not None:
                        pipe.zadd(
                            self.ingestion_invalidation_due_key,
                            {
                                registered_attempt_id: datetime.now(
                                    timezone.utc
                                ).timestamp()
                            },
                        )
                    pipe.hdel(
                        key,
                        "lease_token",
                        "worker_id",
                        "heartbeat_at",
                        "attempt_started_at",
                        "ingestion_attempt_id",
                    )
                    await pipe.execute()
                    return not cancelled and not attempts_exhausted
                except WatchError:
                    continue
        raise JobStoreError("could not recover stale job because its state kept changing")

    async def _mark_terminal(
        self,
        job_id: str,
        status: str,
        *,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[Mapping[str, Any]] = None,
        cancel_reason: Optional[str] = None,
        lease_token: Optional[str] = None,
        successful_ingestion_attempt_id: Optional[str] = None,
    ) -> bool:
        job_id_value = validate_job_id(job_id)
        key = self._job_key(job_id_value)
        lease_token_value = _validate_lease_token(lease_token) if lease_token is not None else None
        successful_attempt_id = (
            _validate_ingestion_attempt_id(successful_ingestion_attempt_id)
            if successful_ingestion_attempt_id is not None
            else None
        )
        for _ in range(_MAX_WATCH_RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    watched_keys = [key, self.queue_key, self.processing_key]
                    if successful_attempt_id is not None:
                        watched_keys.extend(
                            [
                                self.ingestion_invalidation_payload_key,
                                self.ingestion_invalidation_due_key,
                            ]
                        )
                    await pipe.watch(*watched_keys)
                    record = _decode_mapping(await pipe.hgetall(key))
                    if not record:
                        await pipe.unwatch()
                        raise JobNotFoundError(f"unknown job: {job_id_value}")
                    current_status = record.get("status")
                    if current_status in TERMINAL_STATUSES:
                        await pipe.unwatch()
                        if lease_token_value is not None:
                            raise JobLeaseLostError(f"worker lease lost for job: {job_id_value}")
                        return False

                    queued_cancellation = (
                        status == CANCELLED
                        and current_status == QUEUED
                        and lease_token_value is None
                    )
                    owns_lease = (
                        current_status == RUNNING
                        and lease_token_value is not None
                        and hmac.compare_digest(
                            record.get("lease_token", ""), lease_token_value
                        )
                    )
                    if not (queued_cancellation or owns_lease):
                        await pipe.unwatch()
                        raise JobLeaseLostError(f"worker lease lost for job: {job_id_value}")

                    now = utc_now_iso()
                    cancellation_wins = (
                        owns_lease and record.get("cancel_requested") == "1"
                    )
                    terminal_status = CANCELLED if cancellation_wins else status
                    registered_attempt_id = _registered_ingestion_attempt(record)
                    if terminal_status == SUCCEEDED:
                        if registered_attempt_id is None:
                            if successful_attempt_id is not None:
                                await pipe.unwatch()
                                raise JobStoreError(
                                    "successful ingestion attempt was not durably registered"
                                )
                        elif (
                            successful_attempt_id is None
                            or not hmac.compare_digest(
                                registered_attempt_id,
                                successful_attempt_id,
                            )
                            or not await pipe.hexists(
                                self.ingestion_invalidation_payload_key,
                                registered_attempt_id,
                            )
                        ):
                            await pipe.unwatch()
                            raise JobStoreError(
                                "successful ingestion attempt was not durably registered"
                            )
                    fields = {
                        "status": terminal_status,
                        "updated_at": now,
                        "completed_at": now,
                    }
                    if result is not None and not cancellation_wins:
                        fields["result"] = _json_dumps(dict(result))
                    if error is not None and not cancellation_wins:
                        fields["error"] = _json_dumps(dict(error))
                    effective_cancel_reason = (
                        "cancellation requested" if cancellation_wins else cancel_reason
                    )
                    if effective_cancel_reason:
                        fields["cancel_reason"] = str(effective_cancel_reason)[:1000]

                    active_key = await self._matching_active_job_key(
                        pipe,
                        record,
                        job_id_value,
                    )

                    pipe.multi()
                    pipe.hset(key, mapping=fields)
                    pipe.hdel(
                        key,
                        "lease_token",
                        "heartbeat_at",
                        "attempt_started_at",
                        "ingestion_attempt_id",
                    )
                    if terminal_status == SUCCEEDED and registered_attempt_id is not None:
                        pipe.hdel(
                            self.ingestion_invalidation_payload_key,
                            registered_attempt_id,
                        )
                        pipe.zrem(
                            self.ingestion_invalidation_due_key,
                            registered_attempt_id,
                        )
                    elif registered_attempt_id is not None:
                        pipe.zadd(
                            self.ingestion_invalidation_due_key,
                            {
                                registered_attempt_id: datetime.now(
                                    timezone.utc
                                ).timestamp()
                            },
                        )
                    pipe.lrem(self.processing_key, 0, job_id_value)
                    pipe.lrem(self.queue_key, 0, job_id_value)
                    if active_key is not None:
                        pipe.delete(active_key)
                    if self.result_ttl_seconds > 0:
                        pipe.expire(key, self.result_ttl_seconds)
                    await pipe.execute()
                    return True
                except WatchError:
                    continue
        raise JobStoreError("could not finish job because its state kept changing")

    @staticmethod
    def _public_job(
        record: Mapping[str, Any],
        *,
        include_payload: bool,
        include_lease: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "job_id": record["job_id"],
            "kind": record["kind"],
            "status": record["status"],
            "cancel_requested": record.get("cancel_requested") == "1",
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "enqueued_at": record.get("enqueued_at"),
        }
        for key in (
            "owner_id",
            "started_at",
            "attempt_started_at",
            "attempt",
            "completed_at",
            "worker_id",
            "heartbeat_at",
            "cancel_requested_at",
            "cancel_reason",
            "requeue_reason",
            "result",
            "error",
        ):
            if record.get(key) not in (None, ""):
                result[key] = record[key]
        if include_payload:
            result["payload"] = _json_loads(record.get("payload"), {})
        if include_lease and record.get("lease_token"):
            result["lease_token"] = record["lease_token"]
        return result


_default_store: Optional[RedisJobStore] = None


def get_job_store() -> RedisJobStore:
    global _default_store
    if _default_store is None:
        _default_store = RedisJobStore()
    return _default_store


async def create_job(
    kind: str,
    payload: Mapping[str, Any],
    *,
    owner_id: Optional[str] = None,
    coalesce_active: bool = False,
) -> dict[str, Any]:
    return await get_job_store().create_job(
        kind,
        payload,
        owner_id=owner_id,
        coalesce_active=coalesce_active,
    )


async def enqueue_job(
    kind: str,
    payload: Mapping[str, Any],
    *,
    owner_id: Optional[str] = None,
    coalesce_active: bool = True,
) -> dict[str, Any]:
    return await create_job(
        kind,
        payload,
        owner_id=owner_id,
        coalesce_active=coalesce_active,
    )


async def get_job_status(job_id: str) -> Optional[dict[str, Any]]:
    return await get_job_store().get_status(job_id)


async def get_job_result(job_id: str) -> Optional[dict[str, Any]]:
    return await get_job_store().get_result(job_id)


async def request_cancellation(job_id: str) -> Optional[dict[str, Any]]:
    return await get_job_store().request_cancellation(job_id)


async def claim_job(timeout: float = 1.0, worker_id: Optional[str] = None) -> Optional[dict[str, Any]]:
    return await get_job_store().claim_job(timeout=timeout, worker_id=worker_id)


async def heartbeat_job(job_id: str, worker_id: str, lease_token: str) -> bool:
    return await get_job_store().heartbeat_job(job_id, worker_id, lease_token)


async def complete_job(
    job_id: str,
    result_metadata: Mapping[str, Any],
    *,
    lease_token: Optional[str] = None,
    successful_ingestion_attempt_id: Optional[str] = None,
) -> bool:
    return await get_job_store().complete_job(
        job_id,
        result_metadata,
        lease_token=lease_token,
        successful_ingestion_attempt_id=successful_ingestion_attempt_id,
    )


async def fail_job(
    job_id: str,
    error: Mapping[str, Any] | str,
    *,
    lease_token: Optional[str] = None,
) -> bool:
    return await get_job_store().fail_job(job_id, error, lease_token=lease_token)


async def cancel_job(
    job_id: str,
    reason: str = "cancellation requested",
    *,
    lease_token: Optional[str] = None,
) -> bool:
    return await get_job_store().mark_cancelled(
        job_id,
        reason,
        lease_token=lease_token,
    )


async def is_cancellation_requested(job_id: str) -> bool:
    return await get_job_store().is_cancellation_requested(job_id)


async def requeue_stale_jobs(stale_after_seconds: Optional[int] = None) -> int:
    return await get_job_store().requeue_stale_jobs(stale_after_seconds)


async def close_job_store() -> None:
    global _default_store
    if _default_store is not None:
        await _default_store.close()
        _default_store = None
