import importlib.util
import hashlib
import os
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from artifact_store import ArtifactStore


class FakeFastMCP:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.auth = kwargs.get("auth")
        self.tools = []

    def tool(self, function=None, **kwargs):
        def decorate(target):
            self.tools.append(target.__name__)
            return target

        return decorate(function) if function is not None else decorate


class FakeStaticTokenVerifier:
    def __init__(self, tokens, required_scopes):
        self.tokens = tokens
        self.required_scopes = required_scopes


class FakeRequest:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


async def _unused_async(*args, **kwargs):
    return {}


def load_mcp_server(*, backend="redis", auth_token=""):
    fastmcp = types.ModuleType("fastmcp")
    fastmcp.__path__ = []
    fastmcp.FastMCP = FakeFastMCP
    fastmcp_server = types.ModuleType("fastmcp.server")
    fastmcp_server.__path__ = []
    fastmcp_auth = types.ModuleType("fastmcp.server.auth")
    fastmcp_auth.StaticTokenVerifier = FakeStaticTokenVerifier

    browser = types.ModuleType("browser")
    browser.DEFAULT_MAX_CHARS = 300000
    extractors = types.ModuleType("extractors")
    extractors.clamp_int = lambda value, minimum, maximum: max(minimum, min(int(value), maximum))
    github = types.ModuleType("github_connector")
    github.get_github_file = _unused_async
    github.inspect_github_repository = _unused_async
    github.normalize_repository = lambda value: value
    github.search_github = _unused_async
    pipelines = types.ModuleType("pipelines")
    pipelines.build_evidence_pack = lambda results: list(results)
    pipelines.compact_investigation_result = lambda result, **kwargs: dict(result)
    pipelines.explore_url_pipeline = _unused_async
    pipelines.research_pipeline = _unused_async
    redaction = types.ModuleType("redaction")
    redaction.redact_sensitive_text = lambda text: (text, 0)
    searching = types.ModuleType("searching")
    searching.normalize_domain = lambda domain: domain
    shared = types.ModuleType("shared")
    shared.DEFAULT_NAMESPACE = "default"
    shared.IngestRequest = FakeRequest
    shared.QueryRequest = FakeRequest
    shared.delete_source_impl = _unused_async
    shared.get_domain = lambda url: "example.com"
    shared.list_sources_impl = _unused_async
    shared.normalize_namespace = lambda value: value
    shared.rag_ingest_impl = _unused_async
    shared.rag_query_impl = _unused_async
    shared.runtime_retrieval_context = lambda: {"current_date_utc": "2026-07-11"}
    shared.source_stats_impl = _unused_async

    stubs = {
        "fastmcp": fastmcp,
        "fastmcp.server": fastmcp_server,
        "fastmcp.server.auth": fastmcp_auth,
        "browser": browser,
        "extractors": extractors,
        "github_connector": github,
        "pipelines": pipelines,
        "redaction": redaction,
        "searching": searching,
        "shared": shared,
    }
    module_path = Path(__file__).resolve().parents[1] / "mcp_server.py"
    spec = importlib.util.spec_from_file_location(f"mcp_server_test_{uuid.uuid4().hex}", module_path)
    module = importlib.util.module_from_spec(spec)
    environment = {"JOB_BACKEND": backend, "MCP_AUTH_TOKEN": auth_token}
    with patch.dict(sys.modules, stubs, clear=False), patch.dict(os.environ, environment, clear=False):
        spec.loader.exec_module(module)
    return module


class MCPJobIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_research_enqueues_complete_scoped_payload(self):
        server = load_mcp_server()
        job_id = uuid.uuid4().hex
        enqueue = AsyncMock(
            return_value={"job_id": job_id, "kind": "research_web", "status": "queued"}
        )

        with patch.object(server, "enqueue_job", enqueue):
            result = await server.start_research(
                "current evidence",
                mode="deep",
                max_sources=5,
                verify=False,
                namespace="project-a",
                include_memory=True,
                synthesize=True,
            )

        enqueue.assert_awaited_once_with(
            "research_web",
            {
                "query": "current evidence",
                "mode": "deep",
                "max_sources": 5,
                "verify": False,
                "namespace": "project-a",
                "include_memory": True,
                "synthesize": True,
            },
        )
        self.assertEqual(result["job_id"], job_id)
        self.assertFalse(result["terminal"])
        self.assertGreaterEqual(result["retry_after_seconds"], 5)
        instructions = " ".join(result["answering_instructions"])
        self.assertIn("never ask the user", instructions.lower())
        self.assertIn("at most once", instructions.lower())
        self.assertNotIn("report the job ID and check it later", instructions)
        self.assertNotIn("ask the user to call research_job", instructions.lower())
        self.assertIn("retrieval_context", result)

    async def test_research_job_routes_status_metadata_full_result_and_cancel(self):
        server = load_mcp_server()
        job_id = uuid.uuid4().hex

        with patch.object(
            server, "get_job_status", AsyncMock(return_value={"job_id": job_id, "status": "running"})
        ) as status:
            self.assertEqual(
                (await server.research_job("status", job_id, wait_seconds=0))["status"],
                "running",
            )
            status.assert_awaited_once_with(job_id)

        metadata_result = {
            "job_id": job_id,
            "status": "succeeded",
            "result": {"artifact_id": "a", "artifact_path": f"{job_id}/result.json"},
        }
        with patch.object(
            server,
            "get_job_status",
            AsyncMock(return_value={"job_id": job_id, "status": "succeeded"}),
        ), patch.object(
            server, "get_job_result", AsyncMock(return_value=metadata_result)
        ) as get_result:
            self.assertEqual(
                await server.research_job(
                    "result",
                    job_id,
                    include_full_result=False,
                    wait_seconds=0,
                ),
                metadata_result,
            )
            get_result.assert_awaited_once_with(job_id)

        full_result = {"query": "q", "job": {"job_id": job_id}}
        with patch.object(
            server,
            "get_job_status",
            AsyncMock(return_value={"job_id": job_id, "status": "succeeded"}),
        ), patch.object(server, "_load_completed_job", AsyncMock(return_value=full_result)) as load:
            self.assertEqual(
                await server.research_job("result", job_id, wait_seconds=0),
                full_result,
            )
            load.assert_awaited_once_with(job_id)

        cancelled = {"job_id": job_id, "status": "cancelled"}
        with patch.object(server, "request_cancellation", AsyncMock(return_value=cancelled)) as cancel:
            self.assertEqual(await server.research_job("cancel", job_id), cancelled)
            cancel.assert_awaited_once_with(job_id)

    async def test_research_job_long_poll_returns_completed_result_without_busy_loop(self):
        server = load_mcp_server()
        job_id = uuid.uuid4().hex
        running = {"job_id": job_id, "status": "running"}
        succeeded = {"job_id": job_id, "status": "succeeded"}
        full_result = {"query": "q", "job": {"job_id": job_id}}

        with patch.object(
            server, "get_job_status", AsyncMock(return_value=running)
        ), patch.object(
            server, "_wait_for_terminal_job", AsyncMock(return_value=succeeded)
        ) as wait, patch.object(
            server, "_load_completed_job", AsyncMock(return_value=full_result)
        ) as load:
            result = await server.research_job("result", job_id, wait_seconds=12)

        self.assertEqual(result, full_result)
        wait.assert_awaited_once_with(job_id, 12)
        load.assert_awaited_once_with(job_id)

    async def test_research_job_running_response_discourages_same_turn_polling(self):
        server = load_mcp_server()
        job_id = uuid.uuid4().hex
        running = {"job_id": job_id, "status": "running"}

        with patch.object(
            server, "get_job_status", AsyncMock(return_value=running)
        ), patch.object(
            server, "_wait_for_terminal_job", AsyncMock(return_value=running)
        ):
            result = await server.research_job("result", job_id, wait_seconds=5)

        self.assertEqual(result["status"], "running")
        self.assertFalse(result["terminal"])
        self.assertEqual(result["job_id"], job_id)
        instructions = " ".join(result["answering_instructions"])
        self.assertIn("never ask the user", instructions.lower())
        self.assertIn("at most once", instructions.lower())
        self.assertNotIn("report the job ID and check it later", instructions)
        self.assertNotIn("ask the user to call research_job", instructions.lower())
        self.assertEqual(
            result["automatic_continuation"],
            {
                "tool": "research_job",
                "action": "result",
                "job_id": job_id,
                "wait_seconds": server.MCP_JOB_LONG_POLL_SECONDS,
                "maximum_calls": 1,
            },
        )

    async def test_research_job_preserves_queued_status(self):
        server = load_mcp_server()
        job_id = uuid.uuid4().hex
        queued = {"job_id": job_id, "status": "queued"}

        with patch.object(server, "get_job_status", AsyncMock(return_value=queued)):
            result = await server.research_job("status", job_id, wait_seconds=0)

        self.assertEqual(result["status"], "queued")
        self.assertFalse(result["terminal"])

        with patch.object(server, "get_job_status", AsyncMock(return_value=queued)):
            result = await server.research_job("result", job_id, wait_seconds=0)

        self.assertEqual(result["status"], "queued")
        self.assertFalse(result["terminal"])

    async def test_completed_job_loads_full_json_artifact(self):
        server = load_mcp_server()
        job_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(temp_dir)
            artifact = await store.write_json(job_id, {"query": "q", "evidence": [1, 2]})
            stored = {
                "job_id": job_id,
                "status": "succeeded",
                "result": {
                    "artifact_id": artifact["artifact_id"],
                    "artifact_path": artifact["relative_path"],
                },
            }
            with patch.object(server, "get_job_result", AsyncMock(return_value=stored)), patch.object(
                server, "get_artifact_store", return_value=store
            ):
                result = await server._load_completed_job(job_id)

        self.assertEqual(result["evidence"], [1, 2])
        self.assertEqual(result["job"]["artifact_id"], artifact["artifact_id"])
        self.assertEqual(result["job"]["job_id"], job_id)
        self.assertNotIn("artifact_path", result["job"])
        self.assertTrue(result["job"]["result_payload_complete"])
        self.assertTrue(result["result_payload_complete"])
        self.assertTrue(result["artifact_guidance"]["result_payload_complete"])
        self.assertFalse(result["artifact_guidance"]["job_result_artifact_path_exposed"])
        self.assertFalse(
            result["artifact_guidance"]["call_get_research_artifact_for_job_artifact"]
        )
        self.assertIn(
            "Do not call get_research_artifact",
            " ".join(result["answering_instructions"]),
        )

    async def test_full_result_hides_only_job_archive_path(self):
        server = load_mcp_server()
        source_path = "job/source-page.txt"
        result = server._complete_research_result(
            {
                "query": "q",
                "job": {
                    "job_id": "job",
                    "artifact_id": "job:result",
                    "artifact_path": "job/result.json",
                },
                "evidence": [
                    {
                        "artifact_id": "job:source-page",
                        "artifact_path": source_path,
                        "artifact_reference": {
                            "artifact_id": "job:source-page",
                            "artifact_path": source_path,
                        },
                    }
                ],
            }
        )

        self.assertNotIn("artifact_path", result["job"])
        self.assertEqual(result["job"]["artifact_id"], "job:result")
        self.assertTrue(result["job"]["result_payload_complete"])
        self.assertEqual(result["evidence"][0]["artifact_path"], source_path)
        self.assertEqual(
            result["evidence"][0]["artifact_reference"]["artifact_path"],
            source_path,
        )

    async def test_completed_result_keeps_evidence_authoritative_and_hides_deferred_manifest(self):
        server = load_mcp_server()
        evidence = [
            {
                "title": "Current source",
                "url": "https://example.com/current",
                "content": "Fresh current-run evidence",
            }
        ]
        result = server._complete_research_result(
            {
                "query": "current information",
                "evidence": evidence,
                "results": [],
                "memory_results": [],
                "persistence": {
                    "mode": "deferred",
                    "status": "queued",
                    "source_count": 1,
                },
                "_deferred_persistence": {
                    "namespace": "private-project",
                    "sources": [
                        {
                            "artifact_path": "job/private-source.txt",
                            "content": "private staging data",
                        }
                    ],
                },
            }
        )

        self.assertEqual(result["evidence"], evidence)
        self.assertEqual(result["results"], [])
        self.assertEqual(result["memory_results"], [])
        self.assertNotIn("_deferred_persistence", result)
        self.assertEqual(
            result["persistence"],
            {
                "mode": "deferred",
                "status": "queued",
                "source_count": 1,
            },
        )
        instructions = " ".join(result["answering_instructions"]).lower()
        self.assertIn("evidence array is the authoritative", instructions)
        self.assertIn("do not rerun research", instructions)
        self.assertIn("do not poll or delay", instructions)
        self.assertNotIn("private staging data", str(result))
        self.assertNotIn("job/private-source.txt", str(result))

    async def test_sync_research_tool_uses_queue_when_backend_is_redis(self):
        server = load_mcp_server(backend="redis")
        queued_result = {"job_id": uuid.uuid4().hex, "status": "running"}
        enqueue_and_wait = AsyncMock(return_value=queued_result)
        with patch.object(server, "_enqueue_and_wait", enqueue_and_wait):
            result = await server.research_web("question", namespace="project-a")

        self.assertEqual(result, queued_result)
        kind, payload, tool_name = enqueue_and_wait.await_args.args
        self.assertEqual((kind, tool_name), ("research_web", "research_web"))
        self.assertEqual(payload["query"], "question")
        self.assertEqual(payload["namespace"], "project-a")

    async def test_authenticated_inline_research_does_not_create_unowned_artifacts(self):
        server = load_mcp_server(backend="inline", auth_token="top-secret")
        pipeline = AsyncMock(return_value={"query": "question"})
        with patch.object(server, "_authorization_failure", return_value=None), patch.object(
            server, "research_pipeline", pipeline
        ):
            result = await server.research_web("question", namespace="project-a")

        self.assertEqual(result["query"], "question")
        self.assertTrue(result["artifact_guidance"]["result_payload_complete"])
        self.assertTrue(
            result["artifact_guidance"]["source_artifacts_may_contain_additional_content"]
        )
        self.assertFalse(pipeline.await_args.kwargs["persist_source_artifacts"])

    async def test_authenticated_stdio_tool_call_uses_local_trust_boundary(self):
        server = load_mcp_server(backend="inline", auth_token="top-secret")
        pipeline = AsyncMock(return_value={"query": "question"})
        with patch.dict(server.os.environ, {"MCP_TRANSPORT": "stdio"}, clear=False), patch.object(
            server, "research_pipeline", pipeline
        ):
            result = await server.research_web("question", namespace="project-a")

        self.assertEqual(result["query"], "question")
        self.assertFalse(
            result["artifact_guidance"]["call_get_research_artifact_for_job_artifact"]
        )
        self.assertTrue(pipeline.await_args.kwargs["persist_source_artifacts"])

    async def test_authenticated_http_tool_call_still_requires_access_token(self):
        server = load_mcp_server(backend="inline", auth_token="top-secret")
        query = AsyncMock(return_value={"results": []})
        with patch.dict(
            server.os.environ,
            {"MCP_TRANSPORT": "streamable-http"},
            clear=False,
        ), patch.object(server, "rag_query_impl", query):
            result = await server.query_memory("question")

        self.assertEqual(result, {"error": "authentication_required"})
        query.assert_not_awaited()

    async def test_inline_investigation_ingests_redirected_source_identity(self):
        server = load_mcp_server(backend="inline")
        requested_url = "https://start.example/path"
        final_url = "https://docs.example/final"
        explore = AsyncMock(
            return_value={
                "full_text_preview": "redirected content",
                "final_url": final_url,
                "title": "Final page",
            }
        )
        ingest = AsyncMock(return_value={"stored": 1})
        with patch.object(server, "explore_url_pipeline", explore), patch.object(
            server, "rag_ingest_impl", ingest
        ):
            result = await server.investigate_url(
                requested_url,
                "find details",
                auto_ingest=True,
            )

        metadata = ingest.await_args.args[0].metadata
        self.assertEqual(metadata["source"], final_url)
        self.assertEqual(metadata["url"], final_url)
        self.assertEqual(metadata["requested_url"], requested_url)
        self.assertEqual(result["stored_chunks"], 1)

    async def test_polling_failure_preserves_enqueued_job_id(self):
        server = load_mcp_server(backend="redis")
        job_id = uuid.uuid4().hex
        with patch.object(
            server,
            "enqueue_job",
            AsyncMock(return_value={"job_id": job_id, "status": "queued"}),
        ), patch.object(
            server,
            "get_job_status",
            AsyncMock(side_effect=OSError("redis temporarily unavailable")),
        ):
            result = await server._enqueue_and_wait("research_web", {"query": "q"}, "research_web")

        self.assertEqual(result["job_id"], job_id)
        self.assertEqual(result["warning"], "job_status_temporarily_unavailable")

    async def test_coalesced_sync_request_continues_waiting_for_completed_result(self):
        server = load_mcp_server(backend="redis")
        job_id = uuid.uuid4().hex
        completed = {
            "query": "same",
            "evidence": [{"url": "https://example.com/result"}],
        }
        with patch.object(
            server,
            "enqueue_job",
            AsyncMock(
                return_value={
                    "job_id": job_id,
                    "status": "running",
                    "coalesced": True,
                }
            ),
        ), patch.object(
            server,
            "get_job_status",
            AsyncMock(
                side_effect=[
                    {"job_id": job_id, "status": "running"},
                    {"job_id": job_id, "status": "succeeded"},
                ]
            ),
        ) as status, patch.object(
            server,
            "_load_completed_job",
            AsyncMock(return_value=completed),
        ) as load, patch.object(server.asyncio, "sleep", AsyncMock()) as sleep:
            result = await server._enqueue_and_wait(
                "research_web",
                {"query": "same"},
                "research_web",
            )

        self.assertEqual(result, completed)
        self.assertEqual(status.await_count, 2)
        sleep.assert_awaited_once()
        load.assert_awaited_once_with(job_id)

    async def test_disabled_backend_and_queue_failures_return_stable_errors(self):
        disabled = load_mcp_server(backend="inline")
        self.assertEqual((await disabled.start_research("q"))["error"], "durable_jobs_disabled")
        self.assertEqual(
            (await disabled.research_job("status", uuid.uuid4().hex))["error"],
            "durable_jobs_disabled",
        )

        server = load_mcp_server()
        with patch.object(server, "enqueue_job", AsyncMock(side_effect=OSError("redis down"))):
            result = await server.start_research("q")
        self.assertEqual(result["error"], "job_queue_unavailable")
        self.assertTrue(result["retryable"])

        with patch.object(
            server,
            "enqueue_job",
            AsyncMock(side_effect=server.JobQueueFullError("queue cap reached")),
        ):
            result = await server.start_research("q")
        self.assertEqual(result["error"], "job_queue_full")
        self.assertTrue(result["retryable"])

    async def test_research_admission_limit_is_a_structured_tool_response(self):
        server = load_mcp_server()
        limited = server.ResearchAdmissionLimitedError(
            "rolling_window",
            retry_after_seconds=37,
            active_jobs=0,
            max_active=1,
            recent_jobs=2,
            max_new_jobs=2,
            window_seconds=60,
        )
        with patch.object(server, "enqueue_job", AsyncMock(side_effect=limited)):
            sync_result = await server._enqueue_and_wait(
                "research_web",
                {"query": "third request"},
                "research_web",
            )
            durable_result = await server.start_research("third request")

        for result, tool_name in (
            (sync_result, "research_web"),
            (durable_result, "start_research"),
        ):
            self.assertEqual(result["error"], "research_admission_limited")
            self.assertEqual(result["tool"], tool_name)
            self.assertEqual(result["reason"], "rolling_window")
            self.assertTrue(result["retryable"])
            self.assertEqual(result["retry_after_seconds"], 37)
            self.assertEqual(result["limits"]["recent_jobs"], 2)
            self.assertEqual(result["limits"]["max_new_jobs"], 2)
            self.assertIn("retrieval_context", result)

    async def test_anonymous_admission_denial_is_not_marked_retryable(self):
        server = load_mcp_server()
        limited = server.ResearchAdmissionLimitedError(
            "anonymous_disabled",
            retry_after_seconds=60,
            active_jobs=0,
            max_active=1,
            recent_jobs=0,
            max_new_jobs=2,
            window_seconds=60,
        )
        with patch.object(server, "enqueue_job", AsyncMock(side_effect=limited)):
            result = await server.start_research("question")

        self.assertEqual(result["error"], "research_admission_limited")
        self.assertFalse(result["retryable"])
        self.assertIn("authenticated", " ".join(result["answering_instructions"]))

    async def test_inline_backend_does_not_use_redis_admission(self):
        server = load_mcp_server(backend="inline")
        pipeline = AsyncMock(return_value={"query": "local", "evidence": []})
        with patch.object(server, "enqueue_job", AsyncMock()) as enqueue, patch.object(
            server,
            "research_pipeline",
            pipeline,
        ):
            result = await server.research_web("local")

        enqueue.assert_not_awaited()
        pipeline.assert_awaited_once()
        self.assertEqual(result["query"], "local")

    async def test_inline_research_scopes_cache_to_current_principal(self):
        server = load_mcp_server(backend="inline")
        pipeline = AsyncMock(return_value={"query": "local", "evidence": []})

        with patch.object(
            server,
            "_current_principal_id",
            return_value="client-a",
        ), patch.object(server, "research_pipeline", pipeline):
            await server.research_web("local")

        expected_scope = hashlib.sha256(b"owner\x00client-a").hexdigest()
        self.assertEqual(
            pipeline.await_args.kwargs["search_cache_scope"],
            expected_scope,
        )
        self.assertNotIn("client-a", str(pipeline.await_args.kwargs))

    def test_auth_provider_is_optional_and_configures_required_scope(self):
        open_server = load_mcp_server(auth_token="")
        self.assertIsNone(open_server.mcp.auth)

        protected = load_mcp_server(auth_token="top-secret")
        self.assertIsInstance(protected.mcp.auth, FakeStaticTokenVerifier)
        self.assertIn("top-secret", protected.mcp.auth.tokens)
        self.assertEqual(protected.mcp.auth.required_scopes, ["research"])

    async def test_authenticated_client_cannot_read_another_clients_job(self):
        server = load_mcp_server(auth_token="top-secret")
        job_id = uuid.uuid4().hex
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research"],
            claims={"scopes": ["research"], "namespaces": ["*"]},
        )
        with patch.object(server, "_current_access_token", return_value=token), patch.object(
            server,
            "get_job_status",
            AsyncMock(return_value={"job_id": job_id, "status": "running", "owner_id": "client-b"}),
        ), patch.object(server, "request_cancellation", AsyncMock()) as cancel:
            result = await server.research_job("cancel", job_id)

        self.assertEqual(result["error"], "forbidden")
        cancel.assert_not_awaited()

    async def test_authenticated_artifact_path_cannot_traverse_to_another_job(self):
        server = load_mcp_server(auth_token="top-secret")
        owned_id = uuid.uuid4().hex
        victim_id = uuid.uuid4().hex
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research", "artifacts:read"],
            claims={
                "scopes": ["research", "artifacts:read"],
                "namespaces": ["*"],
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(temp_dir)
            victim = await store.write_text(victim_id, "victim secret")
            traversal = f"{owned_id}/../{victim['relative_path']}"
            status = AsyncMock(
                return_value={"job_id": owned_id, "status": "succeeded", "owner_id": "client-a"}
            )
            with patch.object(server, "_current_access_token", return_value=token), patch.object(
                server, "get_job_status", status
            ), patch.object(server, "get_artifact_store", return_value=store):
                result = await server.get_research_artifact(traversal)

        self.assertEqual(result["error"], "artifact_unavailable")
        self.assertNotIn("victim secret", str(result))
        status.assert_not_awaited()

    async def test_artifact_read_error_does_not_expose_storage_root(self):
        server = load_mcp_server(auth_token="top-secret")
        job_id = uuid.uuid4().hex
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research", "artifacts:read"],
            claims={
                "scopes": ["research", "artifacts:read"],
                "namespaces": ["*"],
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(temp_dir)
            with patch.object(server, "_current_access_token", return_value=token), patch.object(
                server,
                "get_job_status",
                AsyncMock(
                    return_value={"job_id": job_id, "status": "succeeded", "owner_id": "client-a"}
                ),
            ), patch.object(server, "get_artifact_store", return_value=store):
                result = await server.get_research_artifact(f"{job_id}/missing.json")

        self.assertEqual(result["error"], "artifact_unavailable")
        self.assertNotIn(temp_dir, str(result))

    async def test_persistent_binding_authorizes_artifact_after_job_metadata_expires(self):
        server = load_mcp_server(auth_token="top-secret")
        job_id = uuid.uuid4().hex
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research", "artifacts:read"],
            claims={"scopes": ["research", "artifacts:read"], "namespaces": ["*"]},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(temp_dir)
            await store.bind_owner_principal(job_id, "client-a")
            artifact = await store.write_text(job_id, "owned evidence")
            with patch.object(server, "_current_access_token", return_value=token), patch.object(
                server, "get_job_status", AsyncMock(return_value=None)
            ), patch.object(server, "get_artifact_store", return_value=store):
                result = await server.get_research_artifact(artifact["relative_path"])

        self.assertEqual(result["content"], "owned evidence")

    async def test_persistent_binding_mismatch_denies_artifact(self):
        server = load_mcp_server(auth_token="top-secret")
        job_id = uuid.uuid4().hex
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research", "artifacts:read"],
            claims={"scopes": ["research", "artifacts:read"], "namespaces": ["*"]},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(temp_dir)
            await store.bind_owner_principal(job_id, "client-b")
            artifact = await store.write_text(job_id, "other evidence")
            with patch.object(server, "_current_access_token", return_value=token), patch.object(
                server,
                "get_job_status",
                AsyncMock(
                    return_value={
                        "job_id": job_id,
                        "status": "succeeded",
                        "owner_id": "client-a",
                    }
                ),
            ), patch.object(server, "get_artifact_store", return_value=store):
                result = await server.get_research_artifact(artifact["relative_path"])

        self.assertEqual(result["error"], "forbidden")
        self.assertNotIn("other evidence", str(result))

    async def test_artifact_owner_binding_file_is_not_exposed(self):
        server = load_mcp_server(auth_token="top-secret")
        job_id = uuid.uuid4().hex
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research", "artifacts:read"],
            claims={"scopes": ["research", "artifacts:read"], "namespaces": ["*"]},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ArtifactStore(temp_dir)
            await store.bind_owner_principal(job_id, "client-a")
            with patch.object(server, "_current_access_token", return_value=token), patch.object(
                server, "get_job_status", AsyncMock(return_value=None)
            ), patch.object(server, "get_artifact_store", return_value=store):
                result = await server.get_research_artifact(f"{job_id}/_owner.json")

        self.assertEqual(result["error"], "forbidden")
        self.assertNotIn("principal_id", str(result))

    async def test_repository_scoped_client_cannot_run_global_github_search(self):
        server = load_mcp_server(auth_token="top-secret")
        token = types.SimpleNamespace(
            client_id="client-a",
            scopes=["research", "github:read"],
            claims={
                "scopes": ["research", "github:read"],
                "namespaces": ["*"],
                "github_repositories": ["owner/allowed"],
            },
        )
        search = AsyncMock(return_value={"results": []})
        with patch.object(server, "_current_access_token", return_value=token), patch.object(
            server, "search_github", search
        ), patch.dict(server.os.environ, {"GITHUB_TOKEN": ""}):
            result = await server.github_research(
                "search",
                query="repo:owner/forbidden secret",
            )

        self.assertEqual(result["error"], "forbidden")
        search.assert_not_awaited()

    async def test_unknown_github_action_is_validated_before_authorization(self):
        server = load_mcp_server(auth_token="top-secret")
        with patch.object(server, "_authorization_failure") as authorize:
            result = await server.github_research("unsupported")

        self.assertEqual(result["valid_actions"], ["search", "inspect", "read"])
        authorize.assert_not_called()

    def test_stdio_startup_has_no_http_only_arguments(self):
        server = load_mcp_server(backend="inline")
        with patch.dict(server.os.environ, {"MCP_TRANSPORT": "stdio"}, clear=False):
            self.assertEqual(server._build_run_kwargs(), {"transport": "stdio"})

    def test_http_startup_enables_strict_host_origin_protection(self):
        server = load_mcp_server(backend="inline")
        environment = {
            "MCP_TRANSPORT": "streamable-http",
            "MCP_HOST": "0.0.0.0",
            "MCP_EXTERNAL_BIND_ADDRESS": "127.0.0.1",
            "MCP_ALLOWED_HOSTS": "127.0.0.1:*,localhost:*,mcp-gateway:*",
            "MCP_ALLOWED_ORIGINS": "https://research.example.com",
            "MCP_PATH": "/mcp",
        }
        with patch.dict(server.os.environ, environment, clear=False):
            kwargs = server._build_run_kwargs()

        self.assertIs(kwargs["host_origin_protection"], True)
        self.assertEqual(
            kwargs["allowed_hosts"],
            ["127.0.0.1:*", "localhost:*", "mcp-gateway:*"],
        )
        self.assertEqual(kwargs["allowed_origins"], ["https://research.example.com"])
        self.assertEqual(kwargs["path"], "/mcp")

        from mcp.server.transport_security import (
            TransportSecurityMiddleware,
            TransportSecuritySettings,
        )

        middleware = TransportSecurityMiddleware(
            TransportSecuritySettings(allowed_hosts=kwargs["allowed_hosts"])
        )
        self.assertTrue(middleware._validate_host("127.0.0.1:8001"))
        self.assertTrue(middleware._validate_host("mcp-gateway:8001"))
        self.assertFalse(middleware._validate_host("attacker.example:8001"))

    def test_http_default_host_allowlist_accepts_port_bearing_hosts_and_ipv6(self):
        server = load_mcp_server(backend="inline")
        with patch.dict(server.os.environ, {}, clear=True):
            settings = server._http_security_settings("0.0.0.0", "127.0.0.1")

        self.assertEqual(
            settings["allowed_hosts"],
            [
                "127.0.0.1",
                "127.0.0.1:*",
                "localhost",
                "localhost:*",
                "[::1]",
                "[::1]:*",
            ],
        )

        from mcp.server.transport_security import (
            TransportSecurityMiddleware,
            TransportSecuritySettings,
        )

        middleware = TransportSecurityMiddleware(
            TransportSecuritySettings(allowed_hosts=settings["allowed_hosts"])
        )
        self.assertTrue(middleware._validate_host("127.0.0.1:8001"))
        self.assertTrue(middleware._validate_host("localhost:8001"))
        self.assertTrue(middleware._validate_host("[::1]:8001"))

    def test_http_startup_rejects_global_host_wildcard_and_invalid_origin(self):
        server = load_mcp_server(backend="inline")
        base = {
            "MCP_TRANSPORT": "streamable-http",
            "MCP_HOST": "127.0.0.1",
            "MCP_EXTERNAL_BIND_ADDRESS": "127.0.0.1",
        }
        with patch.dict(
            server.os.environ,
            {**base, "MCP_ALLOWED_HOSTS": "*", "MCP_ALLOWED_ORIGINS": ""},
            clear=False,
        ), self.assertRaisesRegex(ValueError, "global"):
            server._build_run_kwargs()
        with patch.dict(
            server.os.environ,
            {
                **base,
                "MCP_ALLOWED_HOSTS": "localhost",
                "MCP_ALLOWED_ORIGINS": "https://user:pass@example.com",
            },
            clear=False,
        ), self.assertRaisesRegex(ValueError, "origins"):
            server._build_run_kwargs()


if __name__ == "__main__":
    unittest.main()
