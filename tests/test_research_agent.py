import json
import time
from unittest.mock import AsyncMock

import pytest

import planner
import redaction
import research_agent
import searching


def _evidence(evidence_id=1, url="https://docs.example.com/guide"):
    return {
        "evidence_id": evidence_id,
        "title": "Official guide",
        "url": url,
        "quote": "Install the package with the documented command.",
        "evidence_type": "extracted_page_content",
    }


@pytest.mark.asyncio
async def test_unified_agent_fails_clearly_without_internal_model(monkeypatch):
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: False)

    result = await research_agent.run_research_assistant("Find the current setup guide")

    assert result["status"] == "configuration_required"
    assert result["error"] == "internal_research_model_not_configured"
    assert result["required_settings"] == [
        "RESEARCH_MODEL_BASE_URL",
        "RESEARCH_MODEL_NAME",
    ]
    assert "API_KEY" not in result["detail"]


@pytest.mark.asyncio
async def test_model_plan_is_bounded_redacted_and_deterministically_augmented(monkeypatch):
    monkeypatch.setattr(
        research_agent,
        "_chat",
        AsyncMock(
            return_value=(
                '{"mode":"invalid","use_web_search":false,'
                '"queries":["install product sk-secretvalue12345678901234567890",'
                '"second query","third query","fourth query","fifth query","sixth query"],'
                '"include_images":false}'
            )
        ),
    )
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )

    plan = await research_agent.build_assistant_plan(
        "Find current install instructions and screenshots",
        mode="auto",
    )

    assert plan["mode"] == "technical"
    assert plan["use_web_search"] is True
    assert plan["include_images"] is True
    assert 1 <= len(plan["queries"]) <= 5
    assert all(len(query) <= 180 for query in plan["queries"])
    assert "sk-secretvalue" not in " ".join(plan["queries"])


def test_deterministic_fallback_preserves_public_subject_and_event_date(monkeypatch):
    request = (
        "Summarize today's latest news on the Iran war/conflict as of "
        "March 28, 2026. Identify the most important developments."
    )
    monkeypatch.setattr(
        research_agent,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-08-03"},
    )
    monkeypatch.setattr(
        planner,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-08-03"},
    )

    plan = research_agent.deterministic_assistant_plan(request, "quick")

    assert plan["queries"]
    assert "iran" in plan["queries"][0].lower()
    assert "2026" in plan["queries"][0]
    assert plan["queries"][0].casefold() != "news"


def test_deterministic_fallback_retains_safe_term_before_sentence_period():
    plan = research_agent.deterministic_assistant_plan(
        "Codename Apollo Blue failed with Docker.",
        "quick",
    )

    assert "Docker" in plan["queries"][0]
    assert "Apollo" not in " ".join(plan["queries"])


@pytest.mark.parametrize(
    "private_host",
    [
        "db.prod.internal:5432",
        "api.service.corp",
        "db.home:5432",
        "host.localdomain:6379",
        "localhost:8080",
        "[fd00::1234]:5432",
        "fe80::1",
    ],
)
def test_deterministic_fallback_removes_bare_private_hosts(private_host):
    plan = research_agent.deterministic_assistant_plan(
        f"Fix connection refused from {private_host} in Docker",
        "quick",
    )

    serialized = json.dumps(plan)
    assert private_host not in serialized
    assert "Docker" in plan["queries"][0]
    assert "connection refused" in plan["queries"][0]


@pytest.mark.parametrize(
    "private_endpoint",
    [
        "10.20.30.40:5432/private-route-9472",
        "db.prod.internal:5432/private-route-9472",
        "db.prod.internal:5432/reset(private-route-9472)",
        "[fd00::1234]:5432/private-route-9472",
        "fe80::1/private-route-9472",
    ],
)
def test_deterministic_fallback_removes_private_endpoint_paths(private_endpoint):
    plan = research_agent.deterministic_assistant_plan(
        f"Fix connection refused from {private_endpoint} in Docker",
        "quick",
    )

    serialized = json.dumps(plan)
    assert "private-route-9472" not in serialized
    assert "Docker" in plan["queries"][0]


def test_parenthesized_url_path_never_becomes_a_public_search_query():
    secret_path = "secret-token-value-123456789"
    request = f"Research https://docs.example.com/reset({secret_path})"

    plan = research_agent.deterministic_assistant_plan(request, "quick")

    assert secret_path not in " ".join(plan["queries"])
    assert plan["_execution_urls"] == [
        f"https://docs.example.com/reset({secret_path})"
    ]


def test_model_repeated_url_path_never_becomes_a_public_search_query():
    secret_path = "secret-token-value-123456789"
    request = f"Research Docker using https://docs.example.com/reset({secret_path})"

    plan = research_agent._normalize_plan(
        {
            "mode": "quick",
            "queries": [
                f"Docker reset guide https://docs.example.com/reset({secret_path})"
            ],
        },
        request,
        "quick",
    )

    assert secret_path not in " ".join(plan["queries"])


@pytest.mark.asyncio
async def test_model_assistant_plan_replaces_generic_query_with_public_anchor(monkeypatch):
    request = (
        "Summarize today's latest news on the Iran war/conflict as of "
        "March 28, 2026. Identify the most important developments."
    )
    monkeypatch.setattr(
        research_agent,
        "_chat",
        AsyncMock(
            return_value=json.dumps(
                {
                    "mode": "quick",
                    "use_web_search": True,
                    "queries": ["news"],
                    "include_images": False,
                }
            )
        ),
    )
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )
    monkeypatch.setattr(
        research_agent,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-08-03"},
    )
    monkeypatch.setattr(
        planner,
        "runtime_retrieval_context",
        lambda: {"current_date_local": "2026-08-03"},
    )

    plan = await research_agent.build_assistant_plan(request, mode="quick")

    assert plan["queries"]
    assert "iran" in plan["queries"][0].lower()
    assert all(query.casefold() != "news" for query in plan["queries"])


@pytest.mark.asyncio
async def test_explicit_mode_cannot_be_overridden_by_planning_model(monkeypatch):
    monkeypatch.setattr(
        research_agent,
        "_chat",
        AsyncMock(
            return_value=(
                '{"mode":"deep","use_web_search":true,'
                '"queries":["official product documentation"]}'
            )
        ),
    )
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )

    plan = await research_agent.build_assistant_plan(
        "Find the product documentation",
        mode="quick",
    )

    assert plan["mode"] == "quick"


@pytest.mark.asyncio
async def test_all_unified_model_calls_redact_private_boundary_data(monkeypatch):
    chat = AsyncMock(
        side_effect=[
            (
                '{"mode":"technical","use_web_search":true,'
                '"queries":["official package installation"],"include_images":false}'
            ),
            '{"needs_follow_up":false,"reason":"API_KEY=echoed-secret-value"}',
            "Install the package with the documented command [E1].",
        ]
    )
    monkeypatch.setattr(research_agent, "_chat", chat)
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )
    request = (
        "API_KEY=supersecretvalue123; contact alice@example.com; inspect "
        "/srv/private/error.log and "
        "https://docs.example.com/a%2Fb?z=2&sessionid=SESSIONSECRET&a=one%20two"
    )
    evidence = research_agent._merge_evidence(
        [
            {
                "title": "Guide for alice@example.com",
                "url": "https://docs.example.com/guide",
                "quote": (
                    "Install the package with the documented command. "
                    "TOKEN=anothersecretvalue123 at /srv/private/install.log"
                ),
            }
        ]
    )

    await research_agent.build_assistant_plan(request)
    review = await research_agent._review_evidence(request, evidence)
    answer = await research_agent._write_answer(request, evidence)

    serialized_messages = json.dumps(
        [call.args[0] for call in chat.await_args_list],
        ensure_ascii=True,
    )
    for private_value in (
        "supersecretvalue123",
        "alice@example.com",
        "/srv/private/error.log",
        "SESSIONSECRET",
        "anothersecretvalue123",
        "/srv/private/install.log",
    ):
        assert private_value not in serialized_messages
    assert review["reason"] == "model_found_no_actionable_evidence_gap"
    assert "echoed-secret-value" not in json.dumps(review)
    assert answer["citation_validation"]["valid"] is True


@pytest.mark.asyncio
async def test_advanced_planner_and_synthesis_redact_model_boundary_data(monkeypatch):
    chat = AsyncMock(
        side_effect=[
            '{"queries":[],"subquestions":[]}',
            "Install the package [E1].",
        ]
    )
    monkeypatch.setattr(planner, "_chat", chat)
    monkeypatch.setattr(planner, "research_model_configured", lambda: True)
    monkeypatch.setattr(planner, "PLANNER_ENABLE_SYNTHESIS", True)
    request = (
        "Investigate package behavior for alice@example.com using "
        "API_KEY=supersecretvalue123 and /srv/private/error.log"
    )
    evidence = [
        {
            "evidence_id": 1,
            "title": "Guide for alice@example.com",
            "url": "https://docs.example.com/guide?sessionid=SESSIONSECRET",
            "quote": (
                "Install the package. Bearer ABCDEFGHIJKLMNOPQRSTUV at "
                "/srv/private/install.log"
            ),
        }
    ]

    await planner.build_research_plan(request, "deep")
    report = await planner.synthesize_report(request, evidence)

    assert chat.await_count == 2
    serialized_messages = json.dumps(
        [call.args[0] for call in chat.await_args_list],
        ensure_ascii=True,
    )
    for private_value in (
        "alice@example.com",
        "supersecretvalue123",
        "/srv/private/error.log",
        "SESSIONSECRET",
        "ABCDEFGHIJKLMNOPQRSTUV",
        "/srv/private/install.log",
    ):
        assert private_value not in serialized_messages
    assert report["citation_validation"]["valid"] is True


@pytest.mark.asyncio
async def test_unified_agent_runs_at_most_one_follow_up_and_returns_finished_answer(
    monkeypatch,
):
    plan = {
        "mode": "balanced",
        "queries": ["official product setup documentation"],
        "use_web_search": True,
        "urls": [],
        "github_repositories": [],
        "github_searches": [],
        "include_memory": False,
        "include_images": False,
        "image_query": "product setup",
        "answer_focus": "installation",
        "generated_by": "model:private-model",
    }
    first = {
        "evidence": [_evidence()],
        "completion": {"status": "complete"},
        "plan": {"queries": ["official product setup documentation"]},
        "crawled_sources": [{"url": "https://docs.example.com/guide"}],
        "_deferred_persistence": {
            "sources": [{"job_id": "a" * 32, "artifact_owner_id": "a" * 32}]
        },
    }
    second = {
        "evidence": [_evidence(1, "https://support.example.org/install")],
        "completion": {"status": "complete"},
        "crawled_sources": [{"url": "https://support.example.org/install"}],
    }
    pipeline = AsyncMock(side_effect=[first, second])
    review = AsyncMock(
        return_value={
            "needs_follow_up": True,
            "queries": ["product setup known limitations"],
            "reason": "verify limitations",
        }
    )
    write = AsyncMock(
        return_value={
            "answer_markdown": "Use the documented command ([source](https://docs.example.com/guide)).",
            "citations": [
                {
                    "evidence_id": 1,
                    "title": "Official guide",
                    "url": "https://docs.example.com/guide",
                }
            ],
            "citation_validation": {"valid": True},
            "generated_by": "model:private-model",
        }
    )
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(research_agent, "build_assistant_plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(research_agent, "research_pipeline", pipeline)
    monkeypatch.setattr(research_agent, "_review_evidence", review)
    monkeypatch.setattr(research_agent, "_write_answer", write)
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 1)

    result = await research_agent.run_research_assistant("How do I install the product?")

    assert result["status"] == "complete"
    assert result["answer_markdown"].startswith("Use the documented command")
    assert result["research_summary"]["follow_up"]["attempted"] is True
    assert result["research_summary"]["sources_consulted"] == 2
    assert pipeline.await_count == 2
    review.assert_awaited_once()
    write.assert_awaited_once()
    assert result["_deferred_persistence"]["sources"][0]["job_id"] == "a" * 32


@pytest.mark.asyncio
async def test_auto_mode_uses_bounded_quick_path_for_short_timeout_clients(monkeypatch):
    plan = {
        "mode": "quick",
        "queries": ["official product setup documentation"],
        "use_web_search": True,
        "urls": [],
        "github_repositories": [],
        "github_searches": [],
        "include_memory": False,
        "include_images": False,
        "image_query": "product setup",
        "answer_focus": "installation",
        "generated_by": "model:private-model",
    }
    pipeline = AsyncMock(
        return_value={
            "evidence": [_evidence()],
            "completion": {"status": "complete"},
            "plan": {"queries": ["official product setup documentation"]},
        }
    )
    review = AsyncMock()
    write = AsyncMock(
        return_value={
            "answer_markdown": "Use the documented command.",
            "citations": [{"evidence_id": 1, "url": _evidence()["url"]}],
            "citation_validation": {"valid": True},
            "generated_by": "model:private-model",
        }
    )
    planner = AsyncMock(return_value=plan)
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(research_agent, "RESEARCH_ASSISTANT_AUTO_MODE", "quick")
    monkeypatch.setattr(research_agent, "build_assistant_plan", planner)
    monkeypatch.setattr(research_agent, "research_pipeline", pipeline)
    monkeypatch.setattr(research_agent, "_review_evidence", review)
    monkeypatch.setattr(research_agent, "_write_answer", write)
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 1)

    result = await research_agent.run_research_assistant(
        "How do I install the product?",
        mode="auto",
    )

    assert result["status"] == "complete"
    planner.assert_awaited_once_with(
        "How do I install the product?",
        "quick",
        timeout_seconds=research_agent.RESEARCH_AGENT_QUICK_PLAN_TIMEOUT_SECONDS,
    )
    assert pipeline.await_args.kwargs["mode"] == "quick"
    assert pipeline.await_args.kwargs["verify"] is False
    review.assert_not_awaited()
    write.assert_awaited_once()
    assert (
        write.await_args.kwargs["timeout_seconds"]
        == research_agent.RESEARCH_AGENT_QUICK_SYNTHESIS_TIMEOUT_SECONDS
    )


@pytest.mark.asyncio
async def test_public_web_fallback_redacts_complete_request(monkeypatch):
    plan = {
        "mode": "balanced",
        "queries": [],
        "use_web_search": True,
        "urls": [],
        "github_repositories": [],
        "github_searches": [],
        "include_memory": False,
        "include_images": False,
        "image_query": "product setup",
        "answer_focus": "installation",
        "generated_by": "model:private-model",
    }
    pipeline = AsyncMock(
        return_value={
            "evidence": [_evidence()],
            "completion": {"status": "complete"},
            "plan": {"queries": ["product setup"]},
        }
    )
    write = AsyncMock(
        return_value={
            "answer_markdown": "Answer with a source.",
            "citations": [{"evidence_id": 1, "url": _evidence()["url"]}],
            "citation_validation": {"valid": True},
            "generated_by": "model:private-model",
        }
    )
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(research_agent, "build_assistant_plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(research_agent, "research_pipeline", pipeline)
    monkeypatch.setattr(research_agent, "_write_answer", write)
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 0)

    request = "API_KEY=supersecretvalue\nFind the current product setup guide"
    result = await research_agent.run_research_assistant(request)

    public_query = pipeline.await_args.kwargs["query"]
    assert "supersecretvalue" not in public_query
    assert public_query == "installation"
    assert result["research_summary"]["public_query_redactions_applied"] == 1
    assert write.await_args.args[0] == request
    assert write.await_args.args[1][0]["url"] == _evidence()["url"]


@pytest.mark.asyncio
async def test_planner_failure_uses_deterministic_search_fallback(monkeypatch):
    pipeline = AsyncMock(
        return_value={
            "evidence": [_evidence()],
            "completion": {"status": "complete"},
            "plan": {"queries": ["Docker installation documentation"]},
        }
    )
    write = AsyncMock(
        return_value={
            "answer_markdown": "Documented answer.",
            "citations": [{"evidence_id": 1, "url": _evidence()["url"]}],
            "citation_validation": {"valid": True},
            "generated_by": "model:private-model",
        }
    )
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(
        research_agent,
        "build_assistant_plan",
        AsyncMock(side_effect=TimeoutError()),
    )
    monkeypatch.setattr(research_agent, "research_pipeline", pipeline)
    monkeypatch.setattr(research_agent, "_write_answer", write)
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 0)

    result = await research_agent.run_research_assistant(
        "How do I install Docker for internal project Apollo Blue?"
    )

    assert result["status"] == "complete"
    assert result["research_summary"]["plan"]["generated_by"] == "deterministic-fallback"
    assert result["research_summary"]["planning_warning"] == "internal research model timed out"
    assert "Apollo Blue" not in pipeline.await_args.kwargs["query"]


@pytest.mark.asyncio
async def test_failed_explicit_url_recovers_with_one_web_fallback(monkeypatch):
    plan = {
        "mode": "balanced",
        "queries": [],
        "use_web_search": False,
        "urls": ["https://docs.example.com/missing"],
        "github_repositories": [],
        "github_searches": [],
        "include_memory": False,
        "include_images": False,
        "image_query": "documentation",
        "answer_focus": "",
        "generated_by": "model:private-model",
    }
    pipeline = AsyncMock(
        return_value={
            "evidence": [_evidence()],
            "completion": {"status": "complete"},
            "plan": {"queries": ["documentation"]},
        }
    )
    write = AsyncMock(
        return_value={
            "answer_markdown": "Documented answer.",
            "citations": [{"evidence_id": 1, "url": _evidence()["url"]}],
            "citation_validation": {"valid": True},
            "generated_by": "model:private-model",
        }
    )
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(research_agent, "build_assistant_plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(
        research_agent,
        "_acquire_url",
        AsyncMock(return_value={"error": "not_found", "evidence": []}),
    )
    monkeypatch.setattr(research_agent, "research_pipeline", pipeline)
    monkeypatch.setattr(research_agent, "_write_answer", write)
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 1)

    result = await research_agent.run_research_assistant(
        "Investigate https://docs.example.com/missing"
    )

    assert result["research_summary"]["follow_up"]["attempted"] is True
    assert result["research_summary"]["follow_up"]["reason"] == (
        "initial_acquisition_returned_no_evidence"
    )
    pipeline.assert_awaited_once()


@pytest.mark.asyncio
async def test_structured_github_search_is_scoped_and_uses_requested_kind(monkeypatch):
    search = AsyncMock(
        return_value={
            "type": "search",
            "results": [
                {
                    "name": "Fix the startup error",
                    "url": "https://github.com/example/project/issues/7",
                    "text_match": "The issue is fixed in the current release.",
                }
            ],
        }
    )
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(research_agent, "search_github", search)

    result = await research_agent._acquire_github_search(
        {
            "query": "startup error",
            "kind": "issues",
            "repository": "example/project",
        }
    )

    search.assert_awaited_once_with(
        "startup error",
        kind="issues",
        repository="example/project",
        max_results=8,
    )
    assert result["evidence"][0]["evidence_type"] == "github_search_result"


@pytest.mark.asyncio
async def test_github_code_search_reads_bounded_matching_files(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(
        research_agent,
        "search_github",
        AsyncMock(
            return_value={
                "results": [
                    {
                        "name": "compose.py",
                        "url": "https://github.com/example/project/blob/main/compose.py",
                        "repository": "example/project",
                        "path": "compose.py",
                    }
                ]
            }
        ),
    )
    read_file = AsyncMock(
        return_value={
            "type": "file",
            "repository": "example/project",
            "path": "compose.py",
            "url": "https://github.com/example/project/blob/main/compose.py",
            "content": "def build_compose(): return True",
        }
    )
    monkeypatch.setattr(research_agent, "get_github_file", read_file)

    result = await research_agent._acquire_github_search(
        {"query": "build_compose", "kind": "code", "repository": None}
    )

    read_file.assert_awaited_once_with(
        "example/project",
        "compose.py",
        max_chars=20_000,
    )
    assert any(item["evidence_type"] == "github_file" for item in result["evidence"])


def test_unified_github_policy_enforces_client_and_server_repository_boundaries(
    monkeypatch,
):
    client_policy = {
        "allowed": True,
        "repositories": ["example/allowed"],
    }
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    assert research_agent._github_policy_failure(
        "example/allowed", client_policy
    ) is None
    assert "not allowed" in research_agent._github_policy_failure(
        "example/other", client_policy
    )
    assert "global" in research_agent._github_policy_failure(None, client_policy)
    assert "not authorized" in research_agent._github_policy_failure(
        "example/allowed",
        {"allowed": False, "repositories": ["*"]},
    )

    monkeypatch.setenv("GITHUB_TOKEN", "server-token")
    monkeypatch.setenv("GITHUB_ALLOWED_REPOSITORIES", "example/server-only")
    assert "not allowed" in research_agent._github_policy_failure(
        "example/allowed", client_policy
    )


@pytest.mark.asyncio
async def test_final_answer_requires_valid_evidence_ids_and_renders_clickable_urls(
    monkeypatch,
):
    monkeypatch.setattr(
        research_agent,
        "_chat",
        AsyncMock(return_value="Install it using the documented procedure [E1]."),
    )
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )

    result = await research_agent._write_answer("How do I install it?", [_evidence()])

    assert result["citation_validation"]["valid"] is True
    assert "[Official guide](https://docs.example.com/guide)" in result["answer_markdown"]
    assert result["citations"][0]["url"] == "https://docs.example.com/guide"

    invalid_chat = AsyncMock(return_value="Unsupported answer [E99].")
    monkeypatch.setattr(research_agent, "_chat", invalid_chat)
    with pytest.raises(ValueError, match="citation validation"):
        await research_agent._write_answer("How do I install it?", [_evidence()])
    assert invalid_chat.await_count == 2


@pytest.mark.asyncio
async def test_final_answer_removes_model_authored_active_markdown(monkeypatch):
    monkeypatch.setattr(
        research_agent,
        "_chat",
        AsyncMock(
            return_value=(
                "Documented fact [E1]. "
                "![](http://127.1/private) "
                "[click](javascript:alert(1)) "
                "\u003cimg src='http://169.254.169.254/latest/meta-data/'\u003e"
            )
        ),
    )
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )

    result = await research_agent._write_answer("Verify the fact", [_evidence()])

    answer = result["answer_markdown"]
    assert "[Official guide](https://docs.example.com/guide)" in answer
    assert "javascript:" not in answer
    assert "http://127.1" not in answer
    assert "http://169.254.169.254" not in answer
    assert "\u003cimg" not in answer


def test_model_markdown_sanitizer_neutralizes_multiline_links_and_bare_hosts():
    content = (
        "Documented fact [E1].\n"
        "[click](\n//attacker.com/path)\n"
        "![tracking pixel](\n//images.attacker.com/pixel)\n"
        "Bare host: www.attacker.com and 127.0.0.1"
    )

    sanitized = research_agent._sanitize_model_markdown(content)

    assert "[E1]" in sanitized
    assert "](\n" not in sanitized
    assert "![" not in sanitized
    assert "attacker.com" not in sanitized
    assert "127.0.0.1" not in sanitized
    assert "attacker[.]com" in sanitized
    assert "127[.]0[.]0[.]1" in sanitized


def test_model_markdown_sanitizer_blocks_real_tld_hosts_outside_code():
    content = (
        "Contact attacker@evil.sh or open evil.sh/path and evil.py.\n\n"
        "Keep the literal filename `script.py`.\n\n"
        "```bash\npython script.py\n```"
    )

    sanitized = research_agent._sanitize_model_markdown(content)

    assert "attacker@evil[.]sh" in sanitized
    assert "evil[.]sh/path" in sanitized
    assert "evil[.]py" in sanitized
    assert "`script.py`" in sanitized
    assert "python script.py" in sanitized


@pytest.mark.asyncio
async def test_image_sources_are_integrated_without_embedding_remote_bytes(monkeypatch):
    monkeypatch.setattr(
        research_agent,
        "_chat",
        AsyncMock(return_value="The Product screenshot image result matches [E1]."),
    )
    monkeypatch.setattr(
        research_agent,
        "research_model_config",
        lambda: {"model": "private-model"},
    )
    images = [
        {
            "title": "Product screenshot",
            "image_url": "https://images.example.com/product.jpg",
            "source_url": "https://docs.example.com/screenshots",
        }
    ]

    result = await research_agent._write_answer(
        "Find a product image",
        research_agent._merge_evidence(
            research_agent._image_result_evidence(images)
        ),
        images=images,
    )

    assert "### Image sources" in result["answer_markdown"]
    assert "https://docs.example.com/screenshots" in result["answer_markdown"]
    assert "https://images.example.com/product.jpg" not in result["answer_markdown"]


def test_deferred_persistence_manifest_is_deduplicated_and_globally_capped():
    first = {
        "_deferred_persistence": {
            "sources": [
                {"job_id": f"{value:032x}", "artifact_owner_id": f"{value:032x}"}
                for value in range(12)
            ]
        }
    }
    second = {
        "_deferred_persistence": {
            "sources": [
                {"job_id": f"{value:032x}", "artifact_owner_id": f"{value:032x}"}
                for value in range(8, 24)
            ]
        }
    }

    merged = research_agent._merge_deferred_manifests(first, second)

    assert len(merged["sources"]) == 16
    assert len({item["job_id"] for item in merged["sources"]}) == 16


def test_public_research_task_does_not_reuse_private_request_details():
    plan = {
        "queries": ["official Docker installation documentation"],
        "answer_focus": "installation steps",
    }
    request = (
        "How do I install Docker for internal project Apollo Blue? "
        "My SSN is 123-45-6789 and the log is /srv/private/error.log"
    )

    public_task, redactions = research_agent._public_research_task(request, plan)

    assert public_task == "official Docker installation documentation; installation steps"
    assert "Apollo" not in public_task
    assert "123-45-6789" not in public_task
    assert "/srv/private" not in public_task
    assert redactions >= 3


def test_model_repeated_private_identifiers_are_removed_from_public_queries():
    request = (
        "Find Docker docs for our codename Apollo Blue and customer Jane Doe "
        "at 12 Main Street"
    )
    plan = research_agent._normalize_plan(
        {
            "queries": ["Docker Apollo Blue Jane Doe 12 Main Street"],
            "answer_focus": "Apollo Blue deployment",
        },
        request,
        "auto",
    )

    public_task, redactions = research_agent._public_research_task(request, plan)
    serialized = json.dumps(plan["queries"] + [plan["answer_focus"], public_task])

    assert "Docker" in public_task
    assert "deployment" in public_task
    assert "Apollo" not in serialized
    assert "Jane" not in serialized
    assert "Main Street" not in serialized
    assert redactions > 0


def test_non_ascii_private_identifiers_are_removed_from_public_queries():
    request = "Find Docker docs for private project Проект Зефир and codename 秘密项目"
    plan = research_agent._normalize_plan(
        {
            "queries": ["Docker Проект Зефир 秘密项目 installation documentation"],
            "answer_focus": "Проект Зефир deployment",
        },
        request,
        "auto",
    )

    serialized = json.dumps(
        plan["queries"] + [plan["answer_focus"]],
        ensure_ascii=False,
    )
    assert "Docker" in serialized
    assert "Проект" not in serialized
    assert "Зефир" not in serialized
    assert "秘密项目" not in serialized


def test_repeated_private_identifier_is_removed_from_public_queries():
    request = "Codename Apollo Blue! Apollo Blue fails with Docker."
    plan = research_agent._normalize_plan(
        {
            "queries": ["Apollo Blue fails with Docker"],
            "answer_focus": "Apollo Blue failure",
        },
        request,
        "auto",
    )

    public_task, redactions = research_agent._public_research_task(request, plan)
    serialized = json.dumps(plan["queries"] + [plan["answer_focus"], public_task])

    assert "Docker" in public_task
    assert "Apollo" not in serialized
    assert "Blue" not in serialized
    assert redactions > 0


def test_nfkc_equivalent_private_identifier_is_removed_from_public_queries():
    request = (
        "Codename \uff21\uff50\uff4f\uff4c\uff4c\uff4f \uff22\uff4c\uff55\uff45 failed with Docker."
    )
    plan = research_agent._normalize_plan(
        {
            "queries": ["Docker Apollo Blue failure"],
            "answer_focus": "",
        },
        request,
        "auto",
    )

    public_task, redactions = research_agent._public_research_task(request, plan)

    assert public_task == "Docker failure"
    assert redactions > 0


@pytest.mark.parametrize(
    "user_request, expected",
    [
        (
            "How do I configure an OpenStack private cloud setup?",
            "OpenStack private cloud setup",
        ),
        (
            "How do I configure a Docker private registry setup?",
            "Docker private registry setup",
        ),
    ],
)
def test_legitimate_private_cloud_and_registry_queries_keep_specificity(user_request, expected):
    public_task, redactions = research_agent._public_research_task(
        user_request,
        {"queries": [expected], "answer_focus": ""},
    )

    assert public_task == expected
    assert redactions == 0


@pytest.mark.asyncio
async def test_explicit_url_credentials_stay_internal_to_execution(monkeypatch):
    exact_url = (
        "https://docs.example.com/a%2Fb;jsessionid=PATHSECRET?"
        "z=2&sessionid=ABC123SECRET&token=TOKENVALUE&a=one%20two&page=2"
    )
    request = f"Investigate {exact_url}"
    plan = research_agent.deterministic_assistant_plan(request)
    plan["use_web_search"] = False
    acquire = AsyncMock(
        return_value={
            "url": plan["_execution_urls"][0],
            "evidence": [_evidence()],
        }
    )
    write = AsyncMock(
        return_value={
            "answer_markdown": "Documented answer.",
            "citations": [{"evidence_id": 1, "url": _evidence()["url"]}],
            "citation_validation": {"valid": True},
            "generated_by": "model:private-model",
        }
    )
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(research_agent, "build_assistant_plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(research_agent, "_acquire_url", acquire)
    monkeypatch.setattr(research_agent, "_write_answer", write)
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 0)

    result = await research_agent.run_research_assistant(request)

    execution_url = acquire.await_args.args[0]
    serialized_result = json.dumps(result)
    assert execution_url == exact_url
    assert "ABC123SECRET" in execution_url
    assert "PATHSECRET" in execution_url
    assert "TOKENVALUE" in execution_url
    assert "ABC123SECRET" not in serialized_result
    assert "PATHSECRET" not in serialized_result
    assert "TOKENVALUE" not in serialized_result
    assert "_execution_urls" not in serialized_result
    assert "sessionid=%5BREDACTED%5D" in serialized_result
    assert ";jsessionid=%5BREDACTED%5D" in serialized_result
    assert "token=%5BREDACTED%5D" in serialized_result


def test_evidence_urls_redact_signed_query_parameters():
    result = research_agent._merge_evidence(
        [
            {
                "title": "Signed guide",
                "url": (
                    "https://docs.example.com/guide?access_token=secret-value&"
                    "X-Goog-Credential=cloud-secret&sessionid=session-secret&page=2"
                ),
                "quote": "Documented procedure.",
            }
        ]
    )

    assert "secret-value" not in result[0]["url"]
    assert "cloud-secret" not in result[0]["url"]
    assert "session-secret" not in result[0]["url"]
    assert "access_token=%5BREDACTED%5D" in result[0]["url"]
    assert "sessionid=%5BREDACTED%5D" in result[0]["url"]
    assert "page=2" in result[0]["url"]


def test_url_redaction_covers_matrix_and_semicolon_encoded_credentials():
    value = (
        "https://docs.example.com/guide;sid=PATHSECRET;session=SESSIONPATH?"
        "x=1;token=QUERYSECRET&payload=1%3Bapi_key%3DENCODEDSECRET&page=2"
    )

    sanitized, count = redaction.redact_url_credentials(value)

    assert count >= 4
    assert "PATHSECRET" not in sanitized
    assert "SESSIONPATH" not in sanitized
    assert "QUERYSECRET" not in sanitized
    assert "ENCODEDSECRET" not in sanitized
    assert "page=2" in sanitized


def test_url_redaction_covers_camel_case_array_and_oauth_credentials():
    value = (
        "https://docs.example.com/guide?accessToken=ACCESSSECRET&"
        "clientSecret=CLIENTSECRET&oauthToken=OAUTHSECRET&"
        "token[]=ARRAYSECRET&page=2"
    )

    sanitized, count = redaction.redact_url_credentials(value)

    assert count >= 4
    for secret in ("ACCESSSECRET", "CLIENTSECRET", "OAUTHSECRET", "ARRAYSECRET"):
        assert secret not in sanitized
    assert "page=2" in sanitized


def test_url_redaction_covers_semicolon_phpsessionid():
    value = "https://docs.example.com/guide?x=1;phpsessionid=PHPSECRET&page=2"

    sanitized, count = redaction.redact_url_credentials(value)

    assert count >= 1
    assert "PHPSECRET" not in sanitized
    assert "page=2" in sanitized


def test_model_markdown_sanitizer_handles_many_unmatched_brackets_linearly():
    content = "[" * 12_000 + "unmatched"

    started = time.perf_counter()
    sanitized = research_agent._sanitize_model_markdown(content)
    elapsed = time.perf_counter() - started

    assert sanitized == content
    assert elapsed < 2.0


def test_follow_up_evidence_can_displace_a_saturated_initial_set(monkeypatch):
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_EVIDENCE", 8)
    initial = [
        _evidence(index + 1, f"https://initial.example.com/{index}")
        for index in range(8)
    ]
    follow_up = [_evidence(1, "https://needed.example.org/fix")]

    merged = research_agent._merge_evidence(follow_up, initial)

    assert len(merged) == 8
    assert merged[0]["url"] == "https://needed.example.org/fix"
    assert all(item["url"] != "https://initial.example.com/7" for item in merged)


def test_confidence_uses_only_cited_source_diversity():
    evidence = [
        _evidence(1, "https://one.example/a"),
        _evidence(2, "https://two.example/b"),
    ]
    citations = [
        {"evidence_id": value, "url": f"https://one.example/{value}"}
        for value in range(1, 4)
    ]

    assert research_agent._confidence(evidence, citations, []) == "medium"


def test_citation_validation_rejects_uncited_substantive_segments():
    validation = planner.validate_synthesis_citations(
        "Supported statement [E1].\n\nThis separate factual claim has no citation.",
        [_evidence()],
    )

    assert validation["valid"] is False
    assert validation["uncited_segments"] == [
        "This separate factual claim has no citation."
    ]

    unrelated = planner.validate_synthesis_citations(
        "Delete every production database immediately [E1].",
        [_evidence()],
    )
    assert unrelated["valid"] is False
    assert unrelated["lexically_unsupported_segments"] == [
        "Delete every production database immediately [E1]."
    ]


def test_citation_validation_checks_fenced_commands_against_cited_evidence():
    evidence = [
        {
            **_evidence(),
            "quote": "Start the service with docker compose up -d.",
        }
    ]
    supported = planner.validate_synthesis_citations(
        "Start the service with this command [E1].\n\n```bash\n"
        "docker compose up -d\n```",
        evidence,
    )
    unsupported = planner.validate_synthesis_citations(
        "Start the service with the documented command [E1].\n\n```bash\n"
        "sudo rm -rf /\n```",
        evidence,
    )
    uncited = planner.validate_synthesis_citations(
        "The service has a documented startup procedure [E1].\n\n"
        "# Uncited command\n\n```bash\ndocker compose up -d\n```",
        evidence,
    )
    mixed_block = planner.validate_synthesis_citations(
        "Start the service with this command [E1].\n\n```bash\n"
        "docker compose up -d\n~~~\nsudo rm -rf /\n```",
        evidence,
    )

    assert supported["valid"] is True
    assert unsupported["valid"] is False
    assert unsupported["lexically_unsupported_segments"] == ["sudo rm -rf /"]
    assert uncited["valid"] is False
    assert uncited["uncited_segments"] == [
        "Uncited command",
        "docker compose up -d",
    ]
    assert mixed_block["valid"] is False
    assert mixed_block["lexically_unsupported_segments"] == ["sudo rm -rf /"]


def test_citation_validation_checks_indented_commands_against_cited_evidence():
    evidence = [
        {
            **_evidence(),
            "quote": "Start the service with docker compose up -d.",
        }
    ]
    supported = planner.validate_synthesis_citations(
        "Start the service with this command [E1].\n\n    docker compose up -d",
        evidence,
    )
    unsupported = planner.validate_synthesis_citations(
        "Start the service with the documented command [E1].\n\n\tsudo rm -rf /",
        evidence,
    )

    assert supported["valid"] is True
    assert unsupported["valid"] is False
    assert unsupported["lexically_unsupported_segments"] == ["sudo rm -rf /"]


def test_citation_validation_rejects_short_heading_and_weak_overlap_claims():
    evidence = [
        {
            **_evidence(),
            "quote": "Run docker compose up to start the service.",
        }
    ]
    short_uncited = planner.validate_synthesis_citations(
        "Start the service with docker compose [E1].\n\nDocker changed.",
        evidence,
    )
    factual_heading = planner.validate_synthesis_citations(
        "Start the service with docker compose [E1].\n\n# Docker deletes data",
        evidence,
    )
    weak_overlap = planner.validate_synthesis_citations(
        "Docker permanently deletes every production database [E1].",
        evidence,
    )
    structural_heading = planner.validate_synthesis_citations(
        "## Installation steps\n\nStart the service with docker compose [E1].",
        evidence,
    )

    assert short_uncited["valid"] is False
    assert short_uncited["uncited_segments"] == ["Docker changed."]
    assert factual_heading["valid"] is False
    assert factual_heading["uncited_segments"] == ["Docker deletes data"]
    assert weak_overlap["valid"] is False
    assert weak_overlap["lexically_unsupported_segments"] == [
        "Docker permanently deletes every production database [E1]."
    ]
    assert structural_heading["valid"] is True


@pytest.mark.asyncio
async def test_synthesis_failure_marks_raw_evidence_untrusted_and_omits_image_urls(
    monkeypatch,
):
    plan = {
        "mode": "balanced",
        "queries": ["official product setup"],
        "use_web_search": True,
        "urls": [],
        "github_repositories": [],
        "github_searches": [],
        "include_memory": False,
        "include_images": True,
        "image_query": "product setup screenshot",
        "answer_focus": "installation",
        "generated_by": "model:private-model",
    }
    monkeypatch.setattr(research_agent, "research_model_configured", lambda: True)
    monkeypatch.setattr(
        research_agent,
        "build_assistant_plan",
        AsyncMock(return_value=plan),
    )
    monkeypatch.setattr(
        research_agent,
        "research_pipeline",
        AsyncMock(
            return_value={
                "evidence": [_evidence()],
                "completion": {"status": "complete"},
                "plan": {"queries": plan["queries"]},
            }
        ),
    )
    monkeypatch.setattr(
        research_agent,
        "searxng_image_search",
        AsyncMock(
            return_value=[
                {
                    "title": "Product screenshot",
                    "image_url": "https://images.example.com/product.jpg",
                    "thumbnail_url": "https://images.example.com/thumb.jpg",
                    "source_url": "https://docs.example.com/screenshots",
                    "engine": "bing images",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        research_agent,
        "_write_answer",
        AsyncMock(side_effect=ValueError("invalid synthesis")),
    )
    monkeypatch.setattr(research_agent, "RESEARCH_AGENT_MAX_FOLLOW_UP_ROUNDS", 0)

    result = await research_agent.run_research_assistant("Find the setup screenshot")

    assert result["status"] == "partial"
    assert result["error"] == "research_synthesis_failed"
    assert any("untrusted" in item for item in result["answering_instructions"])
    assert result["images"][0]["source_url"] == "https://docs.example.com/screenshots"
    assert result["images"][0]["direct_image_url_omitted"] is True
    assert "image_url" not in result["images"][0]
    assert "thumbnail_url" not in result["images"][0]


def test_image_results_are_bounded_deduplicated_and_reject_unsafe_urls():
    data = {
        "results": [
            {
                "title": "Valid",
                "url": "https://example.com/page",
                "img_src": "https://images.example.com/image.jpg",
                "thumbnail_src": "https://images.example.com/thumb.jpg",
                "engine": "bing images",
            },
            {
                "title": "Duplicate",
                "url": "https://other.example/page",
                "img_src": "https://images.example.com/image.jpg",
            },
            {
                "title": "Private image",
                "url": "https://example.com/private",
                "img_src": "http://127.0.0.1/private.jpg",
            },
            {
                "title": "Credentialed source",
                "url": "https://user:pass@example.com/page",
                "img_src": "https://images.example.com/other.jpg",
            },
        ]
    }

    result = searching.compact_image_results(data, max_results=12)

    assert len(result) == 1
    assert result[0]["image_url"] == "https://images.example.com/image.jpg"
    assert result[0]["source_url"] == "https://example.com/page"
    assert result[0]["content_trust"] == "untrusted_external_content"


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://redis/image.png",
        "http://127.1/image.png",
        "http://0177.0.0.1/image.png",
        "http://0x7f.0.0.1/image.png",
        "http://10.1/image.png",
        "http://169.254.1/image.png",
        "https://images.example.com/image.png?X-Goog-Credential=secret",
        "http://service.internal/image.png",
        "http://router.lan/image.png",
        "http://host.localdomain/image.png",
        "http://device.home.arpa/image.png",
        "http://service.internal./image.png",
        "http://localhost./image.png",
    ],
)
def test_image_results_reject_internal_hostnames(unsafe_url):
    data = {
        "results": [
            {
                "title": "Unsafe image",
                "img_src": unsafe_url,
                "url": "https://example.com/source",
            }
        ]
    }

    assert searching.compact_image_results(data) == []


@pytest.mark.asyncio
async def test_image_metadata_rejects_dns_private_destinations(monkeypatch):
    resolver = AsyncMock(side_effect=searching.DestinationPolicyError("private"))
    monkeypatch.setattr(searching, "resolve_public_addresses", resolver)
    item = {
        "title": "Rebinding result",
        "image_url": "https://images.example.com/image.jpg",
        "source_url": "https://source.example.com/page",
        "thumbnail_url": None,
    }

    assert await searching._validate_image_result(item) is None


def test_new_model_configuration_fails_closed_without_mixing_legacy_values(monkeypatch):
    monkeypatch.setattr(planner, "PLANNER_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setattr(planner, "PLANNER_MODEL", "legacy-model")
    monkeypatch.setattr(planner, "PLANNER_API_KEY", "legacy-key")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_BASE_URL", "https://new.example/v1")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_NAME", "")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_API_KEY", "")

    config = planner.research_model_config()

    assert config["source"] == "RESEARCH_MODEL_*"
    assert config["base_url"] == "https://new.example/v1"
    assert config["model"] == ""
    assert config["configured"] is False


def test_new_model_url_policy_rejects_plain_http_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(planner, "RESEARCH_MODEL_BASE_URL", "http://model.internal/v1")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_NAME", "private-model")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_API_KEY", "")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_ALLOW_INSECURE_HTTP", False)

    with pytest.raises(RuntimeError, match="RESEARCH_MODEL_ALLOW_INSECURE_HTTP"):
        planner._validated_planner_base_url()

    monkeypatch.setattr(planner, "RESEARCH_MODEL_ALLOW_INSECURE_HTTP", True)
    assert planner._validated_planner_base_url() == "http://model.internal/v1"


@pytest.mark.asyncio
async def test_new_model_stream_enforces_its_response_limit(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"x" * 11

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(planner, "RESEARCH_MODEL_BASE_URL", "https://model.example/v1")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_NAME", "private-model")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_API_KEY", "server-secret")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_MAX_RESPONSE_BYTES", 10)
    monkeypatch.setattr(planner.httpx, "AsyncClient", lambda **_kwargs: Client())

    with pytest.raises(ValueError, match="RESEARCH_MODEL_MAX_RESPONSE_BYTES"):
        await planner._chat([{"role": "user", "content": "q"}])


@pytest.mark.asyncio
async def test_internal_model_calls_reuse_one_http_client_per_event_loop(monkeypatch):
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b'{"choices":[{"message":{"content":"ok"}}]}'

    class Client:
        is_closed = False

        def stream(self, *_args, **_kwargs):
            return Response()

        async def aclose(self):
            self.is_closed = True

    created = []

    def factory(**_kwargs):
        client = Client()
        created.append(client)
        return client

    monkeypatch.setattr(planner, "RESEARCH_MODEL_BASE_URL", "https://model.example/v1")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_NAME", "private-model")
    monkeypatch.setattr(planner, "RESEARCH_MODEL_API_KEY", "server-secret")
    monkeypatch.setattr(planner.httpx, "AsyncClient", factory)

    assert await planner._chat([{"role": "user", "content": "one"}]) == "ok"
    assert await planner._chat([{"role": "user", "content": "two"}]) == "ok"
    assert len(created) == 1

    await planner.close_research_model_client()
    assert created[0].is_closed is True
