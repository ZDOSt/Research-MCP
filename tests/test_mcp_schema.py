import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import mcp_server


def test_current_github_access_policy_forwards_only_bounded_non_secret_claims():
    token = SimpleNamespace(
        scopes=["ignored:fallback"],
        claims={
            "scopes": "research,github:read",
            "github_repositories": [
                "owner/repository",
                "owner/repository",
                "other/project",
                "",
            ],
            "access_token": "must-not-be-forwarded",
        },
    )
    with patch.object(
        mcp_server, "_token_authorization_enabled", return_value=True
    ), patch.object(mcp_server, "_current_access_token", return_value=token):
        policy = mcp_server._current_github_access_policy()

    assert policy == {
        "allowed": True,
        "repositories": ["owner/repository", "other/project"],
    }
    assert "access_token" not in policy


@pytest.mark.asyncio
async def test_compatibility_default_exposes_all_bounded_client_schemas():
    tools = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}
    assert {"research_assistant", "research_job", "research_web"}.issubset(tools)

    assistant = tools["research_assistant"].parameters["properties"]
    assert assistant["request"]["minLength"] == 1
    assert assistant["request"]["maxLength"] == mcp_server.MCP_MAX_QUERY_CHARS
    assert assistant["mode"]["enum"] == [
        "auto",
        "quick",
        "balanced",
        "deep",
        "technical",
        "academic",
    ]
    assert tools["research_job"].parameters["properties"]["action"]["enum"] == [
        "status",
        "result",
        "cancel",
    ]


def _tools_for_profile(profile):
    code = (
        "import asyncio,json,mcp_server; "
        "tools=asyncio.run(mcp_server.mcp.list_tools()); "
        "print(json.dumps({t.name:{'parameters':t.parameters,'description':t.description} for t in tools}))"
    )
    environment = dict(os.environ)
    environment["MCP_TOOL_PROFILE"] = profile
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_advanced_and_all_profiles_control_tool_discovery():
    advanced = _tools_for_profile("advanced")
    advanced_names = {
        "get_research_artifact",
        "github_research",
        "ingest_text",
        "investigate_url",
        "manage_sources",
        "query_memory",
        "research_job",
        "research_web",
        "start_research",
    }
    assert set(advanced) == advanced_names
    assert set(_tools_for_profile("all")) == advanced_names | {"research_assistant"}

    tools = advanced

    research = tools["research_web"]["parameters"]["properties"]
    assert research["mode"]["enum"] == [
        "quick",
        "balanced",
        "deep",
        "technical",
        "academic",
        "local_only",
        "web_only",
    ]
    assert research["max_sources"]["anyOf"][0] == {
        "maximum": 8,
        "minimum": 0,
        "type": "integer",
    }
    for tool_name in ("research_web", "start_research"):
        parameters = tools[tool_name]["parameters"]
        proposed_schema = next(
            branch
            for branch in parameters["properties"]["proposed_queries"]["anyOf"]
            if branch.get("type") == "array"
        )
        assert proposed_schema["minItems"] == 1
        assert proposed_schema["maxItems"] == 5
        assert proposed_schema["items"]["minLength"] == 1
        assert proposed_schema["items"]["maxLength"] == 180
        assert "proposed_queries" not in parameters.get("required", [])

    investigation = tools["investigate_url"]["parameters"]["properties"]
    assert investigation["mode"]["enum"] == ["auto", "targeted", "balanced", "exhaustive"]
    assert investigation["max_chars"]["minimum"] == 10_000
    assert investigation["max_chars"]["maximum"] == 750_000

    assert tools["query_memory"]["parameters"]["properties"]["top_k"]["maximum"] == 30
    assert tools["manage_sources"]["parameters"]["properties"]["action"]["enum"] == [
        "list",
        "stats",
        "delete",
    ]
    assert tools["research_job"]["parameters"]["properties"]["action"]["enum"] == [
        "status",
        "result",
        "cancel",
    ]
    assert tools["github_research"]["parameters"]["properties"]["action"]["enum"] == [
        "search",
        "inspect",
        "read",
    ]
    assert tools["github_research"]["parameters"]["properties"]["kind"]["enum"] == [
        "issues",
        "code",
        "repositories",
    ]
    artifact_chars = tools["get_research_artifact"]["parameters"]["properties"]["max_chars"]
    assert artifact_chars["minimum"] == 1_000
    assert artifact_chars["maximum"] == 250_000


def test_invalid_tool_profile_fails_fast_with_clear_error():
    environment = dict(os.environ)
    environment["MCP_TOOL_PROFILE"] = "invalid"
    completed = subprocess.run(
        [sys.executable, "-c", "import mcp_server"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "MCP_TOOL_PROFILE must be unified, advanced, or all" in completed.stderr


@pytest.mark.asyncio
async def test_unified_tool_discovery_promotes_one_completed_research_call():
    tools = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}

    instructions = mcp_server.mcp.instructions
    assert "Use research_assistant once" in instructions
    assert "citation-checked answer" in instructions
    assert "interactive deadline" in instructions
    assert "MCP cannot force a client model to call tools" in instructions

    description = tools["research_assistant"].description
    assert "Complete a research request in one high-level call" in description
    assert "user's complete" in description
    assert "server-owned interactive deadline" in description
    assert "cited partial evidence" in description
    assert "Present answer_markdown directly" in description
