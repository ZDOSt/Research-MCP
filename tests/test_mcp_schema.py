import pytest

import mcp_server


@pytest.mark.asyncio
async def test_all_public_tools_expose_bounded_client_schemas():
    tools = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}
    assert set(tools) == {
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

    research = tools["research_web"].parameters["properties"]
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

    investigation = tools["investigate_url"].parameters["properties"]
    assert investigation["mode"]["enum"] == ["auto", "targeted", "balanced", "exhaustive"]
    assert investigation["max_chars"]["minimum"] == 10_000
    assert investigation["max_chars"]["maximum"] == 750_000

    assert tools["query_memory"].parameters["properties"]["top_k"]["maximum"] == 30
    assert tools["manage_sources"].parameters["properties"]["action"]["enum"] == [
        "list",
        "stats",
        "delete",
    ]
    assert tools["research_job"].parameters["properties"]["action"]["enum"] == [
        "status",
        "result",
        "cancel",
    ]
    assert tools["github_research"].parameters["properties"]["action"]["enum"] == [
        "search",
        "inspect",
        "read",
    ]
    assert tools["github_research"].parameters["properties"]["kind"]["enum"] == [
        "issues",
        "code",
        "repositories",
    ]
    artifact_chars = tools["get_research_artifact"].parameters["properties"]["max_chars"]
    assert artifact_chars["minimum"] == 1_000
    assert artifact_chars["maximum"] == 250_000


@pytest.mark.asyncio
async def test_tool_discovery_promotes_proactive_research_without_redundant_artifact_reads():
    tools = {tool.name: tool for tool in await mcp_server.mcp.list_tools()}

    instructions = mcp_server.mcp.instructions
    assert "may have changed or needs external verification" in instructions
    assert "even when the user did not explicitly ask to search" in instructions
    assert "answer stable, timeless questions directly" in instructions
    assert "duplicate job-result artifact path is intentionally omitted" in instructions

    research_description = tools["research_web"].description
    assert "Use this proactively" in research_description
    assert "may have changed" in research_description
    assert "did not explicitly ask to search" in research_description
    assert "complete research question or task" in research_description
    assert "effective search queries internally" in research_description

    artifact_description = tools["get_research_artifact"].description
    assert "intentionally omit their" in artifact_description
    assert "duplicate job-result artifact path" in artifact_description
    assert "specifically needed source artifact" in artifact_description
    assert "include_full_result=False" in artifact_description
