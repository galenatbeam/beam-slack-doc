from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from agents.run_context import RunContextWrapper
from mcp.types import CallToolResult, Tool

from agentic_search import (
    AgenticSearch,
    Citation,
    CONFLUENCE_AUTH_ISSUE_MESSAGE,
    DEFAULT_OLLAMA_CHAT_MODEL,
    GITHUB_TOOL_NAME_PREFIX,
    PAYLOAD_PREVIEW_CHAR_LIMIT,
    REQUIRED_SEARCH_TOOL_MISSING_MESSAGE,
    REQUIRED_SEARCH_TOOL_NOT_CALLED_MESSAGE,
    ATLASSIAN_SERVER_KEY,
    MCPServerBinding,
    SHARED_FETCH_TOOL,
    SHARED_SEARCH_TOOL,
    SynthesizedAnswer,
    _is_sensitive_debug_key,
    build_mcp_auth_header,
)


class FakeServer:
    def __init__(self, tools=None, responses=None, error=None, list_tools_error=None, call_errors=None):
        self.tools = tools or []
        self.responses = responses or {}
        self.error = error
        self.list_tools_error = list_tools_error
        self.call_errors = call_errors or {}
        self.calls = []
        self.list_tools_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self, *args, **kwargs):
        self.list_tools_calls += 1
        if self.list_tools_error:
            raise self.list_tools_error
        return self.tools

    async def call_tool(self, tool_name, arguments, meta=None):
        self.calls.append((tool_name, arguments))
        if tool_name in self.call_errors:
            raise self.call_errors[tool_name]
        if self.error:
            raise self.error
        return self.responses[tool_name]


def make_tool(
    name: str,
    properties: dict,
    required: list[str] | None = None,
    *,
    schema_extras: dict | None = None,
) -> Tool:
    input_schema = {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
    if schema_extras:
        input_schema.update(schema_extras)
    return Tool(
        name=name,
        inputSchema=input_schema,
    )


def make_search(**overrides) -> AgenticSearch:
    defaults = {
        "openai_api_key": "test-openai-key",
        "atlassian_email": "bot@example.com",
        "confluence_mcp_api_key": "test-mcp-token",
        "atlassian_cloud_id": "test-cloud-id",
        "model": "gpt-4.1-mini",
    }
    defaults.update(overrides)
    return AgenticSearch(**defaults)


def test_sensitive_debug_key_detection_normalizes_camel_case_variants():
    for key in ["token", "authorization", "api_token", "api-token", "API_TOKEN", "apiToken"]:
        assert _is_sensitive_debug_key(key) is True


def test_build_agent_instructions_marks_github_as_secondary_after_atlassian_search():
    search = make_search(github_pat="test-github-token", github_org="https://github.com/BeamTech")

    instructions = search.build_agent_instructions(
        [SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL, f"{GITHUB_TOOL_NAME_PREFIX}search_repositories"]
    )

    assert "Always inspect Atlassian results from `search` first" in instructions
    assert "prioritize README files first" in instructions
    assert GITHUB_TOOL_NAME_PREFIX in instructions
    assert "`beamtech` org" in instructions


def make_discovered_tools() -> list[Tool]:
    return [
        make_tool(
            SHARED_SEARCH_TOOL,
            {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "cloudId": {"type": "string"},
            },
            required=["query"],
        ),
        make_tool(
            SHARED_FETCH_TOOL,
            {
                "ari": {"type": "string"},
                "cloudId": {"type": "string"},
            },
            required=["ari"],
        ),
    ]


def make_github_tools() -> list[Tool]:
    return [
        make_tool(
            "search_repositories",
            {
                "query": {"type": "string"},
                "owner": {"type": "string"},
            },
            required=["query"],
        )
    ]


def test_build_scoped_mcp_server_sanitizes_schema_before_sdk_tool_conversion():
    search = make_search()
    tool = make_tool(
        SHARED_SEARCH_TOOL,
        {
            "query": {"type": "string", "description": "Natural language query."},
            "filters": {
                "type": "object",
                "properties": {
                    "space": {"type": "string"},
                },
                "required": ["space"],
                "additionalProperties": False,
                "patternProperties": {".*": {"type": "string"}},
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
        required=["query", "filters"],
        schema_extras={"patternProperties": {".*": {"type": "string"}}},
    )

    scoped_server = search._build_scoped_mcp_server(
        MCPServerBinding(
            key=ATLASSIAN_SERVER_KEY,
            name="atlassian-rovo",
            url="https://example.com/mcp",
            server=FakeServer(tools=[tool]),
        )
    )
    public_tools = asyncio.run(scoped_server.list_tools())

    assert [tool.name for tool in public_tools] == [SHARED_SEARCH_TOOL]
    assert public_tools[0].inputSchema == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query."},
            "filters": {
                "type": "object",
                "properties": {"space": {"type": "string"}},
                "required": ["space"],
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            },
        },
        "required": ["query", "filters"],
    }
    serialized_schema = json.dumps(public_tools[0].inputSchema)
    assert "additionalProperties" not in serialized_schema
    assert "patternProperties" not in serialized_schema


class FakeRunResult:
    def __init__(self, synthesized: SynthesizedAnswer, final_output: str | None = None):
        self._synthesized = synthesized
        self.final_output = final_output or synthesized.answer

    def final_output_as(self, _model, raise_if_incorrect_type=True):
        return self._synthesized


async def get_agent_tools(agent):
    return await agent.get_all_tools(RunContextWrapper(context=None))


async def invoke_discovered_tools(agent, tool_calls: list[tuple[str, dict]]) -> None:
    tool_map = {tool.name: tool for tool in await get_agent_tools(agent)}
    for tool_name, arguments in tool_calls:
        await tool_map[tool_name].on_invoke_tool(RunContextWrapper(context=None), json.dumps(arguments))


def make_tool_not_found_error(message: str = "Method not found") -> RuntimeError:
    error = RuntimeError(message)
    error.error = SimpleNamespace(code=-32601, message=message)
    return error


def make_invalid_arguments_error(message: str = "Invalid arguments") -> RuntimeError:
    error = RuntimeError(message)
    error.error = SimpleNamespace(code=-32602, message=message)
    return error


@pytest.fixture
def rovo_search_structured_results_payload() -> CallToolResult:
    return CallToolResult(
        content=[],
        structuredContent={
            "results": [
                {
                    "title": "Beam Benefits Overview",
                    "webLink": "https://example.com/beam-overview",
                    "excerpt": "Beam helps employers manage benefits.",
                    "ari": "ari:cloud:confluence::page/overview",
                    "id": "overview-1",
                }
            ]
        },
    )


@pytest.fixture
def rovo_search_ari_only_payload() -> CallToolResult:
    return CallToolResult(
        content=[],
        structuredContent={
            "items": [
                {
                    "name": "Benefits FAQ",
                    "ari": "ari:cloud:confluence::page/faq",
                    "entityId": "faq-123",
                    "summary": "Frequently asked questions about benefits.",
                }
            ]
        },
    )


def test_build_mcp_auth_header_uses_basic_auth_and_base64_email_and_token():
    header = build_mcp_auth_header(
        env={
            "ATLASSIAN_EMAIL": "bot@example.com",
            "CONFLUENCE_MCP_API_KEY": "service-key",
        }
    )

    expected = base64.b64encode(b"bot@example.com:service-key").decode("ascii")
    assert header == f"Basic {expected}"


def test_legacy_auth_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_HEADER", "Bearer ignored-token")
    monkeypatch.setenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", "ignored-service-key")
    monkeypatch.setenv("ATLASSIAN_API_EMAIL", "bot@example.com")
    monkeypatch.setenv("ATLASSIAN_API_TOKEN", "ignored-api-token")
    monkeypatch.delenv("CONFLUENCE_MCP_API_KEY", raising=False)
    monkeypatch.delenv("ATLASSIAN_EMAIL", raising=False)
    monkeypatch.delenv("ATLASSIAN_CLOUD_ID", raising=False)

    search = AgenticSearch(openai_api_key="test-openai-key")

    result = search.search("What failed?")

    assert search.mcp_auth_header == ""
    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == [
        "ATLASSIAN_EMAIL",
        "CONFLUENCE_MCP_API_KEY",
        "ATLASSIAN_CLOUD_ID",
    ]


def test_missing_configuration_includes_atlassian_email_when_unset(monkeypatch):
    monkeypatch.delenv("ATLASSIAN_EMAIL", raising=False)

    search = AgenticSearch(
        openai_api_key="test-openai-key",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="test-cloud-id",
    )

    assert search.missing_configuration() == ["ATLASSIAN_EMAIL"]


def test_search_returns_configuration_error_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("CONFLUENCE_MCP_API_KEY", raising=False)
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_email="bot@example.com",
        atlassian_cloud_id="test-cloud-id",
    )

    result = search.search("What failed?")

    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == ["CONFLUENCE_MCP_API_KEY"]


def test_search_returns_configuration_error_when_cloud_id_is_missing(monkeypatch):
    monkeypatch.delenv("ATLASSIAN_CLOUD_ID", raising=False)
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
    )

    result = search.search("What failed?")

    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == ["ATLASSIAN_CLOUD_ID"]


def test_search_returns_configuration_error_when_github_pat_is_set_but_org_is_blank(monkeypatch):
    monkeypatch.setenv("GITHUB_ORG", "")
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="test-cloud-id",
        github_pat="test-github-token",
    )

    result = search.search("What failed?")

    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == ["GITHUB_ORG"]


def test_default_mode_lists_tools_and_uses_agent_tool_wrapper(rovo_search_structured_results_payload):
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: rovo_search_structured_results_payload,
            SHARED_FETCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "title": "Beam Benefits Overview",
                    "webLink": "https://example.com/beam-overview",
                    "excerpt": "Detailed fetch result.",
                    "ari": "ari:cloud:confluence::page/overview",
                    "id": "overview-1",
                    "body": "Expanded Beam Benefits documentation.",
                },
            ),
        },
    )
    search = make_search()
    synthesized = SynthesizedAnswer(
        answer="Beam Benefits offers dental, vision, and supplemental benefits.",
        citations=[
            Citation(
                title="Beam Benefits Overview",
                url="https://example.com/beam-overview",
                excerpt="Beam helps employers manage benefits.",
            )
        ],
    )

    async def fake_runner(agent, prompt, **_kwargs):
        assert prompt == "What is Beam Benefits?"
        assert {tool.name for tool in await get_agent_tools(agent)} == {SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL}
        await invoke_discovered_tools(
            agent,
            [
                (SHARED_SEARCH_TOOL, {"query": prompt}),
                (SHARED_FETCH_TOOL, {"ari": "ari:cloud:confluence::page/overview"}),
            ],
        )
        return FakeRunResult(synthesized)

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)) as run_agent,
    ):
        result = search.search("What is Beam Benefits?")

    assert result["answer"].startswith("Beam Benefits offers dental")
    assert result["citations"] == [
        {
            "title": "Beam Benefits Overview",
            "url": "https://example.com/beam-overview",
            "excerpt": "Beam helps employers manage benefits.",
            "identifier": "overview-1",
            "ari": "ari:cloud:confluence::page/overview",
        }
    ]
    assert result["raw_response"]["status"] == "ok"
    assert result["raw_response"]["strategy"] == "agentic_discovery"
    assert result["raw_response"]["allowlisted_tools"] == [SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL]
    assert result["raw_response"]["agent_run_used"] == 1
    assert server.list_tools_calls == 1
    assert server.calls == [
        (
            SHARED_SEARCH_TOOL,
            {
                "query": "What is Beam Benefits?",
                "cloudId": "test-cloud-id",
            },
        ),
        (
            SHARED_FETCH_TOOL,
            {
                "ari": "ari:cloud:confluence::page/overview",
                "cloudId": "test-cloud-id",
            },
        )
    ]
    assert result["raw_response"]["selected_tools"] == [SHARED_FETCH_TOOL, SHARED_SEARCH_TOOL]
    run_agent.assert_awaited_once()


def test_search_discovers_github_tools_and_injects_configured_org(rovo_search_structured_results_payload):
    atlassian_server = FakeServer(
        tools=make_discovered_tools(),
        responses={SHARED_SEARCH_TOOL: rovo_search_structured_results_payload},
    )
    github_server = FakeServer(
        tools=make_github_tools(),
        responses={
            "search_repositories": {
                "title": "beam-slack-doc",
                "url": "https://github.com/beamtech/beam-slack-doc",
                "excerpt": "Slack bot README and implementation.",
            }
        },
    )
    search = make_search(github_pat="test-github-token", github_org="https://github.com/BeamTech")

    async def fake_runner(agent, prompt, **_kwargs):
        assert prompt == "What is Beam Benefits?"
        assert {tool.name for tool in await get_agent_tools(agent)} == {
            SHARED_SEARCH_TOOL,
            SHARED_FETCH_TOOL,
            f"{GITHUB_TOOL_NAME_PREFIX}search_repositories",
        }
        assert "Always inspect Atlassian results from `search` first" in agent.instructions
        await invoke_discovered_tools(
            agent,
            [
                (SHARED_SEARCH_TOOL, {"query": prompt}),
                (
                    f"{GITHUB_TOOL_NAME_PREFIX}search_repositories",
                    {"query": "beam slack bot", "owner": "other-org"},
                ),
            ],
        )
        return FakeRunResult(
            SynthesizedAnswer(
                answer="Combined answer.",
                citations=[
                    Citation(
                        title="Beam Benefits Overview",
                        url="https://example.com/beam-overview",
                    )
                ],
            )
        )

    with (
        patch(
            "agentic_search.MCPServerStreamableHttp",
            side_effect=[atlassian_server, github_server],
        ),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What is Beam Benefits?")

    assert result["raw_response"]["status"] == "ok"
    assert result["raw_response"]["discovered_tools"] == [
        SHARED_SEARCH_TOOL,
        SHARED_FETCH_TOOL,
        f"{GITHUB_TOOL_NAME_PREFIX}search_repositories",
    ]
    assert result["raw_response"]["allowlisted_tools"] == [
        SHARED_SEARCH_TOOL,
        SHARED_FETCH_TOOL,
        f"{GITHUB_TOOL_NAME_PREFIX}search_repositories",
    ]
    assert atlassian_server.list_tools_calls == 1
    assert github_server.list_tools_calls == 1
    assert github_server.calls == [
        (
            "search_repositories",
            {"query": "beam slack bot", "owner": "beamtech"},
        )
    ]


def test_allowlist_excludes_non_search_and_fetch_tools(rovo_search_structured_results_payload):
    server = FakeServer(
        tools=[
            *make_discovered_tools(),
            make_tool("createConfluencePage", {"title": {"type": "string"}}, required=["title"]),
            make_tool("updateConfluencePage", {"id": {"type": "string"}}, required=["id"]),
        ],
        responses={SHARED_SEARCH_TOOL: rovo_search_structured_results_payload},
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        assert {tool.name for tool in await get_agent_tools(agent)} == {SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL}
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(
            SynthesizedAnswer(
                answer="Filtered to search/fetch only.",
                citations=[Citation(title="Beam Benefits Overview", url="https://example.com/beam-overview")],
            )
        )

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What is Beam Benefits?")

    assert result["raw_response"]["status"] == "ok"
    assert result["raw_response"]["discovered_tools"] == [
        SHARED_SEARCH_TOOL,
        SHARED_FETCH_TOOL,
        "createConfluencePage",
        "updateConfluencePage",
    ]
    assert result["raw_response"]["allowlisted_tools"] == [SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL]


def test_search_retries_when_first_run_skips_tools_then_succeeds(rovo_search_structured_results_payload):
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={SHARED_SEARCH_TOOL: rovo_search_structured_results_payload},
    )
    search = make_search()
    attempts = {"count": 0}

    async def fake_runner(agent, prompt, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return FakeRunResult(SynthesizedAnswer(answer="Skipped tools.", citations=[]))
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(
            SynthesizedAnswer(
                answer="Used search on retry.",
                citations=[Citation(title="Beam Benefits Overview", url="https://example.com/beam-overview")],
            )
        )

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)) as run_agent,
    ):
        result = search.search("What is Beam Benefits?")

    assert result["answer"] == "Used search on retry."
    assert result["raw_response"]["status"] == "ok"
    assert result["raw_response"]["agent_run_used"] == 2
    assert server.calls == [
        (
            SHARED_SEARCH_TOOL,
            {"query": "What is Beam Benefits?", "cloudId": "test-cloud-id"},
        )
    ]
    assert run_agent.await_count == 2


def test_search_returns_clear_error_after_two_runs_without_tool_calls(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    server = FakeServer(tools=make_discovered_tools(), responses={})
    search = make_search()

    async def fake_runner(_agent, _prompt, **_kwargs):
        return FakeRunResult(SynthesizedAnswer(answer="No tools used.", citations=[]))

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)) as run_agent,
    ):
        result = search.search("What is Beam Benefits?")

    assert result["answer"] == REQUIRED_SEARCH_TOOL_NOT_CALLED_MESSAGE
    assert result["citations"] == []
    assert result["raw_response"]["status"] == "required_search_tool_not_called"
    assert result["raw_response"]["attempts"] == [
        {
            "run": 1,
            "tool_invocation_count": 0,
            "tool_invocations": [],
            "search_call_count": 0,
            "used_plain_text_fallback": False,
        },
        {
            "run": 2,
            "tool_invocation_count": 0,
            "tool_invocations": [],
            "search_call_count": 0,
            "used_plain_text_fallback": False,
        },
    ]
    assert result["raw_response"]["debug"]["tool_enforcement"] == {
        "runs": result["raw_response"]["attempts"],
        "run_used": None,
    }
    assert server.calls == []
    assert run_agent.await_count == 2


def test_search_fails_fast_when_required_search_tool_is_not_discovered():
    server = FakeServer(
        tools=[
            make_tool(
                SHARED_FETCH_TOOL,
                {"ari": {"type": "string"}, "cloudId": {"type": "string"}},
                required=["ari"],
            ),
            make_tool("createConfluencePage", {"title": {"type": "string"}}, required=["title"]),
        ],
        responses={},
    )
    search = make_search()

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock()) as run_agent,
    ):
        result = search.search("What is Beam Benefits?")

    assert result["answer"] == REQUIRED_SEARCH_TOOL_MISSING_MESSAGE
    assert result["citations"] == []
    assert result["raw_response"] == {
        "status": "required_search_tool_missing",
        "query": "What is Beam Benefits?",
        "model": "gpt-4.1-mini",
        "required_tool": SHARED_SEARCH_TOOL,
        "discovered_tools": [SHARED_FETCH_TOOL, "createConfluencePage"],
        "allowlisted_tools": [SHARED_FETCH_TOOL],
    }
    assert server.calls == []
    run_agent.assert_not_awaited()


def test_default_mode_maps_list_tools_failure_to_confluence_auth_issue(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    error = make_tool_not_found_error("Method not found Authorization=Bearer secret-token")
    error.response = SimpleNamespace(
        status_code=401,
        json=lambda: {
            "error": {
                "code": -32601,
                "message": "Forbidden Basic abc123",
                "data": {"api_token": "top-secret"},
            }
        },
    )
    server = FakeServer(list_tools_error=error)
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "confluence_authentication_error"
    assert result["answer"] == CONFLUENCE_AUTH_ISSUE_MESSAGE
    assert debug["last_step"] == "list_tools"
    assert debug["http_status"] == 401
    assert debug["mcp_error_preview"]["code"] == -32601
    assert "[AgenticSearch debug] error:" in captured.err
    assert '"last_step": "list_tools"' in captured.err
    assert "secret-token" not in captured.err
    assert "abc123" not in captured.err
    assert "top-secret" not in captured.err
    assert "authorization" not in serialized
    assert "api_token" not in serialized


def test_search_uses_explicit_atlassian_cloud_id_for_shared_search_args(
    rovo_search_structured_results_payload,
):
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: rovo_search_structured_results_payload,
        },
    )
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="configured-cloud-id",
        model="gpt-4.1-mini",
    )

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(
            SynthesizedAnswer(
                answer="Configured cloud ID used.",
                citations=[Citation(title="Beam Benefits Overview", url="https://example.com/beam-overview")],
            )
        )

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What is Beam Benefits?")

    assert result["raw_response"]["status"] == "ok"
    assert server.list_tools_calls == 1
    assert server.calls == [
        (
            SHARED_SEARCH_TOOL,
            {
                "query": "What is Beam Benefits?",
                "cloudId": "configured-cloud-id",
            },
        )
    ]


def test_search_overrides_model_supplied_cloud_id_with_configured_cloud_id(
    monkeypatch,
    capsys,
    rovo_search_structured_results_payload,
):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: rovo_search_structured_results_payload,
        },
    )
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="configured-cloud-id",
        model="gpt-4.1-mini",
    )

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(
            agent,
            [(SHARED_SEARCH_TOOL, {"query": prompt, "cloudId": "model-cloud-id"})],
        )
        return FakeRunResult(
            SynthesizedAnswer(
                answer="Configured cloud ID overrode the model value.",
                citations=[Citation(title="Beam Benefits Overview", url="https://example.com/beam-overview")],
            )
        )

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What is Beam Benefits?")

    captured = capsys.readouterr()
    assert result["raw_response"]["status"] == "ok"
    assert server.calls == [
        (
            SHARED_SEARCH_TOOL,
            {
                "query": "What is Beam Benefits?",
                "cloudId": "configured-cloud-id",
            },
        )
    ]
    assert result["raw_response"]["debug"]["warnings"] == [
        {
            "warning_type": "cloud_id_override",
            "tool_name": SHARED_SEARCH_TOOL,
            "provided_cloud_id": "[REDACTED]",
            "configured_cloud_id": "[REDACTED]",
        }
    ]
    assert "[AgenticSearch debug] warning:" in captured.err
    assert "model-cloud-id" not in captured.err
    assert "configured-cloud-id" not in captured.err


def test_search_returns_generic_error_response_when_mcp_tool_fails():
    server = FakeServer(
        tools=make_discovered_tools(),
        error=RuntimeError("boom"),
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        raise AssertionError("tool invocation should have failed")

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What failed?")

    assert result["answer"].startswith("Sorry, I ran into an issue")
    assert result["citations"] == []
    assert result["raw_response"]["status"] == "search_error"
    assert result["raw_response"]["error_type"] == "RuntimeError"


def test_search_omits_debug_block_when_debug_disabled(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_DEBUG", raising=False)
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Debug Doc",
                            "url": "https://example.com/debug-doc",
                            "excerpt": "Debug excerpt.",
                        }
                    ]
                },
            )
        },
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(SynthesizedAnswer(answer="Debug excerpt.", citations=[]))

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("debug me")

    assert "debug" not in result["raw_response"]
    assert "payload_preview" not in json.dumps(result["raw_response"])


def test_search_includes_redacted_debug_block_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "yes")
    search = make_search()
    encoded_credentials = search.mcp_auth_header.removeprefix("Basic ")
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Debug Doc",
                            "url": "https://example.com/debug-doc",
                            "excerpt": "Debug excerpt.",
                        }
                    ],
                    "api_token": "super-secret-token",
                    "apiToken": "camel-secret-token",
                    "authorization": "Bearer secret-token",
                    "note": (
                        "Authorization=Bearer another-secret-token; "
                        f"email={search.atlassian_email}; "
                        f"token={search.confluence_mcp_api_key}; "
                        f"encoded={encoded_credentials}"
                    ),
                },
            )
        },
    )

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(SynthesizedAnswer(answer="Debug excerpt.", citations=[]))

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("debug me")

    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert "bearer" not in serialized
    assert "basic" not in serialized
    assert "api_token" not in serialized
    assert debug["mcp_server_url"] == search.mcp_server_url
    assert debug["mcp_server_name"] == search.mcp_server_name
    assert debug["tool_discovery"] == {
        "called": True,
        "method": "list_tools",
        "tool_names": [SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL],
    }
    sanitized_schemas = {entry["tool_name"]: entry for entry in debug["sanitized_tool_schemas"]}
    assert set(sanitized_schemas) == {SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL}
    assert sanitized_schemas[SHARED_SEARCH_TOOL]["stripped_keys"] == ["$.additionalProperties"]
    assert '"additionalProperties": false' in sanitized_schemas[SHARED_SEARCH_TOOL]["original_schema_preview"]
    assert "additionalProperties" not in sanitized_schemas[SHARED_SEARCH_TOOL]["sanitized_schema_preview"]
    assert debug["calls"][0]["tool_name"] == SHARED_SEARCH_TOOL
    assert debug["calls"][0]["argument_keys"] == ["query", "cloudId"]
    assert "results" in debug["calls"][0]["response_summary"]["top_level_keys"]
    payload_preview = debug["calls"][0]["payload_preview"]
    assert "Debug Doc" in payload_preview
    assert "super-secret-token" not in payload_preview
    assert "camel-secret-token" not in payload_preview
    assert "secret-token" not in payload_preview
    assert search.atlassian_email not in payload_preview
    assert search.confluence_mcp_api_key not in payload_preview
    assert encoded_credentials not in payload_preview
    assert "Bearer" not in payload_preview
    assert "apitoken" not in payload_preview.lower()
    assert "authorization" not in payload_preview.lower()
    assert "api_token" not in payload_preview.lower()


def test_search_truncates_payload_preview_when_debug_enabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    oversized_text = "x" * (PAYLOAD_PREVIEW_CHAR_LIMIT + 500)
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Large Debug Doc",
                            "url": "https://example.com/large-debug-doc",
                            "excerpt": oversized_text,
                        }
                    ]
                },
            )
        },
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(SynthesizedAnswer(answer="Large debug excerpt.", citations=[]))

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("debug me")

    payload_preview = result["raw_response"]["debug"]["calls"][0]["payload_preview"]
    assert len(payload_preview) == PAYLOAD_PREVIEW_CHAR_LIMIT
    assert payload_preview.endswith("…")
    assert oversized_text not in payload_preview


def test_search_debug_call_tool_failure_with_http_error_prints_stderr_error_line(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    error = RuntimeError("tool call failed Authorization=Bearer secret-token")
    error.response = SimpleNamespace(
        status_code=403,
        json=lambda: {
            "error": {
                "code": 403,
                "message": "Forbidden Basic abc123",
                "data": {"api_token": "top-secret", "detail": "invalid cloudId"},
            }
        },
    )
    server = FakeServer(tools=make_discovered_tools(), error=error)
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        raise AssertionError("tool invocation should have failed")

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["last_step"] == f"call_tool:{SHARED_SEARCH_TOOL}"
    assert debug["http_status"] == 403
    assert debug["mcp_error_preview"]["code"] == 403
    assert "tool call failed" in debug["error_message"]
    assert "Forbidden" in debug["mcp_error_preview"]["message"]
    assert "[AgenticSearch debug] error:" in captured.err
    assert '"error_message": "tool call failed' in captured.err
    assert "secret-token" not in captured.err
    assert "abc123" not in captured.err
    assert "top-secret" not in captured.err
    assert "authorization" not in serialized
    assert "api_token" not in serialized


def test_search_debug_redacts_json_string_http_response_body(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    error = RuntimeError("tool call failed")
    error.response = SimpleNamespace(
        status_code=403,
        text='{"api_token":"top-secret","Authorization":"Bearer secret-token","mode":"Basic abc123"}',
    )
    server = FakeServer(tools=make_discovered_tools(), error=error)
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        raise AssertionError("tool invocation should have failed")

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    preview = debug["http_response_preview"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["http_status"] == 403
    assert '"mode": "[REDACTED_AUTH]"' in preview
    assert "top-secret" not in preview
    assert "secret-token" not in preview
    assert "Authorization" not in preview
    assert "Bearer" not in preview
    assert "Basic" not in preview
    assert "top-secret" not in captured.err
    assert "secret-token" not in captured.err
    assert "Authorization" not in captured.err
    assert "Bearer" not in captured.err
    assert "Basic abc123" not in captured.err
    assert "authorization" not in serialized
    assert "api_token" not in serialized


def test_search_debug_call_tool_failure_prints_stderr_error_line(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    server = FakeServer(
        tools=make_discovered_tools(),
        error=RuntimeError("boom Authorization=Bearer secret-token"),
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        raise AssertionError("tool invocation should have failed")

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["last_step"] == f"call_tool:{SHARED_SEARCH_TOOL}"
    assert debug["error_type"] == "RuntimeError"
    assert "boom" in debug["error_message"]
    assert '"error_message": "boom [REDACTED_AUTH]"' in captured.err
    assert "[AgenticSearch debug] call_tool:" in captured.err
    assert "[AgenticSearch debug] error:" in captured.err
    assert "secret-token" not in captured.err
    assert "bearer" not in serialized
    assert "authorization" not in serialized


def test_debug_response_summary_includes_redacted_error_text_for_error_payload(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: SimpleNamespace(
                isError=True,
                structuredContent={
                    "message": "Bad request Authorization=Bearer hidden-token",
                    "data": {"api_token": "hidden-token"},
                },
                content=[],
            )
        },
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(SynthesizedAnswer(answer="Bad request surfaced.", citations=[]))

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("debug me")

    summary = result["raw_response"]["debug"]["calls"][0]["response_summary"]
    serialized = json.dumps(summary).lower()
    assert result["raw_response"]["status"] == "ok"
    assert summary["is_error"] is True
    assert "Bad request" in summary["error_text"]
    assert "hidden-token" not in serialized
    assert "authorization" not in serialized
    assert "api_token" not in serialized


def test_local_mode_configures_async_openai_client_and_chat_completions_api(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={
            SHARED_SEARCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Local Mode Doc",
                            "url": "https://example.com/local-mode",
                            "excerpt": "Local mode works.",
                        }
                    ]
                },
            )
        },
    )
    search = AgenticSearch(
        openai_api_key="",
        llm_base_url="http://localhost:11434/v1",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="test-cloud-id",
    )
    synthesized = SynthesizedAnswer(
        answer="Local mode works.",
        citations=[
            Citation(
                title="Local Mode Doc",
                url="https://example.com/local-mode",
            )
        ],
    )

    async def fake_runner(agent, prompt, **_kwargs):
        tool = next(tool for tool in await get_agent_tools(agent) if tool.name == SHARED_SEARCH_TOOL)
        await tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps(
                {
                    "query": prompt,
                    "max_results": 5,
                    "cloudId": "test-cloud-id",
                }
            ),
        )
        return FakeRunResult(synthesized)

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
        patch("agentic_search.set_default_openai_client") as set_client,
        patch("agentic_search.set_default_openai_api") as set_api,
        patch("agentic_search.set_tracing_disabled") as set_tracing_disabled,
    ):
        result = search.search("Use local mode")

    assert result["raw_response"]["status"] == "ok"
    assert search.openai_api_key == "ollama"
    called_client = set_client.call_args.args[0]
    assert str(called_client.base_url).rstrip("/") == "http://localhost:11434/v1"
    assert called_client.api_key == "ollama"
    assert set_client.call_args.kwargs == {"use_for_tracing": False}
    set_api.assert_called_once_with("chat_completions")
    set_tracing_disabled.assert_called_once_with(True)


def test_local_mode_uses_default_ollama_chat_model_when_openai_model_is_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    search = AgenticSearch(
        openai_api_key="",
        llm_base_url="http://localhost:11434/v1",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="test-cloud-id",
    )

    assert search.model == DEFAULT_OLLAMA_CHAT_MODEL


def test_normalize_documents_handles_rovo_ari_only_results(rovo_search_ari_only_payload):
    search = make_search()

    documents = search._normalize_documents(rovo_search_ari_only_payload, source=SHARED_SEARCH_TOOL)

    assert len(documents) == 1
    assert documents[0].title == "Benefits FAQ"
    assert documents[0].url is None
    assert documents[0].identifier == "faq-123"
    assert documents[0].ari == "ari:cloud:confluence::page/faq"
    assert documents[0].excerpt == "Frequently asked questions about benefits."


def test_search_falls_back_to_ari_citation_when_url_is_missing(rovo_search_ari_only_payload):
    server = FakeServer(
        tools=make_discovered_tools(),
        responses={SHARED_SEARCH_TOOL: rovo_search_ari_only_payload},
    )
    search = make_search()

    async def fake_runner(agent, prompt, **_kwargs):
        await invoke_discovered_tools(agent, [(SHARED_SEARCH_TOOL, {"query": prompt})])
        return FakeRunResult(
            SynthesizedAnswer(
                answer="Benefits FAQ covers common enrollment questions.",
                citations=[],
            )
        )

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
        patch("agentic_search.Runner.run", new=AsyncMock(side_effect=fake_runner)),
    ):
        result = search.search("Where is the FAQ?")

    assert result["citations"] == [
        {
            "title": "Benefits FAQ",
            "url": None,
            "excerpt": "Frequently asked questions about benefits.",
            "identifier": "faq-123",
            "ari": "ari:cloud:confluence::page/faq",
        }
    ]


def test_local_mode_falls_back_to_plain_text_synthesis_when_structured_output_fails():
    search = AgenticSearch(
        openai_api_key="",
        llm_base_url="http://localhost:11434/v1",
        atlassian_email="bot@example.com",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="test-cloud-id",
    )
    documents = [
        search._normalize_documents(
            {
                "results": [
                    {
                        "title": "Fallback Doc",
                        "url": "https://example.com/fallback",
                        "excerpt": "Fallback excerpt.",
                    }
                ]
            },
            source="search",
        )[0]
    ]

    with patch(
        "agentic_search.Runner.run",
        new=AsyncMock(
            side_effect=[
                RuntimeError("structured outputs unsupported"),
                SimpleNamespace(final_output="Plain fallback answer."),
            ]
        ),
    ):
        synthesized = asyncio.run(search._synthesize_answer("fallback", documents))

    assert synthesized.answer == "Plain fallback answer."
    assert synthesized.citations == [
        Citation(
            title="Fallback Doc",
            url="https://example.com/fallback",
            excerpt="Fallback excerpt.",
        )
    ]