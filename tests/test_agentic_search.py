from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, Tool

from agentic_search import (
    AgenticSearch,
    Citation,
    CONFLUENCE_AUTH_ISSUE_MESSAGE,
    DEFAULT_OLLAMA_CHAT_MODEL,
    SHARED_FETCH_TOOL,
    SHARED_SEARCH_TOOL,
    SynthesizedAnswer,
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

    async def list_tools(self):
        self.list_tools_calls += 1
        if self.list_tools_error:
            raise self.list_tools_error
        return self.tools

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        if tool_name in self.call_errors:
            raise self.call_errors[tool_name]
        if self.error:
            raise self.error
        return self.responses[tool_name]


def make_tool(name: str, properties: dict, required: list[str] | None = None) -> Tool:
    return Tool(
        name=name,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    )


def make_search() -> AgenticSearch:
    return AgenticSearch(
        openai_api_key="test-openai-key",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="test-cloud-id",
        model="gpt-4.1-mini",
    )


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


class FakeRunResult:
    def __init__(self, synthesized: SynthesizedAnswer, final_output: str | None = None):
        self._synthesized = synthesized
        self.final_output = final_output or synthesized.answer

    def final_output_as(self, _model, raise_if_incorrect_type=True):
        return self._synthesized


async def invoke_discovered_tools(agent, tool_calls: list[tuple[str, dict]]) -> None:
    tool_map = {tool.name: tool for tool in agent.tools}
    for tool_name, arguments in tool_calls:
        await tool_map[tool_name].on_invoke_tool(None, json.dumps(arguments))


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


def test_confluence_mcp_api_key_builds_bearer_auth_header(monkeypatch):
    monkeypatch.setenv("CONFLUENCE_MCP_API_KEY", "service-key")
    monkeypatch.setenv("MCP_AUTH_HEADER", "Bearer ignored-token")
    monkeypatch.setenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", "ignored-service-key")
    monkeypatch.setenv("ATLASSIAN_API_EMAIL", "bot@example.com")
    monkeypatch.setenv("ATLASSIAN_API_TOKEN", "ignored-api-token")

    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_cloud_id="test-cloud-id",
    )

    assert search.mcp_auth_header == "Bearer service-key"


def test_legacy_auth_env_vars_are_ignored(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_HEADER", "Bearer ignored-token")
    monkeypatch.setenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", "ignored-service-key")
    monkeypatch.setenv("ATLASSIAN_API_EMAIL", "bot@example.com")
    monkeypatch.setenv("ATLASSIAN_API_TOKEN", "ignored-api-token")
    monkeypatch.delenv("CONFLUENCE_MCP_API_KEY", raising=False)
    monkeypatch.delenv("ATLASSIAN_CLOUD_ID", raising=False)

    search = AgenticSearch(openai_api_key="test-openai-key")

    result = search.search("What failed?")

    assert search.mcp_auth_header == ""
    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == [
        "CONFLUENCE_MCP_API_KEY",
        "ATLASSIAN_CLOUD_ID",
    ]


def test_search_returns_configuration_error_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("CONFLUENCE_MCP_API_KEY", raising=False)
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        atlassian_cloud_id="test-cloud-id",
    )

    result = search.search("What failed?")

    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == ["CONFLUENCE_MCP_API_KEY"]


def test_search_returns_configuration_error_when_cloud_id_is_missing(monkeypatch):
    monkeypatch.delenv("ATLASSIAN_CLOUD_ID", raising=False)
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        confluence_mcp_api_key="test-mcp-token",
    )

    result = search.search("What failed?")

    assert result["raw_response"]["status"] == "configuration_error"
    assert result["raw_response"]["missing_variables"] == ["ATLASSIAN_CLOUD_ID"]


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
        assert {tool.name for tool in agent.tools} == {SHARED_SEARCH_TOOL, SHARED_FETCH_TOOL}
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


def test_search_includes_redacted_debug_block_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "yes")
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
    assert debug["calls"][0]["tool_name"] == SHARED_SEARCH_TOOL
    assert debug["calls"][0]["argument_keys"] == ["query", "cloudId"]
    assert debug["calls"][0]["response_summary"]["top_level_keys"] == ["results"]


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
        tool = next(tool for tool in agent.tools if tool.name == SHARED_SEARCH_TOOL)
        await tool.on_invoke_tool(
            None,
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