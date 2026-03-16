from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, Tool

from agentic_search import (
    AgenticSearch,
    Citation,
    CONFLUENCE_SEARCH_TOOL,
    CONFLUENCE_SEARCH_TOOL_FALLBACK,
    DEFAULT_OLLAMA_CHAT_MODEL,
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


def make_tool_not_found_error(message: str = "Method not found") -> RuntimeError:
    error = RuntimeError(message)
    error.error = SimpleNamespace(code=-32601, message=message)
    return error


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


def test_search_calls_static_confluence_cql_tool_without_list_tools():
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Beam Benefits Overview",
                            "url": "https://example.com/beam-overview",
                            "excerpt": "Beam helps employers manage benefits.",
                        }
                    ]
                },
            ),
        },
    )
    search = make_search()
    search._synthesize_answer = AsyncMock(
        return_value=SynthesizedAnswer(
            answer="Beam Benefits offers dental, vision, and supplemental benefits.",
            citations=[
                Citation(
                    title="Beam Benefits Overview",
                    url="https://example.com/beam-overview",
                    excerpt="Beam helps employers manage benefits.",
                )
            ],
        )
    )

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What is Beam Benefits?")

    assert result["answer"].startswith("Beam Benefits offers dental")
    assert result["citations"] == [
        {
            "title": "Beam Benefits Overview",
            "url": "https://example.com/beam-overview",
            "excerpt": "Beam helps employers manage benefits.",
        }
    ]
    assert result["raw_response"]["status"] == "ok"
    assert result["raw_response"]["strategy"] == "confluence"
    assert server.list_tools_calls == 0
    assert server.calls == [
        (
            CONFLUENCE_SEARCH_TOOL,
            {
                "cql": 'text ~ "What is Beam Benefits?"',
                "limit": 5,
                "cloudId": "test-cloud-id",
            },
        )
    ]


def test_search_uses_explicit_atlassian_cloud_id_for_confluence_tool_args():
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Beam Benefits Overview",
                            "url": "https://example.com/beam-overview",
                            "excerpt": "Beam helps employers manage benefits.",
                        }
                    ]
                },
            ),
        },
    )
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        confluence_mcp_api_key="test-mcp-token",
        atlassian_cloud_id="configured-cloud-id",
        model="gpt-4.1-mini",
    )
    search._synthesize_answer = AsyncMock(
        return_value=SynthesizedAnswer(
            answer="Beam Benefits offers dental, vision, and supplemental benefits.",
            citations=[
                Citation(
                    title="Beam Benefits Overview",
                    url="https://example.com/beam-overview",
                )
            ],
        )
    )

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What is Beam Benefits?")

    assert result["raw_response"]["status"] == "ok"
    assert server.list_tools_calls == 0
    assert server.calls == [
        (
            CONFLUENCE_SEARCH_TOOL,
            {
                "cql": 'text ~ "What is Beam Benefits?"',
                "limit": 5,
                "cloudId": "configured-cloud-id",
            },
        )
    ]


def test_search_falls_back_to_secondary_confluence_tool_when_primary_tool_not_found():
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL_FALLBACK: CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Benefits FAQ",
                            "url": "https://example.com/faq",
                            "excerpt": "Frequently asked questions about benefits.",
                        }
                    ]
                },
            ),
        },
        call_errors={CONFLUENCE_SEARCH_TOOL: make_tool_not_found_error()},
    )
    search = make_search()
    search._synthesize_answer = AsyncMock(
        return_value=SynthesizedAnswer(
            answer="Employees can review plan information from the benefits portal.",
            citations=[Citation(title="Benefits FAQ", url="https://example.com/faq")],
        )
    )

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("Where do employees review plan information?")

    assert result["raw_response"]["strategy"] == "confluence"
    assert server.list_tools_calls == 0
    assert server.calls == [
        (
            CONFLUENCE_SEARCH_TOOL,
            {
                "cql": 'text ~ "Where do employees review plan information?"',
                "limit": 5,
                "cloudId": "test-cloud-id",
            },
        ),
        (
            CONFLUENCE_SEARCH_TOOL_FALLBACK,
            {
                "cql": 'text ~ "Where do employees review plan information?"',
                "limit": 5,
                "cloudId": "test-cloud-id",
            },
        ),
    ]


def test_search_returns_no_results_response_when_search_is_empty():
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL: CallToolResult(content=[], structuredContent={"results": []}),
        },
    )
    search = make_search()
    search._synthesize_answer = AsyncMock()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("totally unknown phrase")

    assert result["raw_response"]["status"] == "no_results"
    assert result["citations"] == []
    search._synthesize_answer.assert_not_called()


def test_search_returns_generic_error_response_when_mcp_tool_fails():
    server = FakeServer(
        error=RuntimeError("boom"),
    )
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What failed?")

    assert result["answer"].startswith("Sorry, I ran into an issue")
    assert result["citations"] == []
    assert result["raw_response"]["status"] == "search_error"
    assert result["raw_response"]["error_type"] == "RuntimeError"


def test_search_omits_debug_block_when_debug_disabled(monkeypatch):
    monkeypatch.delenv("AGENTIC_SEARCH_DEBUG", raising=False)
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL: CallToolResult(
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
    search._synthesize_answer = AsyncMock(
        return_value=SynthesizedAnswer(
            answer="Debug excerpt.",
            citations=[Citation(title="Debug Doc", url="https://example.com/debug-doc")],
        )
    )

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("debug me")

    assert "debug" not in result["raw_response"]


def test_search_includes_redacted_debug_block_when_enabled(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "yes")
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL: CallToolResult(
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
    search._synthesize_answer = AsyncMock(
        return_value=SynthesizedAnswer(
            answer="Debug excerpt.",
            citations=[Citation(title="Debug Doc", url="https://example.com/debug-doc")],
        )
    )

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("debug me")

    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert "bearer" not in serialized
    assert "basic" not in serialized
    assert "api_token" not in serialized
    assert debug["mcp_server_url"] == search.mcp_server_url
    assert debug["mcp_server_name"] == search.mcp_server_name
    assert debug["tool_discovery"] == {"skipped": True}
    assert debug["search_tool"] == {
        "tool_name": CONFLUENCE_SEARCH_TOOL,
        "argument_keys": ["cql", "limit", "cloudId"],
    }
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
    server = FakeServer(error=error)
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["last_step"] == f"call_tool:{CONFLUENCE_SEARCH_TOOL}"
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
    server = FakeServer(error=error)
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
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
        error=RuntimeError("boom Authorization=Bearer secret-token"),
    )
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["last_step"] == f"call_tool:{CONFLUENCE_SEARCH_TOOL}"
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
        responses={
            CONFLUENCE_SEARCH_TOOL: SimpleNamespace(
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

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("debug me")

    summary = result["raw_response"]["debug"]["calls"][0]["response_summary"]
    serialized = json.dumps(summary).lower()
    assert result["raw_response"]["status"] == "no_results"
    assert summary["is_error"] is True
    assert "Bad request" in summary["error_text"]
    assert "hidden-token" not in serialized
    assert "authorization" not in serialized
    assert "api_token" not in serialized


def test_local_mode_configures_async_openai_client_and_chat_completions_api():
    server = FakeServer(
        responses={
            CONFLUENCE_SEARCH_TOOL: CallToolResult(
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
    search._synthesize_answer = AsyncMock(
        return_value=SynthesizedAnswer(
            answer="Local mode works.",
            citations=[
                Citation(
                    title="Local Mode Doc",
                    url="https://example.com/local-mode",
                )
            ],
        )
    )

    with (
        patch("agentic_search.MCPServerStreamableHttp", return_value=server),
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