from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, Tool

from agentic_search import (
    ACCESSIBLE_RESOURCES_TOOL,
    AgenticSearch,
    Citation,
    DEFAULT_OLLAMA_CHAT_MODEL,
    SynthesizedAnswer,
)


class FakeServer:
    def __init__(self, tools, responses=None, error=None, list_tools_error=None):
        self.tools = tools
        self.responses = responses or {}
        self.error = error
        self.list_tools_error = list_tools_error
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
        if self.list_tools_error:
            raise self.list_tools_error
        return self.tools

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
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
        mcp_auth_header="Bearer test-mcp-token",
        model="gpt-4.1-mini",
    )


def test_mcp_auth_header_takes_precedence_over_structured_auth_env(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_HEADER", "Bearer preferred-token")
    monkeypatch.setenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", "service-key")
    monkeypatch.setenv("ATLASSIAN_API_EMAIL", "bot@example.com")
    monkeypatch.setenv("ATLASSIAN_API_TOKEN", "api-token")

    search = AgenticSearch(openai_api_key="test-openai-key")

    assert search.mcp_auth_header == "Bearer preferred-token"


def test_service_account_key_builds_bearer_auth_header(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_HEADER", raising=False)
    monkeypatch.setenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", "service-key")
    monkeypatch.delenv("ATLASSIAN_API_EMAIL", raising=False)
    monkeypatch.delenv("ATLASSIAN_API_TOKEN", raising=False)

    search = AgenticSearch(openai_api_key="test-openai-key")

    assert search.mcp_auth_header == "Bearer service-key"


def test_api_token_auth_builds_basic_header(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_HEADER", raising=False)
    monkeypatch.delenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setenv("ATLASSIAN_API_EMAIL", "bot@example.com")
    monkeypatch.setenv("ATLASSIAN_API_TOKEN", "api-token")

    search = AgenticSearch(openai_api_key="test-openai-key")

    expected = base64.b64encode(b"bot@example.com:api-token").decode("ascii")
    assert search.mcp_auth_header == f"Basic {expected}"


def test_search_returns_configuration_error_when_atlassian_auth_is_missing(monkeypatch):
    monkeypatch.delenv("MCP_AUTH_HEADER", raising=False)
    monkeypatch.delenv("ATLASSIAN_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.delenv("ATLASSIAN_API_EMAIL", raising=False)
    monkeypatch.delenv("ATLASSIAN_API_TOKEN", raising=False)

    search = AgenticSearch(openai_api_key="test-openai-key")

    result = search.search("What failed?")

    assert result["raw_response"]["status"] == "configuration_error"
    assert "ATLASSIAN_SERVICE_ACCOUNT_KEY" in result["raw_response"]["missing_variable"]


def test_search_prefers_shared_search_and_fetch_tools():
    server = FakeServer(
        tools=[
            make_tool("search", {"query": {"type": "string"}, "limit": {"type": "integer"}}, ["query"]),
            make_tool("fetch", {"ari": {"type": "string"}}, ["ari"]),
        ],
        responses={
            "search": CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Beam Benefits Overview",
                            "url": "https://example.com/beam-overview",
                            "excerpt": "Beam helps employers manage benefits.",
                            "ari": "ari:cloud:confluence::page/123",
                        }
                    ]
                },
            ),
            "fetch": CallToolResult(
                content=[],
                structuredContent={
                    "title": "Beam Benefits Overview",
                    "url": "https://example.com/beam-overview",
                    "body": "Beam Benefits offers dental, vision, and supplemental benefits.",
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
    assert result["raw_response"]["strategy"] == "shared"
    assert server.calls[0] == ("search", {"query": "What is Beam Benefits?", "limit": 5})
    assert server.calls[1] == ("fetch", {"ari": "ari:cloud:confluence::page/123"})


def test_search_prefers_explicit_mcp_cloud_id_over_discovery():
    server = FakeServer(
        tools=[
            make_tool(
                "search",
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "cloudId": {"type": "string"},
                },
                ["query"],
            ),
            make_tool(
                "fetch",
                {"ari": {"type": "string"}, "cloudId": {"type": "string"}},
                ["ari"],
            ),
            make_tool(ACCESSIBLE_RESOURCES_TOOL, {}),
        ],
        responses={
            "search": CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Beam Benefits Overview",
                            "url": "https://example.com/beam-overview",
                            "excerpt": "Beam helps employers manage benefits.",
                            "ari": "ari:cloud:confluence::page/123",
                        }
                    ]
                },
            ),
            "fetch": CallToolResult(
                content=[],
                structuredContent={
                    "title": "Beam Benefits Overview",
                    "url": "https://example.com/beam-overview",
                    "body": "Beam Benefits offers dental, vision, and supplemental benefits.",
                },
            ),
            ACCESSIBLE_RESOURCES_TOOL: CallToolResult(
                content=[],
                structuredContent={"resources": [{"cloudId": "discovered-cloud-id"}]},
            ),
        },
    )
    search = AgenticSearch(
        openai_api_key="test-openai-key",
        mcp_auth_header="Bearer test-mcp-token",
        mcp_cloud_id="configured-cloud-id",
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
    assert server.calls[0] == (
        "search",
        {
            "query": "What is Beam Benefits?",
            "limit": 5,
            "cloudId": "configured-cloud-id",
        },
    )
    assert server.calls[1] == (
        "fetch",
        {
            "ari": "ari:cloud:confluence::page/123",
            "cloudId": "configured-cloud-id",
        },
    )
    assert all(call[0] != ACCESSIBLE_RESOURCES_TOOL for call in server.calls)


def test_search_discovers_cloud_id_when_mcp_cloud_id_is_unset():
    server = FakeServer(
        tools=[
            make_tool(
                "search",
                {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "cloudId": {"type": "string"},
                },
                ["query"],
            ),
            make_tool(
                "fetch",
                {"ari": {"type": "string"}, "cloudId": {"type": "string"}},
                ["ari"],
            ),
            make_tool(ACCESSIBLE_RESOURCES_TOOL, {}),
        ],
        responses={
            ACCESSIBLE_RESOURCES_TOOL: CallToolResult(
                content=[],
                structuredContent={"resources": [{"cloudId": "discovered-cloud-id"}]},
            ),
            "search": CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Beam Benefits Overview",
                            "url": "https://example.com/beam-overview",
                            "excerpt": "Beam helps employers manage benefits.",
                            "ari": "ari:cloud:confluence::page/123",
                        }
                    ]
                },
            ),
            "fetch": CallToolResult(
                content=[],
                structuredContent={
                    "title": "Beam Benefits Overview",
                    "url": "https://example.com/beam-overview",
                    "body": "Beam Benefits offers dental, vision, and supplemental benefits.",
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
                )
            ],
        )
    )

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What is Beam Benefits?")

    assert result["raw_response"]["status"] == "ok"
    assert server.calls[0] == (ACCESSIBLE_RESOURCES_TOOL, {})
    assert server.calls[1] == (
        "search",
        {
            "query": "What is Beam Benefits?",
            "limit": 5,
            "cloudId": "discovered-cloud-id",
        },
    )
    assert server.calls[2] == (
        "fetch",
        {
            "ari": "ari:cloud:confluence::page/123",
            "cloudId": "discovered-cloud-id",
        },
    )


def test_search_falls_back_to_confluence_tools_when_shared_tools_missing():
    server = FakeServer(
        tools=[
            make_tool("searchConfluenceUsingCqlSearch", {"cql": {"type": "string"}}, ["cql"]),
            make_tool("getConfluencePage", {"id": {"type": "string"}}, ["id"]),
        ],
        responses={
            "searchConfluenceUsingCqlSearch": CallToolResult(
                content=[],
                structuredContent={
                    "results": [
                        {
                            "title": "Benefits FAQ",
                            "url": "https://example.com/faq",
                            "excerpt": "Frequently asked questions about benefits.",
                            "id": "456",
                        }
                    ]
                },
            ),
            "getConfluencePage": CallToolResult(
                content=[],
                structuredContent={
                    "title": "Benefits FAQ",
                    "url": "https://example.com/faq",
                    "body": "Employees can review plan information from the benefits portal.",
                },
            ),
        },
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
    assert server.calls[0][0] == "searchConfluenceUsingCqlSearch"
    assert server.calls[0][1]["cql"] == 'text ~ "Where do employees review plan information?"'
    assert server.calls[1] == ("getConfluencePage", {"id": "456"})


def test_search_returns_no_results_response_when_search_is_empty():
    server = FakeServer(
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
        responses={
            "search": CallToolResult(content=[], structuredContent={"results": []}),
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
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
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
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
        responses={
            "search": CallToolResult(
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
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
        responses={
            "search": CallToolResult(
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
    assert debug["calls"][0]["response_summary"]["top_level_keys"] == ["results", "[REDACTED]"]


def test_search_debug_list_tools_failure_prints_stderr_error_line(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    error = RuntimeError("list_tools failed Authorization=Bearer secret-token")
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
    server = FakeServer(tools=[], list_tools_error=error)
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["last_step"] == "list_tools"
    assert debug["http_status"] == 403
    assert debug["mcp_error_preview"]["code"] == 403
    assert "list_tools failed" in debug["error_message"]
    assert "Forbidden" in debug["mcp_error_preview"]["message"]
    assert "[AgenticSearch debug] error:" in captured.err
    assert '"error_message": "list_tools failed' in captured.err
    assert "secret-token" not in captured.err
    assert "abc123" not in captured.err
    assert "top-secret" not in captured.err
    assert "authorization" not in serialized
    assert "api_token" not in serialized


def test_search_debug_call_tool_failure_prints_stderr_error_line(monkeypatch, capsys):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    server = FakeServer(
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
        error=RuntimeError("boom Authorization=Bearer secret-token"),
    )
    search = make_search()

    with patch("agentic_search.MCPServerStreamableHttp", return_value=server):
        result = search.search("What failed?")

    captured = capsys.readouterr()
    debug = result["raw_response"]["debug"]
    serialized = json.dumps(debug).lower()
    assert result["raw_response"]["status"] == "search_error"
    assert debug["last_step"] == "call_tool:search"
    assert debug["error_type"] == "RuntimeError"
    assert "boom" in debug["error_message"]
    assert "traceback" in debug
    assert '"error_message": "boom [REDACTED_SECRET]"' in captured.err
    assert "[AgenticSearch debug] call_tool:" in captured.err
    assert "[AgenticSearch debug] error:" in captured.err
    assert "secret-token" not in captured.err
    assert "bearer" not in serialized
    assert "authorization" not in serialized


def test_debug_response_summary_includes_redacted_error_text_for_error_payload(monkeypatch):
    monkeypatch.setenv("AGENTIC_SEARCH_DEBUG", "1")
    server = FakeServer(
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
        responses={
            "search": SimpleNamespace(
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
        tools=[make_tool("search", {"query": {"type": "string"}}, ["query"])],
        responses={
            "search": CallToolResult(
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
        mcp_auth_header="Bearer test-mcp-token",
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
        mcp_auth_header="Bearer test-mcp-token",
    )

    assert search.model == DEFAULT_OLLAMA_CHAT_MODEL


def test_local_mode_falls_back_to_plain_text_synthesis_when_structured_output_fails():
    search = AgenticSearch(
        openai_api_key="",
        llm_base_url="http://localhost:11434/v1",
        mcp_auth_header="Bearer test-mcp-token",
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