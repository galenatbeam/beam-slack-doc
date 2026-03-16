from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp.types import CallToolResult, Tool

from agentic_search import (
    AgenticSearch,
    Citation,
    DEFAULT_OLLAMA_CHAT_MODEL,
    SynthesizedAnswer,
)


class FakeServer:
    def __init__(self, tools, responses=None, error=None):
        self.tools = tools
        self.responses = responses or {}
        self.error = error
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def list_tools(self):
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