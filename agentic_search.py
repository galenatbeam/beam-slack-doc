"""Stub Confluence search component powered by an OpenAI agent."""

from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()


class AgenticSearch:
    """Stub for an OpenAI-powered agent that searches Confluence via MCP."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        confluence_space: str | None = None,
        mcp_server_name: str | None = None,
        model: str | None = None,
    ) -> None:
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.confluence_space = confluence_space or os.getenv("CONFLUENCE_SPACE", "")
        self.mcp_server_name = mcp_server_name or os.getenv(
            "MCP_SERVER_NAME", "confluence-search"
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def search(self, query: str) -> Dict[str, Any]:
        """Return a stubbed result until the OpenAI agent + MCP wiring is added."""
        clean_query = query.strip()
        return {
            "query": clean_query,
            "answer": (
                "Stubbed Confluence answer for "
                f"'{clean_query}' in space '{self.confluence_space or 'UNSET'}'. "
                f"Planned MCP server: '{self.mcp_server_name}'."
            ),
            "citations": [],
            "raw_response": {
                "status": "stub",
                "model": self.model,
                "mcp_server_name": self.mcp_server_name,
                "confluence_space": self.confluence_space,
            },
        }

    def build_agent_instructions(self) -> str:
        """Instruction template for the future OpenAI agent implementation."""
        return (
            "You are a documentation search assistant. Use the MCP-backed Confluence "
            f"search tool for space '{self.confluence_space or 'UNSET'}'. Summarize "
            "results, cite relevant pages, and say when information is missing."
        )