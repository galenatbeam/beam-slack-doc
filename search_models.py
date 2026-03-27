"""Shared models for agentic search orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation returned to callers."""

    title: str
    url: str | None = None
    excerpt: str | None = None
    identifier: str | None = None
    ari: str | None = None


class SynthesizedAnswer(BaseModel):
    """Structured model output for final answers."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)


@dataclass(frozen=True)
class ToolStrategy:
    """Selected MCP tool strategy."""

    name: str
    search_tool: str
    fetch_tool: str | None
    uses_cql: bool


@dataclass
class DocumentResult:
    """Normalized documentation result used for synthesis and citations."""

    title: str
    url: str | None = None
    excerpt: str | None = None
    identifier: str | None = None
    ari: str | None = None
    content: str | None = None
    source: str = "search"


@dataclass
class AgentRunOutcome:
    """Successful agent run outcome after search-tool enforcement."""

    synthesized: SynthesizedAnswer
    collected_documents: list[DocumentResult]
    run_used: int


@dataclass(frozen=True)
class MCPServerBinding:
    """Connected MCP server metadata used for tool routing."""

    key: str
    name: str
    url: str
    server: Any


@dataclass(frozen=True)
class DiscoveredToolBinding:
    """Discovered tool plus the originating server it must be called on."""

    server_binding: MCPServerBinding
    tool: Any
    public_name: str

    @property
    def original_name(self) -> str:
        return str(getattr(self.tool, "name", self.public_name))