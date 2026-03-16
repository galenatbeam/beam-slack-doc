"""Agentic Confluence search powered by OpenAI Agents SDK + Atlassian Rovo MCP."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStreamableHttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_MCP_SERVER_URL = "https://mcp.atlassian.com/v1/mcp"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_CHAT_MODEL = "qwen3:4b-instruct"
LOCAL_LLM_API_KEY_PLACEHOLDER = "ollama"
MAX_FETCH_RESULTS = 3
SHARED_SEARCH_TOOL = "search"
SHARED_FETCH_TOOL = "fetch"
CONFLUENCE_SEARCH_TOOL = "searchConfluenceUsingCqlSearch"
CONFLUENCE_FETCH_TOOL = "getConfluencePage"
ACCESSIBLE_RESOURCES_TOOL = "getAccessibleAtlassianResources"
TITLE_KEYS = ("title", "name", "pageTitle")
URL_KEYS = ("url", "webLink", "link", "pageUrl")
EXCERPT_KEYS = ("excerpt", "snippet", "summary", "text")
ID_KEYS = ("id", "pageId", "contentId", "entityId")
ARI_KEYS = ("ari", "resourceAri", "resourceARI", "resourceId")
CONTENT_KEYS = ("body", "content", "markdown", "value")


class Citation(BaseModel):
    """Citation returned to callers."""

    title: str
    url: str
    excerpt: str | None = None


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


class AgenticSearch:
    """Search documentation via Atlassian Rovo MCP and synthesize a cited answer."""

    def __init__(
        self,
        openai_api_key: str | None = None,
        confluence_space: str | None = None,
        mcp_server_name: str | None = None,
        model: str | None = None,
        llm_base_url: str | None = None,
        mcp_server_url: str | None = None,
        mcp_auth_header: str | None = None,
        max_results: int = 5,
    ) -> None:
        self.llm_base_url = (llm_base_url or os.getenv("LLM_BASE_URL", "")).strip()
        env_openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_api_key = (
            openai_api_key
            or env_openai_api_key
            or (LOCAL_LLM_API_KEY_PLACEHOLDER if self.uses_local_llm else "")
        )
        self.confluence_space = confluence_space or os.getenv("CONFLUENCE_SPACE", "")
        self.mcp_server_name = mcp_server_name or os.getenv(
            "MCP_SERVER_NAME", "atlassian-rovo"
        )
        self.model = self._resolve_model(model)
        self.mcp_server_url = mcp_server_url or os.getenv(
            "MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL
        )
        self.mcp_auth_header = mcp_auth_header or os.getenv("MCP_AUTH_HEADER", "")
        self.max_results = max(1, max_results)

    @property
    def uses_local_llm(self) -> bool:
        return bool(self.llm_base_url)

    def search(self, query: str) -> Dict[str, Any]:
        """Search documentation and return a structured answer with citations."""
        clean_query = query.strip()
        if not clean_query:
            return {
                "answer": "Please provide a question or keywords to search the documentation.",
                "citations": [],
                "raw_response": {
                    "status": "invalid_query",
                    "model": self.model,
                    "mcp_server_url": self.mcp_server_url,
                },
            }

        if not self.uses_local_llm and not self.openai_api_key:
            return self._configuration_error_response(clean_query, "OPENAI_API_KEY")

        if not self.mcp_auth_header:
            return self._configuration_error_response(clean_query, "MCP_AUTH_HEADER")

        try:
            return asyncio.run(self._search_async(clean_query))
        except Exception as exc:  # pragma: no cover - exercised via public response path
            return self._error_response(clean_query, "search_error", exc)

    def build_agent_instructions(self) -> str:
        """Instruction template for the answer-synthesis agent."""
        scope = ""
        if self.confluence_space:
            scope = f" Favor documents from Confluence space '{self.confluence_space}'."

        return (
            "You are a documentation search assistant. You are given documentation that was "
            "already retrieved through MCP tools. Answer only from the provided documents. "
            "Keep the answer concise, say clearly when the docs do not fully answer the query, "
            "and cite only documents that were provided using title + URL." + scope
        )

    async def _search_async(self, query: str) -> Dict[str, Any]:
        self._configure_agents_sdk()

        async with self._build_mcp_server() as server:
            tools = await server.list_tools()
            tool_map = {tool.name: tool for tool in tools}
            strategy = self._select_strategy(tool_map)
            if strategy is None:
                raise RuntimeError("No supported Atlassian Rovo search tools were available.")

            cloud_id = await self._resolve_cloud_id(server, tool_map, strategy)
            search_payload = await server.call_tool(
                strategy.search_tool,
                self._build_search_arguments(
                    tool_map[strategy.search_tool],
                    query=query,
                    cloud_id=cloud_id,
                    use_cql=strategy.uses_cql,
                ),
            )
            documents = self._normalize_documents(search_payload, source=strategy.search_tool)

            if not documents:
                return {
                    "answer": (
                        f"I couldn't find relevant documentation for '{query}'. Try adding more "
                        "specific keywords, a product name, or a team-specific term."
                    ),
                    "citations": [],
                    "raw_response": {
                        "status": "no_results",
                        "strategy": strategy.name,
                        "model": self.model,
                        "query": query,
                        "selected_tools": [strategy.search_tool, strategy.fetch_tool],
                    },
                }

            fetch_errors: list[dict[str, str]] = []
            if strategy.fetch_tool and strategy.fetch_tool in tool_map:
                documents = await self._enrich_documents(
                    server,
                    tool_map[strategy.fetch_tool],
                    documents,
                    cloud_id,
                    fetch_errors,
                )

            synthesized = await self._synthesize_answer(query, documents)
            citations = self._normalize_citations(synthesized.citations, documents)

            if not citations:
                citations = self._fallback_citations(documents)

            return {
                "answer": synthesized.answer.strip(),
                "citations": [citation.model_dump() for citation in citations],
                "raw_response": {
                    "status": "ok",
                    "strategy": strategy.name,
                    "model": self.model,
                    "query": query,
                    "selected_tools": [strategy.search_tool, strategy.fetch_tool],
                    "documents": [self._document_debug_view(doc) for doc in documents],
                    "fetch_errors": fetch_errors,
                    "model_output": synthesized.model_dump(),
                },
            }

    def _build_mcp_server(self) -> MCPServerStreamableHttp:
        return MCPServerStreamableHttp(
            name=self.mcp_server_name,
            params={
                "url": self.mcp_server_url,
                "headers": {"Authorization": self.mcp_auth_header},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=2,
        )

    def _select_strategy(self, tool_map: Mapping[str, Any]) -> ToolStrategy | None:
        if SHARED_SEARCH_TOOL in tool_map:
            return ToolStrategy(
                name="shared",
                search_tool=SHARED_SEARCH_TOOL,
                fetch_tool=SHARED_FETCH_TOOL if SHARED_FETCH_TOOL in tool_map else None,
                uses_cql=False,
            )

        if CONFLUENCE_SEARCH_TOOL in tool_map:
            return ToolStrategy(
                name="confluence",
                search_tool=CONFLUENCE_SEARCH_TOOL,
                fetch_tool=(
                    CONFLUENCE_FETCH_TOOL if CONFLUENCE_FETCH_TOOL in tool_map else None
                ),
                uses_cql=True,
            )

        return None

    async def _resolve_cloud_id(
        self,
        server: Any,
        tool_map: Mapping[str, Any],
        strategy: ToolStrategy,
    ) -> str | None:
        selected_tools = [tool_map[strategy.search_tool]]
        if strategy.fetch_tool and strategy.fetch_tool in tool_map:
            selected_tools.append(tool_map[strategy.fetch_tool])

        if not any(self._tool_accepts_param(tool, "cloudId") for tool in selected_tools):
            return None

        if ACCESSIBLE_RESOURCES_TOOL not in tool_map:
            return None

        resources_payload = await server.call_tool(ACCESSIBLE_RESOURCES_TOOL, {})
        for resource in self._iter_candidate_dicts(self._extract_payload(resources_payload)):
            cloud_id = self._first_string(resource, ("cloudId",))
            if cloud_id:
                return cloud_id

        return None

    def _build_search_arguments(
        self,
        tool: Any,
        *,
        query: str,
        cloud_id: str | None,
        use_cql: bool,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {}
        value = self._build_cql_query(query) if use_cql else query
        query_key = self._first_schema_key(
            tool,
            ("cql", "query", "searchQuery", "q", "text", "prompt"),
        )
        if query_key:
            arguments[query_key] = value

        limit_key = self._first_schema_key(
            tool,
            ("limit", "maxResults", "pageSize", "numResults"),
        )
        if limit_key:
            arguments[limit_key] = self.max_results

        space_key = self._first_schema_key(tool, ("space", "spaceKey"))
        if space_key and self.confluence_space and not use_cql:
            arguments[space_key] = self.confluence_space

        if cloud_id and self._tool_accepts_param(tool, "cloudId"):
            arguments["cloudId"] = cloud_id

        if not arguments:
            raise RuntimeError(f"Could not determine search arguments for tool '{tool.name}'.")

        return arguments

    def _build_fetch_arguments(
        self,
        tool: Any,
        document: DocumentResult,
        cloud_id: str | None,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {}

        ari_key = self._first_schema_key(tool, ("ari", "resourceAri", "resourceARI", "resourceId"))
        if ari_key and document.ari:
            arguments[ari_key] = document.ari

        id_key = self._first_schema_key(tool, ("pageId", "id", "contentId", "entityId"))
        if id_key and document.identifier and id_key not in arguments:
            arguments[id_key] = document.identifier

        if cloud_id and self._tool_accepts_param(tool, "cloudId"):
            arguments["cloudId"] = cloud_id

        body_format_key = self._first_schema_key(tool, ("bodyFormat", "format"))
        if body_format_key:
            arguments[body_format_key] = "markdown"

        if not arguments:
            raise RuntimeError(f"Could not determine fetch arguments for tool '{tool.name}'.")

        return arguments

    async def _enrich_documents(
        self,
        server: Any,
        tool: Any,
        documents: Sequence[DocumentResult],
        cloud_id: str | None,
        fetch_errors: list[dict[str, str]],
    ) -> list[DocumentResult]:
        enriched: list[DocumentResult] = []
        for document in documents[:MAX_FETCH_RESULTS]:
            try:
                payload = await server.call_tool(
                    tool.name,
                    self._build_fetch_arguments(tool, document, cloud_id),
                )
                merged = self._merge_document(
                    document,
                    self._normalize_documents(payload, source=tool.name)[:1],
                    self._extract_text_blob(self._extract_payload(payload)),
                )
                enriched.append(merged)
            except Exception as exc:
                fetch_errors.append(
                    {
                        "tool": tool.name,
                        "document": document.title,
                        "error_type": type(exc).__name__,
                    }
                )
                enriched.append(document)

        if len(documents) > MAX_FETCH_RESULTS:
            enriched.extend(documents[MAX_FETCH_RESULTS:])

        return enriched

    async def _synthesize_answer(
        self, query: str, documents: Sequence[DocumentResult]
    ) -> SynthesizedAnswer:
        prompt = json.dumps(
            {
                "query": query,
                "documents": [self._document_prompt_view(doc) for doc in documents],
            },
            ensure_ascii=False,
        )

        try:
            return await self._run_structured_synthesis(prompt)
        except Exception:
            if not self.uses_local_llm:
                raise
            return await self._run_plain_text_synthesis(prompt, documents)

    async def _run_structured_synthesis(self, prompt: str) -> SynthesizedAnswer:
        agent = Agent(
            name="Doc Search Synthesizer",
            instructions=self.build_agent_instructions(),
            model=self.model,
            output_type=SynthesizedAnswer,
        )
        result = await Runner.run(agent, prompt)
        return result.final_output_as(SynthesizedAnswer, raise_if_incorrect_type=True)

    async def _run_plain_text_synthesis(
        self,
        prompt: str,
        documents: Sequence[DocumentResult],
    ) -> SynthesizedAnswer:
        agent = Agent(
            name="Doc Search Synthesizer",
            instructions=(
                self.build_agent_instructions()
                + " Return a concise plain-text answer only. Do not emit JSON or XML."
            ),
            model=self.model,
        )
        result = await Runner.run(agent, prompt)
        answer = getattr(result, "final_output", None)
        answer_text = answer if isinstance(answer, str) else str(answer or "")
        answer_text = answer_text.strip()
        if not answer_text:
            answer_text = (
                "I found relevant documentation, but the local model did not return a usable "
                "structured answer."
            )
        return SynthesizedAnswer(
            answer=answer_text,
            citations=self._fallback_citations(documents),
        )

    def _configure_agents_sdk(self) -> None:
        client_kwargs: dict[str, Any] = {"api_key": self.openai_api_key}
        if self.uses_local_llm:
            client_kwargs["base_url"] = self.llm_base_url

        set_default_openai_client(
            AsyncOpenAI(**client_kwargs),
            use_for_tracing=not self.uses_local_llm,
        )
        set_default_openai_api(
            "chat_completions" if self.uses_local_llm else "responses"
        )
        set_tracing_disabled(self.uses_local_llm)

    def _resolve_model(self, explicit_model: str | None) -> str:
        requested_model = explicit_model or os.getenv("OPENAI_MODEL", "")
        if requested_model:
            return requested_model
        if self.uses_local_llm:
            return DEFAULT_OLLAMA_CHAT_MODEL
        return DEFAULT_MODEL

    def _normalize_documents(self, payload: Any, source: str) -> list[DocumentResult]:
        extracted = self._extract_payload(payload)
        candidates = list(self._iter_candidate_dicts(extracted))
        if isinstance(extracted, dict) and not candidates:
            candidates = [extracted]

        documents: list[DocumentResult] = []
        seen: set[str] = set()
        for candidate in candidates:
            title = self._first_string(candidate, TITLE_KEYS) or self._first_string(
                candidate, EXCERPT_KEYS
            )
            url = self._first_string(candidate, URL_KEYS)
            excerpt = self._first_string(candidate, EXCERPT_KEYS)
            identifier = self._first_string(candidate, ID_KEYS)
            ari = self._first_string(candidate, ARI_KEYS)
            content = self._extract_text_blob(candidate)
            if content == excerpt:
                content = None

            if not title:
                continue

            dedupe_key = url or ari or identifier or title
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            documents.append(
                DocumentResult(
                    title=title,
                    url=url,
                    excerpt=excerpt,
                    identifier=identifier,
                    ari=ari,
                    content=content,
                    source=source,
                )
            )

        return documents[: self.max_results]

    def _normalize_citations(
        self,
        citations: Sequence[Citation],
        documents: Sequence[DocumentResult],
    ) -> list[Citation]:
        by_title = {document.title: document for document in documents}
        normalized: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for citation in citations:
            document = by_title.get(citation.title)
            url = citation.url or (document.url if document else None)
            if not citation.title or not url:
                continue
            key = (citation.title, url)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                Citation(
                    title=citation.title,
                    url=url,
                    excerpt=citation.excerpt or (document.excerpt if document else None),
                )
            )
        return normalized

    def _fallback_citations(self, documents: Sequence[DocumentResult]) -> list[Citation]:
        citations: list[Citation] = []
        for document in documents:
            if document.url:
                citations.append(
                    Citation(
                        title=document.title,
                        url=document.url,
                        excerpt=document.excerpt,
                    )
                )
        return citations[:MAX_FETCH_RESULTS]

    def _merge_document(
        self,
        original: DocumentResult,
        fetched_documents: Sequence[DocumentResult],
        fallback_content: str | None,
    ) -> DocumentResult:
        if not fetched_documents:
            if fallback_content:
                original.content = fallback_content
            return original

        fetched = fetched_documents[0]
        return DocumentResult(
            title=fetched.title or original.title,
            url=fetched.url or original.url,
            excerpt=fetched.excerpt or original.excerpt,
            identifier=fetched.identifier or original.identifier,
            ari=fetched.ari or original.ari,
            content=fetched.content or fallback_content or original.content,
            source=fetched.source,
        )

    def _extract_payload(self, payload: Any) -> Any:
        if hasattr(payload, "structuredContent") and payload.structuredContent is not None:
            return payload.structuredContent

        if hasattr(payload, "content"):
            text_fragments: list[str] = []
            for item in payload.content:
                if getattr(item, "type", None) == "text":
                    text_fragments.append(item.text)
            if len(text_fragments) == 1:
                return self._maybe_parse_json(text_fragments[0])
            if text_fragments:
                return [self._maybe_parse_json(fragment) for fragment in text_fragments]

        return payload

    def _iter_candidate_dicts(self, value: Any):
        if isinstance(value, dict):
            if any(key in value for key in TITLE_KEYS + URL_KEYS + ID_KEYS + ARI_KEYS):
                yield value
            for nested in value.values():
                yield from self._iter_candidate_dicts(nested)
        elif isinstance(value, list):
            for item in value:
                yield from self._iter_candidate_dicts(item)

    def _extract_text_blob(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            for key in CONTENT_KEYS:
                nested = value.get(key)
                text = self._extract_text_blob(nested)
                if text:
                    return text
            return None
        if isinstance(value, list):
            parts = [part for item in value if (part := self._extract_text_blob(item))]
            return "\n\n".join(parts) if parts else None
        return None

    def _tool_accepts_param(self, tool: Any, key: str) -> bool:
        schema = getattr(tool, "inputSchema", {}) or {}
        properties = schema.get("properties", {}) or {}
        return key in properties

    def _first_schema_key(self, tool: Any, options: Sequence[str]) -> str | None:
        schema = getattr(tool, "inputSchema", {}) or {}
        properties = schema.get("properties", {}) or {}
        for option in options:
            if option in properties:
                return option
        return None

    def _first_string(self, mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _maybe_parse_json(self, text: str) -> Any:
        stripped = text.strip()
        if not stripped:
            return stripped
        if stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return stripped
        return stripped

    def _build_cql_query(self, query: str) -> str:
        escaped_query = query.replace("\\", "\\\\").replace('"', '\\"')
        text_clause = f'text ~ "{escaped_query}"'
        if not self.confluence_space:
            return text_clause
        escaped_space = self.confluence_space.replace('"', '\\"')
        return f'space = "{escaped_space}" AND {text_clause}'

    def _document_prompt_view(self, document: DocumentResult) -> dict[str, Any]:
        return {
            "title": document.title,
            "url": document.url,
            "excerpt": document.excerpt,
            "content": document.content,
            "source": document.source,
        }

    def _document_debug_view(self, document: DocumentResult) -> dict[str, Any]:
        debug_document = asdict(document)
        if debug_document.get("content"):
            debug_document["content_preview"] = debug_document["content"][:500]
            debug_document.pop("content")
        return debug_document

    def _configuration_error_response(
        self, query: str, missing_variable: str
    ) -> Dict[str, Any]:
        return {
            "answer": (
                "Search is not configured correctly right now. Please verify the required "
                "OpenAI and MCP environment variables and try again."
            ),
            "citations": [],
            "raw_response": {
                "status": "configuration_error",
                "query": query,
                "missing_variable": missing_variable,
                "model": self.model,
                "mcp_server_url": self.mcp_server_url,
            },
        }

    def _error_response(
        self, query: str, status: str, exc: Exception
    ) -> Dict[str, Any]:
        return {
            "answer": (
                "Sorry, I ran into an issue while searching the documentation. Please try "
                "again in a moment."
            ),
            "citations": [],
            "raw_response": {
                "status": status,
                "query": query,
                "model": self.model,
                "error_type": type(exc).__name__,
            },
        }