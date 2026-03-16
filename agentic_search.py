"""Agentic Confluence search powered by OpenAI Agents SDK + Atlassian Rovo MCP."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
import traceback
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
ATLASSIAN_AUTH_CONFIG = (
    "MCP_AUTH_HEADER or ATLASSIAN_SERVICE_ACCOUNT_KEY or "
    "ATLASSIAN_API_EMAIL + ATLASSIAN_API_TOKEN"
)
DEBUG_ENV_VAR = "AGENTIC_SEARCH_DEBUG"
DEBUG_TRUE_VALUES = {"1", "true", "yes"}
TRACEBACK_CHAR_LIMIT = 2000
ERROR_PREVIEW_CHAR_LIMIT = 300
SECRET_KEY_FRAGMENTS = (
    "auth",
    "authorization",
    "token",
    "secret",
    "password",
    "credential",
    "api_key",
    "apikey",
    "email",
    "bearer",
    "basic",
)
SECRET_VALUE_PATTERNS = (
    (re.compile(r"Bearer\s+[^\s]+", re.IGNORECASE), "[REDACTED_AUTH]"),
    (re.compile(r"Basic\s+[^\s]+", re.IGNORECASE), "[REDACTED_AUTH]"),
    (
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(
            r"(?i)(api[_ -]?token|api[_ -]?key|access[_ -]?token|authorization|password|secret)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        "[REDACTED_SECRET]",
    ),
)


def build_mcp_auth_header(
    explicit_header: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build the Atlassian MCP Authorization header from supported config sources."""
    resolved_env = env or os.environ
    header = (explicit_header or "").strip() or resolved_env.get("MCP_AUTH_HEADER", "").strip()
    if header:
        return header

    service_account_key = resolved_env.get("ATLASSIAN_SERVICE_ACCOUNT_KEY", "").strip()
    if service_account_key:
        return f"Bearer {service_account_key}"

    api_email = resolved_env.get("ATLASSIAN_API_EMAIL", "").strip()
    api_token = resolved_env.get("ATLASSIAN_API_TOKEN", "").strip()
    if api_email and api_token:
        encoded_credentials = base64.b64encode(
            f"{api_email}:{api_token}".encode("utf-8")
        ).decode("ascii")
        return f"Basic {encoded_credentials}"

    return ""


def _env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in DEBUG_TRUE_VALUES


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
        mcp_cloud_id: str | None = None,
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
        self.mcp_auth_header = build_mcp_auth_header(mcp_auth_header)
        self.mcp_cloud_id = (mcp_cloud_id or os.getenv("MCP_CLOUD_ID", "")).strip()
        self.max_results = max(1, max_results)
        self.debug_enabled = _env_flag(os.getenv(DEBUG_ENV_VAR))
        self._debug_state: dict[str, Any] | None = None
        self._debug_last_step: str | None = None

    @property
    def uses_local_llm(self) -> bool:
        return bool(self.llm_base_url)

    def search(self, query: str) -> Dict[str, Any]:
        """Search documentation and return a structured answer with citations."""
        self._reset_debug_state()
        clean_query = query.strip()
        if not clean_query:
            raw_response = {
                "status": "invalid_query",
                "model": self.model,
                "mcp_server_url": self.mcp_server_url,
            }
            return {
                "answer": "Please provide a question or keywords to search the documentation.",
                "citations": [],
                "raw_response": self._with_debug(raw_response),
            }

        if not self.uses_local_llm and not self.openai_api_key:
            return self._configuration_error_response(clean_query, "OPENAI_API_KEY")

        if not self.mcp_auth_header:
            return self._configuration_error_response(clean_query, ATLASSIAN_AUTH_CONFIG)

        try:
            self._start_debug_session()
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

        self._set_last_step("mcp_connect")
        async with self._build_mcp_server() as server:
            self._set_last_step("list_tools")
            tools = await server.list_tools()
            tool_names = [tool.name for tool in tools]
            self._record_debug("list_tools", {"tool_names": tool_names})
            tool_map = {tool.name: tool for tool in tools}
            strategy = self._select_strategy(tool_map)
            if strategy is None:
                raise RuntimeError("No supported Atlassian Rovo search tools were available.")
            self._record_debug(
                "strategy",
                {
                    "strategy": strategy.name,
                    "search_tool": strategy.search_tool,
                    "fetch_tool": strategy.fetch_tool,
                },
            )

            cloud_id = await self._resolve_cloud_id(server, tool_map, strategy)
            self._record_debug("cloud_id", {"cloud_id": cloud_id})
            search_payload = await self._call_tool(
                server,
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
                raw_response = {
                    "status": "no_results",
                    "strategy": strategy.name,
                    "model": self.model,
                    "query": query,
                    "selected_tools": [strategy.search_tool, strategy.fetch_tool],
                }
                return {
                    "answer": (
                        f"I couldn't find relevant documentation for '{query}'. Try adding more "
                        "specific keywords, a product name, or a team-specific term."
                    ),
                    "citations": [],
                    "raw_response": self._with_debug(raw_response),
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

            self._set_last_step("synthesis")
            synthesized = await self._synthesize_answer(query, documents)
            citations = self._normalize_citations(synthesized.citations, documents)

            if not citations:
                citations = self._fallback_citations(documents)

            raw_response = {
                "status": "ok",
                "strategy": strategy.name,
                "model": self.model,
                "query": query,
                "selected_tools": [strategy.search_tool, strategy.fetch_tool],
                "fetch_errors": fetch_errors,
            }
            return {
                "answer": synthesized.answer.strip(),
                "citations": [citation.model_dump() for citation in citations],
                "raw_response": self._with_debug(raw_response),
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

        if self.mcp_cloud_id:
            return self.mcp_cloud_id

        if ACCESSIBLE_RESOURCES_TOOL not in tool_map:
            return None

        resources_payload = await self._call_tool(server, ACCESSIBLE_RESOURCES_TOOL, {})
        for resource in self._iter_nested_dicts(self._extract_payload(resources_payload)):
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
                payload = await self._call_tool(
                    server, tool.name, self._build_fetch_arguments(tool, document, cloud_id)
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

    def _iter_nested_dicts(self, value: Any):
        if isinstance(value, dict):
            yield value
            for nested in value.values():
                yield from self._iter_nested_dicts(nested)
        elif isinstance(value, list):
            for item in value:
                yield from self._iter_nested_dicts(item)

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
        debug_document.pop("content", None)
        return debug_document

    async def _call_tool(
        self,
        server: Any,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        self._set_last_step(f"call_tool:{tool_name}")
        call_debug = {
            "tool_name": tool_name,
            "argument_keys": [self._sanitize_key_name(key) for key in arguments.keys()],
        }
        try:
            payload = await server.call_tool(tool_name, dict(arguments))
        except Exception as exc:
            call_debug["error_type"] = type(exc).__name__
            call_debug["error_message"] = self._preview_text(str(exc))
            self._record_call_debug(call_debug)
            raise

        call_debug["response_summary"] = self._summarize_payload(payload)
        self._record_call_debug(call_debug)
        return payload

    def _reset_debug_state(self) -> None:
        self._debug_state = None
        self._debug_last_step = None

    def _start_debug_session(self) -> None:
        if not self.debug_enabled:
            return
        self._debug_state = {
            "mcp_server_url": self.mcp_server_url,
            "mcp_server_name": self.mcp_server_name,
            "calls": [],
        }
        self._record_debug(
            "mcp_server",
            {
                "mcp_server_url": self.mcp_server_url,
                "mcp_server_name": self.mcp_server_name,
            },
        )

    def _set_last_step(self, step: str) -> None:
        self._debug_last_step = step
        if self._debug_state is not None:
            self._debug_state["last_step"] = step

    def _record_debug(self, label: str, details: Mapping[str, Any]) -> None:
        if not self.debug_enabled:
            return
        sanitized = self._sanitize_debug_object(dict(details))
        if self._debug_state is not None:
            self._debug_state[label] = sanitized
        print(
            f"[AgenticSearch debug] {label}: {json.dumps(sanitized, ensure_ascii=False)}",
            file=sys.stderr,
        )

    def _record_call_debug(self, call_debug: Mapping[str, Any]) -> None:
        if not self.debug_enabled:
            return
        sanitized = self._sanitize_debug_object(dict(call_debug))
        if self._debug_state is not None:
            self._debug_state.setdefault("calls", []).append(sanitized)
        print(
            f"[AgenticSearch debug] call_tool: {json.dumps(sanitized, ensure_ascii=False)}",
            file=sys.stderr,
        )

    def _record_error_debug(self, error_summary: Mapping[str, Any]) -> None:
        if not self.debug_enabled:
            return
        sanitized = self._sanitize_debug_object(dict(error_summary))
        print(
            f"[AgenticSearch debug] error: {json.dumps(sanitized, ensure_ascii=False)}",
            file=sys.stderr,
        )

    def _summarize_payload(self, payload: Any) -> dict[str, Any]:
        extracted = self._extract_payload(payload)
        summary: dict[str, Any] = {"payload_type": type(extracted).__name__}
        is_error = bool(getattr(payload, "isError", False))
        if isinstance(extracted, dict) and extracted.get("isError") is True:
            is_error = True
        if isinstance(extracted, dict):
            keys: list[str] = []
            counts: dict[str, int] = {}
            for key, value in extracted.items():
                safe_key = self._sanitize_key_name(key)
                keys.append(safe_key)
                count = self._summary_count(value)
                if count is not None:
                    counts[safe_key] = count
            summary["top_level_keys"] = keys
            if counts:
                summary["counts"] = counts
        elif isinstance(extracted, list):
            summary["top_level_keys"] = []
            summary["counts"] = {"items": len(extracted)}
        else:
            summary["top_level_keys"] = []

        if is_error:
            summary["is_error"] = True
            error_text = self._extract_payload_error_text(payload, extracted)
            if error_text:
                summary["error_text"] = error_text
        return summary

    def _summary_count(self, value: Any) -> int | None:
        if isinstance(value, (dict, list, tuple, set)):
            return len(value)
        return None

    def _sanitize_debug_object(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, nested in value.items():
                safe_key = self._sanitize_key_name(key)
                if safe_key == "[REDACTED]":
                    sanitized[safe_key] = "[REDACTED]"
                else:
                    sanitized[safe_key] = self._sanitize_debug_object(nested)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_debug_object(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_debug_object(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _sanitize_key_name(self, key: str) -> str:
        normalized = key.lower().replace("-", "_")
        if any(fragment in normalized for fragment in SECRET_KEY_FRAGMENTS):
            return "[REDACTED]"
        return key

    def _redact_text(self, text: str) -> str:
        redacted = text
        for pattern, replacement in SECRET_VALUE_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def _truncate_text(self, text: str, limit: int = ERROR_PREVIEW_CHAR_LIMIT) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    def _preview_text(self, text: str) -> str:
        return self._truncate_text(self._redact_text(text))

    def _preview_debug_value(self, value: Any) -> str | None:
        if value is None:
            return None
        sanitized = self._sanitize_debug_object(value)
        if isinstance(sanitized, str):
            return self._truncate_text(sanitized)
        serialized = json.dumps(sanitized, ensure_ascii=False)
        return self._truncate_text(serialized)

    def _extract_payload_error_text(self, payload: Any, extracted: Any) -> str | None:
        text = self._error_text_from_value(extracted)
        if text:
            return text

        for item in getattr(payload, "content", []) or []:
            if getattr(item, "type", None) == "text":
                text = self._preview_text(getattr(item, "text", ""))
                if text:
                    return text
        return None

    def _error_text_from_value(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return self._preview_text(stripped) if stripped else None
        if isinstance(value, dict):
            for key in ("message", "detail", "details", "error", "reason"):
                nested = value.get(key)
                text = self._error_text_from_value(nested)
                if text:
                    return text
        if isinstance(value, list):
            for item in value:
                text = self._error_text_from_value(item)
                if text:
                    return text
        return None

    def _extract_exception_debug_details(self, exc: Exception) -> dict[str, Any]:
        details: dict[str, Any] = {}
        response = getattr(exc, "response", None)
        response_body = None
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code is not None:
                details["http_status"] = status_code
            response_body = self._extract_response_body(response)
            response_preview = self._preview_debug_value(response_body)
            if response_preview:
                details["http_response_preview"] = response_preview

        mcp_error_preview = self._build_mcp_error_preview(getattr(exc, "error", None))
        if not mcp_error_preview:
            mcp_error_preview = self._build_mcp_error_preview(exc)
        if not mcp_error_preview:
            mcp_error_preview = self._build_mcp_error_preview(response_body)
        if mcp_error_preview:
            details["mcp_error_preview"] = mcp_error_preview
        return details

    def _extract_response_body(self, response: Any) -> Any:
        try:
            return response.json()
        except Exception:
            pass

        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None
        if isinstance(text, str) and text.strip():
            return text.strip()

        try:
            content = getattr(response, "content", None)
        except Exception:
            content = None
        if isinstance(content, bytes):
            decoded = content.decode("utf-8", errors="replace").strip()
            return decoded or None
        if isinstance(content, str) and content.strip():
            return content.strip()
        return None

    def _build_mcp_error_preview(self, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None

        if isinstance(value, dict):
            nested_error = value.get("error")
            if nested_error is not None:
                nested_preview = self._build_mcp_error_preview(nested_error)
                if nested_preview:
                    return nested_preview
            code = value.get("code")
            message = value.get("message")
            data = value.get("data")
        else:
            nested_error = getattr(value, "error", None)
            if nested_error is not None and nested_error is not value:
                nested_preview = self._build_mcp_error_preview(nested_error)
                if nested_preview:
                    return nested_preview
            code = getattr(value, "code", None)
            message = getattr(value, "message", None)
            data = getattr(value, "data", None)

        preview: dict[str, Any] = {}
        if code is not None:
            preview["code"] = code
        if message not in (None, ""):
            preview["message"] = self._preview_debug_value(message)
        if data not in (None, ""):
            preview["data"] = self._preview_debug_value(data)
        return preview or None

    def _error_summary_from_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: snapshot[key]
            for key in (
                "last_step",
                "error_type",
                "error_message",
                "http_status",
                "http_response_preview",
                "mcp_error_preview",
            )
            if key in snapshot
        }

    def _debug_snapshot(self, exc: Exception | None = None) -> dict[str, Any]:
        snapshot = self._sanitize_debug_object(self._debug_state or {})
        if self._debug_last_step:
            snapshot["last_step"] = self._debug_last_step
        if exc is not None:
            snapshot["error_type"] = type(exc).__name__
            snapshot["error_message"] = self._preview_text(str(exc))
            snapshot.update(self._extract_exception_debug_details(exc))
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            snapshot["traceback"] = self._redact_text(formatted[:TRACEBACK_CHAR_LIMIT])
        return snapshot

    def _with_debug(self, raw_response: dict[str, Any], exc: Exception | None = None) -> dict[str, Any]:
        if self.debug_enabled:
            debug_snapshot = self._debug_snapshot(exc)
            raw_response["debug"] = debug_snapshot
            if exc is not None:
                self._record_error_debug(self._error_summary_from_snapshot(debug_snapshot))
        return raw_response

    def _configuration_error_response(
        self, query: str, missing_variable: str
    ) -> Dict[str, Any]:
        raw_response = {
            "status": "configuration_error",
            "query": query,
            "missing_variable": missing_variable,
            "model": self.model,
            "mcp_server_url": self.mcp_server_url,
        }
        return {
            "answer": (
                "Search is not configured correctly right now. Please verify the required "
                "OpenAI and MCP environment variables and try again."
            ),
            "citations": [],
            "raw_response": self._with_debug(raw_response),
        }

    def _error_response(
        self, query: str, status: str, exc: Exception
    ) -> Dict[str, Any]:
        raw_response = {
            "status": status,
            "query": query,
            "model": self.model,
            "error_type": type(exc).__name__,
        }
        return {
            "answer": (
                "Sorry, I ran into an issue while searching the documentation. Please try "
                "again in a moment."
            ),
            "citations": [],
            "raw_response": self._with_debug(raw_response, exc),
        }