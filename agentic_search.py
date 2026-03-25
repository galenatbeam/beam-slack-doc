"""Agentic Confluence search powered by OpenAI Agents SDK + Atlassian Rovo MCP."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
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
from agents.tool import FunctionTool
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_MCP_SERVER_URL = "https://mcp.atlassian.com/v1/mcp"
DEFAULT_MCP_SERVER_NAME = "atlassian-rovo"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_OLLAMA_CHAT_MODEL = "qwen3:4b-instruct"
LOCAL_LLM_API_KEY_PLACEHOLDER = "ollama"
MAX_FETCH_RESULTS = 3
SHARED_SEARCH_TOOL = "search"
SHARED_FETCH_TOOL = "fetch"
CONFLUENCE_SEARCH_TOOL = "searchConfluenceUsingCql"
CONFLUENCE_FETCH_TOOL = "getConfluencePage"
TITLE_KEYS = ("title", "name", "pageTitle")
URL_KEYS = ("url", "webLink", "link", "pageUrl")
EXCERPT_KEYS = ("excerpt", "snippet", "summary", "text")
ID_KEYS = ("id", "pageId", "contentId", "entityId")
ARI_KEYS = ("ari", "resourceAri", "resourceARI", "resourceId")
CONTENT_KEYS = ("body", "content", "markdown", "value")
CONFLUENCE_MCP_API_KEY_ENV = "CONFLUENCE_MCP_API_KEY"
ATLASSIAN_EMAIL_ENV = "ATLASSIAN_EMAIL"
ATLASSIAN_CLOUD_ID_ENV = "ATLASSIAN_CLOUD_ID"
DEBUG_ENV_VAR = "AGENTIC_SEARCH_DEBUG"
DEBUG_TRUE_VALUES = {"1", "true", "yes"}
ERROR_PREVIEW_CHAR_LIMIT = 300
TOOL_CONTENT_CHAR_LIMIT = 4000
PAYLOAD_PREVIEW_CHAR_LIMIT = 1200
CONFLUENCE_AUTH_ISSUE_MESSAGE = (
    "Confluence authentication issue. Please verify the Atlassian MCP "
    "credentials and try again."
)
SENSITIVE_DEBUG_KEYS = {
    "atlassian_email",
    "api_key",
    "api_token",
    "authorization",
    "email",
    "headers",
    "password",
    "secret",
    "token",
}
AUTH_TEXT_PATTERNS = (
    (re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[^\s,;]+"), "[REDACTED_AUTH]"),
    (re.compile(r"(?i)authorization\s*[:=]\s*basic\s+[^\s,;]+"), "[REDACTED_AUTH]"),
    (re.compile(r"(?i)bearer\s+[^\s,;]+"), "[REDACTED_AUTH]"),
    (re.compile(r"(?i)basic\s+[^\s,;]+"), "[REDACTED_AUTH]"),
)
SUPPORTED_TOOL_SCHEMA_KEYS = frozenset(
    {"type", "properties", "required", "description", "items", "enum", "anyOf", "oneOf"}
)


def build_mcp_auth_header(
    api_key: str | None = None,
    email: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Build the Atlassian MCP Authorization header from email + API token."""
    resolved_env = env or os.environ
    resolved_email = (email or resolved_env.get(ATLASSIAN_EMAIL_ENV, "")).strip()
    resolved_api_key = (api_key or resolved_env.get(CONFLUENCE_MCP_API_KEY_ENV, "")).strip()
    if not resolved_email or not resolved_api_key:
        return ""
    credentials = f"{resolved_email}:{resolved_api_key}".encode("utf-8")
    encoded_credentials = base64.b64encode(credentials).decode("ascii")
    return f"Basic {encoded_credentials}"


def _env_flag(value: str | None) -> bool:
    return (value or "").strip().lower() in DEBUG_TRUE_VALUES


def _normalize_debug_key(key: str) -> str:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key.strip())
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.lower().strip("_")


def _is_sensitive_debug_key(key: str) -> bool:
    return _normalize_debug_key(key) in SENSITIVE_DEBUG_KEYS


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
        confluence_mcp_api_key: str | None = None,
        atlassian_email: str | None = None,
        atlassian_cloud_id: str | None = None,
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
        self.mcp_server_name = mcp_server_name or DEFAULT_MCP_SERVER_NAME
        self.model = self._resolve_model(model)
        self.mcp_server_url = mcp_server_url or os.getenv(
            "MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL
        )
        self.atlassian_email = (
            atlassian_email or os.getenv(ATLASSIAN_EMAIL_ENV, "")
        ).strip()
        self.confluence_mcp_api_key = (
            confluence_mcp_api_key or os.getenv(CONFLUENCE_MCP_API_KEY_ENV, "")
        ).strip()
        self.mcp_auth_header = build_mcp_auth_header(
            api_key=self.confluence_mcp_api_key,
            email=self.atlassian_email,
        )
        self.atlassian_cloud_id = (
            atlassian_cloud_id or os.getenv(ATLASSIAN_CLOUD_ID_ENV, "")
        ).strip()
        self.max_results = max(1, max_results)
        self.debug_enabled = _env_flag(os.getenv(DEBUG_ENV_VAR))
        self._debug_state: dict[str, Any] | None = None
        self._debug_last_step: str | None = None

    @property
    def uses_local_llm(self) -> bool:
        return bool(self.llm_base_url)

    def missing_configuration(self) -> list[str]:
        missing: list[str] = []
        if not self.uses_local_llm and not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if not self.atlassian_email:
            missing.append(ATLASSIAN_EMAIL_ENV)
        if not self.confluence_mcp_api_key:
            missing.append(CONFLUENCE_MCP_API_KEY_ENV)
        if not self.atlassian_cloud_id:
            missing.append(ATLASSIAN_CLOUD_ID_ENV)
        return missing

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

        missing = self.missing_configuration()
        if missing:
            return self._configuration_error_response(clean_query, missing)

        try:
            self._start_debug_session()
            return asyncio.run(self._search_async(clean_query))
        except Exception as exc:  # pragma: no cover - exercised via public response path
            return self._error_response(clean_query, "search_error", exc)

    def build_agent_instructions(self, discovered_tool_names: Sequence[str] | None = None) -> str:
        """Instruction template for the answer-synthesis agent."""
        scope = ""
        if self.confluence_space:
            scope = f" Favor documents from Confluence space '{self.confluence_space}'."

        tool_guidance = ""
        if discovered_tool_names:
            tool_guidance = " Available MCP tools: " + ", ".join(discovered_tool_names) + "."

        return (
            "You are a documentation search assistant. Use the discovered MCP tools to answer "
            "the user's question. Start with a search-style tool to find relevant documents. "
            "If a fetch/get tool could improve the answer, call it for the most relevant result. "
            "Rank the evidence, answer only from tool-returned documents, keep the answer "
            "concise, and clearly say when the docs do not fully answer the query. Return "
            "citations using title + URL when available, otherwise title + ARI/ID."
            + tool_guidance
            + scope
        )

    async def _search_async(self, query: str) -> Dict[str, Any]:
        return await self._search_agentic_async(query)

    async def _search_agentic_async(self, query: str) -> Dict[str, Any]:
        self._configure_agents_sdk()

        self._set_last_step("mcp_connect")
        async with self._build_mcp_server() as server:
            try:
                discovered_tools = await self._list_tools(server)
            except Exception as exc:
                return self._confluence_auth_issue_response(query, exc)

            if not discovered_tools:
                raise RuntimeError("No MCP tools were discovered.")

            collected_documents: list[DocumentResult] = []
            agent_tools = self._build_discovered_function_tools(
                server,
                discovered_tools,
                collected_documents,
            )

            self._set_last_step("agent_run")
            synthesized = await self._run_tool_calling_agent(
                query,
                agent_tools,
                [tool.name for tool in discovered_tools if getattr(tool, "name", None)],
                collected_documents,
            )
            citations = self._normalize_citations(synthesized.citations, collected_documents)

            if not citations:
                citations = self._fallback_citations(collected_documents)

            raw_response = {
                "status": "ok",
                "strategy": "agentic_discovery",
                "model": self.model,
                "query": query,
                "discovered_tools": [tool.name for tool in discovered_tools if getattr(tool, "name", None)],
                "selected_tools": sorted({document.source for document in collected_documents}),
            }
            return {
                "answer": synthesized.answer.strip(),
                "citations": [citation.model_dump() for citation in citations],
                "raw_response": self._with_debug(raw_response),
            }

    async def _list_tools(self, server: Any) -> list[Any]:
        self._set_last_step("list_tools")
        tools_result = await server.list_tools()
        tools = getattr(tools_result, "tools", tools_result)
        discovered_tools = list(tools or [])
        self._record_debug(
            "tool_discovery",
            {
                "called": True,
                "method": "list_tools",
                "tool_names": [tool.name for tool in discovered_tools if getattr(tool, "name", None)],
            },
        )
        return discovered_tools

    def _build_discovered_function_tools(
        self,
        server: Any,
        discovered_tools: Sequence[Any],
        collected_documents: list[DocumentResult],
    ) -> list[FunctionTool]:
        return [
            self._build_discovered_function_tool(server, tool, collected_documents)
            for tool in discovered_tools
            if getattr(tool, "name", None)
        ]

    def _build_discovered_function_tool(
        self,
        server: Any,
        tool: Any,
        collected_documents: list[DocumentResult],
    ) -> FunctionTool:
        tool_name = tool.name
        description = getattr(tool, "description", None) or f"MCP tool '{tool_name}'."
        params_json_schema = self._sanitize_tool_input_schema(
            tool_name,
            getattr(tool, "inputSchema", None),
        )

        async def on_invoke_tool(_ctx: Any, input_json: str) -> str:
            arguments = self._prepare_discovered_tool_arguments(
                tool,
                self._parse_tool_input(input_json),
            )
            payload = await self._call_tool(server, tool_name, arguments)
            documents = self._normalize_documents(payload, source=tool_name)
            collected_documents.extend(documents)
            result: dict[str, Any] = {
                "tool_name": tool_name,
                "documents": [self._document_tool_view(document) for document in documents],
            }
            extracted = self._extract_payload(payload)
            text_blob = self._extract_text_blob(extracted)
            if text_blob and not documents:
                result["content"] = self._truncate_text(text_blob, TOOL_CONTENT_CHAR_LIMIT)
            return json.dumps(result, ensure_ascii=False)

        return FunctionTool(
            name=tool_name,
            description=description,
            params_json_schema=params_json_schema,
            on_invoke_tool=on_invoke_tool,
        )

    def _sanitize_tool_input_schema(self, tool_name: str, input_schema: Any) -> dict[str, Any]:
        if not isinstance(input_schema, Mapping):
            return {"type": "object", "properties": {}}

        stripped_keys: list[str] = []
        sanitized = self._sanitize_tool_schema_node(input_schema, "$", stripped_keys)

        if sanitized.get("type") != "object":
            stripped_keys.append("$.type")
            sanitized = {"type": "object", "properties": {}}
        else:
            sanitized["type"] = "object"
            if not isinstance(sanitized.get("properties"), dict):
                sanitized["properties"] = {}

        required = sanitized.get("required")
        if isinstance(required, list):
            filtered_required = [
                item for item in required if isinstance(item, str) and item in sanitized["properties"]
            ]
            if filtered_required:
                sanitized["required"] = filtered_required
            else:
                sanitized.pop("required", None)
        else:
            sanitized.pop("required", None)

        if stripped_keys or sanitized != input_schema:
            self._record_sanitized_tool_schema(tool_name, input_schema, sanitized, stripped_keys)

        return sanitized

    def _sanitize_tool_schema_node(
        self,
        schema: Mapping[str, Any],
        path: str,
        stripped_keys: list[str],
    ) -> dict[str, Any]:
        sanitized: dict[str, Any] = {}

        for key, value in schema.items():
            key_path = self._schema_path(path, str(key))
            if key not in SUPPORTED_TOOL_SCHEMA_KEYS:
                stripped_keys.append(key_path)
                continue

            if key == "type":
                normalized_type = self._normalize_schema_type(value)
                if normalized_type is None:
                    stripped_keys.append(key_path)
                    continue
                sanitized[key] = normalized_type
                continue

            if key == "properties":
                if not isinstance(value, Mapping):
                    stripped_keys.append(key_path)
                    continue
                sanitized[key] = {
                    str(property_name): self._sanitize_tool_schema_child(
                        property_schema,
                        self._schema_path(key_path, str(property_name)),
                        stripped_keys,
                    )
                    for property_name, property_schema in value.items()
                }
                continue

            if key == "required":
                if not isinstance(value, list):
                    stripped_keys.append(key_path)
                    continue
                sanitized[key] = [item for item in value if isinstance(item, str)]
                continue

            if key == "description":
                if isinstance(value, str) and value.strip():
                    sanitized[key] = value
                else:
                    stripped_keys.append(key_path)
                continue

            if key == "enum":
                if not isinstance(value, list):
                    stripped_keys.append(key_path)
                    continue
                sanitized[key] = [
                    item
                    for item in value
                    if item is None or isinstance(item, (str, int, float, bool))
                ]
                continue

            if key == "items":
                if isinstance(value, Mapping):
                    sanitized[key] = self._sanitize_tool_schema_node(value, key_path, stripped_keys)
                elif isinstance(value, list):
                    sanitized[key] = [
                        self._sanitize_tool_schema_child(item, f"{key_path}[{index}]", stripped_keys)
                        for index, item in enumerate(value)
                    ]
                else:
                    stripped_keys.append(key_path)
                continue

            if key in {"anyOf", "oneOf"}:
                if not isinstance(value, list):
                    stripped_keys.append(key_path)
                    continue
                options = [
                    self._sanitize_tool_schema_child(item, f"{key_path}[{index}]", stripped_keys)
                    for index, item in enumerate(value)
                ]
                if options:
                    sanitized[key] = options
                else:
                    stripped_keys.append(key_path)

        if sanitized.get("type") == "object":
            sanitized.setdefault("properties", {})
            required = sanitized.get("required")
            if isinstance(required, list):
                filtered_required = [
                    item for item in required if item in sanitized["properties"]
                ]
                if filtered_required:
                    sanitized["required"] = filtered_required
                else:
                    sanitized.pop("required", None)
        elif "required" in sanitized:
            sanitized.pop("required", None)

        return sanitized

    def _sanitize_tool_schema_child(self, schema: Any, path: str, stripped_keys: list[str]) -> dict[str, Any]:
        if not isinstance(schema, Mapping):
            stripped_keys.append(path)
            return {}
        return self._sanitize_tool_schema_node(schema, path, stripped_keys)

    def _normalize_schema_type(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            non_null_types = [item for item in value if isinstance(item, str) and item != "null"]
            if len(non_null_types) == 1:
                return non_null_types[0]
        return None

    def _schema_path(self, path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    def _record_sanitized_tool_schema(
        self,
        tool_name: str,
        original_schema: Mapping[str, Any],
        sanitized_schema: Mapping[str, Any],
        stripped_keys: Sequence[str],
    ) -> None:
        if not self.debug_enabled:
            return

        unique_stripped_keys = list(dict.fromkeys(stripped_keys))
        entry: dict[str, Any] = {
            "tool_name": tool_name,
            "stripped_keys": unique_stripped_keys,
        }

        original_preview = self._preview_debug_value_deterministic(original_schema)
        if original_preview:
            entry["original_schema_preview"] = original_preview

        sanitized_preview = self._preview_debug_value_deterministic(sanitized_schema)
        if sanitized_preview:
            entry["sanitized_schema_preview"] = sanitized_preview

        sanitized_entry = self._sanitize_debug_object(entry)
        if self._debug_state is not None:
            self._debug_state.setdefault("sanitized_tool_schemas", []).append(sanitized_entry)
        print(
            f"[AgenticSearch debug] sanitized_tool_schema: {json.dumps(sanitized_entry, ensure_ascii=False)}",
            file=sys.stderr,
        )

    def _parse_tool_input(self, input_json: str) -> dict[str, Any]:
        stripped = input_json.strip()
        if not stripped or stripped == "null":
            return {}
        parsed = json.loads(stripped)
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise RuntimeError("Tool arguments must be a JSON object.")
        return parsed

    def _prepare_discovered_tool_arguments(
        self,
        tool: Any,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        prepared = dict(arguments)
        if (
            self.atlassian_cloud_id
            and self._tool_accepts_param(tool, "cloudId")
            and "cloudId" not in prepared
        ):
            prepared["cloudId"] = self.atlassian_cloud_id
        return prepared

    async def _run_tool_calling_agent(
        self,
        query: str,
        tools: Sequence[FunctionTool],
        discovered_tool_names: Sequence[str],
        collected_documents: Sequence[DocumentResult],
    ) -> SynthesizedAnswer:
        try:
            return await self._run_tool_calling_agent_structured(query, tools, discovered_tool_names)
        except Exception:
            if not self.uses_local_llm:
                raise
            return await self._run_tool_calling_agent_plain_text(
                query,
                tools,
                discovered_tool_names,
                collected_documents,
            )

    async def _run_tool_calling_agent_structured(
        self,
        query: str,
        tools: Sequence[FunctionTool],
        discovered_tool_names: Sequence[str],
    ) -> SynthesizedAnswer:
        agent = Agent(
            name="AgenticSearch",
            instructions=self.build_agent_instructions(discovered_tool_names),
            model=self.model,
            tools=list(tools),
            output_type=SynthesizedAnswer,
        )
        result = await Runner.run(agent, query)
        return result.final_output_as(SynthesizedAnswer, raise_if_incorrect_type=True)

    async def _run_tool_calling_agent_plain_text(
        self,
        query: str,
        tools: Sequence[FunctionTool],
        discovered_tool_names: Sequence[str],
        collected_documents: Sequence[DocumentResult],
    ) -> SynthesizedAnswer:
        agent = Agent(
            name="AgenticSearch",
            instructions=(
                self.build_agent_instructions(discovered_tool_names)
                + " Return a concise plain-text answer only. Do not emit JSON or XML."
            ),
            model=self.model,
            tools=list(tools),
        )
        result = await Runner.run(agent, query)
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
            citations=self._fallback_citations(collected_documents),
        )

    def _build_mcp_server(self) -> MCPServerStreamableHttp:
        return MCPServerStreamableHttp(
            name=self.mcp_server_name,
            params={
                "url": self.mcp_server_url,
                "headers": {"Authorization": self.mcp_auth_header},
                "timeout": 10,
            },
            cache_tools_list=False,
            max_retry_attempts=2,
        )

    def _build_natural_language_query(self, query: str) -> str:
        if not self.confluence_space:
            return query
        return f"{query}\n\nPrefer results from Confluence space '{self.confluence_space}' when relevant."

    def _build_shared_search_argument_candidates(self, query: str) -> list[dict[str, Any]]:
        base_arguments = {"query": self._build_natural_language_query(query)}
        candidates: list[dict[str, Any]] = [dict(base_arguments)]

        with_limit = {**base_arguments, "max_results": self.max_results}
        candidates.insert(0, with_limit)

        if self.atlassian_cloud_id:
            candidates.insert(0, {**with_limit, "cloudId": self.atlassian_cloud_id})
            candidates.insert(2, {**base_arguments, "cloudId": self.atlassian_cloud_id})

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        for candidate in candidates:
            fingerprint = tuple(candidate.items())
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            deduped.append(candidate)
        return deduped

    async def _call_shared_search_tool(
        self,
        server: Any,
        argument_candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[str, Any]:
        for index, arguments in enumerate(argument_candidates):
            argument_keys = [key for key in arguments.keys() if not _is_sensitive_debug_key(key)]
            self._record_debug(
                "search_tool",
                {"tool_name": SHARED_SEARCH_TOOL, "argument_keys": argument_keys},
            )
            try:
                payload = await self._call_tool(server, SHARED_SEARCH_TOOL, arguments)
                return SHARED_SEARCH_TOOL, payload
            except Exception as exc:
                if index < len(argument_candidates) - 1 and self._is_tool_argument_error(exc):
                    continue
                raise

        raise RuntimeError("No supported shared search argument shape was accepted.")

    def _is_tool_argument_error(self, exc: Exception) -> bool:
        candidates: list[Any] = [getattr(exc, "error", None), exc]
        response = getattr(exc, "response", None)
        if response is not None:
            candidates.append(self._extract_response_body(response))
        return any(self._contains_invalid_argument_signal(candidate) for candidate in candidates)

    def _contains_invalid_argument_signal(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            if value.get("code") == -32602:
                return True
            return any(
                self._contains_invalid_argument_signal(value.get(key))
                for key in ("error", "message", "detail", "data")
            )
        if isinstance(value, (list, tuple, set)):
            return any(self._contains_invalid_argument_signal(item) for item in value)
        if isinstance(value, str):
            lowered = value.lower()
            return any(
                marker in lowered
                for marker in (
                    "-32602",
                    "invalid params",
                    "invalid arguments",
                    "validation",
                    "unexpected property",
                    "additional properties",
                    "unknown argument",
                    "unknown parameter",
                    "unexpected key",
                )
            )

        if getattr(value, "code", None) == -32602:
            return True

        nested_error = getattr(value, "error", None)
        if nested_error is not None and nested_error is not value:
            if self._contains_invalid_argument_signal(nested_error):
                return True

        return any(
            self._contains_invalid_argument_signal(getattr(value, attr, None))
            for attr in ("message", "detail", "data")
        )

    def _is_tool_not_found_error(self, exc: Exception) -> bool:
        candidates: list[Any] = [getattr(exc, "error", None), exc]
        response = getattr(exc, "response", None)
        if response is not None:
            candidates.append(self._extract_response_body(response))
        return any(self._contains_tool_not_found_signal(candidate) for candidate in candidates)

    def _contains_tool_not_found_signal(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            if value.get("code") == -32601:
                return True
            return any(
                self._contains_tool_not_found_signal(value.get(key))
                for key in ("error", "message", "detail", "data")
            )
        if isinstance(value, (list, tuple, set)):
            return any(self._contains_tool_not_found_signal(item) for item in value)
        if isinstance(value, str):
            lowered = value.lower()
            return any(
                marker in lowered
                for marker in ("-32601", "tool not found", "unknown tool", "method not found")
            )

        if getattr(value, "code", None) == -32601:
            return True

        nested_error = getattr(value, "error", None)
        if nested_error is not None and nested_error is not value:
            if self._contains_tool_not_found_signal(nested_error):
                return True

        return any(
            self._contains_tool_not_found_signal(getattr(value, attr, None))
            for attr in ("message", "detail", "data")
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

        return self.atlassian_cloud_id or None

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
            identifier = self._first_string(candidate, ID_KEYS)
            ari = self._first_string(candidate, ARI_KEYS)
            title = (
                self._first_string(candidate, TITLE_KEYS)
                or identifier
                or ari
                or self._first_string(candidate, EXCERPT_KEYS)
            )
            url = self._first_string(candidate, URL_KEYS)
            excerpt = self._first_string(candidate, EXCERPT_KEYS)
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
            identifier = citation.identifier or (document.identifier if document else None)
            ari = citation.ari or (document.ari if document else None)
            reference = url or ari or identifier
            if not citation.title or not reference:
                continue
            key = (citation.title, reference)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                Citation(
                    title=citation.title,
                    url=url,
                    excerpt=citation.excerpt or (document.excerpt if document else None),
                    identifier=identifier,
                    ari=ari,
                )
            )
        return normalized

    def _fallback_citations(self, documents: Sequence[DocumentResult]) -> list[Citation]:
        citations: list[Citation] = []
        for document in documents:
            if document.url or document.ari or document.identifier:
                citations.append(
                    Citation(
                        title=document.title,
                        url=document.url,
                        excerpt=document.excerpt,
                        identifier=document.identifier,
                        ari=document.ari,
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

    def _document_tool_view(self, document: DocumentResult) -> dict[str, Any]:
        tool_view = self._document_prompt_view(document)
        tool_view["identifier"] = document.identifier
        tool_view["ari"] = document.ari
        return tool_view

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
            "argument_keys": [key for key in arguments.keys() if not _is_sensitive_debug_key(key)],
        }
        try:
            payload = await server.call_tool(tool_name, dict(arguments))
        except Exception as exc:
            call_debug["error_type"] = type(exc).__name__
            call_debug["error_message"] = self._preview_text(str(exc))
            self._record_call_debug(call_debug)
            raise

        call_debug["response_summary"] = self._summarize_payload(payload)
        if self.debug_enabled:
            payload_preview = self._build_payload_preview(payload)
            if payload_preview:
                call_debug["payload_preview"] = payload_preview
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
                if _is_sensitive_debug_key(key):
                    continue
                keys.append(key)
                count = self._summary_count(value)
                if count is not None:
                    counts[key] = count
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
                if _is_sensitive_debug_key(key):
                    continue
                sanitized[key] = self._sanitize_debug_object(nested)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_debug_object(item) for item in value]
        if isinstance(value, tuple):
            return [self._sanitize_debug_object(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value

    def _secret_debug_fragments(self) -> tuple[str, ...]:
        fragments = {
            fragment.strip()
            for fragment in (
                self.atlassian_email,
                self.confluence_mcp_api_key,
                self.mcp_auth_header,
                self.mcp_auth_header.removeprefix("Basic ")
                if self.mcp_auth_header.startswith("Basic ")
                else "",
                f"{self.atlassian_email}:{self.confluence_mcp_api_key}"
                if self.atlassian_email and self.confluence_mcp_api_key
                else "",
            )
            if fragment and fragment.strip()
        }
        return tuple(sorted(fragments, key=len, reverse=True))

    def _redact_text(self, text: str) -> str:
        sanitized_json = self._sanitize_json_text(text)
        if sanitized_json is not None:
            return sanitized_json

        redacted = text
        for pattern, replacement in AUTH_TEXT_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        for secret in self._secret_debug_fragments():
            redacted = redacted.replace(secret, "[REDACTED_SECRET]")
        return redacted

    def _sanitize_json_text(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped or stripped[0] not in "[{":
            return None
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        sanitized = self._sanitize_debug_object(parsed)
        return json.dumps(sanitized, ensure_ascii=False)

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
        if sanitized in ({}, [], ()):
            return None
        if isinstance(sanitized, str):
            return self._truncate_text(sanitized)
        serialized = json.dumps(sanitized, ensure_ascii=False)
        return self._truncate_text(serialized)

    def _build_payload_preview(self, payload: Any) -> str | None:
        preview_source: dict[str, Any] = {}
        is_error = getattr(payload, "isError", None)
        if is_error:
            preview_source["isError"] = True

        structured_content = getattr(payload, "structuredContent", None)
        if structured_content is not None:
            preview_source["structuredContent"] = structured_content

        content_preview = self._normalize_debug_preview_value(getattr(payload, "content", None))
        if content_preview not in (None, [], {}):
            preview_source["content"] = content_preview

        if not preview_source:
            extracted = self._extract_payload(payload)
            if extracted is payload:
                preview_source = {"value": self._normalize_debug_preview_value(payload)}
            else:
                preview_source = {"value": extracted}

        return self._preview_debug_value_deterministic(
            preview_source,
            limit=PAYLOAD_PREVIEW_CHAR_LIMIT,
        )

    def _normalize_debug_preview_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): self._normalize_debug_preview_value(nested)
                for key, nested in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_debug_preview_value(item) for item in value]
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump(exclude_none=True)
            except TypeError:
                dumped = value.model_dump()
            return self._normalize_debug_preview_value(dumped)
        if hasattr(value, "__dict__"):
            nested = {
                key: nested_value
                for key, nested_value in vars(value).items()
                if not key.startswith("_")
            }
            if nested:
                return self._normalize_debug_preview_value(nested)
        return str(value)

    def _preview_debug_value_deterministic(
        self,
        value: Any,
        limit: int = ERROR_PREVIEW_CHAR_LIMIT,
    ) -> str | None:
        if value is None:
            return None
        sanitized = self._sanitize_debug_object(self._normalize_debug_preview_value(value))
        if sanitized in ({}, [], (), ""):
            return None
        if isinstance(sanitized, str):
            return self._truncate_text(sanitized, limit=limit)
        serialized = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
        return self._truncate_text(serialized, limit=limit)

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
        return snapshot

    def _with_debug(self, raw_response: dict[str, Any], exc: Exception | None = None) -> dict[str, Any]:
        if self.debug_enabled:
            debug_snapshot = self._debug_snapshot(exc)
            raw_response["debug"] = debug_snapshot
            if exc is not None:
                self._record_error_debug(self._error_summary_from_snapshot(debug_snapshot))
        return raw_response

    def _configuration_error_response(
        self, query: str, missing_variables: Sequence[str]
    ) -> Dict[str, Any]:
        raw_response = {
            "status": "configuration_error",
            "query": query,
            "missing_variables": list(missing_variables),
            "model": self.model,
            "mcp_server_url": self.mcp_server_url,
        }
        return {
            "answer": (
                "Search is not configured correctly right now. Please verify the required "
                "environment variables and try again."
            ),
            "citations": [],
            "raw_response": self._with_debug(raw_response),
        }

    def _confluence_auth_issue_response(self, query: str, exc: Exception) -> Dict[str, Any]:
        raw_response = {
            "status": "confluence_authentication_error",
            "query": query,
            "model": self.model,
            "error_type": type(exc).__name__,
        }
        return {
            "answer": CONFLUENCE_AUTH_ISSUE_MESSAGE,
            "citations": [],
            "raw_response": self._with_debug(raw_response, exc),
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