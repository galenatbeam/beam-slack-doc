"""Thin MCP server wrapper for namespacing, allowlisting, and schema sanitization."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agents.mcp.server import MCPServer
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, Tool as MCPTool

SUPPORTED_TOOL_SCHEMA_KEYS = frozenset(
    {"type", "properties", "required", "description", "items", "enum", "anyOf", "oneOf"}
)


@dataclass(frozen=True)
class ScopedToolBinding:
    """Public-facing tool metadata mapped back to the underlying MCP tool."""

    public_name: str
    original_name: str
    tool: MCPTool


SanitizedSchemaRecorder = Callable[[str, Mapping[str, Any], Mapping[str, Any], Sequence[str]], None]
ToolExecutor = Callable[
    [ScopedToolBinding, dict[str, Any] | None, dict[str, Any] | None],
    Awaitable[CallToolResult],
]


class ScopedMCPServer(MCPServer):
    """Expose a filtered/namespaced view of an MCP server using the native SDK path."""

    def __init__(
        self,
        server: Any,
        *,
        name_prefix: str = "",
        allowed_tool_names: set[str] | None = None,
        tool_executor: ToolExecutor | None = None,
        sanitized_schema_recorder: SanitizedSchemaRecorder | None = None,
    ) -> None:
        super().__init__(
            use_structured_content=True,
            failure_error_function=None,
            tool_meta_resolver=getattr(server, "tool_meta_resolver", None),
        )
        self.server = server
        self.name_prefix = name_prefix
        self.allowed_tool_names = allowed_tool_names
        self.tool_executor = tool_executor
        self.sanitized_schema_recorder = sanitized_schema_recorder
        self._public_tools: list[MCPTool] | None = None
        self._bindings_by_public_name: dict[str, ScopedToolBinding] = {}
        self._discovered_tool_names: list[str] = []
        self._allowlisted_tool_names: list[str] = []

    @property
    def name(self) -> str:
        return str(getattr(self.server, "name", "mcp-server"))

    @property
    def cached_tools(self) -> list[MCPTool] | None:
        return self._public_tools

    @property
    def discovered_tool_names(self) -> list[str]:
        return list(self._discovered_tool_names)

    @property
    def allowlisted_tool_names(self) -> list[str]:
        return list(self._allowlisted_tool_names)

    async def connect(self):
        await self.server.connect()

    async def cleanup(self):
        await self.server.cleanup()

    async def list_tools(
        self,
        run_context: Any | None = None,
        agent: Any | None = None,
    ) -> list[MCPTool]:
        if self._public_tools is not None:
            return self._public_tools

        try:
            tools_result = await self.server.list_tools(run_context=run_context, agent=agent)
        except TypeError:
            tools_result = await self.server.list_tools()

        tools = list(getattr(tools_result, "tools", tools_result) or [])
        public_tools: list[MCPTool] = []
        bindings_by_public_name: dict[str, ScopedToolBinding] = {}
        discovered_tool_names: list[str] = []
        allowlisted_tool_names: list[str] = []

        for tool in tools:
            original_name = getattr(tool, "name", None)
            if not original_name:
                continue

            public_name = f"{self.name_prefix}{original_name}"
            discovered_tool_names.append(public_name)
            if self.allowed_tool_names is not None and original_name not in self.allowed_tool_names:
                continue

            sanitized_schema = sanitize_tool_input_schema(
                public_name,
                getattr(tool, "inputSchema", None),
                recorder=self.sanitized_schema_recorder,
            )
            public_tools.append(
                tool.model_copy(
                    update={"name": public_name, "inputSchema": sanitized_schema},
                    deep=True,
                )
            )
            bindings_by_public_name[public_name] = ScopedToolBinding(
                public_name=public_name,
                original_name=str(original_name),
                tool=tool,
            )
            allowlisted_tool_names.append(public_name)

        self._public_tools = public_tools
        self._bindings_by_public_name = bindings_by_public_name
        self._discovered_tool_names = discovered_tool_names
        self._allowlisted_tool_names = allowlisted_tool_names
        return public_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        binding = self._bindings_by_public_name.get(tool_name)
        if binding is None:
            await self.list_tools()
            binding = self._bindings_by_public_name.get(tool_name)
        if binding is None:
            raise RuntimeError(f"Tool '{tool_name}' is not available on MCP server '{self.name}'.")

        if self.tool_executor is not None:
            return await self.tool_executor(binding, arguments, meta)

        if meta is None:
            return await self.server.call_tool(binding.original_name, arguments or {})
        return await self.server.call_tool(binding.original_name, arguments or {}, meta=meta)

    async def list_prompts(self) -> ListPromptsResult:
        return await self.server.list_prompts()

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> GetPromptResult:
        return await self.server.get_prompt(name, arguments)


def sanitize_tool_input_schema(
    tool_name: str,
    input_schema: Any,
    *,
    recorder: SanitizedSchemaRecorder | None = None,
) -> dict[str, Any]:
    if not isinstance(input_schema, Mapping):
        return {"type": "object", "properties": {}}

    stripped_keys: list[str] = []
    sanitized = _sanitize_tool_schema_node(input_schema, "$", stripped_keys)
    if sanitized.get("type") != "object":
        stripped_keys.append("$.type")
        sanitized = {"type": "object", "properties": {}}
    else:
        sanitized["type"] = "object"
        if not isinstance(sanitized.get("properties"), dict):
            sanitized["properties"] = {}

    required = sanitized.get("required")
    if isinstance(required, list):
        filtered_required = [item for item in required if isinstance(item, str) and item in sanitized["properties"]]
        if filtered_required:
            sanitized["required"] = filtered_required
        else:
            sanitized.pop("required", None)
    else:
        sanitized.pop("required", None)

    if recorder is not None and (stripped_keys or sanitized != input_schema):
        recorder(tool_name, input_schema, sanitized, stripped_keys)
    return sanitized


def _sanitize_tool_schema_node(
    schema: Mapping[str, Any],
    path: str,
    stripped_keys: list[str],
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in schema.items():
        key_path = f"{path}.{key}" if path else str(key)
        if key not in SUPPORTED_TOOL_SCHEMA_KEYS:
            stripped_keys.append(key_path)
            continue
        if key == "type":
            normalized_type = _normalize_schema_type(value)
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
                str(property_name): _sanitize_tool_schema_child(
                    property_schema,
                    f"{key_path}.{property_name}",
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
            sanitized[key] = [item for item in value if item is None or isinstance(item, (str, int, float, bool))]
            continue
        if key == "items":
            if isinstance(value, Mapping):
                sanitized[key] = _sanitize_tool_schema_node(value, key_path, stripped_keys)
            elif isinstance(value, list):
                sanitized[key] = [
                    _sanitize_tool_schema_child(item, f"{key_path}[{index}]", stripped_keys)
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
                _sanitize_tool_schema_child(item, f"{key_path}[{index}]", stripped_keys)
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
            filtered_required = [item for item in required if item in sanitized["properties"]]
            if filtered_required:
                sanitized["required"] = filtered_required
            else:
                sanitized.pop("required", None)
    elif "required" in sanitized:
        sanitized.pop("required", None)
    return sanitized


def _sanitize_tool_schema_child(schema: Any, path: str, stripped_keys: list[str]) -> dict[str, Any]:
    if not isinstance(schema, Mapping):
        stripped_keys.append(path)
        return {}
    return _sanitize_tool_schema_node(schema, path, stripped_keys)


def _normalize_schema_type(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        non_null_types = [item for item in value if isinstance(item, str) and item != "null"]
        if len(non_null_types) == 1:
            return non_null_types[0]
    return None