"""Helpers for MCP tool argument normalization and scoping policy."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

ToolAcceptsParam = Callable[[Any, str], bool]


def prepare_atlassian_tool_arguments(
    tool: Any,
    arguments: Mapping[str, Any],
    atlassian_cloud_id: str,
    *,
    tool_accepts_param: ToolAcceptsParam,
) -> tuple[dict[str, Any], bool]:
    prepared = dict(arguments)
    if not atlassian_cloud_id or not tool_accepts_param(tool, "cloudId"):
        return prepared, False

    provided_cloud_id = prepared.get("cloudId")
    override_detected = (
        "cloudId" in prepared
        and isinstance(provided_cloud_id, str)
        and provided_cloud_id != atlassian_cloud_id
    )
    prepared["cloudId"] = atlassian_cloud_id
    return prepared, override_detected


def prepare_github_tool_arguments(
    tool: Any,
    arguments: Mapping[str, Any],
    github_org: str,
    *,
    tool_accepts_param: ToolAcceptsParam,
) -> tuple[dict[str, Any], bool]:
    prepared = dict(arguments)
    if not github_org:
        return prepared, False

    github_scope_keys = [
        key for key in ("owner", "org", "organization") if tool_accepts_param(tool, key)
    ]
    override_detected = False
    for key in github_scope_keys:
        provided_value = prepared.get(key)
        normalized_value = normalize_github_org(provided_value if isinstance(provided_value, str) else "")
        if (
            not override_detected
            and key in prepared
            and isinstance(provided_value, str)
            and normalized_value
            and normalized_value != github_org
        ):
            override_detected = True
        prepared[key] = github_org
    return prepared, override_detected


def normalize_github_org(value: str | None) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""

    candidate = re.sub(r"^https?://", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.removeprefix("git@github.com:")
    candidate = candidate.removeprefix("@")
    candidate = candidate.removeprefix("www.")
    candidate = re.split(r"[?#]", candidate, maxsplit=1)[0].strip().strip("/")

    if candidate.lower().startswith("github.com/"):
        candidate = candidate[len("github.com/") :]

    parts = [segment.strip() for segment in candidate.split("/") if segment.strip()]
    if not parts:
        return ""

    normalized = parts[0].removesuffix(".git").strip()
    return normalized.lower()