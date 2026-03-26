"""Command-line wrapper for AgenticSearch."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from agentic_search import AgenticSearch


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Search documentation with AgenticSearch.",
        epilog=(
            "Required MCP config: ATLASSIAN_EMAIL, CONFLUENCE_MCP_API_KEY, and "
            "ATLASSIAN_CLOUD_ID. Atlassian MCP auth uses Basic "
            "base64(ATLASSIAN_EMAIL:CONFLUENCE_MCP_API_KEY). "
            "MCP_SERVER_URL is optional and defaults to https://mcp.atlassian.com/v1/mcp. "
            "Search also requires either "
            "OPENAI_API_KEY (OpenAI-hosted mode) or LLM_BASE_URL=http://localhost:11434/v1 "
            "(local Ollama mode; OPENAI_API_KEY optional). Optional: "
            "OPENAI_MODEL, CONFLUENCE_SPACE, MCP_SERVER_URL, MCP_SERVER_NAME. Set "
            "AGENTIC_SEARCH_DEBUG=1 (also true/yes) to emit MCP debug summaries "
            "to stderr and include raw_response.debug in --json output without printing "
            "auth headers, raw Atlassian credentials, or encoded Basic credentials."
        ),
    )
    parser.add_argument("query", help="Question or keywords to search for")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the full JSON response",
    )
    return parser


def _missing_config(search: AgenticSearch) -> list[str]:
    if hasattr(search, "missing_configuration"):
        return search.missing_configuration()

    missing: list[str] = []
    if not getattr(search, "openai_api_key", ""):
        missing.append("OPENAI_API_KEY")
    if not getattr(search, "atlassian_email", ""):
        missing.append("ATLASSIAN_EMAIL")
    if not getattr(search, "confluence_mcp_api_key", ""):
        missing.append("CONFLUENCE_MCP_API_KEY")
    if not getattr(search, "atlassian_cloud_id", ""):
        missing.append("ATLASSIAN_CLOUD_ID")
    return missing


def _print_default_output(result: dict) -> None:
    print(result.get("answer", ""))
    print("\nCitations:")
    citations = result.get("citations", [])
    if not citations:
        print("- none")
        return

    for citation in citations:
        title = citation.get("title", "Untitled")
        reference = citation.get("url") or citation.get("ari") or citation.get("identifier")
        print(f"- {title}: {reference or '(no link available)'}")


def _print_failure_debug(result: dict) -> None:
    raw_response = result.get("raw_response", {}) or {}
    debug = raw_response.get("debug")
    status = raw_response.get("status", "")
    if not debug or not isinstance(debug, dict) or not status.endswith("error"):
        return

    summary = {
        key: debug[key]
        for key in (
            "last_step",
            "error_type",
            "error_message",
            "http_status",
            "http_response_preview",
            "mcp_error_preview",
        )
        if key in debug
    }
    if not summary:
        summary = debug
    print(
        f"[AgenticSearch debug] result: {json.dumps(summary, ensure_ascii=False)}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    load_dotenv()
    args = build_parser().parse_args(argv)

    search = AgenticSearch()
    missing = _missing_config(search)
    if missing:
        print(
            "Missing required configuration: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    result = search.search(args.query)
    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_default_output(result)
        _print_failure_debug(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())