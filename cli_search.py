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
            "Required env vars: OPENAI_API_KEY and MCP_AUTH_HEADER. "
            "Optional: OPENAI_MODEL, CONFLUENCE_SPACE, MCP_SERVER_URL, MCP_SERVER_NAME."
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
    missing: list[str] = []
    if not search.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not search.mcp_auth_header:
        missing.append("MCP_AUTH_HEADER")
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
        url = citation.get("url", "")
        print(f"- {title}: {url}")


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())