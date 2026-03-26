from __future__ import annotations

import json

import pytest

import cli_search


def test_main_prints_answer_and_citations(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "test-openai-key"
            self.atlassian_email = "bot@example.com"
            self.confluence_mcp_api_key = "test-token"
            self.atlassian_cloud_id = "test-cloud-id"

        def search(self, query: str):
            assert query == "What is Beam Benefits?"
            return {
                "answer": "Beam Benefits offers dental and vision coverage.",
                "citations": [
                    {
                        "title": "Benefits Overview",
                        "url": "https://example.com/benefits-overview",
                    }
                ],
                "raw_response": {"status": "ok"},
            }

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["What is Beam Benefits?"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Beam Benefits offers dental and vision coverage." in captured.out
    assert "Citations:" in captured.out
    assert "Benefits Overview: https://example.com/benefits-overview" in captured.out


def test_main_prints_json_when_requested(monkeypatch, capsys):
    result = {
        "answer": "A JSON answer.",
        "citations": [],
        "raw_response": {"status": "no_results"},
    }

    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "test-openai-key"
            self.atlassian_email = "bot@example.com"
            self.confluence_mcp_api_key = "test-token"
            self.atlassian_cloud_id = "test-cloud-id"

        def search(self, query: str):
            assert query == "query"
            return result

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["--json", "query"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == result


def test_main_returns_error_for_missing_required_config(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = ""
            self.atlassian_email = ""
            self.confluence_mcp_api_key = ""
            self.atlassian_cloud_id = ""

        def search(self, query: str):  # pragma: no cover - should never be called
            raise AssertionError("search should not run without required config")

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["query"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "OPENAI_API_KEY" in captured.err
    assert "ATLASSIAN_EMAIL" in captured.err
    assert "CONFLUENCE_MCP_API_KEY" in captured.err
    assert "ATLASSIAN_CLOUD_ID" in captured.err


def test_main_allows_local_mode_without_real_openai_api_key(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "ollama"
            self.atlassian_email = "bot@example.com"
            self.confluence_mcp_api_key = "test-token"
            self.atlassian_cloud_id = "test-cloud-id"

        def search(self, query: str):
            assert query == "query"
            return {
                "answer": "Local mode answer.",
                "citations": [],
                "raw_response": {"status": "ok"},
            }

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["query"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Local mode answer." in captured.out
    assert captured.err == ""


def test_main_prints_debug_failure_summary_to_stderr_without_json(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "test-openai-key"
            self.atlassian_email = "bot@example.com"
            self.confluence_mcp_api_key = "test-token"
            self.atlassian_cloud_id = "test-cloud-id"

        def search(self, query: str):
            assert query == "query"
            return {
                "answer": "Sorry, search failed.",
                "citations": [],
                "raw_response": {
                    "status": "search_error",
                    "debug": {
                        "last_step": "call_tool:search",
                        "error_type": "RuntimeError",
                        "error_message": "boom",
                        "http_status": 403,
                    },
                },
            }

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["query"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Sorry, search failed." in captured.out
    assert "[AgenticSearch debug] result:" in captured.err
    assert '"last_step": "call_tool:search"' in captured.err
    assert '"error_message": "boom"' in captured.err


def test_main_prints_ari_when_citation_has_no_url(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "test-openai-key"
            self.atlassian_email = "bot@example.com"
            self.confluence_mcp_api_key = "test-token"
            self.atlassian_cloud_id = "test-cloud-id"

        def search(self, query: str):
            assert query == "query"
            return {
                "answer": "Found a result without a URL.",
                "citations": [
                    {
                        "title": "Benefits FAQ",
                        "url": None,
                        "ari": "ari:cloud:confluence::page/faq",
                    }
                ],
                "raw_response": {"status": "ok"},
            }

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["query"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Benefits FAQ: ari:cloud:confluence::page/faq" in captured.out


def test_parser_uses_exit_code_two_for_invalid_usage():
    with pytest.raises(SystemExit) as excinfo:
        cli_search.main([])

    assert excinfo.value.code == 2


def test_help_mentions_agentic_search_debug(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli_search.main(["--help"])

    captured = capsys.readouterr()

    assert excinfo.value.code == 0
    assert "AGENTIC_SEARCH_DEBUG" in captured.out
    assert "ATLASSIAN_EMAIL" in captured.out
    assert "CONFLUENCE_MCP_API_KEY" in captured.out
    assert "ATLASSIAN_CLOUD_ID" in captured.out