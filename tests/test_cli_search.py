from __future__ import annotations

import json

import pytest

import cli_search


def test_main_prints_answer_and_citations(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "test-openai-key"
            self.mcp_auth_header = "Bearer test-token"

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
            self.mcp_auth_header = "Bearer test-token"

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
            self.mcp_auth_header = ""

        def search(self, query: str):  # pragma: no cover - should never be called
            raise AssertionError("search should not run without required config")

    monkeypatch.setattr(cli_search, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli_search, "AgenticSearch", FakeSearch)

    exit_code = cli_search.main(["query"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "OPENAI_API_KEY" in captured.err
    assert "Atlassian auth" in captured.err


def test_main_allows_local_mode_without_real_openai_api_key(monkeypatch, capsys):
    class FakeSearch:
        def __init__(self):
            self.openai_api_key = "ollama"
            self.mcp_auth_header = "Bearer test-token"

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
            self.mcp_auth_header = "Bearer test-token"

        def search(self, query: str):
            assert query == "query"
            return {
                "answer": "Sorry, search failed.",
                "citations": [],
                "raw_response": {
                    "status": "search_error",
                    "debug": {
                        "last_step": "list_tools",
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
    assert '"last_step": "list_tools"' in captured.err
    assert '"error_message": "boom"' in captured.err


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