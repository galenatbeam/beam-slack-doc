from __future__ import annotations

import slack_bot

from slack_bot import PLACEHOLDER_TEXT, RecentEventCache, SlackHttpBot


class ImmediateExecutor:
    def __init__(self) -> None:
        self.submissions = []

    def submit(self, fn, *args, **kwargs):
        self.submissions.append((fn, args, kwargs))
        fn(*args, **kwargs)


class FakeSlackClient:
    def __init__(self) -> None:
        self.post_calls = []
        self.update_calls = []

    def chat_postMessage(self, **kwargs):
        self.post_calls.append(kwargs)
        return {"ts": "999.001"}

    def chat_update(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"ok": True}


class FakeFlaskApp:
    def __init__(self) -> None:
        self.run_calls = []

    def run(self, **kwargs):
        self.run_calls.append(kwargs)


class FakeSearch:
    def __init__(self) -> None:
        self.queries = []

    def search(self, query: str):
        self.queries.append(query)
        return {
            "answer": "Found the answer.",
            "citations": [
                {
                    "title": "Beam Benefits Overview",
                    "url": "https://example.com/beam-overview",
                }
            ],
            "raw_response": {"status": "ok"},
        }


def make_bot(search: FakeSearch | None = None) -> tuple[SlackHttpBot, FakeSearch, ImmediateExecutor]:
    fake_search = search or FakeSearch()
    executor = ImmediateExecutor()
    bot = SlackHttpBot(
        signing_secret="secret",
        bot_token="token",
        agentic_search=fake_search,
        executor=executor,
        event_cache=RecentEventCache(),
        token_verification_enabled=False,
    )
    return bot, fake_search, executor


def test_resolve_thread_ts_prefers_existing_thread_and_falls_back_to_event_ts():
    assert SlackHttpBot.resolve_thread_ts({"thread_ts": "111.222", "ts": "333.444"}) == "111.222"
    assert SlackHttpBot.resolve_thread_ts({"ts": "333.444"}) == "333.444"


def test_handle_app_mention_posts_placeholder_then_updates_it():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_app_mention_event(
        event={
            "channel": "C123",
            "ts": "171.001",
            "text": "<@B123> what is Beam?",
            "user": "U123",
        },
        body={
            "event_id": "evt-1",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions
    assert client.post_calls == [
        {
            "channel": "C123",
            "thread_ts": "171.001",
            "text": PLACEHOLDER_TEXT,
        }
    ]
    assert fake_search.queries == ["<@B123> what is Beam?"]
    assert client.update_calls == [
        {
            "channel": "C123",
            "ts": "999.001",
            "text": "Found the answer.\n\nCitations:\n- Beam Benefits Overview: https://example.com/beam-overview",
        }
    ]


def test_duplicate_event_id_does_not_double_post():
    bot, fake_search, _executor = make_bot()
    client = FakeSlackClient()
    event = {
        "channel": "C123",
        "ts": "171.001",
        "text": "<@B123> what is Beam?",
        "user": "U123",
    }
    body = {
        "event_id": "evt-duplicate",
        "authorizations": [{"user_id": "B123"}],
    }

    bot.handle_app_mention_event(event=event, body=body, client=client)
    bot.handle_app_mention_event(event=event, body=body, client=client)

    assert len(client.post_calls) == 1
    assert len(client.update_calls) == 1
    assert fake_search.queries == ["<@B123> what is Beam?"]


def test_handle_message_event_posts_placeholder_then_updates_it_in_thread():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "ts": "171.001",
            "text": "What is Beam?",
            "user": "U123",
        },
        body={
            "event_id": "evt-message-1",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions
    assert client.post_calls == [
        {
            "channel": "C123",
            "thread_ts": "171.001",
            "text": PLACEHOLDER_TEXT,
        }
    ]
    assert fake_search.queries == ["What is Beam?"]
    assert client.update_calls == [
        {
            "channel": "C123",
            "ts": "999.001",
            "text": "Found the answer.\n\nCitations:\n- Beam Benefits Overview: https://example.com/beam-overview",
        }
    ]


def test_handle_message_event_ignores_thread_replies():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "thread_ts": "171.000",
            "ts": "171.001",
            "text": "What is Beam?",
            "user": "U123",
        },
        body={
            "event_id": "evt-message-thread-reply",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_handle_message_event_ignores_messages_without_question_mark():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "ts": "171.001",
            "text": "Tell me about Beam",
            "user": "U123",
        },
        body={
            "event_id": "evt-message-no-question",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_handle_message_event_ignores_direct_bot_mentions():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "ts": "171.001",
            "text": "<@B123> What is Beam?",
            "user": "U123",
        },
        body={
            "event_id": "evt-message-mention",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_handle_message_event_ignores_message_subtypes():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "subtype": "message_changed",
            "ts": "171.001",
            "text": "What is Beam?",
            "user": "U123",
        },
        body={
            "event_id": "evt-message-subtype",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_handle_message_event_ignores_bot_message_subtype():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "subtype": "bot_message",
            "ts": "171.001",
            "text": "What is Beam?",
            "bot_id": "B999",
        },
        body={
            "event_id": "evt-message-bot-subtype",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_handle_message_event_ignores_bot_messages():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "ts": "171.001",
            "text": "What is Beam?",
            "bot_id": "B999",
        },
        body={
            "event_id": "evt-message-bot",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_handle_message_event_ignores_self_messages():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "C123",
            "channel_type": "channel",
            "ts": "171.001",
            "text": "What is Beam?",
            "user": "B123",
        },
        body={
            "event_id": "evt-message-self",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_duplicate_message_event_id_does_not_double_post():
    bot, fake_search, _executor = make_bot()
    client = FakeSlackClient()
    event = {
        "channel": "C123",
        "channel_type": "channel",
        "ts": "171.001",
        "text": "What is Beam?",
        "user": "U123",
    }
    body = {
        "event_id": "evt-message-duplicate",
        "authorizations": [{"user_id": "B123"}],
    }

    bot.handle_message_event(event=event, body=body, client=client)
    bot.handle_message_event(event=event, body=body, client=client)

    assert len(client.post_calls) == 1
    assert len(client.update_calls) == 1
    assert fake_search.queries == ["What is Beam?"]


def test_handle_message_event_ignores_non_channel_messages():
    bot, fake_search, executor = make_bot()
    client = FakeSlackClient()

    bot.handle_message_event(
        event={
            "channel": "D123",
            "channel_type": "im",
            "ts": "171.001",
            "text": "What is Beam?",
            "user": "U123",
        },
        body={
            "event_id": "evt-message-im",
            "authorizations": [{"user_id": "B123"}],
        },
        client=client,
    )

    assert executor.submissions == []
    assert client.post_calls == []
    assert client.update_calls == []
    assert fake_search.queries == []


def test_run_uses_socket_mode_when_app_token_is_set(monkeypatch):
    started = {}

    class FakeSocketModeHandler:
        def __init__(self, bolt_app, app_token):
            started["bolt_app"] = bolt_app
            started["app_token"] = app_token

        def start(self):
            started["started"] = True

    monkeypatch.setattr(slack_bot, "SocketModeHandler", FakeSocketModeHandler)

    bot, _fake_search, _executor = make_bot()
    bot.app_token = "xapp-test-token"
    app = FakeFlaskApp()

    bot.run(app=app, port=4321, debug=False)

    assert started == {
        "bolt_app": bot.bolt_app,
        "app_token": "xapp-test-token",
        "started": True,
    }
    assert app.run_calls == []


def test_run_falls_back_to_http_mode_without_app_token():
    bot, _fake_search, _executor = make_bot()
    bot.app_token = ""
    app = FakeFlaskApp()

    bot.run(app=app, port=4321, debug=False)

    assert app.run_calls == [{"host": "0.0.0.0", "port": 4321, "debug": False}]


def test_default_agentic_search_wires_github_env_configuration(monkeypatch):
    captured_kwargs = {}

    class FakeAgenticSearch:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def search(self, query: str):
            return {"answer": query, "citations": [], "raw_response": {"status": "ok"}}

    monkeypatch.setattr(slack_bot, "AgenticSearch", FakeAgenticSearch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("CONFLUENCE_SPACE", "BEN")
    monkeypatch.setenv("MCP_SERVER_NAME", "atlassian-rovo")
    monkeypatch.setenv("GITHUB_PAT", "test-github-token")
    monkeypatch.setenv("GITHUB_ORG", "beamtech")
    monkeypatch.setenv("GITHUB_MCP_SERVER_NAME", "github")
    monkeypatch.setenv("GITHUB_MCP_SERVER_URL", "https://api.githubcopilot.com/mcp/readonly")

    bot = SlackHttpBot(
        signing_secret="secret",
        bot_token="token",
        app_token="",
        agentic_search=None,
        token_verification_enabled=False,
    )

    assert isinstance(bot.agentic_search, FakeAgenticSearch)
    assert captured_kwargs == {
        "openai_api_key": "test-openai-key",
        "confluence_space": "BEN",
        "mcp_server_name": "atlassian-rovo",
        "github_pat": "test-github-token",
        "github_org": "beamtech",
        "github_mcp_server_name": "github",
        "github_mcp_server_url": "https://api.githubcopilot.com/mcp/readonly",
    }