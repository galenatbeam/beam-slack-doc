"""Lightweight HTTP Slack bot scaffold."""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agentic_search import AgenticSearch

load_dotenv()

PLACEHOLDER_TEXT = "Searching docs..."
EVENT_ID_TTL_SECONDS = 600
QUESTION_REGEX = re.compile(r"\?")


class RecentEventCache:
    """Small in-memory TTL cache for Slack event deduplication."""

    def __init__(self, ttl_seconds: float = EVENT_ID_TTL_SECONDS, now: Callable[[], float] | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self._now = now or time.monotonic
        self._entries: Dict[str, float] = {}
        self._lock = threading.Lock()

    def add(self, key: str) -> bool:
        now = self._now()
        with self._lock:
            self._purge_expired(now)
            expires_at = self._entries.get(key)
            if expires_at is not None and expires_at > now:
                return False

            self._entries[key] = now + self.ttl_seconds
            return True

    def _purge_expired(self, now: float) -> None:
        expired_keys = [key for key, expires_at in self._entries.items() if expires_at <= now]
        for key in expired_keys:
            self._entries.pop(key, None)


class SlackHttpBot:
    """Minimal Slack Bolt bot exposed through Flask."""

    def __init__(
        self,
        signing_secret: str | None = None,
        bot_token: str | None = None,
        app_token: str | None = None,
        agentic_search: AgenticSearch | None = None,
        executor: ThreadPoolExecutor | None = None,
        event_cache: RecentEventCache | None = None,
        token_verification_enabled: bool = True,
    ) -> None:
        self.signing_secret = signing_secret or os.getenv("SLACK_SIGNING_SECRET", "")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")
        self.app_token = os.getenv("SLACK_APP_TOKEN", "") if app_token is None else app_token
        self.agentic_search = agentic_search or AgenticSearch(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            confluence_space=os.getenv("CONFLUENCE_SPACE", ""),
            mcp_server_name=os.getenv("MCP_SERVER_NAME", "confluence-search"),
            github_pat=os.getenv("GITHUB_PAT"),
            github_org=os.getenv("GITHUB_ORG"),
            github_mcp_server_name=os.getenv("GITHUB_MCP_SERVER_NAME"),
            github_mcp_server_url=os.getenv("GITHUB_MCP_SERVER_URL"),
        )
        self.executor = executor or ThreadPoolExecutor(max_workers=4, thread_name_prefix="slack-search")
        self.event_cache = event_cache or RecentEventCache()
        self.bolt_app = App(
            signing_secret=self.signing_secret,
            token=self.bot_token,
            token_verification_enabled=token_verification_enabled,
        )
        self.slack_handler = SlackRequestHandler(self.bolt_app)
        self.register_handlers()

    def register_handlers(self) -> None:
        @self.bolt_app.command("/search-docs")
        def handle_search_command(ack, body) -> None:  # type: ignore[no-untyped-def]
            prompt = self.extract_prompt(body)
            ack(self.build_slack_message(prompt))

        @self.bolt_app.event("app_mention")
        def handle_app_mention(event, body, client) -> None:  # type: ignore[no-untyped-def]
            self.handle_app_mention_event(event=event, body=body, client=client)

        @self.bolt_app.event("message")
        def handle_message(event, body, client) -> None:  # type: ignore[no-untyped-def]
            self.handle_message_event(event=event, body=body, client=client)

    def handle_app_mention_event(self, event: Dict[str, object], body: Dict[str, object], client: Any) -> None:
        if self.should_ignore_app_mention(event, body):
            return

        event_id = str(body.get("event_id") or "").strip()
        if event_id and not self.event_cache.add(event_id):
            return

        channel = str(event.get("channel") or "").strip()
        if not channel:
            return

        thread_ts = self.resolve_thread_ts(event)
        if not thread_ts:
            return

        prompt = self.extract_prompt({"event": event})
        placeholder = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=PLACEHOLDER_TEXT,
        )
        placeholder_ts = str(placeholder.get("ts") or "").strip()
        if not placeholder_ts:
            return

        self.executor.submit(
            self.complete_app_mention,
            client,
            channel,
            placeholder_ts,
            prompt,
        )

    def handle_message_event(self, event: Dict[str, object], body: Dict[str, object], client: Any) -> None:
        if self.should_ignore_message_event(event, body):
            return

        channel = str(event.get("channel") or "").strip()
        if not self.is_channel_message(event, channel):
            return

        if self.is_thread_reply(event):
            return

        thread_ts = str(event.get("ts") or "").strip()
        if not thread_ts:
            return

        prompt = self.extract_prompt({"event": event})
        if not QUESTION_REGEX.search(prompt):
            return

        event_id = str(body.get("event_id") or "").strip()
        if event_id and not self.event_cache.add(event_id):
            return

        placeholder = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=PLACEHOLDER_TEXT,
        )
        placeholder_ts = str(placeholder.get("ts") or "").strip()
        if not placeholder_ts:
            return

        self.executor.submit(
            self.complete_app_mention,
            client,
            channel,
            placeholder_ts,
            prompt,
        )

    def complete_app_mention(self, client: Any, channel: str, placeholder_ts: str, prompt: str) -> None:
        try:
            text = self.build_response_text(prompt)
        except Exception:
            text = "Sorry, I ran into an issue while searching the docs."

        client.chat_update(
            channel=channel,
            ts=placeholder_ts,
            text=text,
        )

    def create_app(self) -> Flask:
        app = Flask(__name__)

        @app.get("/healthz")
        def healthcheck() -> tuple[Dict[str, bool], int]:
            return {"ok": True}, 200

        @app.post("/slack/events")
        @app.post("/slack/commands")
        def slack_requests():
            return self.slack_handler.handle(request)

        return app

    def should_use_socket_mode(self) -> bool:
        return self.app_token.startswith("xapp-")

    def create_socket_mode_handler(self) -> SocketModeHandler:
        return SocketModeHandler(self.bolt_app, self.app_token)

    def run(self, app: Flask | None = None, port: int | None = None, debug: bool = True) -> None:
        if self.should_use_socket_mode():
            self.create_socket_mode_handler().start()
            return

        flask_app = app or self.create_app()
        flask_app.run(host="0.0.0.0", port=port or int(os.getenv("PORT", "3000")), debug=debug)

    def build_slack_message(self, prompt: str) -> Dict[str, str]:
        return {"response_type": "ephemeral", "text": self.build_response_text(prompt)}

    def build_response_text(self, prompt: str) -> str:
        if not prompt:
            return "Slack bot stub is running. Send a slash command or app mention with text."

        result = self.agentic_search.search(prompt)
        return self.format_search_result(result)

    @staticmethod
    def format_search_result(result: Dict[str, object]) -> str:
        answer = str(result.get("answer") or "").strip()
        if not answer:
            answer = "Sorry, I couldn't find an answer."

        lines = [answer]
        citations = result.get("citations")
        if isinstance(citations, list):
            citation_lines = [
                line
                for citation in citations
                if (line := SlackHttpBot.format_citation_line(citation))
            ]
            if citation_lines:
                lines.extend(["", "Citations:", *citation_lines])

        return "\n".join(lines)

    @staticmethod
    def format_citation_line(citation: object) -> str | None:
        if not isinstance(citation, dict):
            return None

        title = str(citation.get("title") or "Untitled").strip()
        target = citation.get("url") or citation.get("ari") or citation.get("identifier")
        if target:
            return f"- {title}: {target}"
        return f"- {title}"

    @staticmethod
    def resolve_thread_ts(event: Dict[str, object]) -> str:
        value = event.get("thread_ts") or event.get("ts")
        return str(value or "").strip()

    @staticmethod
    def is_thread_reply(event: Dict[str, object]) -> bool:
        thread_ts = str(event.get("thread_ts") or "").strip()
        ts = str(event.get("ts") or "").strip()
        return bool(thread_ts and thread_ts != ts)

    @staticmethod
    def is_channel_message(event: Dict[str, object], channel: str | None = None) -> bool:
        channel_type = str(event.get("channel_type") or "").strip()
        resolved_channel = channel if channel is not None else str(event.get("channel") or "").strip()
        return channel_type == "channel" or resolved_channel.startswith("C")

    @staticmethod
    def should_ignore_slack_event(event: Dict[str, object], body: Dict[str, object]) -> bool:
        if event.get("bot_id"):
            return True

        authorized_user_id = SlackHttpBot.extract_authorized_user_id(body)
        event_user_id = str(event.get("user") or "").strip()
        return bool(authorized_user_id and event_user_id == authorized_user_id)

    @staticmethod
    def should_ignore_app_mention(event: Dict[str, object], body: Dict[str, object]) -> bool:
        if event.get("subtype") == "bot_message":
            return True

        return SlackHttpBot.should_ignore_slack_event(event, body)

    @staticmethod
    def should_ignore_message_event(event: Dict[str, object], body: Dict[str, object]) -> bool:
        if event.get("subtype"):
            return True

        if SlackHttpBot.should_ignore_slack_event(event, body):
            return True

        text = SlackHttpBot.extract_prompt({"event": event})
        bot_user_id = SlackHttpBot.extract_authorized_user_id(body)
        return SlackHttpBot.contains_direct_bot_mention(text, bot_user_id)

    @staticmethod
    def contains_direct_bot_mention(text: str, bot_user_id: str) -> bool:
        if not text or not bot_user_id:
            return False

        return f"<@{bot_user_id}>" in text

    @staticmethod
    def extract_authorized_user_id(body: Dict[str, object]) -> str:
        authorizations = body.get("authorizations")
        if not isinstance(authorizations, list) or not authorizations:
            return ""

        first_authorization = authorizations[0]
        if not isinstance(first_authorization, dict):
            return ""

        return str(first_authorization.get("user_id") or "").strip()

    @staticmethod
    def extract_prompt(payload: Dict[str, object]) -> str:
        if payload.get("text"):
            return str(payload["text"]).strip()

        event = payload.get("event") or {}
        if isinstance(event, dict) and event.get("text"):
            return str(event["text"]).strip()

        return ""


bot = SlackHttpBot()
app = bot.create_app()


def main() -> None:
    port = int(os.getenv("PORT", "3000"))
    bot.run(app=app, port=port, debug=True)


if __name__ == "__main__":
    main()