"""Lightweight HTTP Slack bot scaffold."""

from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

from agentic_search import AgenticSearch

load_dotenv()


class SlackHttpBot:
    """Minimal Slack Bolt bot exposed through Flask."""

    def __init__(
        self,
        signing_secret: str | None = None,
        bot_token: str | None = None,
        agentic_search: AgenticSearch | None = None,
    ) -> None:
        self.signing_secret = signing_secret or os.getenv("SLACK_SIGNING_SECRET", "")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN", "")
        self.agentic_search = agentic_search or AgenticSearch(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            confluence_space=os.getenv("CONFLUENCE_SPACE", ""),
            mcp_server_name=os.getenv("MCP_SERVER_NAME", "confluence-search"),
        )
        self.bolt_app = App(
            signing_secret=self.signing_secret,
            token=self.bot_token,
        )
        self.slack_handler = SlackRequestHandler(self.bolt_app)
        self.register_handlers()

    def register_handlers(self) -> None:
        @self.bolt_app.command("/search-docs")
        def handle_search_command(ack, body) -> None:  # type: ignore[no-untyped-def]
            prompt = self.extract_prompt(body)
            ack(self.build_slack_message(prompt))

        @self.bolt_app.event("app_mention")
        def handle_app_mention(event, say) -> None:  # type: ignore[no-untyped-def]
            prompt = self.extract_prompt({"event": event})
            say(text=self.build_slack_message(prompt)["text"])

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

    def build_slack_message(self, prompt: str) -> Dict[str, str]:
        if not prompt:
            return {
                "response_type": "ephemeral",
                "text": "Slack bot stub is running. Send a slash command or app mention with text.",
            }

        result = self.agentic_search.search(prompt)
        return {"response_type": "ephemeral", "text": result["answer"]}

    @staticmethod
    def extract_prompt(payload: Dict[str, object]) -> str:
        if payload.get("text"):
            return str(payload["text"]).strip()

        event = payload.get("event") or {}
        if isinstance(event, dict) and event.get("text"):
            return str(event["text"]).strip()

        return ""


app = SlackHttpBot().create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    app.run(host="0.0.0.0", port=port, debug=True)