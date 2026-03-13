# Lightweight Python Slack Bot

This repository contains a minimal scaffold for a Slack bot with two components:

1. `slack_bot.py` — a lightweight Slack Bolt bot served through Flask.
2. `agentic_search.py` — a stubbed OpenAI-agent search component intended to query Confluence through an MCP search tool.

## Files

- `slack_bot.py`: exposes Slack Bolt HTTP endpoints and `/healthz`
- `agentic_search.py`: placeholder search class for OpenAI + MCP + Confluence
- `requirements.txt`: minimal Python dependencies

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SLACK_SIGNING_SECRET="your-signing-secret"
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export OPENAI_API_KEY="your-openai-key"
export CONFLUENCE_SPACE="YOURSPACE"
export MCP_SERVER_NAME="confluence-search"
python slack_bot.py
```

The app will start on `http://localhost:3000` by default.

## Slack endpoints

- `POST /slack/events` — receives Slack Events API payloads via Bolt
- `POST /slack/commands` — receives slash commands via Bolt
- `GET /healthz` — simple health check

## Current behavior

- Uses `slack_bolt` with Flask adapter wiring
- Responds to `/search-docs` slash commands with a stubbed answer from `AgenticSearch`
- Responds to `app_mention` events with the same stubbed answer
- Lets Bolt handle Slack request verification and URL verification flows

## Next steps

- Wire `AgenticSearch.search()` to the OpenAI Agents SDK or Responses API
- Connect an MCP server that can search your Confluence space
- Add richer Slack formatting, citations, and async processing as needed