# Lightweight Python Slack Bot

This repository contains a minimal scaffold for a Slack bot with three components:

1. `slack_bot.py` — a lightweight Slack Bolt bot served through Flask.
2. `agentic_search.py` — an OpenAI Agents SDK search component for Atlassian Rovo / Confluence MCP.
3. `cli_search.py` — a small CLI wrapper around `AgenticSearch.search()`.

## Files

- `slack_bot.py`: exposes Slack Bolt HTTP endpoints and `/healthz`
- `agentic_search.py`: structured documentation search for OpenAI + MCP + Confluence
- `cli_search.py`: CLI entrypoint for local documentation searches
- `requirements.txt`: minimal Python dependencies

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SLACK_SIGNING_SECRET="your-signing-secret"
export SLACK_BOT_TOKEN="xoxb-your-bot-token"
export OPENAI_API_KEY="your-openai-key"
export ATLASSIAN_API_EMAIL="you@example.com"
export ATLASSIAN_API_TOKEN="your-atlassian-api-token"
export MCP_CLOUD_ID="your-atlassian-cloud-id"
export CONFLUENCE_SPACE="YOURSPACE"
export MCP_SERVER_NAME="atlassian-rovo"
python slack_bot.py
```

The app will start on `http://localhost:3000` by default.

You can also put these values in a `.env` file; both `slack_bot.py` and `cli_search.py`
load it automatically.

## CLI usage

Required environment variables:

- Atlassian auth: choose one
  - `MCP_AUTH_HEADER` (backward-compatible override; full `Authorization` value)
  - `ATLASSIAN_SERVICE_ACCOUNT_KEY` (`Authorization: Bearer <key>`)
  - `ATLASSIAN_API_EMAIL` + `ATLASSIAN_API_TOKEN` (`Authorization: Basic <base64(email:api_token)>`, constructed automatically)
- `MCP_CLOUD_ID` for bot/non-interactive flows so the client pins the intended Atlassian site

OpenAI-hosted mode also requires:

- `OPENAI_API_KEY`

Optional environment variables:

- `OPENAI_MODEL` (defaults to `gpt-4.1-mini`)
- `LLM_BASE_URL` (set to `http://localhost:11434/v1` for local Ollama mode)
- `CONFLUENCE_SPACE`
- `MCP_SERVER_URL` (defaults to `https://mcp.atlassian.com/v1/mcp`)
- `MCP_SERVER_NAME` (defaults to `atlassian-rovo`)

If `MCP_CLOUD_ID` is omitted, the app falls back to Atlassian resource discovery when a tool accepts `cloudId`, but setting `MCP_CLOUD_ID` is recommended for deterministic bot behavior.

### Atlassian API-token auth

For bot/CI/non-interactive use cases, prefer Atlassian API-token auth instead of OAuth.

Personal API token example:

```bash
export ATLASSIAN_API_EMAIL="you@example.com"
export ATLASSIAN_API_TOKEN="your-atlassian-api-token"
export MCP_CLOUD_ID="your-atlassian-cloud-id"
```

Service account key example:

```bash
export ATLASSIAN_SERVICE_ACCOUNT_KEY="your-service-account-key"
export MCP_CLOUD_ID="your-atlassian-cloud-id"
```

Backward-compatible manual header override:

```bash
export MCP_AUTH_HEADER="Bearer <your-atlassian-token>"
export MCP_CLOUD_ID="your-atlassian-cloud-id"
```

### Local LLM mode (Ollama)

To run the CLI against a local OpenAI-compatible Ollama endpoint, set:

```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export ATLASSIAN_API_EMAIL="you@example.com"
export ATLASSIAN_API_TOKEN="your-atlassian-api-token"
export MCP_CLOUD_ID="your-atlassian-cloud-id"
```

In local mode:

- `OPENAI_API_KEY` is optional; the app uses a safe internal placeholder if it is unset
- `OPENAI_MODEL` is optional; if unset, the default local chat model is `qwen3:4b-instruct`
- MCP retrieval still uses the Atlassian Rovo MCP server configuration above; only the LLM backend changes

Run a search from the command line:

```bash
python cli_search.py "What is Beam Benefits?"
```

Print the full JSON response instead:

```bash
python cli_search.py --json "What is Beam Benefits?"
```

### Troubleshooting AgenticSearch

If a CLI search fails and you need a safe trace, enable debug mode:

```bash
AGENTIC_SEARCH_DEBUG=1 python cli_search.py --json "What is Beam Benefits?"
```

When enabled, AgenticSearch writes a redacted MCP trace to stderr and adds
`raw_response.debug` to the JSON response. The debug payload includes tool names,
selected strategy, `cloudId`, call argument keys, and response summaries only—no
content body previews.

## Slack endpoints

- `POST /slack/events` — receives Slack Events API payloads via Bolt
- `POST /slack/commands` — receives slash commands via Bolt
- `GET /healthz` — simple health check

## Current behavior

- Uses `slack_bolt` with Flask adapter wiring
- Responds to `/search-docs` slash commands with an `AgenticSearch` answer
- Responds to `app_mention` events with the same `AgenticSearch` answer
- Lets Bolt handle Slack request verification and URL verification flows

## Next steps

- Tune `AgenticSearch.search()` prompts and retrieval behavior for your docs
- Add richer Slack formatting, citations, and async processing as needed