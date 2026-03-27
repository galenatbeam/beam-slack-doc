# Lightweight Python Slack Bot

This repository contains a minimal scaffold for a Slack bot with three components:

1. `slack_bot.py` — a lightweight Slack Bolt bot served through Flask.
2. `agentic_search.py` — an OpenAI Agents SDK search component that searches Atlassian Rovo / Confluence first, then optionally falls back to GitHub MCP readonly tools for README/code context.
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
export SLACK_APP_TOKEN="xapp-your-app-level-token"  # optional; enables Socket Mode
export OPENAI_API_KEY="your-openai-key"
export ATLASSIAN_EMAIL="you@example.com"
export CONFLUENCE_MCP_API_KEY="your-confluence-mcp-api-key"
export ATLASSIAN_CLOUD_ID="your-atlassian-cloud-id"
export CONFLUENCE_SPACE="YOURSPACE"
export MCP_SERVER_NAME="atlassian-rovo"
export GITHUB_PAT="your-github-pat"              # optional GitHub readonly fallback
export GITHUB_ORG="beamtech"                     # optional; defaults to beamtech
python slack_bot.py
```

The app will start on `http://localhost:3000` by default when `SLACK_APP_TOKEN` is unset.

You can also put these values in a `.env` file; both `slack_bot.py` and `cli_search.py`
load it automatically. See `.env.example` for a ready-to-copy template.

## Slack bot run modes

### Socket Mode (recommended for local MVP testing)

Set `SLACK_APP_TOKEN` to run the existing Bolt app via Socket Mode instead of receiving events over HTTP.

1. In your Slack app settings, enable **Socket Mode**.
2. Create an **app-level token** with the `connections:write` scope. The token value starts with `xapp-`.
3. Export `SLACK_APP_TOKEN` alongside your existing `SLACK_BOT_TOKEN` and search/MCP env vars.
4. Run:

```bash
python slack_bot.py
```

Notes:

- Socket Mode does **not** require ngrok for local mention testing.
- The existing `app_mention` flow is unchanged: the bot posts a thread placeholder, runs search asynchronously, then updates that placeholder.

## Slack app configuration

Required bot token scopes:

- `app_mentions:read`
- `channels:history`
- `chat:write`
- `commands`

Required event subscriptions:

- `app_mention`
- `message.channels`

If you use Socket Mode locally, your app-level token also needs `connections:write`.

### HTTP mode (fallback)

If `SLACK_APP_TOKEN` is unset, `slack_bot.py` keeps the existing Flask HTTP mode:

- `POST /slack/events`
- `POST /slack/commands`
- `GET /healthz`

## CLI usage

Required environment variables:

- `ATLASSIAN_EMAIL` — Atlassian account email used for MCP Basic auth
- `CONFLUENCE_MCP_API_KEY` — Atlassian personal API token used with `ATLASSIAN_EMAIL`
- `ATLASSIAN_CLOUD_ID` — pins the Atlassian tenant/site when the shared Rovo `search` tool accepts `cloudId`

Default agentic mode also requires:

- `OPENAI_API_KEY`

Optional environment variables:

- `OPENAI_MODEL` (defaults to `gpt-4.1-mini`)
- `LLM_BASE_URL` (set to `http://localhost:11434/v1` for local Ollama mode)
- `CONFLUENCE_SPACE`
- `MCP_SERVER_URL` (defaults to `https://mcp.atlassian.com/v1/mcp`)
- `MCP_SERVER_NAME` (defaults to `atlassian-rovo`)
- `GITHUB_PAT` (enables GitHub MCP readonly fallback)
- `GITHUB_ORG` (defaults to `beamtech`; URLs like `https://github.com/beamtech` are normalized)
- `GITHUB_MCP_SERVER_URL` (defaults to `https://api.githubcopilot.com/mcp/readonly`)
- `GITHUB_MCP_SERVER_NAME` (defaults to `github`)

### Optional GitHub MCP readonly fallback

When `GITHUB_PAT` is set, AgenticSearch can also use GitHub's remote MCP server as a
secondary source for `github.com/beamtech`.

- Atlassian/Confluence is always searched first.
- GitHub is only used when the Atlassian results are incomplete or point to code-level details.
- GitHub retrieval is scoped to the configured org and should prioritize README files before source code.

Optional GitHub env setup:

```bash
export GITHUB_PAT="your-github-pat"
export GITHUB_ORG="beamtech"
export GITHUB_MCP_SERVER_URL="https://api.githubcopilot.com/mcp/readonly"  # optional override
export GITHUB_MCP_SERVER_NAME="github"                                      # optional override
```

### Atlassian MCP auth

This repo now uses a single MCP auth shape for bot/CI/non-interactive use cases.

```bash
export ATLASSIAN_EMAIL="you@example.com"
export CONFLUENCE_MCP_API_KEY="your-confluence-mcp-api-key"
export ATLASSIAN_CLOUD_ID="your-atlassian-cloud-id"
export MCP_SERVER_URL="https://mcp.atlassian.com/v1/mcp"  # optional override
```

The MCP server receives:

`Authorization: Basic <base64(ATLASSIAN_EMAIL:CONFLUENCE_MCP_API_KEY)>`

### Local LLM mode (Ollama)

To run the CLI against a local OpenAI-compatible Ollama endpoint, set:

```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export ATLASSIAN_EMAIL="you@example.com"
export CONFLUENCE_MCP_API_KEY="your-confluence-mcp-api-key"
export ATLASSIAN_CLOUD_ID="your-atlassian-cloud-id"
```

In local mode:

- `OPENAI_API_KEY` is optional; the app uses a safe internal placeholder if it is unset
- `OPENAI_MODEL` is optional; if unset, the default local chat model is `qwen3:4b-instruct`
- MCP retrieval still uses the Atlassian Rovo MCP server configuration above; only the LLM backend changes
- Default behavior is discovery-first: `list_tools()` -> agent tool-calls search/fetch tools -> answer with citations
- If `list_tools()` fails (including MCP `-32601`), AgenticSearch returns a Confluence authentication error instead of falling back to a static tool allowlist

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

When enabled, AgenticSearch writes an MCP debug summary to stderr and adds
`raw_response.debug` to the JSON response. The debug payload includes discovered
tool names from `list_tools()`, called tool names, call argument keys, and
response summaries. Auth headers, raw Atlassian email/token values, encoded
Basic credentials, and content body previews are not logged.

## Slack endpoints

- `POST /slack/events` — receives Slack Events API payloads via Bolt
- `POST /slack/commands` — receives slash commands via Bolt
- `GET /healthz` — simple health check

## Current behavior

- Uses `slack_bolt` with Flask adapter wiring
- Responds to `/search-docs` slash commands with an `AgenticSearch` answer
- Responds to `app_mention` events with the same `AgenticSearch` answer
- Responds to top-level `message.channels` events containing `?` with the same async placeholder → threaded reply flow
- Ignores `message.channels` thread replies for the `?` trigger to avoid polluting existing threads
- Searches Atlassian first and optionally uses GitHub MCP readonly tools as a secondary source for Beamtech README/code details
- Lets Bolt handle Slack request verification and URL verification flows

## Next steps

- Tune `AgenticSearch.search()` prompts and retrieval behavior for your docs
- Add richer Slack formatting, citations, and async processing as needed