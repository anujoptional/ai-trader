# AI Trader Project Instructions

## Project goal

Build an AI-assisted intraday trading research and execution system.

Sole active broker and historical-data source: Zerodha Kite.
Groww access and derived local datasets are retired.

## Start here (current state)

A live **paper** trading system runs every weekday (Windows scheduled task
"AI Trader paper run" at 08:30 IST, manual Kite login, `ai_trader.cli.run_paper`
with GPT-6 Luna as the main AI and GPT-6.1 Sol and Claude Sonnet 5.5 as shadow
AIs). Before any work, read `docs/handover.txt` section "CURRENT STATE": daily
operation, setup, what git tracks, live results, research summary, open issues.
Details: `docs/AI_LAYER.md`. Do not break the daily run: keep the gates green
(`ruff format --check .`, `ruff check .`, `pytest`) and ask before changing the
prompt, AI models or exits.

## Architecture

Before making architectural or cross-module changes, read
`docs/ARCHITECTURE.md`. It is the canonical description of the intended system.

## Safety rules

1. Shadow/paper trading is the default mode.
2. Never place a live trade unless live trading has been explicitly enabled.
3. Never hard-code API keys, secrets, passwords, TOTP secrets, or access tokens.
4. Secrets must come from environment variables or an approved local secret store.
5. Never print secrets in logs.
6. Never commit the .env file.
7. GPT/LLM output must never bypass deterministic risk controls.
8. Position sizing must be determined by deterministic code, not by the LLM.
9. Every trading decision should eventually be journaled.
10. Broker-specific logic must remain behind the read-only broker abstraction. Only Zerodha Kite is active; do not reintroduce Groww or mix provider datasets.
11. Codex must never read, print, inspect, cat, grep, or otherwise expose the
    contents of `.env` or any other credential file. Codex may write code that
    loads environment variables, but it must never inspect their values.

## Development approach

Keep the initial architecture simple.

Do not add Docker, Kubernetes, Redis, MCP, databases other than SQLite, or large agent frameworks unless there is a demonstrated requirement.

Prefer:
- Python 3.12
- typed Python
- small modules
- tests
- structured logging
- deterministic risk logic

Do not implement live order placement during the initial development phase.
