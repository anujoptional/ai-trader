# AI Trader Project Instructions

## Project goal

Build an AI-assisted intraday trading research and execution system.

Initial broker: Groww.
Possible future broker: Zerodha Kite.

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
10. Broker-specific logic must remain behind a broker abstraction so Groww can later be replaced by Kite.
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
