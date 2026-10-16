"""How the agent talks to each provider: Anthropic Messages (Claude) and OpenAI
Responses (GPT). The turn loop is the same for both; an API object turns the
conversation into a request and a reply into tool calls, text and cost."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ai_trader.ai import anthropic_client, openai_client
from ai_trader.ai.anthropic_client import (
    CLAUDE_EFFORTS,
    WEB_SEARCH_TOOL,
    AnthropicMessages,
)
from ai_trader.ai.decision import provider
from ai_trader.ai.openai_client import PRICES, OpenAIResponses
from ai_trader.config import (
    REASONING_EFFORTS,
    load_anthropic_settings,
    load_openai_settings,
)

MAX_SEARCHES_PER_REQUEST = 5
EFFORTS = tuple(dict.fromkeys((*REASONING_EFFORTS, *CLAUDE_EFFORTS)))


def model_problem(model: str, effort: str) -> str | None:
    """Why ``model`` with ``effort`` cannot run the agent, or None."""
    if model not in PRICES:
        return f"model must be one of {', '.join(PRICES)}"
    efforts = CLAUDE_EFFORTS if provider(model) == "anthropic" else REASONING_EFFORTS
    if effort not in efforts:
        return f"effort for {model} must be one of {', '.join(efforts)}"
    return None


def make_client(
    model: str, timeout_seconds: float
) -> AnthropicMessages | OpenAIResponses:
    """The provider's client for ``model``, with its key from .env."""
    if provider(model) == "anthropic":
        key = load_anthropic_settings().api_key.get_secret_value()
        return AnthropicMessages(key, timeout_seconds=timeout_seconds)
    key = load_openai_settings().api_key.get_secret_value()
    return OpenAIResponses(key, timeout_seconds=timeout_seconds)


@dataclass(frozen=True)
class Call:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Reply:
    items: list[dict[str, Any]]
    calls: list[Call]
    text: str
    cost: float
    usage: Any
    searches: list[str] = field(default_factory=list)
    paused: bool = False
    truncated: bool = False


class ClaudeApi:
    """Messages API: tool_use / tool_result blocks, adaptive thinking, one moving
    cache breakpoint (the last block of the last user message)."""

    def __init__(
        self,
        model: str,
        effort: str,
        tools: Sequence[Mapping[str, Any]],
        *,
        thinking: bool = True,
        max_tokens: int = 16_000,
        web_search: bool = True,
        search_domains: tuple[str, ...] = (),
    ) -> None:
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.tools: list[Any] = list(tools)
        if web_search:
            search: dict[str, Any] = {
                "type": WEB_SEARCH_TOOL,
                "name": "web_search",
                "max_uses": MAX_SEARCHES_PER_REQUEST,
            }
            if search_domains:
                search["allowed_domains"] = list(search_domains)
            self.tools.append(search)

    def user(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": [{"type": "text", "text": text}]}

    def assistant(self, text: str) -> dict[str, Any]:
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    def request(self, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        for message in messages:
            if message["role"] == "user" and not isinstance(message["content"], str):
                for block in message["content"]:
                    block.pop("cache_control", None)
        last = messages[-1]
        if last["role"] == "user" and not isinstance(last["content"], str):
            last["content"][-1]["cache_control"] = {"type": "ephemeral"}
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "tools": self.tools,
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if self.thinking:
            body["thinking"] = {"type": "adaptive"}
        return body

    def parse(self, payload: Mapping[str, Any]) -> Reply:
        content = list(payload.get("content") or [])
        stop = payload.get("stop_reason")
        return Reply(
            items=[{"role": "assistant", "content": content}],
            calls=[
                Call(str(b.get("id")), str(b.get("name")), dict(b.get("input") or {}))
                for b in content
                if b.get("type") == "tool_use"
            ],
            text=" ".join(
                (b.get("text") or "").strip()
                for b in content
                if b.get("type") == "text" and (b.get("text") or "").strip()
            ),
            cost=anthropic_client.parse_usage([payload]).cost_usd(self.model),
            usage=payload.get("usage"),
            searches=[
                str(s["query"])
                for s in anthropic_client.web_searches(content)
                if s.get("query")
            ],
            paused=stop == "pause_turn",
            truncated=stop == "max_tokens",
        )

    def results(self, outputs: list[tuple[Call, str, bool]]) -> list[dict[str, Any]]:
        blocks = []
        for call, text, error in outputs:
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": text,
            }
            if error:
                block["is_error"] = True
            blocks.append(block)
        return [{"role": "user", "content": blocks}]


class GptApi:
    """Responses API: function calls and their outputs, nothing stored at OpenAI
    (reasoning travels encrypted with the conversation), automatic prompt caching."""

    def __init__(
        self,
        model: str,
        effort: str,
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int = 16_000,
        web_search: bool = True,
        cache_key: str = "autonomous-agent",
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.cache_key = cache_key
        self.web_search = web_search
        self.tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
                "strict": False,
            }
            for t in tools
        ]
        if web_search:
            self.tools.append({"type": "web_search"})

    def user(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def assistant(self, text: str) -> dict[str, Any]:
        return {"role": "assistant", "content": text}

    def request(self, system: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        include = ["reasoning.encrypted_content"]
        if self.web_search:
            include.append("web_search_call.action.sources")
        return {
            "model": self.model,
            "instructions": system,
            "input": messages,
            "tools": self.tools,
            "reasoning": {"effort": self.effort},
            "max_output_tokens": self.max_tokens,
            "store": False,
            "include": include,
            "prompt_cache_key": self.cache_key,
        }

    def parse(self, payload: Mapping[str, Any]) -> Reply:
        output = [dict(item) for item in payload.get("output") or []]
        calls = []
        for item in output:
            if item.get("type") != "function_call":
                continue
            try:
                args = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"unreadable_arguments": str(item.get("arguments"))[:500]}
            calls.append(
                Call(
                    str(item.get("call_id")),
                    str(item.get("name")),
                    args if isinstance(args, dict) else {},
                )
            )
        texts = [
            (part.get("text") or "").strip()
            for item in output
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        ]
        incomplete = (payload.get("incomplete_details") or {}).get("reason")
        return Reply(
            items=output,
            calls=calls,
            text=" ".join(t for t in texts if t),
            cost=openai_client.parse_usage(payload).cost_usd(self.model),
            usage=payload.get("usage"),
            searches=[
                str(s["query"])
                for s in openai_client.web_searches(payload)
                if s.get("query")
            ],
            truncated=incomplete == "max_output_tokens",
        )

    def results(self, outputs: list[tuple[Call, str, bool]]) -> list[dict[str, Any]]:
        return [
            {"type": "function_call_output", "call_id": call.id, "output": text}
            for call, text, _ in outputs
        ]
