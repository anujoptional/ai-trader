"""Minimal Anthropic Messages API client with the OpenAI client's interface.

Standard-library HTTP, the key only in the ``x-api-key`` header, errors carry status
codes and Anthropic's error type, never request bodies or headers. Returns the same
``StructuredResponse`` as ``OpenAIResponses.structured`` so ``OpenAIDecider`` drives
either provider: a JSON-schema answer (``output_config.format``), effort
(``output_config.effort``), server-side web search with each query and source URL,
and usage priced from ``PRICES`` (web search USD 10 per 1,000).
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from ai_trader.ai.openai_client import (
    RETRYABLE,
    OpenAIError,
    StructuredResponse,
    Usage,
    _retry_after,
    _urlopen,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")
WEB_SEARCH_TOOL = "web_search_20250305"
MAX_SEARCHES = 5
SEARCH_FIRST = "Use web_search at least once before you answer."
# Long server-tool turns can pause; the request is resent to let the model finish.
MAX_CONTINUATIONS = 3
# Anthropic's structured outputs reject some JSON-schema keywords; the answer is
# validated (and confidence clamped) in code anyway.
UNSUPPORTED_SCHEMA_KEYS = frozenset({"minimum", "maximum"})


class AnthropicError(OpenAIError):
    """A failed or unusable Claude response; carries no secrets or payloads."""


def claude_schema(schema: Any) -> Any:
    if isinstance(schema, Mapping):
        return {
            key: claude_schema(value)
            for key, value in schema.items()
            if key not in UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [claude_schema(value) for value in schema]
    return schema


def parse_usage(payloads: list[Mapping[str, Any]]) -> Usage:
    total = dict.fromkeys(("input", "cached", "written", "output", "searches"), 0)
    for payload in payloads:
        usage = payload.get("usage") or {}
        cached = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        total["input"] += int(usage.get("input_tokens") or 0) + cached + written
        total["cached"] += cached
        total["written"] += written
        total["output"] += int(usage.get("output_tokens") or 0)
        server = usage.get("server_tool_use") or {}
        total["searches"] += int(server.get("web_search_requests") or 0)
    return Usage(
        input_tokens=total["input"],
        cached_tokens=total["cached"],
        cache_write_tokens=total["written"],
        output_tokens=total["output"],
        reasoning_tokens=0,
        web_searches=total["searches"],
    )


def web_searches(content: list[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Each web search: its query and the result URLs, in order."""
    queries: dict[str, dict[str, Any]] = {}
    for block in content:
        if block.get("type") == "server_tool_use" and block.get("name") == "web_search":
            queries[str(block.get("id"))] = {
                "type": "search",
                "query": (block.get("input") or {}).get("query"),
                "url": None,
                "sources": [],
            }
        elif block.get("type") == "web_search_tool_result":
            entry = queries.get(str(block.get("tool_use_id")))
            results = block.get("content")
            if entry is not None and isinstance(results, list):
                entry["sources"] = [
                    r.get("url")
                    for r in results
                    if isinstance(r, Mapping) and r.get("url")
                ]
    return tuple(queries.values())


def output_json(content: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The JSON answer: text after the last search, which citations may split."""
    tail: list[str] = []
    for block in content:
        if block.get("type") == "text":
            tail.append(block.get("text") or "")
        elif block.get("type") in ("server_tool_use", "web_search_tool_result"):
            tail = []
    for text in ("".join(tail), *reversed(tail)):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise AnthropicError("Claude output contained no JSON object")


class AnthropicMessages:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 90.0,
        attempts: int = 3,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key.strip():
            raise AnthropicError("Anthropic API key is empty")
        self._key = api_key
        self._timeout = timeout_seconds
        self._attempts = attempts
        self._transport = transport or _urlopen
        self._sleep = sleep
        self._clock = clock

    def structured(
        self,
        *,
        model: str,
        instructions: str,
        user_input: str,
        schema: Mapping[str, Any],
        reasoning_effort: str | None = None,
        max_output_tokens: int = 4000,
        web_search: bool = False,
        require_search: bool = False,
        **_: Any,
    ) -> StructuredResponse:
        output: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": claude_schema(schema)}
        }
        if reasoning_effort:
            output["effort"] = reasoning_effort
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "system": instructions,
            "messages": [{"role": "user", "content": user_input}],
            "output_config": output,
        }
        if web_search:
            body["tools"] = [
                {
                    "type": WEB_SEARCH_TOOL,
                    "name": "web_search",
                    "max_uses": MAX_SEARCHES,
                }
            ]
            if require_search:
                # Claude 5.5 rejects a forced tool_choice ("any"/"tool"): ask instead.
                body["system"] = f"{instructions}\n\n{SEARCH_FIRST}"
        started = self._clock()
        payloads, content = [], []
        for _ in range(MAX_CONTINUATIONS + 1):
            payload = self._post(body)
            payloads.append(payload)
            content += list(payload.get("content") or [])
            if payload.get("stop_reason") != "pause_turn":
                break
            body = body | {
                "messages": body["messages"]
                + [{"role": "assistant", "content": payload.get("content") or []}]
            }
        last = payloads[-1]
        if last.get("stop_reason") not in ("end_turn", "stop_sequence"):
            raise AnthropicError(f"Claude stopped early ({last.get('stop_reason')})")
        return StructuredResponse(
            data=output_json(content),
            usage=parse_usage(payloads),
            model=str(last.get("model") or model),
            response_id=str(last.get("id") or ""),
            latency_seconds=self._clock() - started,
            searches=web_searches(content),
        )

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        data = json.dumps(body).encode("utf-8")
        failure = "no attempt"
        for attempt in range(self._attempts):
            request = urllib.request.Request(
                API_URL,
                data=data,
                method="POST",
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": API_VERSION,
                    "content-type": "application/json",
                },
            )
            try:
                raw = self._transport(request, self._timeout)
                return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as error:
                kind = _error_type(error)
                failure = f"HTTP {error.code}" + (f" {kind}" if kind else "")
                if error.code not in RETRYABLE and error.code != 529:
                    break
                if error.code in (429, 529) and attempt + 1 < self._attempts:
                    self._sleep(_retry_after(error, 5.0 * 2**attempt))
                    continue
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
                failure = type(error).__name__
            if attempt + 1 < self._attempts:
                self._sleep(min(20.0, 2.0 * 2**attempt))
        raise AnthropicError(f"Anthropic request failed: {failure}")


def _error_type(error: urllib.error.HTTPError) -> str:
    """Anthropic's machine-readable error type; never the message or request."""
    try:
        detail = json.loads(error.read().decode("utf-8")).get("error") or {}
    except Exception:
        return ""
    value = detail.get("type") or ""
    return str(value)[:60] if isinstance(value, str) else ""
