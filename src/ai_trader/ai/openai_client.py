"""Minimal OpenAI Responses API client: strict JSON-schema outputs, usage and cost.

Standard-library HTTP keeps the dependency surface small. The API key is read from
``OpenAISettings`` and only ever placed in the Authorization header; errors carry
status codes, never request bodies or headers.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

API_URL = "https://api.openai.com/v1/responses"
RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})
# 429s that waiting cannot fix: the account needs credit or a higher limit.
BILLING_CODES = frozenset(
    {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}
)

# USD per 1M tokens (standard tier, <=272K context) from developers.openai.com
# pricing, 2026-10-11; web search adds USD 10 per 1,000 calls plus content tokens.
PRICES = {
    "gpt-6-astra": {
        "input": 10.00,
        "cached": 1.00,
        "cache_write": 12.50,
        "output": 50.00,
    },
    "gpt-6.1-sol": {
        "input": 2.00,
        "cached": 0.10,
        "cache_write": 2.50,
        "output": 10.00,
    },
    "gpt-6-sol": {"input": 2.00, "cached": 0.20, "cache_write": 2.50, "output": 10.00},
    "gpt-6-luna": {"input": 0.10, "cached": 0.01, "cache_write": 0.125, "output": 0.50},
    "gpt-5.6-luna": {
        "input": 0.20,
        "cached": 0.02,
        "cache_write": 0.25,
        "output": 1.20,
    },
    "gpt-5-mini": {"input": 0.25, "cached": 0.025, "cache_write": 0.25, "output": 2.00},
    # Anthropic, platform.claude.com/docs/en/about-claude/pricing, 2026-10-13;
    # web search is also USD 10 per 1,000. Haiku 5.5 rate for prompts <=100K.
    "claude-sonnet-5-5": {
        "input": 2.00,
        "cached": 0.10,
        "cache_write": 2.50,
        "output": 10.00,
    },
    "claude-opus-5-5": {
        "input": 4.00,
        "cached": 0.20,
        "cache_write": 5.00,
        "output": 20.00,
    },
    "claude-haiku-5-5": {
        "input": 0.10,
        "cached": 0.01,
        "cache_write": 0.125,
        "output": 0.50,
    },
}
WEB_SEARCH_USD_PER_CALL = 0.010


class OpenAIError(RuntimeError):
    """A failed or unusable model response; carries no secrets or payloads."""


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    web_searches: int

    def cost_usd(self, model: str) -> float:
        price = PRICES.get(model)
        if price is None:
            raise OpenAIError(f"No price configured for model {model}")
        uncached = max(
            self.input_tokens - self.cached_tokens - self.cache_write_tokens, 0
        )
        return (
            uncached * price["input"]
            + self.cached_tokens * price["cached"]
            + self.cache_write_tokens * price["cache_write"]
            + self.output_tokens * price["output"]
        ) / 1_000_000 + self.web_searches * WEB_SEARCH_USD_PER_CALL


@dataclass(frozen=True, slots=True)
class StructuredResponse:
    data: Mapping[str, Any]
    usage: Usage
    model: str
    response_id: str
    latency_seconds: float
    searches: tuple[dict[str, Any], ...] = ()


def web_searches(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """What each web_search_call did: action type, query or page, source URLs."""
    found = []
    for item in payload.get("output") or []:
        if item.get("type") != "web_search_call":
            continue
        action = item.get("action") or {}
        found.append(
            {
                "type": action.get("type"),
                "query": action.get("query") or action.get("queries"),
                "url": action.get("url"),
                "sources": [
                    source.get("url")
                    for source in action.get("sources") or []
                    if isinstance(source, Mapping) and source.get("url")
                ],
            }
        )
    return tuple(found)


def parse_usage(payload: Mapping[str, Any]) -> Usage:
    usage = payload.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or {}
    searches = sum(
        1
        for item in payload.get("output") or []
        if item.get("type") == "web_search_call"
    )
    return Usage(
        input_tokens=int(usage.get("input_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        cache_write_tokens=int(
            details.get("cache_write_tokens")
            or details.get("cache_creation_tokens")
            or 0
        ),
        output_tokens=int(usage.get("output_tokens") or 0),
        reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
        web_searches=searches,
    )


def _charges_cache_writes(model: str) -> bool:
    price = PRICES.get(model)
    return price is not None and price["cache_write"] > price["input"]


def output_json(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The schema-constrained JSON from a completed response, or a clear error."""
    if payload.get("status") != "completed":
        reason = (payload.get("incomplete_details") or {}).get("reason")
        raise OpenAIError(f"Response not completed ({payload.get('status')}, {reason})")
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "refusal":
                raise OpenAIError("Model refused the request")
            if content.get("type") == "output_text":
                try:
                    data = json.loads(content.get("text") or "")
                except json.JSONDecodeError as error:
                    raise OpenAIError("Model output was not valid JSON") from error
                if not isinstance(data, dict):
                    raise OpenAIError("Model output was not a JSON object")
                return data
    raise OpenAIError("Response contained no output text")


class OpenAIResponses:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 45.0,
        attempts: int = 3,
        transport: Callable[[urllib.request.Request, float], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key.strip():
            raise OpenAIError("OpenAI API key is empty")
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
        schema_name: str,
        schema: Mapping[str, Any],
        reasoning_effort: str | None = None,
        max_output_tokens: int = 4000,
        web_search: bool = False,
        require_search: bool = False,
        cache_key: str | None = None,
    ) -> StructuredResponse:
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if web_search:
            body["tools"] = [{"type": "web_search"}]
            body["include"] = ["web_search_call.action.sources"]
            if require_search:
                body["tool_choice"] = "required"
        if cache_key:
            body["prompt_cache_key"] = cache_key
        if _charges_cache_writes(model):
            # GPT-5.6+ implicitly writes the whole prompt to cache at 1.25x input.
            # Every packet is unique and the shared prefix is below the 1,024-token
            # minimum, so explicit mode with no breakpoints turns the writes off.
            body["prompt_cache_options"] = {"mode": "explicit"}
        started = self._clock()
        payload = self._post(body)
        return StructuredResponse(
            data=output_json(payload),
            usage=parse_usage(payload),
            model=str(payload.get("model") or model),
            response_id=str(payload.get("id") or ""),
            latency_seconds=self._clock() - started,
            searches=web_searches(payload),
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
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                raw = self._transport(request, self._timeout)
                return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as error:
                code = _error_code(error)
                failure = f"HTTP {error.code}" + (f" {code}" if code else "")
                if error.code not in RETRYABLE or code in BILLING_CODES:
                    break
                if error.code == 429 and attempt + 1 < self._attempts:
                    self._sleep(_retry_after(error, 5.0 * 2**attempt))
                    continue
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
                failure = type(error).__name__
            if attempt + 1 < self._attempts:
                self._sleep(min(20.0, 2.0 * 2**attempt))
        raise OpenAIError(f"OpenAI request failed: {failure}")


def _urlopen(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _retry_after(error: urllib.error.HTTPError, default: float) -> float:
    """The server's Retry-After (seconds) when given, else ``default``; at most 60."""
    try:
        value = float((error.headers or {}).get("retry-after") or default)
    except (TypeError, ValueError):
        value = default
    return min(60.0, max(0.5, value))


def _error_code(error: urllib.error.HTTPError) -> str:
    """OpenAI's machine-readable error code or type; never the message or request."""
    try:
        detail = json.loads(error.read().decode("utf-8")).get("error") or {}
    except Exception:
        return ""
    value = detail.get("code") or detail.get("type") or ""
    return str(value)[:60] if isinstance(value, str) else ""
