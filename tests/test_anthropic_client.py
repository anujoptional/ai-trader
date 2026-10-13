import io
import json
import urllib.error

import pytest

from ai_trader.ai.anthropic_client import (
    SEARCH_AGAIN,
    AnthropicError,
    AnthropicMessages,
)
from ai_trader.ai.decision import DELIVERY_V4, OpenAIDecider
from ai_trader.cli.run_paper import shadow_spec
from ai_trader.config import ConfigurationError, load_anthropic_settings
from tests.test_ai_decision import request

ANSWER = {"decisions": [{"symbol": "TCS", "p_up_first": 1.4, "verdict": "BUY"}]}
UNSEARCHED = {"decisions": [{"symbol": "TCS", "p_up_first": 0.2, "verdict": "NO_BUY"}]}


def message(content: list[dict], stop: str = "end_turn", searches: int = 1) -> bytes:
    return json.dumps(
        {
            "id": "msg_1",
            "model": "claude-sonnet-5-5",
            "stop_reason": stop,
            "content": content,
            "usage": {
                "input_tokens": 10_000,
                "cache_read_input_tokens": 2_000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 200,
                "server_tool_use": {"web_search_requests": searches},
            },
        }
    ).encode()


SEARCHED = [
    {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": {"query": "TCS share news"},
    },
    {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {"type": "web_search_result", "url": "https://example.com/a"},
            {"type": "web_search_result", "url": "https://example.com/b"},
        ],
    },
    # Citations can split the answer across text blocks.
    {"type": "text", "text": json.dumps(ANSWER)[:20]},
    {"type": "text", "text": json.dumps(ANSWER)[20:]},
]


def test_claude_request_searches_and_prices_like_openai() -> None:
    sent = []

    def transport(req, timeout):
        sent.append((dict(req.header_items()), json.loads(req.data)))
        return message(SEARCHED)

    decider = OpenAIDecider(
        AnthropicMessages("sk-ant-test", transport=transport),
        "claude-sonnet-5-5",
        reasoning_effort="low",
        web_search=True,
        require_search=True,
        policy=DELIVERY_V4,
    )
    result = decider.decide(request("TCS"))
    headers, body = sent[0]
    assert len(sent) == 1  # it searched, so it is not asked again
    assert headers["X-api-key"] == "sk-ant-test"
    assert headers["Anthropic-version"] == "2023-06-01"
    assert "tool_choice" not in body  # Claude 5.5 rejects a forced tool choice
    assert body["system"].startswith(DELIVERY_V4.instructions)
    assert body["system"].endswith("Use web_search at least once before you answer.")
    assert body["tools"][0]["name"] == "web_search"
    assert body["output_config"]["effort"] == "low"
    schema = body["output_config"]["format"]["schema"]
    assert "minimum" not in json.dumps(schema)
    fields = schema["properties"]["decisions"]["items"]["properties"]
    assert list(fields) == ["symbol", "p_up_first", "verdict", "reason"]
    verdict = result.verdicts[0]
    assert (verdict.verdict, verdict.confidence) == ("BUY", 1.0)  # clamped in code
    assert verdict.source == "anthropic:claude-sonnet-5-5"
    searches = json.loads(result.response_json)["web_searches"]
    assert searches[0]["query"] == "TCS share news"
    assert searches[0]["sources"] == ["https://example.com/a", "https://example.com/b"]
    # 10,000 uncached at $2 + 2,000 cached at $0.10 + 200 out at $10, + 1 search.
    assert result.cost_usd == pytest.approx(0.02 + 0.0002 + 0.002 + 0.01)
    assert result.usage.input_tokens == 12_000 and result.usage.web_searches == 1


def test_claude_paused_turn_is_resumed_and_usage_summed() -> None:
    replies = [
        message(SEARCHED[:2], stop="pause_turn"),
        message(SEARCHED[2:], searches=0),
    ]
    sent = []

    def transport(req, timeout):
        sent.append(json.loads(req.data))
        return replies[len(sent) - 1]

    response = AnthropicMessages("k", transport=transport).structured(
        model="claude-sonnet-5-5",
        instructions="i",
        user_input="u",
        schema={},
        web_search=True,
    )
    assert sent[1]["messages"][-1] == {"role": "assistant", "content": SEARCHED[:2]}
    assert response.data == ANSWER and response.usage.output_tokens == 400
    assert response.searches[0]["sources"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def structured(transport, *, require_search: bool = True):
    return AnthropicMessages("k", transport=transport, sleep=lambda s: None).structured(
        model="claude-sonnet-5-5",
        instructions="i",
        user_input="u",
        schema={},
        web_search=True,
        require_search=require_search,
    )


def replying(*replies: bytes):
    sent: list[dict] = []

    def transport(req, timeout):
        sent.append(json.loads(req.data))
        return replies[len(sent) - 1]

    return transport, sent


UNSEARCHED_TEXT = [{"type": "text", "text": json.dumps(UNSEARCHED)}]


def test_claude_is_asked_again_when_it_answers_without_searching() -> None:
    transport, sent = replying(message(UNSEARCHED_TEXT, searches=0), message(SEARCHED))
    response = structured(transport)
    assert len(sent) == 2
    assert sent[1]["messages"] == [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": UNSEARCHED_TEXT},
        {"role": "user", "content": SEARCH_AGAIN},
    ]
    assert response.data == ANSWER  # the searched answer replaces the first
    assert [s["query"] for s in response.searches] == ["TCS share news"]
    assert response.usage.input_tokens == 24_000  # both requests are billed
    assert response.usage.web_searches == 1


def test_claude_retry_happens_once_and_never_loses_the_answer() -> None:
    second = [{"type": "text", "text": json.dumps(ANSWER)}]
    transport, sent = replying(
        message(UNSEARCHED_TEXT, searches=0), message(second, searches=0)
    )
    response = structured(transport)
    assert len(sent) == 2 and response.data == ANSWER and response.searches == ()

    calls: list[int] = []

    def retry_fails(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            return message(UNSEARCHED_TEXT, searches=0)
        raise urllib.error.URLError("down")

    response = structured(retry_fails)
    assert response.data == UNSEARCHED and response.usage.input_tokens == 12_000

    transport, sent = replying(message(UNSEARCHED_TEXT, searches=0))
    assert structured(transport, require_search=False).data == UNSEARCHED
    assert len(sent) == 1


def test_claude_errors_name_the_type_and_retry_only_overload() -> None:
    sleeps: list[float] = []

    def failing(code: int, kind: str):
        def transport(req, timeout):
            body = json.dumps({"error": {"type": kind, "message": "secret detail"}})
            raise urllib.error.HTTPError(
                req.full_url, code, "x", {}, io.BytesIO(body.encode())
            )

        return transport

    def call(transport) -> None:
        AnthropicMessages("k", transport=transport, sleep=sleeps.append).structured(
            model="claude-sonnet-5-5", instructions="i", user_input="u", schema={}
        )

    with pytest.raises(AnthropicError, match="HTTP 400 invalid_request_error$"):
        call(failing(400, "invalid_request_error"))
    assert sleeps == []
    with pytest.raises(AnthropicError, match="HTTP 529 overloaded_error"):
        call(failing(529, "overloaded_error"))
    assert len(sleeps) == 2
    with pytest.raises(AnthropicError, match="stopped early"):
        call(lambda r, t: message([{"type": "text", "text": "{"}], stop="max_tokens"))


def test_claude_failure_is_no_buy() -> None:
    def broken(req, timeout):
        raise urllib.error.URLError("down")

    client = AnthropicMessages("k", transport=broken, sleep=lambda s: None)
    result = OpenAIDecider(client, "claude-sonnet-5-5", policy=DELIVERY_V4).decide(
        request("TCS")
    )
    assert result.verdicts[0].verdict == "NO_BUY" and result.status.startswith("error")


def test_shadow_accepts_claude_with_claude_efforts() -> None:
    assert shadow_spec("claude-sonnet-5-5:low") == ("claude-sonnet-5-5", "low")
    assert shadow_spec("gpt-6.1-sol:low") == ("gpt-6.1-sol", "low")
    with pytest.raises(Exception, match="EFFORT"):
        shadow_spec("claude-sonnet-5-5:minimal")


def test_anthropic_settings_need_the_key_and_share_the_budget() -> None:
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
        load_anthropic_settings({})
    settings = load_anthropic_settings(
        {"ANTHROPIC_API_KEY": "sk-ant-x", "AI_DAILY_BUDGET_USD": "4"}
    )
    assert settings.daily_budget_usd == 4 and settings.timeout_seconds == 90
    assert "sk-ant-x" not in repr(settings)
