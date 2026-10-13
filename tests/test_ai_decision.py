import io
import json
import sqlite3
import urllib.error
from datetime import date, datetime

import pytest

from ai_trader.ai.decision import (
    DELIVERY,
    DELIVERY_V3,
    DELIVERY_V4,
    Candidate,
    DecisionRequest,
    OpenAIDecider,
    RuleDecider,
    validate,
)
from ai_trader.ai.journal import SCHEMA, DecisionJournal
from ai_trader.ai.openai_client import OpenAIError, OpenAIResponses, parse_usage
from ai_trader.ai.packet import delivery_block
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import ConfigurationError, load_openai_settings

AS_OF = datetime(2026, 10, 12, 10, 5, tzinfo=INDIA_TIMEZONE)


def completed(decisions: list[dict], usage: dict | None = None) -> bytes:
    return json.dumps(
        {
            "id": "resp_1",
            "model": "gpt-6-luna",
            "status": "completed",
            "output": [
                {"type": "web_search_call"},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": json.dumps({"decisions": decisions}),
                        }
                    ],
                },
            ],
            "usage": usage
            or {
                "input_tokens": 2000,
                "input_tokens_details": {"cached_tokens": 1500},
                "output_tokens": 300,
                "output_tokens_details": {"reasoning_tokens": 200},
            },
        }
    ).encode()


def request(*symbols: str) -> DecisionRequest:
    return DecisionRequest(
        AS_OF,
        {"nifty_day": 0.004},
        tuple(Candidate(s, {"movement_probability": 0.4}) for s in symbols),
    )


def test_client_sends_strict_schema_and_prices_usage() -> None:
    sent = []

    def transport(req, timeout):
        sent.append(req)
        return completed(
            [{"symbol": "TCS", "verdict": "LONG", "confidence": 0.6, "reason": "x"}]
        )

    client = OpenAIResponses("sk-test", transport=transport, sleep=lambda s: None)
    result = client.structured(
        model="gpt-6-luna",
        instructions="i",
        user_input="{}",
        schema_name="s",
        schema={"type": "object"},
        reasoning_effort="low",
        web_search=True,
    )
    body = json.loads(sent[0].data)
    assert body["text"]["format"]["strict"] is True and body["store"] is False
    assert body["tools"] == [{"type": "web_search"}]
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    assert "sk-test" not in sent[0].data.decode()
    assert result.usage.cached_tokens == 1500 and result.usage.web_searches == 1
    expected = (500 * 0.10 + 1500 * 0.01 + 300 * 0.50) / 1e6 + 0.01
    assert result.usage.cost_usd("gpt-6-luna") == pytest.approx(expected)


def test_client_leaves_caching_alone_where_writes_are_free() -> None:
    sent = []

    def transport(req, timeout):
        sent.append(json.loads(req.data))
        return completed([])

    OpenAIResponses("k", transport=transport).structured(
        model="gpt-5-mini",
        instructions="i",
        user_input="{}",
        schema_name="s",
        schema={"type": "object"},
    )
    assert "prompt_cache_options" not in sent[0]


def test_client_retries_rate_limits_but_not_bad_requests() -> None:
    calls = []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("u", 429, "rate", {}, None)
        return completed([])

    sleeps = []
    OpenAIResponses("k", transport=flaky, sleep=sleeps.append).structured(
        model="gpt-6-luna", instructions="", user_input="", schema_name="s", schema={}
    )
    assert len(calls) == 2 and sleeps == [5.0]  # rate limits back off longer

    def no_credit(req, timeout):
        calls.append(1)
        body = io.BytesIO(b'{"error": {"code": "credit_balance_exhausted"}}')
        raise urllib.error.HTTPError("u", 429, "quota", {}, body)

    calls.clear()
    with pytest.raises(OpenAIError, match="HTTP 429 credit_balance_exhausted"):
        OpenAIResponses("k", transport=no_credit, sleep=sleeps.append).structured(
            model="gpt-6-luna",
            instructions="",
            user_input="",
            schema_name="s",
            schema={},
        )
    assert len(calls) == 1  # waiting cannot fix an empty balance

    def dropped(req, timeout):
        calls.append(1)
        if len(calls) == 2:
            raise ConnectionResetError("peer closed")
        return completed([])

    OpenAIResponses("k", transport=dropped, sleep=sleeps.append).structured(
        model="gpt-6-luna", instructions="", user_input="", schema_name="s", schema={}
    )
    assert len(calls) == 3

    def bad(req, timeout):
        raise urllib.error.HTTPError("u", 400, "bad", {}, None)

    with pytest.raises(OpenAIError, match="HTTP 400"):
        OpenAIResponses("k", transport=bad, sleep=sleeps.append).structured(
            model="gpt-6-luna",
            instructions="",
            user_input="",
            schema_name="s",
            schema={},
        )


def test_incomplete_or_refused_responses_are_errors() -> None:
    assert parse_usage({}).input_tokens == 0
    for payload in (
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
        {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal"}]}],
        },
    ):
        client = OpenAIResponses(
            "k", transport=lambda r, t, p=payload: json.dumps(p).encode()
        )
        with pytest.raises(OpenAIError):
            client.structured(
                model="gpt-6-luna",
                instructions="",
                user_input="",
                schema_name="s",
                schema={},
            )


def test_validation_fills_missing_and_rejects_unknown() -> None:
    verdicts = validate(
        request("TCS", "INFY", "SBIN"),
        {
            "decisions": [
                {"symbol": "TCS", "verdict": "SHORT", "confidence": 3, "reason": "r"},
                {"symbol": "INFY", "verdict": "BUY", "confidence": 0.9, "reason": "r"},
                {"symbol": "HACK", "verdict": "LONG", "confidence": 0.9, "reason": "r"},
            ]
        },
        "openai:test",
    )
    assert [(v.symbol, v.verdict, v.confidence) for v in verdicts] == [
        ("TCS", "SHORT", 1.0),
        ("INFY", "NO_TRADE", 0.5),
        ("SBIN", "NO_TRADE", 0.5),
    ]


def test_delivery_policy_only_buys_or_skips() -> None:
    verdicts = validate(
        request("TCS", "INFY", "SBIN"),
        {
            "decisions": [
                {"symbol": "TCS", "verdict": "BUY", "confidence": 0.7, "reason": "r"},
                {"symbol": "INFY", "verdict": "LONG", "confidence": 0.9, "reason": "r"},
            ]
        },
        "openai:test",
        DELIVERY,
    )
    assert [(v.symbol, v.verdict) for v in verdicts] == [
        ("TCS", "BUY"),
        ("INFY", "NO_BUY"),
        ("SBIN", "NO_BUY"),
    ]
    enum = DELIVERY.schema["properties"]["decisions"]["items"]["properties"]
    assert enum["verdict"]["enum"] == ["BUY", "NO_BUY"]


def test_delivery_decider_sends_its_policy_and_journals_its_version(tmp_path) -> None:
    sent = []

    def transport(req, timeout):
        sent.append(json.loads(req.data))
        return completed(
            [{"symbol": "TCS", "verdict": "BUY", "confidence": 0.6, "reason": "x"}]
        )

    client = OpenAIResponses("k", transport=transport)
    result = OpenAIDecider(client, "gpt-6-luna", policy=DELIVERY).decide(request("TCS"))
    assert sent[0]["instructions"] == DELIVERY.instructions
    assert sent[0]["text"]["format"]["schema"] == DELIVERY.schema
    assert sent[0]["prompt_cache_key"] == DELIVERY.version
    assert result.verdicts[0].verdict == "BUY"
    assert result.prompt_version == DELIVERY.version
    journal = DecisionJournal(tmp_path / "journal.sqlite")
    journal.record("c1", AS_OF, "openai", result)
    assert journal.completed("c1", result.request_json) is None  # other prompt
    reused = journal.completed("c1", result.request_json, DELIVERY.version)
    assert reused is not None and reused.verdicts == result.verdicts
    assert json.loads(result.response_json)["decisions"][0]["verdict"] == "BUY"
    assert reused.response_json == result.response_json
    journal.close()


def test_journal_adds_the_reply_column_to_older_journals(tmp_path) -> None:
    old_schema = SCHEMA.replace(",\n    response_json TEXT", "")
    assert "response_json" not in old_schema
    db = sqlite3.connect(tmp_path / "old.sqlite")
    db.executescript(old_schema)
    db.close()
    client = OpenAIResponses(
        "k",
        transport=lambda r, t: completed(
            [{"symbol": "TCS", "verdict": "NO_BUY", "confidence": 0.5, "reason": "x"}]
        ),
    )
    result = OpenAIDecider(client, "gpt-6-luna", policy=DELIVERY).decide(request("TCS"))
    journal = DecisionJournal(tmp_path / "old.sqlite")
    journal.record("c1", AS_OF, "openai", result)
    reused = journal.completed("c1", result.request_json, DELIVERY.version)
    assert reused is not None and reused.response_json == result.response_json
    journal.close()


def test_delivery_decider_fails_closed_to_no_buy() -> None:
    def broken(req, timeout):
        raise urllib.error.URLError("down")

    client = OpenAIResponses("k", transport=broken, sleep=lambda s: None)
    result = OpenAIDecider(client, "gpt-6-luna", policy=DELIVERY).decide(request("TCS"))
    assert result.verdicts[0].verdict == "NO_BUY" and result.status.startswith("error")


def test_v3_requires_web_search_and_logs_what_it_searched() -> None:
    sent = []

    def transport(req, timeout):
        sent.append(json.loads(req.data))
        payload = json.loads(
            completed(
                [{"symbol": "TCS", "verdict": "BUY", "confidence": 0.6, "reason": "x"}]
            )
        )
        payload["output"][0] = {
            "type": "web_search_call",
            "action": {
                "type": "search",
                "query": "TCS share news",
                "sources": [{"type": "url", "url": "https://example.com/tcs"}],
            },
        }
        return json.dumps(payload).encode()

    client = OpenAIResponses("k", transport=transport)
    decider = OpenAIDecider(
        client, "gpt-6-luna", web_search=True, require_search=True, policy=DELIVERY_V3
    )
    result = decider.decide(request("TCS"))
    assert sent[0]["tool_choice"] == "required"
    assert sent[0]["include"] == ["web_search_call.action.sources"]
    assert sent[0]["instructions"] == DELIVERY_V3.instructions
    searches = json.loads(result.response_json)["web_searches"]
    assert searches[0]["query"] == "TCS share news"
    assert searches[0]["sources"] == ["https://example.com/tcs"]
    assert result.usage.web_searches == 1 and result.verdicts[0].verdict == "BUY"


def test_v4_asks_for_the_probability_before_the_verdict() -> None:
    items = DELIVERY_V4.schema["properties"]["decisions"]["items"]
    assert list(items["properties"]) == ["symbol", "p_up_first", "verdict", "reason"]
    assert items["required"] == ["symbol", "p_up_first", "verdict", "reason"]
    assert "- p_up_first: your probability" in DELIVERY_V4.instructions
    assert "confidence" not in DELIVERY_V4.instructions
    older = DELIVERY_V3.schema["properties"]["decisions"]["items"]
    assert list(older["properties"]) == ["symbol", "verdict", "confidence", "reason"]
    verdicts = validate(
        request("TCS", "INFY"),
        {
            "decisions": [
                {
                    "symbol": "TCS",
                    "p_up_first": 0.31,
                    "verdict": "NO_BUY",
                    "reason": "r",
                },
                {"symbol": "INFY", "verdict": "BUY", "confidence": 0.9, "reason": "r"},
            ]
        },
        "s",
        DELIVERY_V4,
    )
    assert (verdicts[0].verdict, verdicts[0].confidence) == ("NO_BUY", 0.31)
    # an answer without the v4 field is not trusted: it fails closed
    assert (verdicts[1].verdict, verdicts[1].reason) == ("NO_BUY", "missing")


def test_delivery_block_shows_fractions_in_percent() -> None:
    block = delivery_block(
        0.81234,
        {"ret5": 0.0123, "rsi14": 61.0, "gap": float("nan")},
        frozenset({"ret5", "gap"}),
    )
    assert block == {
        "p_up1pct_2h": 0.8123,
        "ret5_pct": 1.23,
        "rsi14": 61.0,
        "gap_pct": None,
    }


def test_decider_fails_closed_and_respects_budget() -> None:
    def broken(req, timeout):
        raise urllib.error.URLError("down")

    client = OpenAIResponses("k", transport=broken, sleep=lambda s: None)
    result = OpenAIDecider(client, "gpt-6-luna").decide(request("TCS"))
    assert result.verdicts[0].verdict == "NO_TRADE" and result.status.startswith(
        "error"
    )
    spent = OpenAIDecider(client, "gpt-6-luna", budget_left=lambda: 0.0)
    assert spent.decide(request("TCS")).status == "budget_exhausted"


def test_rule_baseline_needs_agreeing_evidence() -> None:
    req = DecisionRequest(
        AS_OF,
        {},
        (
            Candidate(
                "A", {}, {"imbalance_total": 0.4}, {"buildup_1d": "long_buildup"}
            ),
            Candidate(
                "B", {}, {"imbalance_total": 0.4}, {"buildup_1d": "short_buildup"}
            ),
            Candidate("C", {}, None, None),
        ),
    )
    assert [v.verdict for v in RuleDecider().decide(req).verdicts] == [
        "LONG",
        "NO_TRADE",
        "NO_TRADE",
    ]


def test_journal_records_cycles_and_daily_spend(tmp_path) -> None:
    client = OpenAIResponses(
        "k",
        transport=lambda r, t: completed(
            [{"symbol": "TCS", "verdict": "LONG", "confidence": 0.6, "reason": "x"}]
        ),
    )
    result = OpenAIDecider(client, "gpt-6-luna").decide(request("TCS"))
    journal = DecisionJournal(tmp_path / "journal.sqlite")
    journal.record("c1", AS_OF, "openai", result)
    journal.set_outcome("c1", "TCS", 0.004)
    assert journal.spent_usd(datetime.now().astimezone().date()) >= 0
    assert journal.spent_usd(date(2000, 1, 1)) == 0
    reused = journal.completed("c1", result.request_json)
    assert reused is not None and reused.verdicts == result.verdicts
    assert reused.usage == result.usage and reused.cost_usd == result.cost_usd
    assert journal.completed("c1", result.request_json + " ") is None
    journal.record("c1", AS_OF, "openai", result)  # a retry replaces, not duplicates
    journal.close()


def test_openai_settings_require_key_and_parse_options() -> None:
    with pytest.raises(ConfigurationError):
        load_openai_settings({})
    settings = load_openai_settings(
        {
            "OPENAI_API_KEY": "sk-x",
            "OPENAI_MODEL": "gpt-6.1-sol",
            "AI_DAILY_BUDGET_USD": "2",
        }
    )
    assert settings.model == "gpt-6.1-sol" and settings.daily_budget_usd == 2.0
    assert "sk-x" not in repr(settings)
    assert settings.reasoning_effort == "low"
    medium = {"OPENAI_API_KEY": "sk-x", "OPENAI_REASONING_EFFORT": " Medium "}
    assert load_openai_settings(medium).reasoning_effort == "medium"
    with pytest.raises(ConfigurationError, match="OPENAI_REASONING_EFFORT"):
        load_openai_settings(
            {"OPENAI_API_KEY": "sk-x", "OPENAI_REASONING_EFFORT": "med"}
        )
