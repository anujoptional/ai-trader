import json
from datetime import datetime

from ai_trader.ai.agent import AGENT, AGENT_SCHEMA, SessionAgent
from ai_trader.ai.anthropic_client import AnthropicError, AnthropicMessages
from ai_trader.ai.decision import Candidate, DecisionRequest
from ai_trader.ai.openai_client import StructuredResponse, Usage
from ai_trader.clock import INDIA_TIMEZONE

USAGE = Usage(
    input_tokens=1000,
    cached_tokens=800,
    cache_write_tokens=100,
    output_tokens=50,
    reasoning_tokens=0,
    web_searches=1,
)


def request(*symbols: str, minute: int = 40) -> DecisionRequest:
    stocks = [{"symbol": s, "company": f"{s} Ltd"} for s in symbols]
    as_of = datetime(2026, 10, 16, 9, minute, 5, tzinfo=INDIA_TIMEZONE)
    return DecisionRequest(
        as_of,
        {},
        tuple(Candidate(s, {}) for s in symbols),
        document={"as_of": as_of.isoformat(), "stocks": stocks},
    )


class FakeClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return StructuredResponse(
            data=answer,
            usage=USAGE,
            model="claude-opus-5-5",
            response_id="msg",
            latency_seconds=1.0,
            searches=({"type": "search", "query": "q", "url": None, "sources": []},),
        )


def test_the_agent_keeps_one_conversation_and_closes_on_request() -> None:
    client = FakeClient(
        [
            {
                "decisions": [{"symbol": "TCS", "verdict": "LONG"}],
                "close": [],
                "notes": "market weak, careful",
            },
            {
                "decisions": [
                    {"symbol": "INFY", "verdict": "SHORT"},
                    {"symbol": "XYZ", "verdict": "LONG"},  # not asked: ignored
                ],
                "close": ["TCS", "TCS"],
                "notes": "took profit idea",
            },
        ]
    )
    closed, logged = [], []
    books = iter([{"open": []}, {"open": [{"symbol": "TCS", "side": "LONG"}]}])
    agent = SessionAgent(
        client,
        "claude-opus-5-5",
        book=lambda: next(books),
        close=lambda symbol: closed.append(symbol) or "closed LONG @ Rs 1",
        log=logged.append,
    )
    first = agent.decide(request("TCS"))
    second = agent.decide(request("INFY", minute=45))
    assert [v.verdict for v in first.verdicts] == ["LONG"]
    assert [(v.symbol, v.verdict) for v in second.verdicts] == [("INFY", "SHORT")]
    assert closed == ["TCS"]  # once, though named twice
    assert agent.turns == 2
    one, two = client.calls
    assert one["history"] == () and one["cache"] is True
    assert one["schema"] is AGENT_SCHEMA and one["instructions"] == AGENT.instructions
    assert one["require_search"] is False and one["web_search"] is True
    turn = json.loads(one["user_input"])
    assert turn["book"] == {"open": []}
    assert turn["candidates"] == [{"symbol": "TCS", "company": "TCS Ltd"}]
    assert turn["time"] == "2026-10-16T09:40:05+05:30"
    history = two["history"]
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[0]["content"] == one["user_input"]
    assert json.loads(history[1]["content"])["notes"] == "market weak, careful"
    reply = json.loads(second.response_json)
    assert reply["closed"] == {"TCS": "closed LONG @ Rs 1"}
    assert reply["web_searches"][0]["query"] == "q"
    assert second.prompt_version == AGENT.version and second.cost_usd > 0
    assert "notes: took profit idea" in logged
    assert "close TCS: closed LONG @ Rs 1" in logged


def test_a_failed_turn_trades_nothing_and_is_not_remembered() -> None:
    client = FakeClient([AnthropicError("timed out")])
    agent = SessionAgent(client, "claude-opus-5-5")
    result = agent.decide(request("TCS"))
    assert [v.verdict for v in result.verdicts] == ["NO_TRADE"]
    assert result.status.startswith("error") and agent.turns == 0


def test_a_spent_budget_asks_nothing() -> None:
    client = FakeClient([])
    agent = SessionAgent(client, "claude-opus-5-5", budget_left=lambda: 0.0)
    result = agent.decide(request("TCS"))
    assert result.status == "budget_exhausted" and not client.calls


def test_claude_continues_a_conversation_with_the_new_turn_cached() -> None:
    sent = []

    def transport(req, timeout):
        sent.append(json.loads(req.data))
        return json.dumps(
            {
                "id": "msg",
                "model": "claude-opus-5-5",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": '{"decisions": []}'}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode()

    history = (
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "{}"},
    )
    AnthropicMessages("k", transport=transport).structured(
        model="claude-opus-5-5",
        instructions="i",
        user_input="turn 2",
        schema={},
        history=history,
        cache=True,
    )
    messages = sent[0]["messages"]
    assert messages[:2] == list(history)
    assert messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "turn 2",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }
