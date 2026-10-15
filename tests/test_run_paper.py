from datetime import datetime

from ai_trader.ai.decision import DIRECTION, RESEARCH
from ai_trader.ai.journal import DecisionJournal
from ai_trader.cli.run_paper import ai_decider, closed_today, single_instance
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import AnthropicSettings, OpenAISettings


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=INDIA_TIMEZONE)


class Holidays:
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows, self.error = rows or [], error

    def get_json(self, path, params):
        assert (path, params) == ("/api/holiday-master", {"type": "trading"})
        if self.error:
            raise self.error
        return {"CM": self.rows, "FO": []}


def test_only_one_run_at_a_time(tmp_path) -> None:
    first = single_instance(tmp_path / "run.lock")
    assert first is not None
    assert single_instance(tmp_path / "run.lock") is None
    first.close()
    again = single_instance(tmp_path / "run.lock")
    assert again is not None
    again.close()


def test_no_run_on_weekends_holidays_or_after_the_close() -> None:
    dussehra = Holidays([{"tradingDate": "20-Oct-2026", "description": "Dussehra"}])
    assert closed_today(at(13, 8, 30), at(13, 15, 31), dussehra) is None
    assert closed_today(at(20, 8, 30), at(20, 15, 31), dussehra) == (
        "NSE trading holiday (Dussehra)"
    )
    assert "Saturday" in closed_today(at(17, 8, 30), at(17, 15, 31), Holidays())
    assert "over" in closed_today(at(13, 16), at(13, 15, 31), Holidays())
    # The holiday list is best effort: when NSE fails, the run goes ahead.
    down = Holidays(error=OSError("down"))
    assert closed_today(at(20, 8, 30), at(20, 15, 31), down) is None


def test_each_ai_gets_the_chosen_prompt_timeout_and_budget(tmp_path) -> None:
    openai = OpenAISettings(api_key="sk-test", daily_budget_usd=5.0)
    anthropic = AnthropicSettings(api_key="sk-ant-test")
    journal = DecisionJournal(tmp_path / "decisions.sqlite")
    sol = ai_decider(
        openai, "gpt-6.1-sol", "medium", journal, policy=RESEARCH, timeout=150
    )
    assert sol._policy is RESEARCH and sol._effort == "medium"
    assert sol._web_search and sol._require_search
    assert sol._client._timeout == 150 and sol._budget_left() == 5.0
    opus = ai_decider(
        openai, "claude-opus-5-5", "medium", journal, anthropic, budget=30.0
    )
    assert opus._policy is DIRECTION and opus._budget_left() == 30.0
    assert opus._client._timeout == anthropic.timeout_seconds
    journal.close()
