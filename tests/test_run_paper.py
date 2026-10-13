from datetime import datetime

from ai_trader.cli.run_paper import closed_today, single_instance
from ai_trader.clock import INDIA_TIMEZONE


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
