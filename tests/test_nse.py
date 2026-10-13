from datetime import date

import pytest

from ai_trader.news.nse import (
    NseClient,
    NseError,
    announcement,
    corporate_action,
    parse_nse_time,
)


class FakeResponse:
    def __init__(self, status: int, payload: object = None) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None, timeout: float = 0) -> object:
        self.calls.append((url, params))
        return self.responses.pop(0)


def client(responses: list[FakeResponse]) -> tuple[NseClient, FakeSession, list]:
    session = FakeSession(responses)
    sleeps: list[float] = []
    ticks = iter(range(1000))
    nse = NseClient(session, sleep=sleeps.append, clock=lambda: float(next(ticks)))
    return nse, session, sleeps


def test_client_primes_reprimes_on_403_and_parses_rows() -> None:
    row = {
        "symbol": "RELIANCE",
        "desc": "Outcome of Board Meeting",
        "an_dt": "13-Oct-2026 16:05:12",
        "attchmntFile": "https://nsearchives.nseindia.com/x.pdf",
        "attchmntText": "Results approved",
        "sm_name": "Reliance Industries Limited",
    }
    nse, session, _ = client(
        [
            FakeResponse(200),
            FakeResponse(403),
            FakeResponse(200),
            FakeResponse(200, [row, {"symbol": ""}]),
        ]
    )
    items = nse.announcements(date(2026, 10, 13), date(2026, 10, 13), "RELIANCE")
    assert [item.symbol for item in items] == ["RELIANCE"]
    assert items[0].announced_at.isoformat() == "2026-10-13T16:05:12+05:30"
    assert session.calls[-1][1] == {
        "index": "equities",
        "from_date": "13-10-2026",
        "to_date": "13-10-2026",
        "symbol": "RELIANCE",
    }
    assert sum("corporate-filings" in url for url, _ in session.calls) == 2
    assert session.headers["Referer"].endswith("corporate-filings-announcements")


def test_client_gives_up_after_bounded_attempts() -> None:
    nse, _, sleeps = client([FakeResponse(500) for _ in range(9)])
    with pytest.raises(NseError, match="HTTP 500"):
        nse.corporate_actions(date(2020, 1, 1), date(2020, 12, 31))
    assert len([s for s in sleeps if s >= 2]) == 4


def test_record_normalization_handles_missing_fields() -> None:
    assert parse_nse_time("-") is None
    assert announcement({"symbol": "TCS"}) is None
    action = corporate_action(
        {
            "symbol": "TCS",
            "subject": "Bonus 1:1",
            "exDate": "17-Jan-2024",
            "series": "EQ",
        }
    )
    assert action is not None and action.ex_date == date(2024, 1, 17)
    assert action.record_date is None
