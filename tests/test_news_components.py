import json
from datetime import date, datetime, timedelta

from ai_trader.ai.openai_client import OpenAIResponses
from ai_trader.cli.fetch_nse_filings import fetch
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.news.features import categorize, event_context
from ai_trader.news.nse import Announcement, BoardMeeting, NseError
from ai_trader.news.store import NewsStore
from ai_trader.news.websearch import web_digest

AS_OF = datetime(2026, 10, 12, 10, 0, tzinfo=INDIA_TIMEZONE)


def item(symbol: str, at: datetime, subject: str, dissemination: str | None = None):
    raw = {"exchdisstime": dissemination} if dissemination else {}
    return Announcement(
        f"{symbol}{at}{subject}", symbol, "Co", subject, "text", "", at, raw
    )


def test_store_reads_only_what_was_public(tmp_path) -> None:
    store = NewsStore(tmp_path / "news.sqlite")
    store.add_announcements(
        [
            item("TCS", AS_OF - timedelta(hours=2), "Financial Results"),
            item("TCS", AS_OF - timedelta(hours=1), "Trading Window closure"),
            item("TCS", AS_OF + timedelta(minutes=5), "Award of order"),
            # Announced before but disseminated after the decision time.
            item(
                "TCS",
                AS_OF - timedelta(minutes=1),
                "Acquisition",
                "12-Oct-2026 10:03:00",
            ),
        ]
    )
    assert (
        store.add_announcements(
            [item("TCS", AS_OF - timedelta(hours=2), "Financial Results")]
        )
        == 0
    )
    store.add_meetings(
        [
            BoardMeeting(
                "m1",
                "TCS",
                date(2026, 10, 13),
                "Financial Results",
                "",
                AS_OF - timedelta(days=5),
                {},
            ),
            BoardMeeting(
                "m2",
                "TCS",
                date(2026, 10, 14),
                "Fund raising",
                "",
                AS_OF + timedelta(hours=1),
                {},
            ),
        ]
    )
    context = event_context(store, "TCS", AS_OF, AS_OF - timedelta(hours=18))
    assert context["since_prev_close"] == 2
    assert context["material_since_prev_close"] == 1
    assert context["items"][0]["category"] == "results"
    assert [m["date"] for m in context["board_meetings_next_3_days"]] == ["2026-10-13"]
    store.close()


def test_categories() -> None:
    assert categorize("Outcome of Board Meeting") == "results"
    assert categorize("Bagging/Receiving of orders/contracts") == "order_or_contract"
    assert categorize("Copy of Newspaper Publication") == "routine"
    assert categorize("Something new") == "other"


def test_actions_are_fetched_in_month_ranges(tmp_path) -> None:
    store = NewsStore(tmp_path / "news.sqlite")
    calls = []

    class Client:
        def corporate_actions(self, start, end):
            calls.append((start, end))
            return []

    fetch(
        Client(),
        store,
        date(2026, 9, 1),
        date(2026, 10, 15),
        ("actions",),
        progress=lambda message: None,
    )
    assert calls == [
        (date(2026, 9, 1), date(2026, 10, 1)),
        (date(2026, 10, 2), date(2026, 10, 15)),
    ]
    assert len(store.fetched_days("actions")) == 45
    store.close()


def test_fetch_resumes_and_records_failures(tmp_path) -> None:
    store = NewsStore(tmp_path / "news.sqlite")

    class Client:
        def __init__(self):
            self.calls = []

        def announcements(self, start, end):
            self.calls.append(start)
            if start == date(2026, 10, 2):
                raise NseError("HTTP 503")
            return [
                item(
                    "TCS",
                    datetime(2026, 10, start.day, 12, tzinfo=INDIA_TIMEZONE),
                    "Updates",
                )
            ]

    client = Client()
    added = fetch(
        client, store, date(2026, 10, 1), date(2026, 10, 3), ("announcements",)
    )
    assert added == {"announcements": 2}
    assert store.fetched_days("announcements") == {date(2026, 10, 1), date(2026, 10, 3)}
    fetch(client, store, date(2026, 10, 1), date(2026, 10, 3), ("announcements",))
    assert client.calls.count(date(2026, 10, 1)) == 1  # finished days are skipped
    store.close()


def test_web_digest_uses_search_tool_and_schema() -> None:
    sent = []

    def transport(request, timeout):
        sent.append(json.loads(request.data))
        return json.dumps(
            {
                "status": "completed",
                "output": [
                    {"type": "web_search_call"},
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"items": [], "summary": "none"}),
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 9000, "output_tokens": 200},
            }
        ).encode()

    digest, response = web_digest(
        OpenAIResponses("k", transport=transport),
        "gpt-6-luna",
        "TCS",
        "Tata Consultancy",
        AS_OF,
    )
    assert digest == {"items": [], "summary": "none"}
    assert sent[0]["tools"] == [{"type": "web_search"}]
    assert response.usage.web_searches == 1
