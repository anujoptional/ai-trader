from datetime import date
from decimal import Decimal

import pytest

pytest.importorskip("torch")  # the validator imports research scripts that need it

from scripts.validate_paper_inputs import (  # noqa: E402
    EXIT_MINUTE,
    book_fill,
    plain_exit,
    source_ages,
    store_target,
    url_date,
)

DAY = date(2026, 10, 14)


def test_url_dates_are_read_in_common_formats_and_never_in_the_future() -> None:
    assert url_date("https://x.in/news/2026/10/14/story", DAY) == DAY
    assert url_date("https://x.in/a-b-20261013.html", DAY) == date(2026, 10, 13)
    assert url_date("https://x.in/markets/oct-9-2026-stocks", DAY) == date(2026, 10, 9)
    assert url_date("https://x.in/14-october-2026/x", DAY) == DAY
    assert url_date("https://x.in/item/12345678", DAY) is None
    assert url_date("https://x.in/2026/10/15/tomorrow", DAY) is None
    ages = source_ages(
        ["https://a/2026/10/14/x", "https://b/2026/08/01/y", "https://c/z"], DAY
    )
    assert ages == {
        "sources": 3,
        "dated_in_url": 2,
        "today": 1,
        "within_7_days": 1,
        "older_than_30_days": 1,
    }


def bar(price: float, high: float | None = None, low: float | None = None):
    return [price, high or price, low or price, price, 100.0]


def test_plain_exit_takes_the_target_after_the_entry_minute_else_15_00() -> None:
    rows = [bar(100.0)] * 375
    rows[30] = bar(100.0, high=102.0)  # the entry minute: never checked
    rows[31] = bar(100.5, high=101.0, low=99.0)
    assert plain_exit(rows, 30, "LONG", Decimal("101")) == (31, Decimal("101"), "tp")
    assert plain_exit(rows, 30, "SHORT", Decimal("99")) == (31, Decimal("99"), "tp")
    rows[32] = bar(98.5)
    assert plain_exit(rows, 31, "SHORT", Decimal("99")) == (
        32,
        Decimal("98.5"),
        "tp_gap",
    )
    flat = [bar(100.0)] * 344 + [bar(100.4)] + [bar(110.0)] * 30  # 15:00 on: ignored
    assert plain_exit(flat, 10, "LONG", Decimal("101")) == (
        EXIT_MINUTE,
        Decimal("100.4"),
        "time_1500",
    )


def test_book_fill_walks_its_side_of_the_book() -> None:
    raw = {
        "asks": [["100", 1000, 1], ["100.1", 500, 1]],
        "bids": [["99.95", 2001, 3]],
    }
    quantity, fill, target = book_fill(raw, "LONG", Decimal("0.05"))
    assert (quantity, fill, target) == (2000, Decimal("100.05"), Decimal("101.10"))
    quantity, fill, target = book_fill(raw, "SHORT", Decimal("0.05"))
    assert (quantity, fill, target) == (2001, Decimal("99.95"), Decimal("98.95"))
    half = book_fill(raw, "LONG", Decimal("0.05"), Decimal("0.005"))
    assert half == (2000, Decimal("100.05"), Decimal("100.60"))  # 100.55025 up a tick


def test_targets_come_from_each_stores_own_settings(tmp_path) -> None:
    from ai_trader.paper.broker import PaperBroker, PaperConfig
    from ai_trader.paper.store import PaperStore

    store = PaperStore(tmp_path / "half.sqlite")
    PaperBroker(store, PaperConfig(target=Decimal("0.005")))
    assert store_target(store) == Decimal("0.005")
    assert store_target(PaperStore(tmp_path / "new.sqlite")) == Decimal("0.01")
