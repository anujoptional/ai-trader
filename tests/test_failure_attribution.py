"""A broker failure must say which of two different things went wrong.

Every call into Groww can fail in two ways that look identical from outside and
call for opposite responses. Either the request never got through -- the
connection or the service -- in which case the same run tomorrow may simply
work. Or Groww answered and the reply is not the shape this module reads, in
which case tomorrow's run fails the same way and somebody has to change code
first. One message for both told an operator which instrument failed and never
which of those two it was, so the only way to find out was to run it again and
watch.

**Exception type cannot tell them apart here, which is why the split is
structural.** Groww intermittently answers a perfectly valid request with a
plain-text ``404 page not found`` body, and the SDK reports that as
``ValueError`` -- the same class ``resolve_instrument`` raises when it rejects a
payload. Any classifier keyed on the exception would put those two in the same
bucket. Where the failure happened is the only thing that separates them, so
``_request`` wraps the call and ``_reading`` wraps everything after it, and the
message follows from which of the two was running.

The tests below establish four things: that the two failures do not read alike,
that both still carry the identification the old single message carried, that
the wording is one wording across every path rather than six near-misses, and
that neither branch shows an operator the underlying exception's text -- a
property with two code paths to get wrong now instead of one.
"""

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError

from ai_trader.broker import CandleInterval, Instrument
from ai_trader.broker import groww as groww_module
from ai_trader.broker.groww import (
    GrowwAuthenticationError,
    GrowwBroker,
    GrowwBrokerError,
)
from ai_trader.clock import INDIA_TIMEZONE
from ai_trader.config import GrowwSettings

_INSTRUMENT = Instrument(exchange="NSE", trading_symbol="RELIANCE")

_START = datetime(2026, 9, 14, 10, 0, tzinfo=INDIA_TIMEZONE)
_END = datetime(2026, 9, 14, 10, 1, tzinfo=INDIA_TIMEZONE)

_LEAK = "ucc-42"
"""Stands in for whatever an exception or a payload might be carrying.

Not a real account identifier. It only has to be a string that could not
arrive in a message by coincidence, so that finding it in one is proof it was
copied there. Both things it stands in for are real.

An SDK exception may quote the request that produced it, headers included.
That is the unanswered side, and it is why ``from None`` is at every site.

The unreadable side is less obvious and worse. Every payload model here sets
``extra="ignore"``, so the account fields Groww sends alongside the data --
``ucc`` among them -- are dropped rather than read. But a pydantic failure
quotes ``input_value``, and ``input_value`` is the *raw* dict, extras included.
The fields the model was careful not to keep are sitting in the validation
error, one ``str(exception)`` from an operator's terminal.

**It is short on purpose, and that is not cosmetic.** Pydantic elides the
middle of a long ``input_value``, so a twenty-seven character sentinel comes
back as ``'leake...d-client-code-sentinel'`` -- plainly leaked, and no longer
findable by substring. A guard written that way reports clean while the value
walks past it. Six characters survive intact, and
``test_the_sentinel_would_have_shown_up_if_it_were_copied`` checks that against
the payloads actually used below rather than against a convenient shorter one.
"""

_UNREACHABLE = ValueError(f"Extra data: line 1 column 5 (char 4) [{_LEAK}]")
"""The decode failure Groww's plain-text 404 produces, carrying a sentinel."""


@dataclass(frozen=True)
class _Path:
    """One call into Groww, and a way to fail it on either side of the line.

    ``unreadable`` is a reply Groww could return today: well-formed enough to
    arrive, wrong enough that this module cannot read it. That is the point --
    the payload branch has to be reached by answering badly, never by refusing
    to answer, or the test would be proving nothing about where the split is.

    Each one carries ``ucc`` as well, so that the leak guard below has something
    to find on every path. ``echoing_model`` names the model whose validation
    error would quote it, and is ``None`` where the path fails with a bare
    ``raise`` that could not echo anything today -- there the sentinel is there
    to fail the day somebody makes one of those messages more helpful.
    """

    name: str
    attribute: str
    unreadable: object
    call: Callable[[GrowwBroker], object]
    echoing_model: type[BaseModel] | None = None


_PATHS = (
    _Path(
        name="get_user_profile",
        attribute="get_user_profile",
        unreadable={"ucc": _LEAK},
        call=lambda broker: broker.get_user_profile(),
        echoing_model=groww_module._GrowwProfilePayload,
    ),
    _Path(
        name="get_ltp",
        attribute="get_ltp",
        unreadable={"NSE_SOMETHINGELSE": 1234.5, "ucc": _LEAK},
        call=lambda broker: broker.get_ltp((_INSTRUMENT,)),
    ),
    _Path(
        name="get_quote",
        attribute="get_quote",
        unreadable={"last_price": 1234.5, "ucc": _LEAK},
        call=lambda broker: broker.get_quote(_INSTRUMENT),
        echoing_model=groww_module._GrowwQuotePayload,
    ),
    _Path(
        name="get_historical_candles",
        attribute="get_historical_candles",
        unreadable={"candles": "not-a-list", "ucc": _LEAK},
        call=lambda broker: broker.get_historical_candles(
            instrument=_INSTRUMENT,
            start=_START,
            end=_END,
            interval=CandleInterval.ONE_MINUTE,
        ),
    ),
    _Path(
        name="resolve_instrument",
        attribute="get_instrument_by_groww_symbol",
        unreadable={
            "exchange": "NSE",
            "exchange_token": "2885",
            "trading_symbol": "RELIANCE",
            "groww_symbol": "NSE-SOMETHINGELSE",
            "segment": "CASH",
            "ucc": _LEAK,
        },
        call=lambda broker: broker.resolve_instrument("NSE-RELIANCE"),
    ),
)

_IDS = tuple(path.name for path in _PATHS)

_ECHOING = tuple(path for path in _PATHS if path.echoing_model is not None)
_ECHOING_IDS = tuple(path.name for path in _ECHOING)


def _no_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the backoff. Eight real attempts would sleep for twelve seconds."""
    monkeypatch.setattr(groww_module.time, "sleep", lambda _seconds: None)


def _fails(path: _Path, rigging: str, value: object) -> tuple[GrowwBrokerError, Mock]:
    """Run one path against a client rigged to fail, and return the failure."""
    client = Mock()
    setattr(getattr(client, path.attribute), rigging, value)

    with pytest.raises(GrowwBrokerError) as caught:
        path.call(GrowwBroker(client))

    return caught.value, client


def _unanswered(path: _Path) -> tuple[GrowwBrokerError, Mock]:
    return _fails(path, "side_effect", _UNREACHABLE)


def _unreadable(path: _Path) -> tuple[GrowwBrokerError, Mock]:
    return _fails(path, "return_value", path.unreadable)


def _shared_opening(first: str, second: str) -> str:
    """Everything two messages say before they part company."""
    for index, (left, right) in enumerate(zip(first, second, strict=False)):
        if left != right:
            return first[:index]
    return first


def _shown(exception: BaseException) -> str:
    """Everything an operator sees when this failure reaches a terminal.

    Wider than ``str(exception)``, and the width is the whole point. ``raise ...
    from None`` does not detach what it suppressed: ``__context__`` still holds
    the original exception, sentinel and all, and only the *printer* is told to
    skip it. So the message alone cannot distinguish a suppressed chain from an
    unsuppressed one, because in both cases the message is identical and in both
    cases the caught exception is still hanging off the object.

    ``__cause__`` cannot distinguish them either, and that is the trap worth
    naming. It reads like the obvious check and is vacuous: ``from None`` sets
    ``__cause__`` to ``None``, and a bare ``raise`` inside an ``except`` block
    leaves it at ``None`` as well, while printing the entire chain. Asserting
    ``__cause__ is None`` therefore passes whether or not the suppression is
    there -- which it did, until removing every ``from None`` from the module
    left all twenty-five tests green. Rendering is what tells them apart.
    """
    return "".join(traceback.format_exception(exception))


# --------------------------------------------------------------------------
# The distinction itself.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_the_two_failures_do_not_read_alike(
    path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One exception type, two messages -- and the shared part is preserved.

    The second assertion is the backwards-compatible half. Splitting the
    message would be no use if it cost the identification the old one carried,
    so whatever the single message said about *what* failed has to survive
    unchanged into both branches, and only the explanation may differ.
    """
    _no_delay(monkeypatch)

    unanswered, _ = _unanswered(path)
    unreadable, _ = _unreadable(path)

    assert str(unanswered) != str(unreadable)
    assert "failed" in _shared_opening(str(unanswered), str(unreadable))


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_each_failure_says_what_to_do_about_it(
    path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which message is which, pinned by the only thing a reader wants.

    Without this the suite would pass just as happily with the two sentences
    swapped, and a run that needs a code change would be read as one worth
    retrying. The phrases are quoted rather than compared against the constants
    because a test that restates the expression it is checking checks nothing.
    """
    _no_delay(monkeypatch)

    unanswered, _ = _unanswered(path)
    unreadable, _ = _unreadable(path)

    assert "a later run may succeed" in str(unanswered)
    assert "retrying will reproduce it" in str(unreadable)


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_the_two_failures_are_told_apart_by_where_they_happened(
    path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The structural claim the messages rest on, checked rather than assumed.

    A message saying a later run may succeed is only honest if this run really
    did try again and get nowhere, and a message saying retrying reproduces it
    is only honest if the reply arrived. Those are the two call counts below.
    Were the split drawn anywhere else -- normalization inside the retry, say
    -- both sentences would still be printed and both would be wrong.
    """
    _no_delay(monkeypatch)

    _, unanswered_client = _unanswered(path)
    _, unreadable_client = _unreadable(path)

    assert getattr(unanswered_client, path.attribute).call_count == (
        groww_module._CALL_ATTEMPTS
    )
    assert getattr(unreadable_client, path.attribute).call_count == 1


# --------------------------------------------------------------------------
# What must not reach the message, on either branch.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", _PATHS, ids=_IDS)
def test_neither_branch_shows_what_it_caught(
    path: _Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two code paths now where there was one, and the same rule on both.

    Splitting the message doubled the number of places a caught exception could
    reach an operator, so the rule that held for the single message has to be
    checked twice. The sentinel arrives by a different route on each side --
    inside the exception on one, inside the payload on the other -- and ``_LEAK``
    above says what each of those stands for.

    The message and the rendered traceback are both checked because they leak
    independently. A message can be clean while the chained traceback prints
    everything anyway, which is what ``from None`` at each site exists to stop
    and what ``_shown`` explains is not observable any other way.

    The last two assertions are the non-vacuity half, and they are what make the
    two above them mean anything: there really is a suppressed exception behind
    each of these, and on the unanswered side it really is carrying the sentinel.
    Without them a module that raised from nowhere at all would pass.
    """
    _no_delay(monkeypatch)

    unanswered, _ = _unanswered(path)
    unreadable, _ = _unreadable(path)

    assert _LEAK not in str(unanswered)
    assert _LEAK not in str(unreadable)
    assert _LEAK not in _shown(unanswered)
    assert _LEAK not in _shown(unreadable)
    assert _LEAK in str(unanswered.__context__)
    assert unreadable.__context__ is not None


@pytest.mark.parametrize("path", _ECHOING, ids=_ECHOING_IDS)
def test_the_sentinel_would_have_shown_up_if_it_were_copied(path: _Path) -> None:
    """Non-vacuity for the test above, which otherwise proves only absence.

    Assertions that a string is missing are worth nothing until something shows
    the string was there to be missed, and this is the half that is easy to get
    wrong. An earlier version of it validated a short stand-in payload rather
    than the one the test above actually sends. It passed, the guard above
    passed, and a deliberately leaked message slipped through both -- because
    pydantic had elided the middle of the longer real payload and left the
    sentinel unfindable while leaving the value perfectly readable.

    So it runs the payload from ``_PATHS``, unmodified. If a model or a payload
    ever grows past the eliding width, this fails and says so, instead of the
    guard above quietly going blind.

    ``model_fields`` is the other half of the point: ``ucc`` is not a field, the
    model drops it, and pydantic reports it anyway.
    """
    assert path.echoing_model is not None

    with pytest.raises(ValidationError) as rejected:
        path.echoing_model.model_validate(path.unreadable)

    assert _LEAK in str(rejected.value)
    assert "ucc" not in path.echoing_model.model_fields


def test_an_unreachable_groww_carries_the_sentinel_too() -> None:
    """Non-vacuity for the other branch, whose carrier is the exception."""
    assert _LEAK in str(_UNREACHABLE)


# --------------------------------------------------------------------------
# Authentication, which can fail a third way and must not claim otherwise.
# --------------------------------------------------------------------------


def _settings() -> GrowwSettings:
    return GrowwSettings(
        totp_token="test-token-value",
        totp_secret="test-secret-value",
    )


def test_authentication_tells_the_two_failures_apart_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one path not in the table above, holding the same property.

    It is separate because it is a classmethod that constructs the client
    rather than using one, not because anything about it is exempt. The
    unreadable case is a token Groww returned that is not a token.
    """
    _no_delay(monkeypatch)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.return_value = "654321"

        api_class.get_access_token.side_effect = _UNREACHABLE
        with pytest.raises(GrowwAuthenticationError) as unanswered:
            GrowwBroker.authenticate(_settings())

        api_class.get_access_token.side_effect = None
        api_class.get_access_token.return_value = {"access_token": _LEAK}
        with pytest.raises(GrowwAuthenticationError) as unreadable:
            GrowwBroker.authenticate(_settings())

    assert "a later run may succeed" in str(unanswered.value)
    assert "retrying will reproduce it" in str(unreadable.value)
    assert _LEAK not in _shown(unanswered.value)
    assert _LEAK not in _shown(unreadable.value)


def test_the_unanswered_sentence_claims_only_that_nothing_came_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Why that sentence names no cause, demonstrated by the case that forbids it.

    ``authenticate`` mints its TOTP inside the operation, deliberately, so that
    a retry crossing a thirty-second window uses the code for the window it
    lands in. The cost is that a secret ``pyotp`` cannot decode fails *before*
    any request leaves this machine, and lands in the same branch as an outage.

    Structurally the branch knows one fact -- nothing was read back -- and that
    fact is true here. "The connection or the service is at fault" would not
    be, and would send somebody to check their network over a typo in ``.env``.
    So the sentence stops where the knowledge stops. The third assertion is the
    point: this is a limit that is stated, not one that is papered over.
    """
    _no_delay(monkeypatch)

    with (
        patch("ai_trader.broker.groww.pyotp.TOTP") as totp_class,
        patch("ai_trader.broker.groww.GrowwAPI") as api_class,
    ):
        totp_class.return_value.now.side_effect = ValueError("Non-base32 digit found")

        with pytest.raises(GrowwAuthenticationError) as caught:
            GrowwBroker.authenticate(_settings())

    message = str(caught.value)

    assert "no reply was read" in message
    assert api_class.get_access_token.call_count == 0
    assert "connection" not in message
