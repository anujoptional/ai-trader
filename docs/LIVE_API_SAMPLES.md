# Live Groww API call/response samples

Captured against the live NSE session on **Monday 2026-09-21** with Python
3.12.13, instrument `Instrument(exchange="NSE", trading_symbol="RELIANCE")`.

This document exists so that work on the broker and market-data layers is not
blocked when the market is closed. It records, for every Groww API this system
calls: the wrapper signature, the raw SDK signature and its exact keyword
arguments, the real wire payload, the normalized dataclass that comes out, and
the conventions (symbol spelling, timezone, numeric type) that connect them.

Everything below is a transcript of a real call. Where a value was derived
rather than observed, it says so.

> **Values only. No credentials.** Both capture scripts redact
> credential-shaped keys recursively, and the authenticated broker object is
> replaced by the literal string `"<GrowwBroker>"` before serialisation. One
> account-identifying field escaped the raw script's filter — the profile's
> `ucc` — and is written here as `"<10-digit UCC>"`. See
> [Redaction and the `ucc` leak](#redaction-and-the-ucc-leak).

## Capture provenance

| Artefact | Contents | Written (IST) |
|---|---|---|
| `api_capture.py` / `api_capture.json` | normalized surface: what our wrappers return | 15:35:56 |
| `api_capture_raw.py` / `api_capture_raw.json` | pre-normalisation wire payloads | 15:33:27 |

Both scripts were run from the system temp directory, outside the repository,
because they call `broker._client` directly and exist only to produce this
document. They are not part of the package and are not linted or tested. The
JSON files are keyed alphabetically (`json.dump(..., sort_keys=True)`), not
chronologically.

The capture began at 15:30:35 IST, five minutes after the 15:30 close, except
for the quote burst and the profile call which straddled it. Post-close
behaviour is therefore documented alongside each API rather than assumed.

## The headline: the retry layer is what makes this work

In the same minute, with the same authenticated client:

| Call path | Calls | Succeeded | Failed |
|---|---|---|---|
| `broker.get_quote(...)` — wrapped in `_retry_broker_call` | 40 | **40** | 0 |
| `broker._client.get_quote(...)` — raw, single attempt | 1 | 0 | **1** |
| `broker._client.get_ltp(...)` — raw, single attempt | 1 | 0 | **1** |
| `broker._client.get_user_profile()` — raw, single attempt | 1 | 0 | **1** |
| `broker._client.get_historical_candles(...)` — raw, single attempt | 1 | 0 | **1** |

All four raw failures were identical:

```
requests.exceptions.JSONDecodeError: Extra data: line 1 column 5 (char 4)
  File ".venv\Lib\site-packages\requests\models.py", line 1120, in json
    raise RequestsJSONDecodeError(e.msg, e.doc, e.pos)
```

That error string is the fingerprint of a plain-text `404 page not found` body
being handed to a JSON parser: it consumes `404` as a number and stops at
character 4, the space before `page`. It is not a real 404 for the resource —
the identical call succeeds moments later. Bare round trips measured 0.141 s
(LTP), 0.172 s (profile), 0.203 s (quote) and 0.312 s (historical), so these
are fast failures, not timeouts.

The 40-call burst ran 15:30:38 → 15:31:32, 54 s wall clock. Its own
`time.sleep(0.25)` accounts for 10 s, leaving ~44 s inside the calls — about
1.1 s per call against a ~0.2 s bare round trip. That gap is retry backoff, and
it is the price of the 40/40 success rate.

The second script retried each raw call up to 40 times to get a payload at all.
It needed **5 attempts for the quote** (4 failures first) and 1 attempt each for
LTP, historical and profile — consistent with the documented clustering of
failures into short outages rather than a uniform error rate.

This is the live justification for `_CALL_ATTEMPTS = 8` with a delay doubling
to a `_MAX_RETRY_DELAY_SECONDS = 2.0` cap (`src/ai_trader/broker/groww.py`),
giving roughly twelve seconds of attempt budget. See
[`ARCHITECTURE.md` section 11](ARCHITECTURE.md) for the measurement that set
those numbers.

**Consequence for any new broker call: wrap it in `_retry_broker_call`.** An
unwrapped Groww call is a coin flip.

## Conventions that connect raw to normalized

### Three symbol spellings

Groww uses a different symbol format on almost every endpoint. The wrappers
hide this; anything calling the SDK directly must not get it wrong.

| Endpoint | Spelling | Produced by |
|---|---|---|
| `get_quote` | `trading_symbol="RELIANCE"` + `exchange="NSE"`, separate | — |
| `get_ltp` | `"NSE_RELIANCE"` (underscore) | `_live_symbol` (`groww.py`) |
| `get_historical_candles` | `"NSE-RELIANCE"` (hyphen) | `_historical_symbol` (`groww.py`) |
| `get_instrument_by_groww_symbol` (streaming) | `"NSE-RELIANCE"` (hyphen) | `_historical_symbol` |

The LTP *response* is keyed by the same underscore form it was asked for:
`{"NSE_RELIANCE": "1247.4"}`.

### Timezones

* Every Groww request that takes a time string takes **naive IST**.
  `_groww_datetime` converts to `Asia/Kolkata` and formats
  `"%Y-%m-%d %H:%M:%S"` — no offset is sent.
* Every normalized `datetime` this system produces is **UTC and
  timezone-aware**.
* `last_trade_time` arrives as an **epoch integer**.
  `_groww_epoch_datetime` accepts either seconds or milliseconds, choosing by
  magnitude against a sanity window of 2000-01-01 to 2100-01-01, and returns
  UTC. The captured value `1789984697` was in the seconds range and normalized
  to `2026-09-21T09:58:17+00:00`, i.e. 15:28:17 IST.
* Historical candle timestamps arrive as **naive IST ISO-8601 strings**
  (`"2026-09-21T09:15:00"`) and normalize to UTC
  (`2026-09-21T03:45:00+00:00`).

Session boundaries in IST: 09:15–15:30, which is 03:45–10:00 UTC, 375
one-minute slots.

### Numbers: float on the wire, `Decimal` in the system

Every price on the wire is a JSON float. The capture scripts record floats via
`repr()`, which is why prices appear as quoted strings in the JSON files —
`"1247.4"` in the raw capture was the float `1247.4`.

Normalisation to `Decimal` happens in two different places:

* **quote and historical candles** — pydantic, via the `Decimal`-annotated
  fields of `_GrowwQuotePayload` / `_GrowwQuoteOHLC` and the candle normalizer.
* **LTP** — the `_decimal` helper in `groww.py`, which goes through
  `Decimal(str(value))` and rejects `bool`.

`volume` is the exception: it is an `int` on the wire and stays an `int`
(`OHLCVCandle.volume`, `MarketQuote.volume`).

The float → `Decimal` conversion is the layer boundary. Downstream of the
broker there are no floats.

## `authenticate`

```python
GrowwBroker.authenticate(settings: GrowwSettings) -> Self   # groww.py
```

Captured 15:30:36.517 IST, **0.61 s**, returned a live `GrowwBroker`.

The broker object is never serialised — it holds the live session. No
credential material appears in any capture artefact.

Two behaviours worth knowing:

* The TOTP is regenerated **per retry attempt**, so a retry that crosses a
  30-second TOTP window uses the code belonging to the window it lands in.
* The SDK prints status text during `GrowwAPI(access_token)` construction; it
  is captured and discarded with `redirect_stdout(StringIO())` so CLI output
  stays limited to what the CLI explicitly prints.

On failure every underlying cause is replaced by
`GrowwAuthenticationError("Groww authentication failed.")`, raised `from None`.
**The `from None` is deliberate and it does suppress the root cause** — this is
why diagnosing a Groww-side problem requires probing the raw SDK, as the two
capture scripts do.

## `get_user_profile`

```python
broker.get_user_profile() -> BrokerProfile                  # groww.py
broker._client.get_user_profile() -> dict[str, Any]         # raw
```

Captured 15:30:38.702 IST, **2.187 s** (post-close: still served).

### Raw wire payload — 6 keys

```json
{
  "active_segments": ["CASH", "FNO", "COMMODITY"],
  "bse_enabled": true,
  "ddpi_enabled": false,
  "nse_enabled": true,
  "ucc": "<10-digit UCC>",
  "vendor_user_id": "<redacted str>"
}
```

### Normalized `BrokerProfile`

| Field | Declared type | Captured value | From |
|---|---|---|---|
| `exchange_enablement` | `Mapping[str, bool]` | `{"NSE": true, "BSE": true}` | `nse_enabled`, `bse_enabled` |
| `active_segments` | `tuple[str, ...]` | `("CASH", "FNO", "COMMODITY")` | `active_segments` |
| `ddpi_enabled` | `bool` | `false` | `ddpi_enabled` |

`exchange_enablement` is a `MappingProxyType`, built explicitly with the two
keys `"NSE"` and `"BSE"` — it is not a pass-through of the wire payload. The
normalized capture records it as the placeholder `"<mappingproxy>"` because the
capture script's serialiser has no branch for that type; the real content is the
two-key mapping above, which is fixed by the code and not by the response.

**4 of the 6 wire keys are consumed.** `_GrowwProfilePayload` is declared with
`ConfigDict(extra="ignore")` and names only `nse_enabled`, `bse_enabled`,
`active_segments`, `ddpi_enabled`, so `ucc` and `vendor_user_id` are dropped at
the pydantic boundary and never reach `BrokerProfile`. That is the reason the
`ucc` exposure described below was confined to an ad-hoc script.

## `get_ltp`

```python
broker.get_ltp(instruments: Sequence[Instrument]) -> tuple[LastTradedPrice, ...]
                                                            # groww.py
broker._client.get_ltp(
    exchange_trading_symbols=("NSE_RELIANCE",),
    segment="CASH",
    timeout: int | None = None,
) -> dict[str, Any]                                         # raw
```

Post-close: still served.

### Raw wire payload

```json
{"NSE_RELIANCE": "1247.4"}
```

A flat mapping of requested symbol to price. Nothing else — no timestamp, no
volume, no staleness indicator. An LTP served after the close is
indistinguishable from a live one.

### Normalized `LastTradedPrice`

| Field | Declared type | Value |
|---|---|---|
| `instrument` | `Instrument` | `Instrument(exchange="NSE", trading_symbol="RELIANCE")` |
| `price` | `Decimal` | `Decimal("1247.4")` |

The wrapper enforces **1 to 50 instruments per request** and zips the response
back onto the input with `strict=True`, so a symbol missing from the response
raises rather than silently shortening the result. Failure surfaces as
`GrowwMarketDataError`.

## `get_quote`

```python
broker.get_quote(instrument: Instrument) -> MarketQuote      # groww.py
broker._client.get_quote(
    trading_symbol="RELIANCE",
    exchange="NSE",
    segment="CASH",
    timeout: int | None = None,
) -> dict[str, Any]                                          # raw
```

Captured 15:30:38.878 IST and then 39 more times through 15:31:32.

### Raw wire payload — 27 keys

```json
{
  "average_price": null,
  "bid_price": null,
  "bid_quantity": null,
  "day_change": 21.0,
  "day_change_perc": 1.7123287671232876,
  "depth": {
    "buy":  [{"orderCount": 0, "price": 0.0, "quantity": 0}, ... x4],
    "sell": [{"orderCount": 0, "price": 0.0, "quantity": 0}, ... x4]
  },
  "high_trade_range": null,
  "implied_volatility": null,
  "last_price": 1247.4,
  "last_trade_quantity": 11,
  "last_trade_time": 1789984697,
  "low_trade_range": null,
  "lower_circuit_limit": 1210.0,
  "market_cap": null,
  "offer_price": null,
  "offer_quantity": null,
  "ohlc": {"close": 1226.4, "high": 1249.1, "low": 1232.5, "open": 1234.1},
  "oi_day_change": 0.0,
  "oi_day_change_percentage": 0.0,
  "open_interest": null,
  "previous_open_interest": null,
  "total_buy_quantity": 0,
  "total_sell_quantity": 0,
  "upper_circuit_limit": 1284.8,
  "volume": 10003083,
  "week_52_high": 1611.8,
  "week_52_low": 1226.4
}
```

### 🔴 `ohlc.close` is the PREVIOUS close, not the current one

This is the single most dangerous naming trap in the Groww surface. The wrapper
maps `ohlc.close` to `MarketQuote.previous_close`, and the current price is
`last_price`.

The payload proves it arithmetically: `day_change 21.0` and
`day_change_perc 1.7123287671232876`, and
`21.0 / 1226.4 = 0.017123287671232876...`. The denominator is `ohlc.close`, so
`ohlc.close` is the reference the change is measured *from* — the previous
close. `last_price 1247.4 = 1226.4 + 21.0` confirms it from the other side.

### Normalized `MarketQuote`

| Field | Declared type | Captured value | From wire |
|---|---|---|---|
| `instrument` | `Instrument` | NSE RELIANCE | the argument |
| `last_price` | `Decimal` | `1247.4` | `last_price` |
| `last_trade_at` | `datetime` | `2026-09-21T09:58:17+00:00` | `last_trade_time` |
| `open` | `Decimal` | `1234.1` | `ohlc.open` |
| `high` | `Decimal` | `1249.1` | `ohlc.high` |
| `low` | `Decimal` | `1232.5` | `ohlc.low` |
| `previous_close` | `Decimal` | `1226.4` | **`ohlc.close`** |
| `volume` | `int` | `10003083` | `volume` |
| `day_change` | `Decimal` | `21.0` | `day_change` |
| `day_change_percent` | `Decimal` | `1.7123287671232876` | `day_change_perc` |

**6 of the 27 wire keys are consumed.** `_GrowwQuotePayload` declares
`last_price`, `last_trade_time`, `ohlc`, `volume`, `day_change`,
`day_change_perc`; `extra="ignore"` silently drops the other 21 — including
`depth`, both circuit limits, the 52-week range, `total_buy_quantity` /
`total_sell_quantity`, `last_trade_quantity` and every options field. Anything
that later wants order-book depth or circuit limits must extend the payload
model; the data is on the wire already.

### Post-close behaviour

The quote endpoint keeps serving the frozen end-of-day snapshot. Across all 40
calls `distinct_volumes` was exactly `[10003083]` — one value — and the last
response was byte-identical to the first. `last_trade_at` stayed pinned at
15:28:17 IST. `depth` was all zeros and `total_buy_quantity` /
`total_sell_quantity` were `0`.

Nothing in the payload says "stale". A consumer that needs freshness must
compare `last_trade_at` against its own clock.

### This is the only live volume Groww serves

`volume` here is the **running cumulative session total**, not a per-minute
figure. It is the source the `VolumePoller`
(`src/ai_trader/market/volume_poller.py`) differences to give live candles a
volume, because the tick stream does not provide one — see
[`create_ltp_stream`](#create_ltp_stream) below.

## `get_historical_candles`

```python
broker.get_historical_candles(
    instrument: Instrument,
    start: datetime,            # timezone-aware, required
    end: datetime,              # timezone-aware, required
    interval: CandleInterval,
) -> tuple[OHLCVCandle, ...]                                 # groww.py

broker._client.get_historical_candles(
    exchange="NSE",
    segment="CASH",
    groww_symbol="NSE-RELIANCE",
    start_time="2026-09-21 09:15:00",   # naive IST
    end_time="2026-09-21 15:33:27",     # naive IST
    candle_interval="1minute",          # the STRING, not the int 1
    timeout: int | None = None,
) -> dict[str, Any]                                          # raw
```

Captured 15:31:35.565 IST, **2.25 s**, **362 candles**. Post-close: still
served.

`_validate_period` rejects a naive `start` or `end` and rejects `start >= end`
before any network call. `CandleInterval.ONE_MINUTE` maps to the string
`"1minute"`; passing the integer `1` does not work.

### Raw wire payload — 5 keys

```json
{
  "candles": [ [timestamp, open, high, low, close, volume], ... ],
  "closing_price": 1247.4,
  "end_time": "2026-09-21 15:28:00",
  "interval_in_minutes": 1,
  "start_time": "2026-09-21 09:15:00"
}
```

Candle rows are **positional lists of six elements**, not objects. Element 0 is
a naive IST ISO-8601 string; elements 1–5 are open, high, low, close, volume.

First three rows as captured (truncated to four elements by the capture
script — see [Known artefacts](#known-artefacts-in-the-capture-files)):

```
["2026-09-21T09:15:00", 1234.1, 1238.6, 1234.1, ...]
["2026-09-21T09:16:00", 1236.8, 1243.5, 1236.8, ...]
["2026-09-21T09:17:00", 1242.0, 1243.6, 1241.5, ...]
```

### Normalized `OHLCVCandle`

| Field | Declared type | First candle | Last candle |
|---|---|---|---|
| `timestamp` | `datetime` | `2026-09-21T03:45:00+00:00` | `2026-09-21T09:58:00+00:00` |
| `open` | `Decimal` | `1234.1` | `1247.4` |
| `high` | `Decimal` | `1238.6` | `1247.4` |
| `low` | `Decimal` | `1234.1` | `1247.4` |
| `close` | `Decimal` | `1236.5` | `1247.4` |
| `volume` | `int` | `252906` | `531468` |

Cross-checking the raw row against the normalized candle confirms both the
field order and the timezone conversion: raw
`["2026-09-21T09:15:00", 1234.1, 1238.6, 1234.1, ...]` → normalized
`timestamp 03:45 UTC, open 1234.1, high 1238.6, low 1234.1`.

### Candles are stamped at their OPEN

The first candle of a session that opens at 09:15 IST is stamped 09:15, not
09:16. A close-stamped convention would put the first candle at 09:16.

### 🔴 The session is NOT contiguous

`spacing_seconds` — the set of gaps between consecutive candles, computed over
all 361 adjacent pairs — was exactly:

```json
[60, 780]
```

One gap of **780 s = 13 minutes**, meaning **12 candles are simply absent**
from the middle of an otherwise one-minute series. The arithmetic closes:

```
last open-stamp 09:58 UTC = 15:28 IST
09:15 .. 15:28 inclusive        = 374 one-minute slots
minus the 12 missing candles    = 362      ← the count actually returned
```

**Any consumer must handle gaps.** It cannot assume candle *n+1* starts one
minute after candle *n*, and it cannot infer elapsed time from an index. The
feature engine is built on this assumption and was validated against exactly
this session.

### The endpoint lags the wall clock

`payload.end_time` is the window Groww actually served, not the window
requested. Requested end 15:33:27 IST; served end **15:28:00** — a lag of over
five minutes. The 15:31 call reported the same 15:28 end. The last two minutes
of the session (15:29, 15:30) had still not appeared at 15:33.

A consumer that needs the current minute must not expect it from this endpoint.

## `create_ltp_stream`

```python
broker.create_ltp_stream(instruments: Sequence[Instrument]) -> GrowwLtpStream
                                                            # groww.py
```

Attempted 15:31:35 IST, after the close.

```
elapsed:     260.765 s
error_type:  GrowwStreamError
error:       "Groww stream connection failed."
ticks:       0
```

Raised from the `GrowwFeed(self._client)` construction inside
`create_ltp_stream`. During those four-plus minutes the growwapi SDK
emitted 61 empty `Error: ` lines on its own logger — noise from the SDK, not
from this system, and worth filtering (`grep -v "^Error: $"`) when reading a
capture log.

> **Both numbers above are now historical.** The 2026-09-21 capture read this
> failure as a post-close artefact. Re-running it *during* live trading on
> 2026-09-22 produced the identical failure, which falsifies that reading: the
> cause is server-side and unrelated to market hours. The 260 s is also no
> longer reachable — `create_ltp_stream` now bounds the connect at 30 s. See
> [Market-open addendum](#market-open-addendum-2026-09-22).

The `error_type` above is what the code raised on 2026-09-21 and is left as
captured. That class has since been split: transport failures like this one now
raise `GrowwStreamConnectionError`, a subclass of `GrowwStreamError`, so a
stream supervisor can tell a dropped connection worth reconnecting for from a
payload or callback failure that would only replay. An identical capture today
would read `GrowwStreamConnectionError`, and `except GrowwStreamError` still
catches it.

~~**Post-close, the stream fails slowly and completely.** Four minutes is long
enough that a naive caller will look hung. Plan for it.~~

**Superseded.** The failure is not post-close behaviour; it reproduces
identically mid-session. The "looks hung" half was real and is now fixed at the
source — see [Market-open addendum](#market-open-addendum-2026-09-22).

### Which step failed

`create_ltp_stream` resolves each instrument to its `exchange_token` **before**
the `try` block that raises. A resolution failure would have surfaced as
`GrowwMarketDataError("Groww instrument lookup failed.")` instead. Because the
capture shows a stream error rather than a market-data one, instrument
resolution still succeeded after the close; it was the `GrowwFeed(self._client)`
construction that consumed the 260 s and failed.

### 🔴 Live ticks carry no usable volume

`MarketTick.cumulative_volume` is `int | None` and is **`None` on every live
tick**. This is not a bug and not an outage.

Groww transports volume over the stream as a **protobuf double**, whose unset
value arrives as `0.0`. `groww_stream._cumulative_volume` maps that zero to
`None` deliberately: treating `0` as a real differencing baseline would
attribute an entire session's volume to a single minute the first time a real
value arrived.

Measured during live trading on 2026-09-21: absent from all 114 ticks sampled.

This is precisely why `VolumePoller` exists. It polls
[`get_quote`](#get_quote) for the running session total and stamps it onto
passing ticks, so the already-tested `CumulativeVolumeTracker` →
`CandleBuilder` path lights up without any change to those classes.

### Normalized `MarketTick`

Field list per `MarketTick` in `src/ai_trader/broker/__init__.py`. No tick
sample was captured post-close; the live shape was validated separately on the
same day.

| Field | Declared type | Live behaviour |
|---|---|---|
| `instrument` | `Instrument` | as subscribed |
| `price` | `Decimal` | populated |
| `timestamp` | `datetime` | UTC, timezone-aware |
| `cumulative_volume` | `int \| None` | **always `None`** — see above |

## Error surface

Every wrapper catches broadly and re-raises one of these, `from None`:

| Exception | Raised by |
|---|---|
| `GrowwAuthenticationError` | `authenticate` |
| `GrowwProfileError` | `get_user_profile` |
| `GrowwMarketDataError` | `get_ltp`, `get_quote`, `get_historical_candles`, `resolve_instrument` |
| `GrowwStreamConnectionError` | `create_ltp_stream` (feed construction only) |

All five derive from `GrowwBrokerError(RuntimeError)`, and
`GrowwStreamConnectionError` additionally derives from `GrowwStreamError`, which
is the class the stream path raises when a failure is *not* the transport — a
payload shape this module no longer recognises, or a consumer callback that
raised. The split is what lets a supervisor reconnect on the first and stop on
the second. The messages are fixed
strings and carry no detail about what actually went wrong. **Retries happen
below this layer** — by the time one of these is raised, eight attempts over
roughly twelve seconds have already failed, so it indicates a genuine outage or
a real error rather than the usual flakiness.

Normalisation deliberately sits **outside** the retry, so a genuine schema
change surfaces immediately instead of being retried eight times.

## Post-close summary

Behaviour observed between 15:30 and 15:36 IST on a trading day:

| API | Post-close | Notes |
|---|---|---|
| `authenticate` | works | 0.61 s |
| `get_user_profile` | works | 2.19 s |
| `get_ltp` | works | last traded price, no staleness marker |
| `get_quote` | works | frozen EOD snapshot, `depth` all zeros |
| `get_historical_candles` | works | completed session, `end_time` lags by 5+ min |
| `create_ltp_stream` | **fails** | `GrowwStreamConnectionError` after 260.765 s, 0 ticks |

Everything except the stream can be exercised outside market hours. ~~Stream
work needs a live weekday session.~~ **Not true** — the stream fails the same
way during live trading, so waiting for market hours does not unblock it. See
[Market-open addendum](#market-open-addendum-2026-09-22).

## Market-open addendum (2026-09-22)

Everything above was captured five minutes *after* the close. This section was
captured **during live trading** on Tuesday 2026-09-22, roughly 12:50–13:10 IST,
on `RELIANCE` and `TCS`. It exists to settle the questions the post-close
capture could only guess at, and to correct the two places where it guessed
wrong.

### What the post-close capture got wrong

| Post-close claim | Market-open reality |
|---|---|
| The 260 s stream failure is post-close behaviour | It reproduces **identically mid-session**. Market hours are irrelevant. |
| "Stream work needs a live weekday session" | A live session does **not** unblock it. The fault is on Groww's side. |

Both claims were reasonable inferences from a post-close-only capture. Neither
survived contact with an open market. Treat any remaining "post-close" framing
in this document as *observed post-close*, not as *caused by the close*.

### 🔴 The live feed was down server-side — the outage has since ended

**Read this section as dated history, not as current state.** On 2026-09-24 the
same transport delivered 143 real NATS ticks; see *Live-feed addendum
(2026-09-24)* at the end of this document for what a working feed does. The
diagnosis below is kept because it is still the right reading of these
symptoms, and because the failure path it exercised is the one the reconnect
layer was built for — but a fresh session must not conclude from it that tick
work is blocked. It is not.

Authentication succeeds and `generate_socket_token` returns 200 (6/6 attempts).
The NATS handshake then never completes:

```
client  -> CONNECT {...authenticated...} + PING
server  -> PING          # never +OK, never PONG, never -ERR
server  -> (silence)
```

`connect()` returns with `last_error=None`, `is_connected=False`,
`status=4 (CONNECTING)` — permanently. DNS, TLS and the WebSocket upgrade were
each verified healthy independently. Server pods identified themselves as
`apex-nats-socket-gateway-server-rollout-*`, NATS `2.11.0-dev`.

**This was a Groww server-side regression**, and the sequel confirms it: it
cleared between 2026-09-22 and 2026-09-24 with no client change whatsoever, on
a client whose stream code had not been touched. It is recorded here so nobody
spends another session re-diagnosing a client that is behaving correctly.

### Transport architecture (why it is not a plain WebSocket)

Groww's live feed is **NATS over WebSocket Secure** at
`wss://socket-api.groww.in`, driven by the vendor's `growwapi/groww/nats_client.py`.
Three vendor facts that shape our code:

- `NatsClient.__init__` **connects synchronously** — `GrowwFeed(client)` blocks
  in its constructor until connect succeeds or the retry budget is spent.
- That budget is nats-py's default: `max_reconnect_attempts=60` at
  `reconnect_time_wait≈2 s`, which is where the ~260 s came from. It is not
  configurable through the surface Groww exposes.
- `GrowwFeed._key()` mints a fresh `os.urandom(32)` seed per construction, so
  the class-level `_nats_clients` cache **never hits** and grows unboundedly.

### The fix, measured against the live outage

`create_ltp_stream` now bounds the connect at 30 s (`_STREAM_CONNECT_TIMEOUT_SECONDS`)
on an abandoned daemon thread, and contains the vendor's log storm into the
raised error rather than letting it reach stderr.

| | Before | After |
|---|---|---|
| `check_market_state` | 4 m 22.750 s | **33.8 s / 34.0 s** |
| `check_stream` | ~4 m | **33.3 s** |
| `check_features --live --live-seconds 90` | 4 m 22 s (blew the 90 s budget) | **33.7 s** |
| stderr | ~61 empty `Error:` lines + `WinError 6` traceback | **one actionable line** |
| exit code | 1 | 1 |

The single line is `Groww live feed unreachable; the stream connection failed.`
A single connect attempt is made deliberately: reconnect policy belongs to
`StreamSupervisor`, which already owns backoff, and retrying here would multiply
the vendor's own 60-attempt schedule.

### The outage is not market-hours related

Reproduced three times, twice of them squarely mid-session: post-close on
2026-09-21, and again at 14:06 IST on 2026-09-22 (`check_stream`, 36.862 s,
exit 1) with the exchange trading and REST quotes flowing normally throughout.
A failed stream connect therefore says nothing about whether the market is
open, and must not be read as "the market may be closed" — that message belongs
only to a connect that *succeeded* and then delivered no ticks.

### Driving StreamSupervisor against the outage

The outage is itself a usable test of a component whose job is surviving
connection failure, so the real `GrowwBroker` was driven through
`StreamSupervisor` with `retry_on=(GrowwStreamConnectionError,)`,
`max_consecutive_failures=3`, `base_delay_seconds=1.0`, at 14:07 IST on
2026-09-22:

```
factory calls   3, at 14:07:51 / 14:08:23 / 14:08:55   (gaps 32.02 s, 32.02 s)
elapsed         94.05 s   = 3 x 30 s connect + 1 s + 2 s backoff + overhead
failures        3
sessions        0
reconnects      0
stopped_because 'gave_up'
last_error      GrowwStreamConnectionError: Groww stream connection timed out
                after 30 seconds; 22 transport errors reported.
```

Nothing escaped onto the caller. `reconnects=0` is correct rather than a bug:
the counter tracks successful re-opens, and `opens` is incremented only after
the factory returns, so three failed opens are recorded in `failures` and leave
`opens=0`. That also explains `sessions=0`.

This makes the give-up bound, the backoff schedule, `StreamReport` accuracy and
error containment live facts rather than stub facts. The success side was
closed out on 2026-09-24 — 14 sessions, 6 reconnects, 143 ticks — in the
addendum at the end of this document.

### REST call timings, market open

| Call | Market open | Post-close (2026-09-21) |
|---|---|---|
| `authenticate` | 2.6 s | 0.61 s |
| `get_quote` | 2.9 s | — |
| `get_historical_candles` | 2.5 s | — |

Authentication is measurably slower during trading hours. Budget for seconds,
not milliseconds.

### `get_historical_candles` is *fresh* during trading hours

The post-close capture measured a 5+ minute lag. Live, over a 90-minute window:

- **Lag behind wall clock: 0.9 minutes.**
- **Zero non-60 s gaps** — the intraday session is contiguous, unlike the
  post-close capture's fragmented view.
- **Zero zero-volume minutes** for RELIANCE.
- The **last candle is the in-progress, partial minute.**

That last point is the trap. A partial minute is structurally identical to a
complete one, so feeding it to the engine reports a fraction of a minute's
volume as though it described the whole minute.

**We are safe by construction:** every CLI backfills via
`find_recent_completed_session`, which reads a *completed* session, so the
in-progress minute cannot reach the engine. Anyone driving the engine from
*today's* historical data intraday must drop the last candle themselves.

### `get_quote` during trading, and the volume path

Sampled at ~1.2 s over 150 s:

- `last_trade_at` lags wall clock by **min 0.6 s, median 2.4 s, max 8.2 s**.
- Exchange timestamps occasionally go **backwards**: 1 late tick in 111 when
  timestamping from `last_trade_at`, 0 when using wall clock.
- Session volume **moves backwards** on a small fraction of reads: 3 of 70
  (2026-09-22) and 8 of 147 (2026-09-21) — roughly 4–5%, consistently.

That last figure is the whole reason `VolumePoller._accept` rejects regressions.
A single accepted stale total makes `CumulativeVolumeTracker` read a restarted
counter, which blanks that minute's volume and disables strict session VWAP for
the rest of the day.

### `VolumePoller`, measured live

Two runs on 2026-09-22 (60 s × 2 instruments, then 150 s × 1):

| Metric | Value |
|---|---|
| polls | 52, then 70 |
| failures | **0** and **0** |
| regressions rejected | 2, then 3 |
| stale stamps | **0** and **0** |
| accepted readings monotonic | **yes**, both instruments |
| round duration | **2.15–2.32 s** against a 2.0 s interval |

The 2.7 s round figure quoted elsewhere in this repo came from 2026-09-21, when
Groww's transient-404 rate was higher. **Treat the round as 2.1–2.7 s**, not as
a single number. Staleness is bounded by the round, not the interval.

### End-to-end validation without the live feed

Because only the NATS transport is down, the **production** tick path was driven
with live REST data: real prices and real polled volume totals through the real
`MarketState` → `CandleBuilder` → `CumulativeVolumeTracker` → `FeatureEngine`.

A live minute closed at 13:03 IST and produced:

```
O=1241.4 H=1242.1 L=1241.0 C=1242.1 V=72814   core_ready=True
```

Volume was **differenced from polled cumulative totals across a real minute
boundary** — the exact mechanism the live feed will use. The path works.

`session_volume` came back `None`, which is **correct, not a defect**:
`SessionContext` withholds every session aggregate unless the session was
observed from inside its opening range, and this probe joined at 13:03. The
behaviour is pinned by `test_a_mid_session_start_stays_withheld_for_the_rest_of_the_day`,
which also documents VWAP as the deliberate exception — and VWAP was indeed
populated while the session family was not.

### Developing without a live market

Everything below is reproducible outside market hours, which is what keeps this
project unblocked:

| Layer | How to exercise it off-hours |
|---|---|
| auth, profile, LTP, quote | work post-close; quote is a frozen EOD snapshot |
| historical candles | work any time; use a completed session |
| candle building, volume differencing, features | replay a completed session through `MarketState` |
| stream transport | drive `StreamSupervisor` with a fake factory shaped to the measured profile below |
| stream *failure handling* | fully covered by tests; no market needed |

Nothing is gated on the market being open any more. The last item that was —
proof that real NATS ticks flow — was closed on 2026-09-24, and the point of
the addendum below is to leave behind not just *that* it worked but a profile
precise enough to build a fake against. A component that behaves correctly
against a fake that goes silent for half of its 60 s windows, hands back a
candle series with minutes missing, and occasionally reports a volume total
lower than the one before it, is a component that has met the real feed's worst
observed behaviour without waiting for 09:15.

### Decision: no REST fallback ingestion mode

Groww's historical REST endpoint is fresh (0.9 min), gapless and complete, so a
REST-driven ingestion fallback is *technically* viable. It was considered and
**deliberately not built**, for two reasons:

1. It is new development, outside the current scope of hardening what exists.
2. A 1-minute-delayed bar feed is not a tick feed. Substituting one for the
   other behind the same interface would report delayed data under a live
   label — precisely the "unavailable beats approximately right" failure that
   ARCHITECTURE §8 exists to prevent.

Recorded here as a validated option with its caveats, not as a plan.

## Live-feed addendum (2026-09-24): what a working feed actually does

The 2026-09-22 outage lifted. Two fifteen-minute `check_scanner --live` windows
on `RELIANCE` were run on Thursday 2026-09-24, at 14:04–14:19 and 14:29–14:44
IST, both with `--max-atr-multiple 30`. Between them they delivered **295 real
NATS ticks** through `StreamSupervisor` into 23 live candles.

Two runs rather than one, deliberately. A single window tells you what happened
once; the value of this section is the figures that *repeated*, because those
are the ones worth building against.

| | Run A | Run B |
|---|---|---|
| ticks delivered | 143 | 152 |
| live candles built | 11 | 12 |
| late / out-of-order ticks | 0 | 0 |
| ticks stamped from `last_trade_at` | 143 | 152 |
| stale stamps | 0 | 0 |
| supervisor sessions (60 s windows) | 14 | 15 |
| **silent sessions** | **7** | **7** |
| reconnects | 6 | 7 |
| stream failures | 2 | 0 |
| volume polls | 413 | 415 |
| poll failures | 0 | 0 |
| poll regressions | 9 (2.2%) | 17 (4.1%) |
| `session_volume` at end | 10,020,651 | 11,195,906 |
| `stopped_because` | `deadline` | `deadline` |

Run A ended with a residual `stream_error` — a 30 s connect timeout reporting
seven transport errors, last `Disconnected` — while run B ended with
`stream_error: null` and no failures at all. The transport is therefore
intermittently flaky rather than reliably healthy or reliably broken, and a run
that reports failures is not evidence the outage has returned.

### 🔴 Silence is the normal case, not the exception

**Half of all one-minute stream windows deliver nothing.** Seven of fourteen in
run A, seven of fifteen in run B — the same absolute count in both, twenty-five
minutes apart, on the most liquid stock on the exchange during the middle of a
trading session. `RELIANCE` unquestionably traded during those minutes. The
silence is the transport's, not the market's.

This is the single most important fact in this document for anyone building on
the feed. A component that assumes a subscription, once opened, keeps producing
is wrong about this feed roughly half the time.

It is also why `StreamSupervisor` is not optional and not merely an
outage-survival measure. Its silent-session tripwire — rebuild a stream that
opened cleanly and then went quiet — fires every other minute under entirely
normal conditions. Without it a live session goes deaf within minutes and
reports no error, because nothing failed.

Reading the counters correctly matters here, and the definitions are not
obvious from their names (`market/stream.py`):

- `sessions` counts completed `collect()` windows, **not** transports opened. A
  productive session *keeps* its transport and loops; the default window is
  `DEFAULT_SESSION_SECONDS = 60.0`, so ~15 sessions in a 900 s run is the
  design, not churn.
- A **silent** session discards its transport, so the next iteration re-opens.
- `reconnects = max(opens - 1, 0)`, and `opens` increments only *after* the
  factory returns — a failed open never increments it.

Run B reconciles exactly on those rules: 7 silent sessions → 7 re-opens, plus
the initial open = 8, giving `reconnects = 7`. The reconnects in these runs are
almost entirely the silence tripwire doing its job, not failure recovery.

Run A is the instructive one. It also had 7 silent sessions but reports only 6
reconnects, which on the rules above means one re-open never succeeded — and
indeed it is the run that finished holding a `stream_error`. The reading is
that its last re-open was still failing when the deadline arrived. Worth
internalizing: **`reconnects` counts successes, so it undercounts effort**, and
a run can end with both a healthy tick count and an unresolved connect error
without either contradicting the other.

### 🔴 The live candle series has holes

Live minutes go missing. Run A produced no candle for 14:09, 14:10 or 14:12
IST; run B none for 14:38 or 14:40:

```
run A   14:06 14:07 14:08  ····  14:11  ····  14:13 14:14 ... (3 minutes absent)
run B   14:35 14:36 14:37  ····  14:39  ····  14:41 14:42 ... (2 minutes absent)
```

The mechanism is in `CandleBuilder._add_tick_locked`, and it is deliberate:
**candle closing is tick-driven, with no timer and no gap filling.** A working
minute is finalized only when a tick bearing a *later* minute arrives. So a
minute in which no tick was received is never emitted at all — it does not
appear as an empty candle, a zero-volume candle, or a flag. It simply is not
there.

Three consequences that downstream layers must be built for:

1. **The series is not one-candle-per-minute.** Any code that treats adjacent
   candles as adjacent minutes is wrong. `Candle` carries its own
   `start_time`/`end_time`; use them rather than an index.
2. **A candle's emission can lag its minute-end without bound.** The 14:08
   candle in run A was emitted when the first 14:11 tick arrived, roughly three
   minutes after the minute it describes ended. The lag is bounded by the
   silence, not by 60 s.
3. **Indicator periods are counted in candles, not minutes.** A fourteen-period
   ATR over a holed series spans more than fourteen minutes of wall clock. This
   is not a defect — it is the standard convention — but it means a live ATR and
   a backfilled ATR at the same timestamp can legitimately differ if the live
   series lost minutes the historical endpoint later fills in.

Point 3 is the one place where the "a scan at *t* equals the scan that would
have been live at *t*" property is genuinely qualified, and it is qualified by
the feed rather than by the scanner. The historical endpoint is gapless; a live
session is not. Replay over history is therefore replaying a *cleaner* series
than the live system sees.

The trailing partial minute is discarded rather than flushed, for the reason in
`check_scanner`: a candle covering part of a minute is indistinguishable
downstream from one covering all of it, and a scanner fed one would score a
fraction of a minute's volume as though it described the whole.

### Volume polling at steady state

The `VolumePoller` was untroubled across both runs: 413 and 415 polls, **zero
failures**, zero stale stamps, every one of the 295 ticks stamped from
`last_trade_at`.

The poll round confirms the earlier estimate from an independent session:
900 s / 413 polls = **2.18 s**, and 900 / 415 = 2.17 s, against the previously
measured 2.15–2.32 s. "Treat the round as 2.1–2.7 s" holds.

Non-monotonic cumulative-volume readings — a poll returning a total *lower*
than the one before it — continue at a similar rate: 2.2% and 4.1% here,
against 4.3% (3/70) and 5.4% (8/147) previously. **Budget for 2–5% of polls
regressing.** It is normal vendor behaviour, the differencing layer already
absorbs it, and it is not an error condition.

### The scanner over live candles

Both runs scanned every live candle they built, with the cost screen on:

| | Run A | Run B |
|---|---|---|
| cycles / considered | 11 / 11 | 12 / 12 |
| `not_ready` | 0 | 0 |
| `unreachable` | 0 | 0 |
| feasibility | `reachable: 11` | `reachable: 12` |
| candidates | 1 | 4 |
| rules that fired | `band_mean_reversion` | `band_mean_reversion`, `range_breakout`, `vwap_reversion` |

`not_ready: 0` throughout is expected rather than impressive: the engine was
already warm from the backfill of the previous session, so the fourteen-period
indicators were defined on the first live candle. A live run started cold would
report `not_ready` for its first thirteen candles.

The exported CSV shows the per-name cost hurdle landing just *under* the stated
0.2% on every live row — `0.0019987` at a close of 1222.7, `0.0019983` at
1223.7. That is the ceil-sizing consequence visible in live data: 82 shares of
a 1222.7 stock turn over 100,261 rupees, slightly above the one-lakh clip, so
the capped brokerage spreads over more turnover and the hurdle comes in
fractionally below the clip figure. Screening at the clip figure would be too
strict, never too loose.

### Teardown noise on stderr

Run B exited cleanly but printed to stderr:

```
Task was destroyed but it is pending!
task: <Task pending name='Task-119'
       coro=<NatsClient._unsubscribe() ...>
```

This is the vendor's asyncio teardown, not our error: `NatsClient._unsubscribe`
is still pending when the loop is torn down. Run A printed nothing. **It is
intermittent, it does not affect the exit code, and it is not a failure.** Do
not chase it; do not treat a clean exit with this line on stderr as a bad run.

### Building a fake that matches this

The point of the measurements above is that none of them need a live market to
reproduce. A fake stream factory driven at these parameters puts a component in
front of the feed's real observed behaviour at any hour:

| Parameter | Value to use | Source |
|---|---|---|
| session window | 60 s | `DEFAULT_SESSION_SECONDS` |
| P(session delivers nothing) | **~0.5** | 7/14 and 7/15 |
| ticks per productive window | **~19–20** | 143/7, 152/8 |
| mean tick interval while flowing | ~3 s | ~20 ticks / 60 s |
| tick payload volume | **absent** — poll for it | vendor limitation |
| `last_trade_at` lag | 0.6 s min / 2.4 s median / 8.2 s max | earlier session |
| poll round | 2.1–2.7 s | 2.18 s, 2.17 s, and prior runs |
| P(poll total regresses) | **0.02–0.05** | 9/413, 17/415, and prior |
| minutes absent per 15 min | 2–3 | observed directly |
| connect failure | intermittent; 30 s timeout | run A had 2, run B had 0 |

A component that stays correct against a fake that goes silent half the time,
occasionally reports a volume total lower than the previous one, hands back
candles with minutes missing, and intermittently refuses to connect has met
this feed's observed worst behaviour without waiting for 09:15.

What such a fake still cannot give you is a *surprise* — a payload shape or
failure mode nobody has seen yet. That is the standing reason to re-run a live
window occasionally even when nothing appears to need it, and to extend this
table when one disagrees with it.



Three values in the JSON files are artefacts of the capture scripts, not of
Groww. They are listed here so a future reader does not treat them as data.

1. **`api_capture.json` → `get_user_profile.result.exchange_enablement:
   "<mappingproxy>"`.** The serialiser has no branch for `MappingProxyType` and
   falls through to `f"<{type(value).__name__}>"`. The real value is
   `{"NSE": true, "BSE": true}`, fixed by the `MappingProxyType` literal in
   `get_user_profile`.

2. **`api_capture_raw.json` → `raw_get_historical_candles.candle_count: 4`.**
   The true count is **362**. The script's `scrub()` truncates every list to
   its first four elements, and `main()` then counted the already-scrubbed
   list.

3. **`api_capture_raw.json` → `last_candle: ["2026-09-21T09:18:00", ...]`.**
   Same cause. It is the 4th row of the session, not the last, and each row is
   itself cut from six elements to four — which is why close and volume are
   missing from the raw candle rows quoted above. The normalized capture has
   the real last candle and the full six fields.

## Redaction and the `ucc` leak

Both capture scripts redact dictionary keys and dataclass fields whose names
match a credential hint list, recursively. `api_capture.py` additionally never
serialises the broker object.

**One field escaped: the raw profile's `ucc`,** a 10-digit Unique Client Code
that identifies the trading account. The raw script's hints include `"client"`
and `"account"`, but the literal key `"ucc"` matches neither. `vendor_user_id`
was caught (it matches `"user_id"`); `ucc` was not.

* It is written in this document as `"<10-digit UCC>"`.
* It is **not** in the repository — `api_capture_raw.json` lives in the system
  temp directory, outside the working tree.
* It never enters the system's own types: `_GrowwProfilePayload` uses
  `ConfigDict(extra="ignore")` and declares only the four fields it needs, so
  `ucc` is dropped at the pydantic boundary and `BrokerProfile` cannot carry it.

If these scripts are ever re-run, add `"ucc"` to the hint list first. More
generally: a name-based redaction filter fails silently on any key it does not
recognise, so every value copied out of a capture must be reviewed by eye
before it lands in a document.

## Regenerating this

The scripts are not in the repository. To recapture, write a script that:

1. Calls `load_groww_settings()` and `GrowwBroker.authenticate(...)`.
   **Never read or print `.env`** — loading environment variables in code is
   permitted; inspecting their values is not (`AGENTS.md` rule 11).
2. Records the normalized return of each wrapper, plus a burst of `get_quote`
   to measure the retry layer.
3. Records the raw `broker._client.*` payloads, retrying each until one
   returns — a single raw call usually fails.
4. Scrubs credential-shaped keys recursively **and** reviews the output by eye.
5. Writes outside the repository.

Run it on a weekday before 15:30 IST if the stream matters; any time otherwise.
