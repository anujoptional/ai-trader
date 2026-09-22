# Feature accuracy validation

The feature engine's job is to turn candles into 47 numbers that a scanner will
later trade on. Tests prove those numbers are *stable* and that unavailable
inputs stay unavailable; they do not prove the numbers are *right*, because a
test written from the same understanding as the code agrees with the code even
when both are wrong.

This document records an independent check of that second question, and the
conventions a reader comparing against TradingView or any charting package will
trip over. It exists so the comparison does not have to be re-derived, and so a
difference against a chart can be classified as expected or investigated
without a live market.

## What was checked

All 47 derived features were recomputed by a separate float64 implementation
written from textbook definitions rather than from the engine's source, and
compared element-wise against the engine's Decimal output over 362 real
one-minute RELIANCE candles from the 2026-09-21 NSE session (09:15 to 15:29
IST). Only rows where both sides produced a value were compared, so a feature
withheld for want of history was never scored against a reference that had
invented one.

**All 47 agree.** Typical agreement is 1e-13 relative or better, which is
float64 noise rather than a difference in method. The three features that
first appeared to disagree were each traced to a fault in the reference, not
in the engine:

| Apparent mismatch | Cause | Resolution |
| --- | --- | --- |
| `bollinger_percent_b_20` | Reference lost precision to float64 cancellation. At the worst row the absolute deviation is 2.1e-10 against a %B of 0.01 — the relative figure is large only because the denominator is small. | Engine is the more accurate of the two. Not a defect. |
| `volume_ratio_20` | Reference used TradingView's convention, which includes the current bar in its own baseline. | Matches to 1.3e-15 once the lagged baseline is used. Deliberate divergence — see below. |
| `session_range_pct` | Reference divided the session range by the session open. | Engine divides by the session low, as its docstring states. Matches to 2.1e-14. |

Worst relative deviations, once the reference was corrected, by family:

```
ema9/21/50            5.5e-16 .. 2.6e-15      sma20                 1.8e-16
rsi14                 1.4e-13                 bollinger_upper/lower 1.8e-13
macd/signal/histogram 3.0e-10 .. 5.1e-11      bollinger_bandwidth   5.5e-10
true_range / atr14    1.4e-12 / 1.6e-13       vwap / deviation      7.3e-16 / 2.1e-10
plus_di / minus_di    2.8e-13 / 3.8e-13       rolling_high/low_20   exactly 0
adx14                 2.4e-13                 obv                   exactly 0
return_1/5/15         ~3.7e-12                session aggregates    exactly 0
```

## Comparing against TradingView

The conventions already match the common charting default in every case where
one exists, so a chart comparison is meaningful rather than apples-to-oranges:

- **RSI, ATR, DI and ADX use Wilder smoothing**, not a simple moving average.
  This is what TradingView's built-in `ta.rsi`, `ta.atr` and `ta.adx` use.
- **EMAs are seeded with the SMA of the first `period` closes.** Charting
  packages that seed from the first close instead will differ for the first few
  dozen bars and converge after that; the difference is a seeding choice, not
  an error in either.
- **Bollinger Bands use population standard deviation** (ddof=0) at 2σ around
  a 20-period SMA, which is how the indicator is defined and what charting
  packages compute.
- **VWAP uses the typical price** `(high + low + close) / 3`, anchored to the
  session, matching TradingView's session VWAP.
- **MACD is 12/26/9** on closes.

Three differences are expected and are **not** defects:

1. **`volume_ratio_20` excludes the current bar from its own baseline.** It
   compares this candle's volume against the mean of the *previous* twenty,
   where TradingView's usual `volume / sma(volume, 20)` includes the current
   bar in the denominator. On a volume spike the two diverge sharply — the
   lagged form reports a larger ratio, because the spike has not been allowed
   to dilute its own baseline. The reasoning is in `_volume_ratio`'s docstring
   in `features/engine.py`. Neither is time-of-day adjusted, so neither is a
   true RVOL.

2. **Session aggregates are withheld unless the session was observed from its
   opening range.** An engine first attached at 11:00 has not seen the day's
   high, so it reports `None` for the whole session rather than the highest
   price it happens to have seen. A chart always shows a value. VWAP is the
   documented exception and is computed from whatever the engine has seen.

3. **Volume-derived features go unavailable rather than approximate.** A single
   unknown volume in a window withholds the whole value, and one unknown volume
   disables strict session VWAP for the rest of the day. A chart backfills from
   the exchange and will show a number where this engine shows `None`. That
   asymmetry is the point — see ARCHITECTURE section 8.

### A row to spot-check by hand

Final candle of the 2026-09-21 session, RELIANCE NSE, one-minute, timestamps in
UTC (add 5:30 for IST):

```
start 2026-09-21T09:58:00Z   end 2026-09-21T09:59:00Z
OHLC  1247.4 / 1247.4 / 1247.4 / 1247.4      volume 531468

ema9    1246.873350   rsi14   55.348082   sma20   1247.375000
ema21   1247.001863   macd    -0.028439   bb_up   1248.787622
ema50   1246.494934   atr14    0.576205   bb_low  1245.962378
adx14     21.667573   vwap  1240.955698   obv     1120524
session_high 1248.9   session_low 1232.5   session_volume 9885947
```

`volume_ratio_20` on this candle is 11.99 — the closing-minute spike, and a
good row on which to observe divergence 1 above, since TradingView's
convention will report a visibly smaller figure.

## Reproducing

```bash
python -m ai_trader.cli.check_features --export-csv features.csv
```

That writes one row per candle at full precision, which is the input the
comparison consumed. The reference implementation itself was a throwaway
script outside the repository and was not kept.

## Why this is not in the test suite

The reference used numpy and pandas. Both are present in `.venv`, but only as
transitive dependencies of `growwapi` — `constraints.txt` records that AGENTS.md
rules them out as things this codebase may use, and nothing under `src/` imports
either. Adding a test that imports them would quietly convert a broker SDK's
dependency into one of ours.

Making this a standing CI guard therefore needs a reference written against the
standard library alone. That is worth doing — it would catch a regression in an
indicator that no current test would notice — but it is new development rather
than validation of what exists, and has not been started.

TA-Lib and pandas-ta were considered as third-party references and are not
installable here: `files.pythonhosted.org` is unreachable from this machine
(`SSLV3_ALERT_HANDSHAKE_FAILURE`) even though the index itself resolves.

## What this does and does not establish

It establishes that the arithmetic is right: every feature matches an
independently written implementation of its textbook definition over a real
session, and the Decimal path is at least as accurate as float64 everywhere,
better where cancellation bites.

It does not establish that the feature *set* is the right one to trade on, that
the parameters (9/21/50, 14, 12/26/9, 20) suit intraday NSE equities, or that
the engine behaves correctly on a session it has not seen — a halt, a corporate
action, an illiquid symbol with minute gaps. Those are questions for the replay
engine, which does not exist yet.
