# RELIANCE Score Research Preparation

Status checked 2026-09-30. **The authorized September 1-28 data remains frozen.**
The first [signed target-time study](signed_score_v1/README.md) has analysed
September 1-4 only and saved an opt-in prototype. The original strategy is
unchanged. The later [63-session probability study](../tp_probability_2026q3/README.md)
uses September 21-28 for final evaluation under the user's expanded date
authorization. Those dates are no longer an unused holdout for ongoing research.
This does not imply calibrated scores, measured fills or live-trading approval.

## Data Gate

Requested: NSE:RELIANCE, one-minute candles, September 1-28 inclusive, with
August 31 used only for feature warm-up. The declared calendar has 19 evaluation
sessions; September 14 is Ganesh Chaturthi, not a missing trading session.

The freeze used cached Groww data, with no network calls or credential loading:

- 6,897 evaluation candles across 19 sessions, plus 367 August 31 warm-up candles.
- Every session has all 360 one-minute bars before the 15:15 square-off.
- Duplicate checks, INR 0.10 price-grid checks and core feature warm-up passed.
- No candles were invented and Yahoo data was not substituted for Groww.

The manifest is now present beside this file. Sparse bars after square-off and
unavailable volume remain as supplied and are recorded in its per-session audit.
The evaluation protocol excludes any label whose horizon exceeds square-off.

September 29 is excluded by the user's explicit approval on 2026-09-30, not by
an implicit fallback. The earlier September 1-29 attempt was refused because
Groww's day request ended at 13:59 and its 14:00-15:30 request returned zero bars,
leaving 75 missing pre-cutoff minutes. That blocker no longer affects this window.

The command below verifies the baseline, including its original source hashes.
An intentional scorer change correctly reports code drift; do not rewrite that
historical manifest. Child experiments check parent data/config hashes and
record their own source hashes, as the signed target-time report does.

```powershell
.venv/Scripts/python.exe -m ai_trader.cli.prepare_scanner_research --plan backtests/reliance_sep2026/plan.json --verify
```

The snapshot location is `data/research/reliance_sep2026`. Its manifest records
CSV hashes, requested-span sidecar hashes, session coverage, the plan hash, the
resolved strategy hash and decision-code hashes. Data and raw external responses
remain under gitignored `data/`; the compact evidence and protocol live here.

## Shared Settings

`strategy.json` is a complete schema-versioned configuration accepted by BOTH
`ai_trader.cli.backtest --strategy-config PATH` and
`ai_trader.cli.check_scanner --strategy-config PATH`. Strategy overrides cannot
be mixed with the file. Decimal parameters are strings, missing/unknown fields
are rejected, and both outputs report the same resolved SHA256.

All five built-in rules are bound to the immutable `StrategyConfig.scoring`.
Thresholds and the three component weights are explicit; default scores are
unchanged. Feature definitions remain shared production code and are pinned by
code hashes, not duplicated in an experimental implementation.

The baseline's instrument tick is **INR 0.10**, not the legacy INR 0.05 fallback.
[NSE CMTR67133](https://nsearchives.nseindia.com/content/circulars/CMTR67133.pdf)
sets that band for prices above INR 1,000 through INR 5,000, reviewed monthly.
RELIANCE's cached August-end close was INR 1,277. This profile is specific to
this instrument and month; it is not a universal tick change.

For later secondary outcome evaluation, this is a reproducible replay command.
It was not run during preparation, to avoid inspecting holdout outcomes:

```powershell
.venv/Scripts/python.exe -m ai_trader.cli.backtest --symbols RELIANCE --start 2026-09-01 --end 2026-09-28 --warmup-start 2026-08-31 --cache data/research/reliance_sep2026 --offline --strategy-config backtests/reliance_sep2026/strategy.json --latency-seconds 1 --half-spread 0.0002 --slippage 0.0001 --no-write
```

Warm-up updates features only: no warm-up candidates, positions, trades or scored
sessions. The replay API rejects warm-up that overlaps the evaluated window.
Replay fill assumptions are separate from the shared strategy: live observes
fills, replay models them. A strategy hash alone does not establish equal inputs,
warm-up, portfolio state or live-vs-historical candle availability.

## Declared Evaluation

The original protocol is in `plan.json`. The revised signed speed-to-target
objective is separately declared in [its protocol](signed_score_v1/protocol.json),
leaving this baseline protocol intact:

| Partition | Dates | Expected Sessions |
| --- | --- | --- |
| Warm-up only | August 31 | 1 |
| Development | September 1-18 | 13 |
| Validation | September 21-23 | 3 |
| Tuning holdout | September 24-28 | 3 |

Primary label: 15-minute direction-adjusted close-to-close return. Secondary
horizons: 5 and 30 minutes, with favourable/adverse excursions and stated-cost
sensitivity. Labels use subsequent candles only, require complete exact horizons,
and cannot extend past square-off or overnight. They are opportunity labels,
not a claim that orders fill at the signal close.

Evaluate raw production-rule scores before portfolio suppression, feasibility
filtering and top-N; retain feasibility as a separate observation. Analyse LONG
and SHORT separately and by rule. Report score-bucket monotonicity, rank
correlation, saturation, distinct-session counts and session-block uncertainty.
Adjacent minutes are correlated: no random row splits or per-row confidence
claims. Select parameters on development, check validation, then evaluate the
holdout once after freezing the choice. Earlier aggregate backtests included
September 24/25; this is a newly reserved tuning holdout, not pristine history.

## Timestamp Validation

`feature_parity.json` records an actual online Yahoo-vs-Groww comparison over
September 23-24. Both start from the same requested epoch. Yahoo prices have
only their small binary representation residue rounded to paisa; no missing
rows or volumes are filled. The saved public response and every per-feature
timestamp comparison are under `data/research/`.

- Independent 60-digit formulas checked all 17 scanner inputs: **23,891 numeric
  comparisons passed** across 723 Groww bars and 722 Yahoo bars.
- These formulas are now retained in `scripts/validate_scanner_features.py` and
  tested on ordinary, flat-price and missing-volume fixtures.
- At 722 common timestamps, closes agreed exactly in 562 cases and volume in
  66. The largest close difference was INR 1.90; the largest RSI difference was
  11.9344 points. Opening-range bounds agreed wherever available.
- A one-minute shift does not explain it: exact close matches at offsets
  -1/0/+1 were 96/562/104.

Example: September 24 bar **10:06 IST**, decision **10:07 IST**. Rounded for
display; the CSV retains full precision:

| Feature | Production on Groww | Reference on Yahoo |
| --- | --- | --- |
| EMA9 | 1238.147425 | 1238.133143 |
| RSI14 | 45.520020 | 45.826039 |
| ATR14 | 0.613969 | 0.595011 |
| ADX14 | 26.123709 | 25.705122 |
| Opening-range high | 1241.20 | 1241.20 |

**Arithmetic validation passed; exact cross-feed feature parity did not.** This
is not a comparison with TradingView's displayed indicators. The two real
providers supply different candles/volumes, so equal formulas need not produce
equal values. Neither feed has been proven exchange ground truth. Keep Groww as
the declared research source; do not adjust formulas to conceal feed differences.
Live-vs-historical input fidelity remains a separate validation concern.

Reproduce offline from the saved external response:

```powershell
.venv/Scripts/python.exe -m scripts.validate_scanner_features --start 2026-09-23 --end 2026-09-24 --reference data/research/reliance_yahoo_20260923_24.json --output backtests/reliance_sep2026/feature_parity.json --overwrite
```

Spread, slippage and latency in the protocol are stress scenarios, not measured
facts. Published fee schedules have not been reconciled with a contract note.
No preparation or backtest result authorizes live trading.

## Verification

The completed preparation code passed the repository gates on 2026-09-30:
`pytest -q`: 906 passed; `ruff check .`: clean; `ruff format --check .`:
98 files already formatted. No source changes were committed.