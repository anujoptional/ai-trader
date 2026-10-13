# AI Decision Layer And Data Components

Research and shadow mode only. Nothing here places, modifies or sizes orders;
verdicts are journaled, and exits, sizing and risk stay in deterministic code.

## Current System: Paper Trading (October 12-13)

Live paper trading, delivery (CNC) longs only; no order is ever placed. The
consolidated current state, daily operation, setup, results and next steps are
in `docs/handover.txt` (section "CURRENT STATE"); this section is the detail.

1. **Scanner** (`scanner/upside.py`, model `data/models/scanner_up1_by_close_v1`):
   every 5 minutes from 09:40 to 14:55, P(+1% above the buy price before the
   session ends) for every stock in the universe that is not held and not already
   bought today; results sorted by probability, candidates at or above 0.60.
   Universe (from October 13): the top 500 by NSE market cap (NSE's daily
   `mcapDDMMYYYY.csv`, `scripts/prepare_mcap_universe.py`), plus 150 reserves; each
   scan uses the 500 highest-ranked stocks it can score, so a reserve replaces any
   stock that cannot be. A minute without trades (Kite has no bar for it) is the
   previous close with zero volume, today and in history; a stock with more than
   10% such minutes is skipped as too illiquid. Speed: a live Kite tick stream builds
   every stock's minute bars, so the whole universe is pre-screened in about a
   second; stocks within 5 points of the threshold, the top 10 and the 5 best F&O
   stocks are then re-scored on Kite's historical bars (the validated path) before
   anything reaches the AI. Without a usable stream the scan fetches every stock
   (about 4 minutes) and the result seeds the stream's bars. Session end: 15:29, or
   15:14 for F&O stocks from 2026-08-03 (NSE closing auction).
2. **Components** for each candidate: order book (Kite depth); futures (F&O
   stocks only): premium over the stock now, days to expiry, OI of the near and
   next months together now vs the last close, and the previous session's OI
   change and build-up (both contracts' Kite day bars); NSE filings since the
   previous close and over 7 days, with NSE's own subject as the category and
   routine subjects (trading window, depository certificates, newspaper copies,
   shareholder meetings, analyst-meet notices and similar) not counted as
   material; and the intraday, daily and market features. Every value is as of
   the request: multi-day changes and levels are measured at the current price,
   and anything from the previous session is named `*_prev_session`.
3. **AI** (`DELIVERY_V4`, prompt `delivery-v4-2026-10-13`, from the evening of Oct
   13; `delivery-v3-2026-10-12` from 14:40 on Oct 12, v2 before): BUY or NO_BUY
   per candidate, from facts only. The request carries no
   scanner output (no probability, stocks in alphabetical order), plain field names
   (`ai/packet.py` `V3_*`), company and industry from the Nifty 500 list, and the
   order book, futures OI and NSE filings. Web search is required on every call
   (OpenAI `tool_choice: required`; Claude by instruction, and a Claude reply
   without a search is asked once more); each search's query and source URLs are
   saved with the reply. Each answer gives `p_up_first` (the AI's probability that
   the price rises 1% before it falls 1%) before the verdict and a short reason;
   v3 named that field `confidence` and put it after the verdict, which Luna
   sometimes answered as confidence in NO_BUY. NO_BUY is used only when a call
   fails, an answer is missing or the daily budget is spent. Every request and
   answer is journaled.
4. **Paper execution** (`paper/`): a BUY becomes a buy of whole shares worth up to
   Rs 2 lakh at the visible asks; a +1% sell limit and a -1% stop are placed at
   once, no time limit. Exits are checked every minute on completed one-minute bars
   with the research rules (a minute opening beyond a level fills at the open; a
   minute reaching both counts as the stop). Limits: one open position per stock,
   one buy per stock per session, 25 open positions, Rs 50 lakh capital. Charges:
   intraday if sold the same session, delivery otherwise.
5. **Portfolio and report** (`data/paper/`): `paper.sqlite` holds every position
   and an event trail (scans, verdicts, fills, refusals); the scanner skips held
   stocks. `report/` holds `trades.csv` (every trade, P&L, cumulative),
   `open_positions.csv` (holdings marked at the latest price, net if sold now),
   `daily.csv`, `summary.json` and `report.txt`, rewritten after every change.
6. **AI log** (`data/paper/decisions.sqlite`): per call, the exact request
   (`cycles.request_json`), the model's full reply (`cycles.response_json`), tokens,
   cost and latency; per candidate, verdict, confidence and reason (`verdicts`).
   The run log `data/paper/logs/<YYYYMMDD>.log` also prints every verdict with its
   reason, and any crash traceback.
7. **Shadow AIs** (`--shadow MODEL:EFFORT`, repeatable): another model/effort is
   asked exactly the same questions in the same cycle (same request objects when
   the candidate lists match) and keeps its own portfolio, journal and report under
   `data/paper/shadows/MODEL_EFFORT/`; the main AI's behaviour is unchanged. It skips
   only its own holdings, so it can be asked about stocks the main AI holds.
   Claude models (`claude-sonnet-5-5`, `claude-haiku-5-5`, `claude-opus-5-5`; needs
   `ANTHROPIC_API_KEY`) run through `ai/anthropic_client.py` with the same prompt,
   JSON schema, web search (queries and URLs journaled) and effort. Claude 5.5
   rejects a forced tool choice, so the search is required by instruction. Every AI
   has its own `AI_DAILY_BUDGET_USD` cap. Daily comparison of all AIs (calls, cost,
   verdicts, hindsight +1%/-1% first, Brier score, paper P&L):
   `python -m scripts.compare_ai_decisions --day YYYY-MM-DD` after 15:31.
   The run log shows one block per decision cycle: market, scanner picks and the
   next best, the facts each stock gets, each AI's searches, verdicts and cost,
   `>>> PAPER BUY` / `<<< PAPER SELL` lines, each AI's portfolio, and a day
   summary at the end.
   Auto-start: Windows task `AI Trader paper run` (weekdays 08:30, only while the
   user is signed in) runs `scripts/start_paper_day.cmd`. It opens the Kite sign-in
   and waits until 14:30 (`--login-wait 360`); weekends, NSE holidays
   (`/api/holiday-master`), starts after 15:31 and a second copy
   (`data/paper/run.lock`) exit at once. Kite sign-in itself stays manual.
8. **Input capture and validation**: each decision cycle saves its raw inputs
   (minute bars of candidates and the top 10, index bars, every order-book
   snapshot, futures OI, skip reasons, executions) to
   `data/paper/capture/<date>/<HHMM>.json`, and complete bars at the end of the
   run (`final_bars.json`). `scripts/validate_paper_inputs.py` checks them against
   NSE's official end-of-day files, NSE's index and announcement APIs, the research
   feature code and Kite's final bars, and dry-runs the paper buy and exit on every
   captured live order book (`--day --nse <bhavcopy dir> --previous --output`).

Production scanner: the tested goal-(a) recipe refitted on 2020-01 to 2025-06,
early stopping 2025 H2, calibration 2026 H1. Holdout Jul-Sep 2026 (never used):
AUC 0.742 (tested model 0.743), calibration error 1.8 points (tested 2.5); the
60-70% band hit 66.2% (22 estimates per session on 213 stocks), 70-80% hit 70.4%.

Validation (October 12, before the first live session):

- Live scanner inputs equal the research features (parity test on synthetic data,
  and 0.0 difference on six real candidates in the replay).
- Two-session offline replay of the real session code on all 213 F&O stocks
  (`scripts/replay_paper_session.py`, mechanical BUY-everything decider,
  `data/paper_validation/replay_buyall_final`): 22 trades (14 wins), net
  +Rs 9,429, every exit equal to an independent loop over the raw minute files,
  report totals exact, no errors.
- Real AI on 30 September (`data/paper_validation/replay_openai_20260930`): 12
  calls, all valid, prompt `delivery-v2-2026-10-12`, $0.0034 in total, 4.7 s
  average (9.1 s max); NO_BUY to all 29 candidates ("stop-first risk"), so no
  trades. Buy-all made 7 trades that day (4 wins, +Rs 1,159). This matches the
  research above: at 0.60 the candidates reached +1% before -1% only 48.8% of the
  time, so live sessions may well end with no trades.
- Tests: `test_upside_scanner.py`, `test_paper.py`, `test_paper_session.py`.

Verified live on October 12-13: Kite sign-in, the history load (650 stocks in
about 9 minutes; a same-day restart reuses `data/paper/cache`), the tick stream
(4.05 million ticks, no reconnects on October 13), scans in 6-7 seconds on the
stream (261 seconds for a full fetch), and every input against NSE's official
files (October 12 and 13; on October 13 the live pre-screen was within 3.3 points
of the exact re-score for 350 stocks, inside the 5-point shortlist margin). Not
yet seen live: a paper fill, because no AI has said BUY.
The scanner was trained on F&O stocks only; for the other stocks in the top-500
universe its probabilities are untested.

Start: automatic (Windows task `AI Trader paper run`, weekdays 08:30, see item
7); by hand, `scripts\start_paper_day.cmd` or:

```powershell
.\.venv\Scripts\python.exe -m ai_trader.cli.run_paper --shadow gpt-6.1-sol:low --shadow claude-sonnet-5-5:low
```

Live results (details in `docs/handover.txt`, section 7):

| Day | Scans | Decisions | BUY | Hindsight (+1% first / -1% first) | AI cost |
|---|---|---|---|---|---|
| Oct 12 (scans 12:25-14:55) | 28 | Luna low 57, Sol medium 24 | 0 | Luna's: 27 / 22 (6 no data) | $0.04 / $0.50 |
| Oct 13 (from 12:00) | 36 | 25 per AI (5 stocks) | 0 | 8 / 17 | Luna low $0.17, Sol low $0.70, Claude Sonnet low $0.55 |

October 13: every pick had fallen 4-8% that day and sat near its low while
NIFTY rose; all three AIs declined all 25, correctly in hindsight (buying each
stock's first pick: 2 wins, 3 losses, -Rs 2,594). Brier score of the stated
confidence: Luna 0.249, Sol 0.221, Claude 0.225 (guessing the day's base rate:
0.218). Fixed the same evening (prompt v4): Luna had twice given its confidence
in NO_BUY instead of the probability of +1% first (0.65, 0.82); on that exact
request v4 gave Luna 0.26, Sol 0.34, Claude 0.43. Claude, which skipped the web
search in 3 of 13 calls, is now asked again when it does not search.

## What The AI Receives, Field By Field (proven October 13)

The "AI seat" audit (night of October 13) started from what the AI reads and
worked back to Kite. Every October 13 request (13 requests, 25 stock decisions)
was rebuilt with the current code (`f5366d5` + `85aee5a`) from the inputs Kite
delivered live. Each value was then proven in four layers:

1. **Formula**: recomputed by separate code from the exact Kite inputs used. It
   must match to the sent precision (3 decimals; futures to 1e-9 before rounding).
2. **Inputs**: those Kite inputs against independent sources. NSE's official files
   must match exactly: equity and F&O bhavcopies, index closes, 52-week list,
   price bands, corporate actions, symbol changes, the announcements API. Yahoo
   Finance minute bars, a separate vendor, must agree within tick-level noise.
3. **Freshness**: every value is as of the request (`as_of`).
4. **End to end**, for information: the same values computed from independent
   inputs only (Yahoo prices on NSE history).

A field can be computed correctly and still be stale. The earlier checks matched
values to the research code and NSE closes. That passed the old `ret5d`, sent
as "5-day change": it was correct for its definition but ended at the previous
close. For PAYTM at 12:45 it showed +0.36%, Oct 12 close Rs 1,690 / Oct 5 close
Rs 1,684. The current field is -4.78%: Rs 1,613.00 at 12:44 (Kite = Yahoo) over
the Oct 6 NSE close of Rs 1,694. Every field below was checked both ways: correct
value, and true as of `as_of`.

| Block: field | Definition (as computed) | Kite source | Independent proof | Result |
|---|---|---|---|---|
| market: `nifty_change_today_pct`, `_15min_pct` | NIFTY at the last completed minute vs previous close / 15 minutes earlier | index minute bars, daily closes | formula; NIFTY closes = NSE; vs Yahoo ^NSEI | 13/13 exact; 64/64 closes; Yahoo median 0.002 pt |
| market: `nifty_change_20d_pct`, `nifty_vs_50d_avg_pct` | NIFTY now vs close 20 sessions ago / mean of last 50 closes | same | same; calendar = NSE (60 sessions) | 13/13 exact |
| market: `india_vix`, `_change_today_pct` | VIX at the last minute; vs previous close | VIX minute bars, daily | previous close = NSE; vs Yahoo ^INDIAVIX | exact; Yahoo median 1.75 bp (VIX is recomputed continuously) |
| today: `last_price` | close of the last completed minute | minute bars (a minute without trades = previous close, zero volume) | formula; Yahoo same minute; quote LTP inside Yahoo's minute range | 25/25; 25/25 |
| today: `minutes_to_close` | minutes of continuous trading left: to 15:15 for F&O stocks (closing auction from 2026-08-03), else 15:30 | NFO list | F&O flag = NSE F&O list; Kite's F&O minute bars end at 15:14 (Yahoo: then only a 14-share print and the auction) | 25/25 |
| today: `change_today_pct`, `gap_at_open_pct`, `change_since_open_pct` | price vs previous close; open vs previous close; price vs open | quote, minute bars, daily | previous close = NSE 205/205; first-minute open = NSE open 42/42 | 25/25 exact |
| today: `change_5min/15min/60min_pct` | price vs close 5/15/60 minutes earlier | minute bars | formula; Kite vs Yahoo minute closes median 0.0 bp | 25/25 exact |
| today: `vs_nifty_today_pct` | stock's change today minus NIFTY's | both | formula | 25/25 |
| today: `vs_vwap_pct`, `vs_day_high_pct`, `position_in_day_range` | price vs VWAP (typical price x volume); vs today's high so far; (price - low)/(high - low) | minute bars | day high/low = NSE official 38/38 non-F&O and 4/4 lows F&O (one F&O high was set in the closing auction, after any decision) | 25/25 exact |
| today: `day_range_vs_avg_range` | (high - low)/price over the 20-day ATR | minute bars, daily | formula | 25/25 |
| today: `volatility_20min_pct`, `volatility_vs_usual` | sqrt of the sum of squared 1-minute log returns over the last 20 minutes; over the median of 20 sessions' 09:15-14:59 volatility scaled to 20 minutes | minute bars, minute history (25 sessions) | history sessions dated by matching Yahoo prices 95/95 | 25/25 exact |
| today: `volume_vs_usual` | volume so far over the mean volume to the same minute in 20 sessions | same | Kite vs Yahoo session volumes within 0.6% (95/95) | 25/25 exact |
| daily: `change_5d/20d/60d_pct`, `vs_50d_avg_pct`, `vs_200d_avg_pct`, `vs_250d_high_pct`, `vs_nifty_20d_pct` | price now vs close N sessions ago / mean of N closes / max(250 highs, today's high); 20-day change minus NIFTY's | daily history (adjusted, see below) | formula; history vs NSE (below); 250-session high vs NSE's 52-week list | 25/25 exact |
| daily: `avg_daily_range_20d_pct`, `volume_5d_vs_20d`, `avg_daily_turnover_20d_cr` | ATR(20) / previous close; mean volume 5 vs 20 sessions; mean close x volume, 20 sessions, crore | daily history | formula; history = NSE | 25/25 |
| order_book: `spread_bps`, `imbalance_*`, `microprice_vs_mid_bps`, `impact_buy/sell_2L_bps` | from the 5 best levels; impact = average fill of a Rs 2 lakh market order vs the mid | Kite quote at the decision | formula on the captured quote; 0 empty levels in 5,790; quote 0-3.4 s before `as_of`, last trade 0-5 s before the quote | 25/25 exact; no historical depth exists anywhere, so only internal checks are possible |
| order_book: `to_upper/lower_circuit_pct` | distance to Kite's circuit limits | quote | limits = NSE close x (1 +- band), band from NSE's list (F&O 10%), rounded inward to the tick | 24/24 stocks exact |
| futures_oi: `days_to_expiry` | calendar days from today to the near-month expiry | NFO instruments | = NSE expiry date | 20/20 |
| futures_oi: `premium_now_pct` | near-month future mid / stock mid - 1, same quote batch | quotes | formula | 187/187 (36 cycles) |
| futures_oi: `oi_change_today_pct`, `buildup_today` | near + next OI now vs both at the last close; build-up from the near month's move since its close (dead bands 0.2% price, 1% OI) | quotes, day bars | formula; previous OI per contract = NSE 374/374 | 187/187; no independent intraday OI exists (at 14:55, 4 of 5 within 3% of NSE's end-of-day OI, max 4.7%) |
| futures_oi: `*_prev_session`, `oi_change_5_sessions_pct` | both contracts' day bars; withheld when the window starts before the last expiry (Sep 29) | day bars | Kite futures bars = NSE F&O bhavcopy: settlement, OI, volume = contracts x lot, 480/480 | 187/187 |
| nse_filings: counts, subjects, meetings, actions | since the previous close (15:30), last 7 days, NSE subjects, material = not routine, board meetings next 3 days, corporate actions today/tomorrow (`material_filings` lists the same filings) | NSE APIs (stored copy) | fresh NSE read, only filings public by `as_of` | 25/25 for each of the 6 fields |
| `company`, `industry` | Nifty 500 list | universe file | company = NSE filings' company name | 25/25 |
| freshness | — | — | `as_of` minus the last bar's close; minus quote times; history ends on the previous session | at most 29 s; at most 3.4 s (stock quotes), 2 s (futures); 25/25 |

End to end, from Yahoo prices on NSE history alone, changes and averages agree
within 0.0002-0.004 points in the median. The worst case, 0.16 point for CEMPRO
at 13:10, was one minute where the two vendors' last ticks differed. Fields that
use the day's high or low, VWAP, volatility or volume differ more
(`vs_day_high_pct` median 0.10 point). Yahoo's 09:15 bar leaves out the opening
trades, so it misses highs set at the open, such as CEMPRO's Rs 1,179, which Kite
and NSE both show. Vendor ticks also differ, and Yahoo has only 19 of the 20
history sessions.

**Kite's daily history is adjusted for corporate actions.** Every row matched NSE
after accounting for that. Without a later action: 147,209 of 147,209 rows equal
NSE exactly (open, high, low, close, volume; all 650 stocks; renames from NSE's
`symbolchange.csv`). Rows before an action: 9,985 of 9,987 equal NSE's raw prices
times the factor of each later ex-date. Two TATAINVEST closes from October 2025
differ by 0.04%. How Kite treats each kind of action:

- splits and bonuses: the exact ratio, 25/25;
- rights: the theoretical ex-rights price from NSE's terms (issue price = premium
  + face value), 4/4;
- dividends of at least 2% of the price: 1 - dividend / cum-date close, 49/49;
  TATASTEEL (2.02%) and INDIAMART (regular + special) were left unadjusted;
- every other action (dividends under 2%, buybacks, meetings): unchanged,
  716/716.

OPEN, not changed (needs the user):

- Kite's three demerger adjustments are wrong. Kite's factor vs the market's
  (NSE ex-date open / cum close): VEDL 0.523 vs 0.374, TMPV 0.689 vs 0.605,
  GUJENERGY none vs 0.853. HINDUNILVR, GENUSPOWER and TVSHLTD were left
  unadjusted, but their gaps are under 2%. The false drop distorts these stocks'
  long-window fields until it leaves the window. At the Oct 12 close VEDL showed
  `vs_200d_avg` -16.3% (market basis -2.6%) and `vs_250d_high` -36.3% (-26.4%);
  GUJENERGY -32.9% (-27.3%) and -48.3% (-41.5%); TMPV `vs_250d_high` -41.2%
  (-33.1%).
- Kite's dividend adjustment differs from the research basis the scanner was
  trained on (`scripts/build_price_basis.py`: NSE raw prices, splits, bonuses and
  demergers, no dividends). It moves `vs_200d_avg` / `vs_250d_high` by more than
  1 point for 31-32 of 600 stocks and `change_60d` for 13. For example,
  CHENNPETRO's 60-day change is 23.4% including its Rs 54 dividend, 18.2% on
  price alone.
- A full-fetch cycle (first scan after a late login, or no stream) takes 3-5
  minutes, and the request is stamped after it. Its bars are then that old while
  `as_of` says now. All Oct 12 requests were 3.4-5.3 minutes behind (no stream
  existed then). On Oct 13 the 12:00 full fetch took 5.3 minutes but had no
  candidates; every Oct 13 request was within 29 s.
- Late prints may revise the decision minute after it was used: 1 of 25 on Oct 13
  (PAYTM 13:10, close 1,595.0 used vs 1,595.7 final).

Re-run (this machine; the scripts are in the untracked `.scratch/`):
`ai_seat_rebuild.py` needs a Kite login and rebuilds the requests.
`ai_seat_fetch.py` downloads NSE and Yahoo data into
`data/validation_20261013/independent/`. Then run `ai_seat_final.py`,
`ai_seat_adjust.py` and `ai_seat_explain.py`. Results:
`data/validation_20261013/ai_seat/final.txt` (summary), `final.json`,
`adjustments.json`, `explained.json`. Yahoo keeps minute data for 30 days only.
For a live day, `scripts/validate_paper_inputs.py` remains the routine check.
The audit also found and fixed one bug (`85aee5a`). The futures bar parser
rejected NSE settlement closes outside the day's traded range (SAIL26OCTFUT on
Sep 24: low 185.06, close 184.53), so SAIL had no futures facts at all.

## Current Design: Delivery Pipeline (October 12)

The user's design, delivery (CNC) trades only:

- **A. Scanner plus direction components.** The scanner is a CatBoost model fitted
  for one question: will the price touch +1% above the buy price within the next two
  hours? Candidates are stocks whose calibrated probability is above 0.75. Direction
  components stay independent: futures open interest, the order book, NSE filings,
  and the indicator features themselves.
- **B. AI.** One BUY or NO_BUY per candidate (`DELIVERY` policy in `ai/decision.py`,
  prompt `delivery-v1-2026-10-12`); the live decider may also use web news.
- **Sell rule.** The first of +1% or -1% from the buy price, with no time limit.

Evaluation script: `scripts/research_delivery_pipeline.py` (tests in
`tests/test_delivery_pipeline.py`). Only components with history are tested: the
order book (Kite keeps no history) and web news (a search cannot be limited to past
dates) are not; open interest is the previous session's futures data, because
intraday OI has no history either.

How it is tested: 213 F&O stocks, decisions every 5 minutes from 09:40 to 13:00 (so
the two-hour window ends inside the session). Inputs: 33 intraday, daily and market
indicators; two that duplicate others (|Spearman| > 0.9: 20-day realised volatility
~ ATR, RSI ~ distance from the 50-day average) are dropped. Scanner fit on 2020-2023,
early stopping on 2024 H1, Platt calibration on 2024 H2 (isotonic was rejected
because on 2024 H1 it put 2,023 rows at 0.955 that hit only 79%); test January 2025
to September 2026; the AI only from June 2026, after its knowledge cutoff. Each
trade buys Rs 2 lakh in whole shares at the next minute's open; at most one buy per
stock per day and none while the stock is held; the sale fills at the level, or at
a minute's open when the price opens beyond it, and a minute touching both levels
counts as the stop. Charges: Zerodha intraday when sold the same session, delivery
(STT 0.1% each side, stamp duty, DP fee) otherwise. 40 random trades were
recomputed independently from the raw minute files with no mismatch.

Results at the specified 0.75 threshold
(`data/research/delivery_pipeline_20261012/report.json`, test January 2025 to
September 2026; net returns per trade after charges, 95% day-resampled range):

| Component | Result |
|---|---|
| Scanner quality | AUC 0.705 (base rate 16.8%); above 0.75, 82.5% did touch +1% within two hours (mean forecast 78%) |
| Scanner as the buy signal | 452 stock-moments on 68 of 431 days; 200 trades; 47.0% reached +1% before -1%; -0.12% per trade [-0.33, -0.01]; -Rs 46.7K; every trade closed the same session |
| Random entries, same days | 401 trades; 45.9%; -0.20% per trade |
| Open interest (previous session) | No useful pattern: neutral buildup 36 trades +0.11%, long buildup 14 trades -0.20%; "buy on long buildup or short covering" 35 trades -0.14% |
| NSE filings | Material filing since the previous close 82 trades -0.08%; none 118 trades -0.14% |
| Features | Position in the day's range (AUC 0.63) and distance from the day's high (0.61) separate winners on 200 trades, unconfirmed (31 inputs tested); a CatBoost model on all features scores AUC 0.52 and buys everything |
| AI (gpt-6-luna, low and medium) | Only 18 rounds in June-September 2026, five stocks after bad news (result gap-downs, a CEO resignation); both efforts said NO_BUY to all 18; buying all five would have lost 0.26% per trade |
| Order book, web news | Not tested (no history) |

The scanner finds stocks that will touch +1% soon, but it does so by finding
volatile stocks, which touch -1% first about half the time, so buying its
candidates loses roughly the charges. The 0.75 threshold leaves too few candidates
(five stock-days in the AI window) to judge the AI.

Diagnostic at 0.60 (`data/research/delivery_pipeline_20261012_p60/report.json`,
same code, only to give each component more trades; not the design):

| Component | Result, test January 2025 to September 2026 |
|---|---|
| Scanner | 7,717 stock-moments on 298 of 431 days; 63.3% touched +1% within two hours (mean forecast 65.5%) |
| Scanner as the buy signal | 1,524 trades; 48.8% reached +1% first; -0.085% per trade [-0.159, -0.011]; -Rs 258K; 99% closed the same session |
| Random entries, same days | 5,677 trades; 49.0%; -0.129% [-0.223, -0.038] (more overnight holds, so more delivery charges) |
| Open interest | Futures existed for 916 of the 1,524 trades (608 came before the stock entered F&O); win rate 45-51% in every buildup group; every OI input AUC 0.49-0.53; the buildup rule 233 trades -0.07% |
| NSE filings | Material filing since the previous close 567 trades -0.084%; none 957 trades -0.086% |
| Features | Every input AUC between 0.46 and 0.53; the two that looked useful at 0.75 fell to 0.52-0.53 |
| AI | 267 rounds (288 candidates) in June-September 2026: NO_BUY to all at low and medium effort ($0.04 and $0.05); buying all 74 would have made +0.075% per trade [-0.20, +0.33] |

Conclusion: the scanner's top inputs are ATR and recent volatility, so it predicts
how much a stock will move, not which way; with equal +1% and -1% exits, only
direction matters, and none of the historical components predicts it. The AI
declines every candidate, so the full pipeline makes no trades. Next steps: let
the AI return its probability of +1% before -1% for every candidate (the buy
decision then becomes a threshold in code, and its judgement can be measured
even when it declines); record the live order book and intraday OI, the two
components history cannot test; and consider fitting the scanner to the trade
outcome itself, although the features model suggests little skill (AUC 0.52).

## Scanner Goals Study (October 12)

Scanner only, four goals, each with its own CatBoost model on the same 31
indicator inputs (`scripts/research_scanner_goals.py`, tests in
`tests/test_scanner_goals.py`; report `data/research/scanner_goals_20261012_v2/
report.json`). An estimate is one stock at one 5-minute check (09:40 to 14:55,
buy at the next minute's open): about 13,500 per session. Fit 2020-2023, early
stopping 2024 H1, calibration 2024 H2 (raw, Platt or isotonic, chosen on 2024 H1
log loss), test January 2025 to September 2026 (431 sessions). A minute reaching
both levels counts as the down move; outcomes still open at the data end or
crossing a missing/flagged session are left out of accuracy (0.2% of (b), 0.5% of
(c), 2.7% of (d) test estimates).

Closing auction: from 2026-08-03 NSE ends continuous trading in F&O stocks at
15:15 (auction 15:15-15:35), so the minute data stops at 15:14. Goal (a) uses that
session end, and the models' time input is the minutes left to it. The first run
(`scanner_goals_20261012`) treated those 42 sessions as incomplete and is
superseded.

| Goal (test) | Base rate | AUC | Skill vs base rate | Calibration error | Highest estimate |
|---|---|---|---|---|---|
| (a) +1% touched before the session ends | 23.3% | 0.732 | 11.4% | 0.9 points | 88% |
| (b) +1% before -1%, no time limit | 49.6% | 0.511 | -0.1% | 1.7 points | 61% |
| (c) +2% before -2% | 49.0% | 0.504 | -0.2% | 2.2 points | 57% |
| (d) +5% before -5% | 48.9% | 0.472 | -0.2% | 2.5 points | 53% |

Goal (a) by band (test; per session = estimates per session, stocks = distinct
stocks per session with at least one estimate in the band):

| Band | Per session | Stocks | Mean estimate | Actual |
|---|---|---|---|---|
| 0-10% | 2,099 | 155 | 6.7% | 4.3% |
| 10-20% | 4,051 | 201 | 15.1% | 13.6% |
| 20-30% | 3,367 | 206 | 24.7% | 24.8% |
| 30-40% | 2,139 | 172 | 34.6% | 34.9% |
| 40-50% | 1,177 | 113 | 44.4% | 44.7% |
| 50-60% | 531 | 56 | 54.2% | 54.2% |
| 60-70% | 157 | 19 | 63.8% | 63.6% |
| 70-80% | 22.4 | 3.7 | 73.1% | 71.7% |
| 80-90% | 0.9 | 0.2 | 81.8% | 81.0% |

The top bands held in 2025 (70-80%: 76% and 71% actual) but were too high in
2026 (70-80%: 64% actual, 80-90%: 74%).

Goals (b)-(d) put 99.8% of estimates between 40% and 60%. The two populated bands:
(b) 40-50% 5,646 per session, actual 48.8%; 50-60% 7,868, actual 50.2% (its 16
estimates per session above 60% hit 49.9%). (c) 3,655 at 47.9%; 9,890 at 49.3%.
(d) 5,251 at 50.4%; 8,294 at 47.8%. The small 2025 H1 edge (50-60% band 55%
actual) reversed in 2026 (48%, 45%, 42%); the inputs carry no stable direction
information. For comparison, the delivery scanner (+1% within two hours) on the
same test: 2,777 / 3,277 / 1,593 / 698 / 245 / 69 / 14.8 estimates per session in
the 0-70% bands, each within 2.2 points of actual; 70-75% 2.1 per session at 66%
(72% estimated); 75-80% 0.8 at 79%; 80-90% 0.2 at 94%.

Independent recheck: 1,000 sampled test labels (250 per goal) recomputed from
the raw minute files with separate code (`.scratch/recheck_goal_labels.py`): 0
mismatches.

## Components

| Layer | Code | Source | History |
|---|---|---|---|
| Movement scanner | `scanner/movement.py`, model from `scripts/e2e_research.py` | Kite minutes | 2020-2026 (corrected basis) |
| Order book | `orderbook/` (features, SQLite recorder), `cli/record_orderbook.py` | Kite `quote()` five-level depth | live only; Kite has no historical depth |
| Open interest | `oi/features.py`, `cli/download_futures_oi.py` | Paper run: near and next contracts' Kite day bars (OI) + live quotes. Research: Kite continuous near-month futures (day, OI) | needs one Kite login to download |
| Filings | `news/nse.py`, `news/store.py`, `news/features.py`, `cli/fetch_nse_filings.py` | NSE announcements, actions, board meetings | Dec 2024 onward stored (320,524 announcements) |
| Web news | `news/websearch.py` | OpenAI `web_search` tool | live only (search cannot be pinned to the past) |
| Decision | `ai/decision.py`, `ai/packet.py`, `ai/cycle.py`, `ai/journal.py`, `ai/openai_client.py` | all of the above | |
| Live loop | `cli/run_decisions.py` | Kite + NSE + OpenAI | shadow mode |

Every packet is built as of the decision time: today's bars up to the decision
close, daily OI from the previous session, filings disseminated at or before the
decision (later of NSE's announcement and dissemination times), and board
meetings only once their intimation was public. Filing text is untrusted: the
prompt tells the model to treat it as data, and the output is a strict JSON schema
validated in code (one verdict per candidate; anything missing or malformed,
any timeout, refusal or budget exhaustion becomes NO_TRADE).

## Inputs Needed

1. `.env` at the repository root (never committed; the code reads it, nobody prints it):

   ```text
   OPENAI_API_KEY=sk-...            # required for the OpenAI decider
   OPENAI_MODEL=gpt-6-luna          # optional; this is the default
   OPENAI_REASONING_EFFORT=low      # optional; default low (medium/high lost money in replay; see Evaluation)
   AI_DAILY_BUDGET_USD=5            # optional hard cap per IST day; default 5
   OPENAI_WEB_SEARCH=false          # optional; true lets the live decider search
   OPENAI_TIMEOUT_SECONDS=45        # optional
   ```

   `KITE_API_KEY` and `KITE_API_SECRET` are already required by the Kite tools.
2. A Kite browser sign-in for each Kite command (callback on 127.0.0.1:8765).
3. Once: the futures-OI history download (about 5 minutes after sign-in).
4. Each market day, to collect what history cannot provide: run the order-book
   recorder, and refresh filings.

## Commands

```powershell
$env:PYTHONPATH = (Get-Location).Path
# once: futures OI history (Kite sign-in)
.\.venv\Scripts\python.exe -m ai_trader.cli.download_futures_oi --universe data/universe_fno_20261011/universe.json --start 2020-01-01 --end 2026-10-09 --output data/oi_kite_20261011
# daily: NSE filings (no sign-in; resumable)
.\.venv\Scripts\python.exe -m ai_trader.cli.fetch_nse_filings --start 2026-10-12 --end 2026-10-12 --refetch-after 2026-10-12
# market hours: depth + futures OI recorder (Kite sign-in; about 200 MB/day at 30 s)
.\.venv\Scripts\python.exe -m ai_trader.cli.record_orderbook --universe data/universe_fno_20261011/universe.json --interval 30
# market hours: shadow decisions every 5 minutes, 09:40-13:30 (Kite sign-in; refreshes daily OI itself)
.\.venv\Scripts\python.exe -m ai_trader.cli.run_decisions --universe data/universe_fno_20261011/universe.json --model data/research/e2e_oi_openai_20261011/scanner_movement_v1.cbm
# evaluation: OpenAI deciders on every post-cutoff historical cycle (Jun-Sep 2026); model:effort pairs; --resume reuses journaled answers
.\.venv\Scripts\python.exe -m scripts.e2e_research --candles data/candles_fno_basis_20261011 --audit data/research/data_quality_fno_20261011/basis_audit/exclusions.json --news data/news/nse_filings.sqlite --oi data/oi_kite_20261011 --output data/research/e2e_openai_<date> --openai-cycles 1000 --openai-models gpt-6-luna:low,gpt-6-luna:medium,gpt-6.1-sol:low --openai-budget-usd 3 --openai-workers 3
# delivery design (October 12): scanner, components, AI, cumulative; about 10 minutes plus AI calls
.\.venv\Scripts\python.exe -m scripts.research_delivery_pipeline --candles data/candles_fno_basis_20261011 --audit data/research/data_quality_fno_20261011/basis_audit/exclusions.json --oi data/oi_kite_20261011 --news data/news/nse_filings.sqlite --output data/research/delivery_pipeline_<date> --threshold 0.75 --openai-models gpt-6-luna:low,gpt-6-luna:medium
# scanner only, four goals (October 12); about 12 minutes, no API calls
.\.venv\Scripts\python.exe -m scripts.research_scanner_goals --candles data/candles_fno_basis_20261011 --audit data/research/data_quality_fno_20261011/basis_audit/exclusions.json --output data/research/scanner_goals_<date>
```

`run_decisions` uses the rule baseline when no OpenAI key is configured.
Decisions are journaled to `data/journal/decisions.sqlite` with the exact input,
model, prompt version, token usage, cost and latency.

## OpenAI Cost

Prices (USD per 1M tokens, standard tier, fetched October 2026): gpt-6-luna
0.10 input / 0.50 output; gpt-6.1-sol 2 / 10; gpt-6-astra 10 / 50. Web search:
USD 10 per 1,000 calls plus the retrieved content billed as input tokens. Batch
and Flex are half price (for evaluation, not live use).

Prompt caching: GPT-5.6 and later write the whole prompt to cache by default at
1.25x the input price. Every packet is unique and the shared prefix (instructions
and schema, about 400 tokens) is below the 1,024-token caching minimum, so the
writes are never reused. The client therefore sends
`prompt_cache_options: {mode: explicit}` with no breakpoints, which turns them off.
The evaluation below ran before this fix and paid about 13% extra ($0.03 luna,
$0.66 sol); 30 later calls on both models showed zero cache writes.

Measured with the real API (688 replay calls per row; sol at medium on 10 calls
whose packets average 2,064 tokens). Accepted effort values are none, minimal,
low, medium, high, xhigh and max (each model supports a subset); anything else
stops at start-up.

| Model, effort | Input tokens | Output tokens (reasoning) | Latency | Cost per call |
|---|---|---|---|---|
| gpt-6-luna, low | 1,909 | 266 (22) | 4.5 s | $0.00032 |
| gpt-6-luna, medium | 1,909 | 703 (469) | 6.7 s | $0.00054 |
| gpt-6-luna, high | 1,909 | 1,145 (910) | 10.2 s | $0.00076 |
| gpt-6.1-sol, low | 1,909 | 270 (15) | 6.2 s | $0.0065 |
| gpt-6.1-sol, medium | 2,064 | 329 (67) | 7.6 s | $0.0074 |

Live packets also carry the order-book block and intraday OI, which replay cannot
have. Adding typical blocks to the 688 real packets costs 523 more tokens (five
candidates), so a live call is about 2,430 input tokens. Live cost with the
measured outputs (21 trading days a month):

| Model, effort | Per call | 47 calls/day (5 min) | Per month | 8 calls/day (30 min), per month |
|---|---|---|---|---|
| gpt-6-luna, low | $0.00038 | $0.018 | $0.37 | $0.06 |
| gpt-6-luna, medium | $0.00059 | $0.028 | $0.59 | $0.10 |
| gpt-6-luna, high | $0.00082 | $0.038 | $0.80 | $0.14 |
| gpt-6.1-sol, low | $0.0076 | $0.36 | $7.5 | $1.27 |
| gpt-6.1-sol, medium | $0.0081 | $0.38 | $8.0 | $1.37 |
| gpt-6-astra, low (not measured; sol-like output assumed) | $0.038 | $1.78 | $37 | $6.4 |

Web search, if enabled on every decision call and used once per call (assume
5,000 retrieved tokens): +$0.0105 per call with luna, +$0.020 with 6.1-sol,
+$0.060 with astra, i.e. +$10.4, +$19.7 or +$59 per month at 47 calls/day.
Searching once per new candidate per day instead (10-20 searches) costs about
$2-4 per month plus tokens. Replay evaluation: all 688 post-cutoff cycles cost
$0.23 (luna low), $0.37 (luna medium), $0.53 (luna high) and $5.14 (sol low);
the luna-low and sol runs predate the cache fix ($0.21 and $4.49 without it);
earlier cycles are not valid OpenAI tests because the model may remember them.
The daily budget cap (`AI_DAILY_BUDGET_USD`) is enforced from the journal before
every call; the OpenAI usage page is the authoritative bill.

For scale: Zerodha charges on one Rs 2 lakh intraday round trip are about Rs 118.

Per candidate (measured from the journals, `.scratch/ai_unit_costs.py`): the
delivery runs send about one candidate per call, so a verdict takes 2.2-2.5 s and
$0.00014 at luna low, 2.8-3.6 s and $0.00018 at luna medium. Intraday rounds of
five candidates per call: luna low 0.9 s and $0.00007, medium 1.35 s and $0.00011,
high 2.05 s and $0.00015, gpt-6.1-sol low 1.24 s and $0.0015 per candidate.

What the journal keeps for every call: the exact input packet, model, prompt
version, tokens, cost and latency; per candidate the verdict, confidence and a
short reason (under 30 words). OpenAI does not return the model's internal
reasoning, only its token count. Two options if more is wanted: request OpenAI's
reasoning summary (`reasoning.summary`, may need organisation verification, billed
as output), or ask for a longer structured rationale in the answer itself.

## Component Validation

- Futures OI history: 213/213 underlyings downloaded (2020-01-01 to 2026-10-09,
  `data/oi_kite_20261011`, manifest with SHA-256). The expiry calendar (last
  Thursday, last Tuesday from September 2025, holidays moved earlier) matches
  every one of the 81 contract switches seen in the data: the median OI ratio is
  0.29 on expiry day and 16x on the next session, and no other day jumps.
- Depth adapter: a real Kite `quote()` for 6 stocks and 12 futures parsed 18/18
  (`data/research/kite_depth_check_20261011.json`); on a Sunday the book is empty,
  so level parsing is covered by tests and still needs one market-hours check.
- OpenAI: 688 replay calls per model, all completed with valid strict-schema
  answers (50 luna calls first hit the rate limit at six workers and succeeded on
  resume at two or three). The client retries 429s with backoff but stops at once
  on billing errors such as `credit_balance_exhausted`.

## Evaluation So Far

`data/research/e2e_oi_openai_20261011/report.json` (with futures OI; the 638-cycle
first pass is kept as `report_first_pass.json`). Setup: the CatBoost movement
scanner picks the top 5 at eight times a day; entry next-minute open, exit 14:59,
Rs 2 lakh per trade, Zerodha charges both sides. The 95% intervals resample whole
trading days (2,000 draws), because trades on one day share the market.

- Scanner (two-sided 1% within 60 minutes): test AUC 0.775 (2025-2026), top-5
  precision 53% against a 15.7% base rate. Movement is predictable; direction is not.
- 2025-2026 test, net bp per trade [95% interval]: futures-OI buildup rule
  -8.9 [-16.8, -1.5] over 2,976 trades and contrarian -9.2 [-16.1, -2.5] reliably
  lose; always long -5.1 [-14.4, 4.1], momentum -2.6 [-9.3, 4.3], random
  -2.1 [-9.0, 4.7], momentum with a fresh filing +0.1 [-8.9, 9.1]; the CatBoost
  direction model with OI features (AUC 0.519) +4.1 [-10.1, 18.9] over 787 trades
  but -5.1 in 2026.
- OpenAI on all 688 post-cutoff cycles (2026-06-01 to 2026-09-30, where the candle
  data ends: 86 trading days, 3,440 candidate
  decisions), low effort, no web search:

| Decider | Trades (long) | Hit | Net bp per trade [95%] | Total | API cost |
|---|---|---|---|---|---|
| gpt-6-luna | 131 (62) | 51.9% | +5.3 [-20.0, 32.0] | +Rs 13.8K | $0.23 |
| gpt-6.1-sol | 20 (10) | 55.0% | +9.7 [-55.9, 77.4] | +Rs 3.9K | $5.14 |
| momentum, same cycles | 860 | 48.3% | +1.3 [-13.4, 15.8] | +Rs 21.6K | - |
| random, same cycles | 860 | 48.7% | -2.6 [-15.2, 10.5] | -Rs 44.1K | - |
| direction model, same cycles | 167 | 55.7% | -5.4 [-36.0, 25.4] | -Rs 18.0K | - |
| OI buildup rule, same cycles | 771 | 45.4% | -13.2 [-26.5, -0.4] | -Rs 203K | - |

No decider has a reliable edge: both OpenAI intervals include zero (luna's mean
is positive in 65% of resamples, sol's in 61%), and luna fell from +8.1 bp on
the first 638 cycles to +5.3 bp on all 688. Luna is selective (204 LONG/SHORT
verdicts, 56% right on direction before charges) and about 20 times cheaper than
sol, which rarely trades. At this spread, confirming a +10 bp edge needs about 1,800
trades (80% power), so forward shadow results alone will take months; better
direction inputs (live depth, intraday OI, filing content) are the faster lever.

Reasoning effort, gpt-6-luna on the same 688 cycles (byte-identical packets):

| Effort | LONG/SHORT verdicts | Right before charges | Trades | Hit | Net bp per trade [95%] | Total | Long only: trades, net bp [95%] |
|---|---|---|---|---|---|---|---|
| low | 77 / 127 | 56.4% | 131 | 51.9% | +5.3 [-20.0, 32.0] | +Rs 13.8K | 62, +6.4 [-31.4, 43.9] |
| medium | 241 / 202 | 49.9% | 257 | 42.0% | -20.3 [-41.1, 1.5] | -Rs 104.2K | 150, -25.8 [-53.9, 2.7] |
| high | 253 / 228 | 52.6% | 270 | 45.9% | -11.5 [-30.5, 8.0] | -Rs 62.3K | 149, -23.9 [-48.4, 0.6] |

More reasoning made luna trade twice as often on evidence no better than a coin
flip, so it lost more after charges; its long picks did worse than buying every
candidate (always long -6.1 bp, random long -6.4, momentum long -4.8 on the same
cycles). Low effort stays the setting; medium or high would need new direction
inputs and a fresh test first. Sol's 10 long trades netted -42.2 bp.
