# AI Trader

AI-assisted intraday trading research system for Indian equities.

## Current State: Live Paper Trading (October 15, 2026)

An AI-assisted **paper** trader for NSE cash equities: intraday longs and short
sales. **No live orders exist**: the read-only broker abstraction has no order
methods. Zerodha Kite Connect is the sole broker and market-data source; daily
history is put on NSE's official price basis (NSE's daily bhavcopies and
corporate actions, the basis the scanner was trained on).

Every weekday a Windows scheduled task starts the session at 08:30 IST; the
user logs in to Kite once in the browser (automated Kite login is not allowed).
Every 5 minutes from 09:40 to 14:55 a CatBoost scanner scores the top 500
stocks by market cap for P(a 1% move either way before the close). Every stock
at 90% or more goes, as its symbol and company name only, to two AIs: GPT-6.1
Sol medium (main) and Claude Opus 5.5 medium (shadow). Each AI researches the
stock on the web and answers LONG, SHORT or NO_TRADE: will the price at 15:00 be
above or below the price now. Each AI keeps its own paper portfolio: Rs 2 lakh
per trade, 0.5% target, else out at 15:00, no stop; no capital, position or
per-stock limits and no AI spend cap (October 15 setup). No research so far has
found a profitable edge.

### Start Here

| Need | Read |
|---|---|
| Binding rules | [AGENTS.md](AGENTS.md) |
| Current state, daily operation, setup, git policy, live results, research summary, next steps | [docs/handover.txt](docs/handover.txt), section "CURRENT STATE" |
| Paper trading and the AI layer in detail, costs, October 11-12 research tables | [docs/AI_LAYER.md](docs/AI_LAYER.md) |
| System contracts (section 0 is the implemented system) | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Data quality, features, live API samples, October 3-7 training | [docs/BROKER_DATA_QUALITY.md](docs/BROKER_DATA_QUALITY.md), [docs/FEATURE_VALIDATION.md](docs/FEATURE_VALIDATION.md), [docs/LIVE_API_SAMPLES.md](docs/LIVE_API_SAMPLES.md), [docs/TP_ONLY_TRAINING.md](docs/TP_ONLY_TRAINING.md) |

### Quick Start

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -c constraints.txt -e ".[dev,research,live]"
Copy-Item .env.example .env   # then fill in the keys; never commit .env
.\scripts\start_paper_day.cmd   # the daily run (the scheduled task runs the same)
# after 15:31: compare the AIs with hindsight
.\.venv\Scripts\python.exe -m scripts.compare_ai_decisions --day 2026-10-15 --paper-dir data/paper/research_tp05
```

`scripts/start_paper_day.cmd` is the same run as the scheduled task uses. The
gates are `ruff format --check .`, `ruff check .` and `pytest`. Git tracks the
code, docs, the production scanner model and the stock universe; candles,
research outputs, the NSE filings database and paper-trading state stay local
(see [What Git Includes](#what-git-includes)).

### Archive: October 9 Research State

The subsections below describe the research phase before paper trading (Kite
migration, CatBoost TP studies, live validation). They are dated evidence, not
current setup: the Quick Start above supersedes their setup commands, and their
"not built" statements predate the paper trader.

Offline research/replay and bounded read-only live diagnostics exist.
Start with offline tests or the saved capture; do not restart completed fits,
data migration, cloud cleanup or authentication as routine session setup.

### Model Result

**October 10 scanner model:** a fresh CatBoost for 09:40-13:30 IST calls (same
13 features plus side, raw probability of a 0.20% touch within 60 minutes, 511
trees, trained through December 2025) narrowly beats the October 5 model below
and is the preferred research model. P >= 0.75 calls touch TP by 15:00 85.2% of
the time but lose -0.10% per call after estimated charges. See
[docs/TP_ONLY_TRAINING.md](docs/TP_ONLY_TRAINING.md) and
[scripts/research_tp_scanner.py](scripts/research_tp_scanner.py).

Eight fresh Kite CatBoost candidates and one seed repeat completed locally in
544.853 total fit seconds, with no new Azure training job. Validation selected
**depth 4, learning rate 0.03, leaf L2 20, 462 trees**. Exact weights were frozen
before chronological calibration and evaluation; the model is not wired into
the runtime rules scanner. The [complete report](docs/TP_ONLY_TRAINING.md) contains
all variants, method choices, metrics, limitations and artifact fingerprints.
Saved weights: [data/research/tp_catboost_kite_20261005/final/best.cbm](data/research/tp_catboost_kite_20261005/final/best.cbm).
Retain that file with its preprocessing, model/selection locks and calibration;
"best" means the validation winner among the tested candidates, not a guarantee
of a profitable policy. These are local Git-ignored artifacts, not a cloud backup.

| Evaluation, April 1-September 28, 2026 | Log Loss | Brier | ROC AUC |
|---|---:|---:|---:|
| Raw winner, predeclared comparator | 0.668633 | 0.237987 | 0.600981 |
| Selected side-calibrated winner | 0.669464 | 0.238368 | 0.601297 |

Evaluation covers **686,112 actions / 123 sessions**. Lower loss/Brier is better:
**calibration worsened later probability accuracy**, despite helping on its
March selection window. Both outputs beat frozen base-rate controls. Raw
P >= 0.80 produced 3,600 touches in 4,480 actions (80.36%), but these overlapping
forecasts are not executed trades or an 80% profitable strategy. All saved
predictions reproduce exactly in the
[independent audit](data/research/tp_catboost_kite_20261005/verification_v2.json).

The [follow-up horizon audit](data/research/tp_probability_horizons_kite_20261005/report.json)
confirmed that the 3,600 touches are within 60 minutes. The same 4,480 raw
P >= 0.80 actions hit 3,901 times by 15:00 (87.08%) and 3,941 times by recorded
continuous-session close (87.97%). Overall one-hour probability calibration
error is 0.90 percentage points across ten bins; longer-horizon probabilities
are not calibrated merely by keeping the same raw score. Full band-by-band
results and the distinction between 15:00 and session close are in the report.

The [frequency/return follow-up](data/research/tp_frequency_returns_kite_20261006/report.json)
counts 343,056 scored candle-times, each with two directional forecasts. P >= 0.80
occurs at 3,030 distinct candle-times, averaging 24.6 per session across all 50
stocks, before any position or duplicate-signal limits. The 579 high-score calls
that never hit TP by 15:00 average **-1.393% gross at the cutoff**. Combining
those misses with +0.20% for each TP gives an idealized mean **-0.00588% before
costs** per high-score opportunity, not portfolio P&L. Good probability
calibration alone does not demonstrate a profitable strategy.

The selected model, raw comparator, calibration and evaluated periods are now
frozen research artifacts. Do not retune against the same evaluation or treat
it as a pristine future holdout. No orders are enabled.

### Market-Hours Validation

The October 7 capture received **13,819 moving ticks across eight instruments**
at 12:39:57-12:56:17 IST, producing 128 complete candles and feature rows.
Five eligible target records at **12:45** produced ten LONG/SHORT forecasts.
After one controlled disconnect, all ten later records were withheld until
reconciliation, as required. Saved-tick replay was exact. Unlike October 6,
this run establishes bounded **in-window inference**, not just late-call plumbing.

The independent review passed **229,133 generic numeric checks** plus 66,526
unavailable-value checks over warm-up/live bars. A new independent 60-digit TP
reference matched all **195 model-input values** across 15 records. The five
issued records' raw/calibrated forecasts matched subsequent-history forecasts.
The existing rules/feasibility path also ran offline over 80 target snapshots,
producing 41 repeatable diagnostic candidates, without positions or orders.

**Built-layer conclusion: the tested ingestion, calculations, inference and
safeguards behave as expected.** The feed looked healthy in the bounded runs;
unattended trading readiness is a different scope. Automatic post-gap state
repair is still absent, so the tested veto must remain in place.

The **74/120 exact OHLCV matches** compare our quote-derived candle with Kite's
historical candle for the **same stock and exact minute**. They are not a bad-tick
rate: all 120 timestamps exist, and 30 index rows have expected unknown-versus-
zero volume. Among 90 stock/ETF candles, 74 matched all OHLCV fields, 88 matched
close and 86 matched volume. The largest stock-price difference in this sample
was INR 0.30, about 0.0222%. Small differences can still affect sensitive rules.

The [construction investigation](data/kite_validation_20261007/candle_construction_analysis.json)
reproduced all 192 candles across both tapes independently and found no observed
aggregation arithmetic error. Some historical extrema were never received;
volume offsets cancel between adjacent minutes. Uniform boundary shifts or
filtering updates made agreement worse, so no algorithm change was adopted.
Exchange-time versus last-trade-time semantics remain untested with the saved
fields. This limitation stays visible in scanner research; it is not a reason
to restart general validation or wait for another market session.

The healthy WebSocket remains open across 60-second collection windows. The
disconnect was deliberately injected to test the built reconnect/veto path;
there were no unexpected disconnects or silent windows in the October 7 run.

See the [consolidated readiness receipt](data/kite_validation_20261007/technical_readiness.json),
[detailed findings](docs/BROKER_DATA_QUALITY.md),
[capture](data/kite_live_validation_20261007/report.json) and
[current independent review](data/kite_live_validation_20261007_review_v3/report.json).
All 3,592,876 historical labels/outcomes and 108 retained NSE daily checks were
reverified; all 686,112 frozen forecasts reproduced again. The training data,
weights and calibration remain unchanged; no new research fits or orders occurred.

**Development does not need to wait for market hours.** The
[offline live-tick guide](docs/LIVE_API_SAMPLES.md#offline-development-first)
documents every retained artifact, actual tick examples, field types, warm-up,
reconnect handling, expected results and remaining live-only acceptance checks.
Replay the saved evidence from the repository root, choosing a fresh output path:

```powershell
& .\.venv\Scripts\python.exe -m scripts.validate_kite_live `
  --review-capture data/kite_live_validation_20261007 `
  --output data/kite_live_offline_review_NEW
```

No login, credentials or market connection is needed. Keep `--review-capture`:
without it the command starts a new live capture. Offline replay was verified
with authentication, credential loading, browser launch and sockets blocked.
The expected review status remains `review_complete_not_end_to_end_approved`;
the October 7 tape yields 128 independently checked candles and five eligible
model records. Reports retain their original whole-product/exact-parity statuses;
the later acceptance of the built research baseline does not rewrite those
results or turn unbuilt services into completed capabilities.

### Active Data

The active, immutable [dataset manifest](data/candles/dataset.json) covers
January 1, 2024 through September 30, 2026: **66 instruments, 16,787,986 usable
minute candles and 45,052 daily candles**. Prices retain Kite's vendor-adjusted
basis, not original cash-execution prices. Missing candles are absent rows,
not explicit null candles. The [validation report](data/kite_migration_20261005/validation.json)
records 494 missing minute slots and 26 quarantined instrument-days; nothing
was imputed. Its calendar retains quarantined days rather than compressing
prior-session lookbacks. It is an observed provider calendar, not independently
certified coverage of every exchange date.

Including quarantines as whole instrument-days, the completeness recheck finds
**7,544 excluded slots out of 16,795,530 expected, or 0.0449%**. Model-input
instruments lose 0.0260%. The research builder separately excludes 5.31% of
candidate times for feature/warm-up or future-window unavailability. These
different denominators are explained in [the quality report](docs/BROKER_DATA_QUALITY.md).

The new [TP dataset](data/research/tp_only_kite_20261005_final/manifest.json)
contains **3,592,876 actions in 200 shards**. All labels, first-touch times,
adverse excursions and terminal returns were independently checked against
the Kite candles. Daily OHLCV for 54 stocks/ETFs on two recent dates passed
108 official NSE comparisons: OHLC within INR 0.10 and all volumes exact.
Daily checks do not certify every minute.

Old Groww-derived datasets, trained local artifacts and backtest outputs were
deleted through the [verified cleanup receipt](data/kite_migration_20261005/retirement.json):
1,188 local files totaling 1,626,923,933 bytes. The old raw cache, daily inputs,
trained weights, local training bundles and derived reports are absent.
Do not reuse their model results as Kite performance. The fresh model above
uses only the replacement Kite data.

Remote cleanup also completed: **753 old blobs / 384,257,861 bytes deleted and
eight old Azure jobs archived**, with zero remaining blobs in the two inventoried
containers and zero allocated training nodes. The
[cloud receipt](data/azure_cleanup_20261005/receipt.json) records the exact scope
and readback. Workspace, storage, key vault, monitoring, datastores, reusable
code/environments and the min-zero/max-three CPU cluster remain. Azure Files
shares were preserved, not content-audited. Archive retains job metadata; old
payloads are no longer available for reruns. Supporting resources can still bill.
The [migration details](docs/BROKER_DATA_QUALITY.md) distinguish validated data
from remaining live-readiness requirements.

### Local Setup

Run commands from the repository root. Python 3.12 is required by
[pyproject.toml](pyproject.toml) and [.python-version](.python-version); another
minor version is not a supported substitute. Use the existing uv-managed venv.
On a fresh checkout only, create it and install the normal development extras:

```powershell
uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -c constraints.txt -e ".[dev,research]"
```

Do not recreate an existing working environment unnecessarily. For the retained
CatBoost audit or full saved-capture review, add the training dependencies:

```powershell
uv pip install --python .venv/Scripts/python.exe -c constraints.txt -e ".[dev,training]"
```

The `training` extra includes research dependencies, CatBoost and retained
neural research libraries; installing it does not run a fit. The `dev` extra
contains pytest/Ruff. [constraints.txt](constraints.txt) pins selected packages,
but does not itself install every package listed. This machine uses the approved
`packagefeedproxy.microsoft.io` mirror. Do not inspect or print feed credentials.
This uv venv has no pip module; use uv for installs and the explicit interpreter
below instead of Windows' bare `python` alias. On Linux, use `.venv/bin/python`.

No broker settings are needed for ordinary tests or offline replay. A fresh clone
does not include the Git-ignored data/model artifacts: full replay needs the
retained fixture, frozen final model and protected manifests described in the
[retention guide](docs/LIVE_API_SAMPLES.md#retention-and-portability). Missing
artifacts require the approved retained copy, not disabled integrity checks or
automatic re-download/retraining. Synthetic tests remain usable without them.

### Read-Only Access

These commands contact Kite and initiate private authentication; they are not
offline setup checks. The declared `kiteconnect==5.2.2` dependency is required.
Privately configure `KITE_API_KEY` and `KITE_API_SECRET`. The paid Connect app's
redirect is `http://127.0.0.1:8765/kite/callback`. No access-token variable is
needed for browser login; tokens stay in process memory. Never share secrets
or callback URLs in chat, logs or screenshots. The assistant must not inspect
`.env` or other credential files. Old Groww variables are unused.

```powershell
& .\.venv\Scripts\python.exe -m ai_trader.cli.check_kite
& .\.venv\Scripts\python.exe -m ai_trader.cli.check_market_data
& .\.venv\Scripts\python.exe -m ai_trader.cli.check_stream
```

Each authenticated command opens a bounded private browser login. Complete
login and 2FA in that browser. Profile output contains only safe enablement
fields; unavailable DDPI status is `unknown`. Kite full-mode streaming supplies
exchange timestamps and native cumulative volume. No REST volume is spliced in.
History access succeeded for the full download. The authenticated read-only
check on October 5 at 22:18 IST also **passed profile, LTP, quote and streaming**.
It received one RELIANCE snapshot with native volume and one NIFTY snapshot.
The initial stream failure was resolved with explicit TLS hostname/peer
verification using the existing system CA store, without disabling verification
or installing certificates. See the [runtime check](data/kite_migration_20261005/live_check.json).
These were after-hours snapshots, not proof of moving market-hours prices,
live/history candle parity or unattended execution reliability. No orders exist.
The subsequent October 6 market-hours run and its limits are summarized above.
Any repeat should start early enough for completed model decisions before 14:15
and use a new evidence directory, never the promoted training cache.

### Reproducible Research

These are separate, deliberate data operations, not a startup script or a new
authorization to rebuild the frozen study. Use unused staging/output paths for
each version and preserve the completed receipts. Four isolated clients share
a single 2.5-request/second history budget; increasing threads does not bypass
it. Checkpoints and promoted files are hash-verified. The active cache refuses
in-place refreshes and ordinary loads spanning known gaps or quarantines.
Only explicitly gap-aware consumers may request incomplete ranges.

**Authenticated download into a new stage**, without promotion or retirement:

```powershell
& .\.venv\Scripts\python.exe -m ai_trader.cli.migrate_kite_data `
  --universe data/candles/universe.json --stage data/kite_NEW `
  --start 2024-01-01 --end 2026-09-30 --workers 4
```

**Offline dataset build** from the existing verified cache/daily inputs, into a
new research directory. This does not train or select a model:

```powershell
& .\.venv\Scripts\python.exe -m scripts.prepare_tp_only `
  --universe data/candles/universe.json --cache data/candles `
  --daily-dir data/research/tp_only_daily_kite_20261005 `
  --output data/research/tp_only_kite_NEW --start 2024-01-01 --end 2026-09-28
```

**Offline cache/label validation** with retained NSE references, writing a NEW
report directory. The validator writes `validation.json` beneath `--stage`, so
never point this repeat at the completed migration directory:

```powershell
& .\.venv\Scripts\python.exe -m ai_trader.cli.migrate_kite_data `
  --stage data/kite_validation_NEW --validate-cache data/candles `
  --research data/research/tp_only_kite_20261005_final `
  --nse-reference data/kite_migration_20261005/nse
```

The target remains a 0.20% directional touch within 60 minutes, without stock
identity inputs. The intended exit is **TP or 15:00 IST, no stop-loss and no
forced 60-minute exit**. Labels can extend past 15:00; that is not an execution
policy. Legacy stop-based replay is a separate diagnostic. Cash sizing on
adjusted historical prices, point-in-time ticks/membership, market-hours
streaming parity and execution reliability are not certified by this migration.

The exact current model is under `data/research/tp_catboost_kite_20261005/final`;
the fresh baseline and all trials/repeat are retained beside it. To audit without
training or broker access, choose an unused output filename:

```powershell
& .\.venv\Scripts\python.exe -m scripts.report_tp_catboost `
  --data data/research/tp_only_kite_20261005_final `
  --study data/research/tp_catboost_kite_20261005 `
  --baseline data/research/tp_catboost_kite_20261005_baseline `
  --output data/research/tp_catboost_kite_20261005/verification_NEXT.json
```

Data and models are Git-ignored local artifacts, not cloud backups. Keep the
current Kite manifests/shards, study sources/locks/weights and validation receipts.
Do not rerun completed training or cleanup commands as a routine session start.

### Source Map

| Surface | Entry Point / Owner | Scope |
|---|---|---|
| Broker boundary, login and full-mode streaming | [src/ai_trader/broker](src/ai_trader/broker) | Kite-only normalized read APIs; no orders |
| Stream supervision, native volume and candles | [src/ai_trader/market](src/ai_trader/market) | Market state and connection recovery, not automatic history reconciliation |
| Immutable historical cache | [src/ai_trader/history/store.py](src/ai_trader/history/store.py) | Hashes, provenance and fail-closed ordinary reads |
| Generic features | [src/ai_trader/features/engine.py](src/ai_trader/features/engine.py) | 63-field rule/replay path with explicit readiness |
| TP features/labels | [src/ai_trader/features/tp_only.py](src/ai_trader/features/tp_only.py) | Separate 13-feature-plus-side research contract |
| Training and bounded model comparison | [scripts/train_tp_only.py](scripts/train_tp_only.py), [scripts/research_tp_catboost.py](scripts/research_tp_catboost.py) | Deliberate versioned research, not a service loop |
| Frozen evaluation and outcome audits | [scripts/report_tp_catboost.py](scripts/report_tp_catboost.py), [scripts/analyze_tp_no_stop.py](scripts/analyze_tp_no_stop.py) | Offline reproduction and diagnostic returns |
| Saved-tick replay / live validation | [scripts/validate_kite_live.py](scripts/validate_kite_live.py), [tests/test_live_path.py](tests/test_live_path.py) | `--review-capture` is offline; default mode authenticates |
| Other diagnostic CLIs and regression tests | [src/ai_trader/cli](src/ai_trader/cli), [tests](tests) | Some commands are authenticated; check the documented mode before running |

### Verification Gates

Latest full suite after the October 10 scanner model study: **1,011 tests
passed in 81.17 seconds**. Ruff lint and the 124-file format check passed.
The model audit, live capture/review and cleanup
receipt above record their own checks; the earlier
[migration verification](data/kite_migration_20261005/verification.json) remains
a dated data/runtime record, not the new model result.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
& .\.venv\Scripts\python.exe -m ruff check .
& .\.venv\Scripts\python.exe -m ruff format --check .
git diff --check
```

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs format/lint/tests on
Ubuntu and Windows with `dev,research` extras and constraints. CI does not
install training extras or provision the Git-ignored frozen model/capture/cache.
The local model and saved-tick audits are separate evidence, not implied by
green CI. The current live-path file has 21 passing tests, including independent
TP-reference missingness, flat-price, future-data and corruption controls.
Earlier October 6 test counts remain dated evidence, not the latest gate.

### Next: Scanner Research

Use the [scanner research handoff](docs/ARCHITECTURE.md#scanner-research-handoff)
and [docs/handover.txt](docs/handover.txt). Freeze a new research protocol, compare
rules and frozen raw/calibrated model controls on the same eligible opportunities,
and measure candidate quality, coverage, false signals, TP-or-15:00 outcomes,
loss tails and costs. The selected scanner policy has not yet been specified.
Previously examined dates are not a new untouched holdout, and a high touch rate
is not profitable execution. Preserve the existing data/model baseline and no-stop
contract; do not automatically retrain, reauthenticate or implement future services.

The [later operational gates](docs/ARCHITECTURE.md#acceptance-gates-and-open-decisions)
remain separate from starting scanner research. All test and capture counts above
are dated October 7 evidence; the October 9 update changes documentation only.

<details>
<summary>Archived pre-migration README and study history, not current setup instructions</summary>

The remainder records the earlier Groww implementation and studies. Its provider
commands, data paths, model results and live-verification claims are retired;
use the current instructions above. Historical results are not Kite results.

**The goal is medium-frequency intraday trading: many small round trips per
session on liquid names, each exited as soon as it is a little ahead of what the
round trip cost.** Both directions are in scope. A trade buys the smallest whole
number of shares worth at least ₹1,00,000 and targets **0.2% above the buy
price** — gross, so after a round trip costing roughly 0.0827% at that size it
keeps about 0.1173%, near ₹117.32. Small per trade, and the thesis is that it
repeats often enough across a session to matter. Whether it actually does is
what the replay engine (roadmap item 7) measures — it is built, it has run, and
its first answer was **no**: at the default thresholds, over 61 sessions of 15
NSE symbols, the rule set loses at every stop multiple swept. See
[`backtests/README.md`](backtests/README.md). Nothing here has traded.

**New to this repository?** Read
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) first — its opening gives the
reading order for everything else, and its sections 2.3 and 11 tell you what
exists today versus what is still to build.

[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) describes the intended system in
full: the pipeline, each layer's contract, and which downstream decisions are
still open. [`docs/handover.txt`](docs/handover.txt) carries the current state,
environment gotchas, and known debt.
[`docs/LIVE_API_SAMPLES.md`](docs/LIVE_API_SAMPLES.md) records a real
request/response sample for every Groww API this system calls, captured against
a live session, so broker work is not blocked when the market is closed.
[`docs/FEATURE_VALIDATION.md`](docs/FEATURE_VALIDATION.md) records the
retained independent accuracy check of all 63 derived features, the
diagnostic dataset/NN schema and information-quality review, and the conventions
that differ from a charting package.
[`backtests/README.md`](backtests/README.md) records what the replay engine has
actually measured, and how to re-run it.
[`AGENTS.md`](AGENTS.md) holds the binding safety rules.

**Current TP-only research:**
[`docs/TP_ONLY_TRAINING.md`](docs/TP_ONLY_TRAINING.md) describes the new 50-stock
dataset, completed seven-model comparison and local CatBoost refinement.
A 607-tree depth-four CatBoost was selected before chronological calibration and
evaluation. Calibration helped the March selection score but worsened later loss
versus raw probabilities. Both outputs are frozen research artifacts, not approved
live models; Azure training compute scaled to zero after the comparison batch.
The target is a 0.20% directional price touch within 60 minutes, ignoring SL and
excluding stock identity. This is separate from the older TP-before-SL studies
and is not a production trading policy. The protocol includes exact settings,
evaluation and probability-band tables, research sources and budget limits.
The latest clarification is **TP or 15:00 IST only, with no stop-loss**. On the
same 32,451 raw CatBoost calls, TP touched within the literal next hour in 68.78%
of cases and by 15:00 in 77.86%; the 7,184 non-hits averaged -0.7669% gross at
15:00. The protocol separates late hourly hits after square-off and records
LONG/SHORT results. The earlier 2-ATR execution simulation is a different scenario,
not the user's requested exit rule. CatBoost remains a research artifact; the
runtime scanner and legacy strategy defaults are unchanged.

## Development setup

Python 3.12 is required — the code uses PEP 695 generic syntax, and
`pyproject.toml` pins `>=3.12,<3.13`. The environment is managed by
[uv](https://docs.astral.sh/uv/), which reads the interpreter version from
`.python-version` and fetches it if the machine does not already have it.

One command builds everything:

```bash
scripts/bootstrap.sh
```

That creates `.venv`, installs the project in editable mode with its dev
extras, and bounds every version by `constraints.txt` so a fresh machine gets
the package set the gates were last validated against rather than whatever PyPI
is serving today. It is safe to re-run.

### The three gates

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
```

On Windows the interpreter is `.venv/Scripts/python.exe`. Invoke it by path on
both platforms rather than activating the venv and typing `python`: on Windows a
bare `python` is intercepted by the Store alias and will not be this
interpreter.

Note that `.venv` is created by uv and therefore has **no `pip` module** —
`python -m pip install ...` fails with `No module named pip`. Use
`uv pip install --python .venv/bin/python ...` if you need to add something by
hand.

The command examples further down this file abbreviate the interpreter to
`python`. Each one means the venv interpreter above.

### Reproducibility

`constraints.txt` records the exact versions the gates were last run against,
regenerated with `uv pip freeze`. It is a constraints file, not a requirements
file: `pyproject.toml` still decides *what* is installed, and this only bounds
*which version* of whatever the resolver picks. Entries for packages that are
not selected are ignored, which is what makes a set captured on Windows safe to
apply unchanged on Linux.

A `uv.lock` would be stricter and is the intended endpoint, but generating one
needs network access to `files.pythonhosted.org`, which the current development
machine cannot reach.

### Continuous integration

`.github/workflows/ci.yml` runs the same three gates on every push and pull
request, across both `ubuntu-latest` and `windows-latest`. The two-OS matrix is
not ceremony: this codebase resolves `Asia/Kolkata` to find NSE session
boundaries, and `zoneinfo` reads that from the operating system on Linux but
from the `tzdata` package on Windows. No CI step supplies broker credentials,
and none needs to — every test stubs the broker, so a test that reaches the
network fails there.

Copy `.env.example` to `.env` for local configuration. Never commit `.env` or
place real credentials in `.env.example`.

## Check Groww read-only access

After adding the required Groww TOTP environment variables locally, run the
profile check manually:

```bash
python -m ai_trader.cli.check_groww
```

This command authenticates with Groww and retrieves only the user profile. Its
output is restricted to exchange enablement, active segments, and DDPI status.

## Check Groww market data

Run the live market-data check manually:

```bash
python -m ai_trader.cli.check_market_data
```

This prints latest prices for NSE RELIANCE and NIFTY, plus a normalized detailed
quote for RELIANCE.

Any of these checks may pause for a few seconds before printing. Groww answers
roughly half of all requests with a malformed body that is not a real error —
the identical call succeeds moments later — so every broker call is retried
automatically, including the login that each check performs first. A check that
ultimately fails has exhausted eight attempts over about twelve seconds, which
is a genuine outage rather than the usual flakiness. A failure during login
prints `Groww authentication failed.`, which is distinct from the per-check
messages below and points at the broker or your credentials, not at the data.

Run the historical market-data check manually:

```bash
python -m ai_trader.cli.check_historical_data
```

This searches backward for a recent completed NSE session, requesting one-minute
RELIANCE CASH candles from 09:15 to 09:30 Asia/Kolkata time. Weekends are skipped
locally, and empty weekday results are treated as potential exchange holidays.
The search stops after at most 10 weekdays. Output contains only the selected
trading date, candle count, and the first and last candles; normalized candle
timestamps are printed in IST, which is the only zone this system reasons in.

## Kite Historical Data

Kite browser login and NSE RELIANCE historical access were verified on
October 5, 2026. This is **historical-data onboarding only**: Groww remains the
runtime broker, and Kite has no order, portfolio, quote or streaming integration.

The app needs the paid **Connect** plan and Redirect URL
`http://127.0.0.1:8765/kite/callback`. Both are configured for the current app.
Privately configure `KITE_API_KEY` and `KITE_API_SECRET` in local `.env`, retaining
the Groww entries. Never put their values or a login callback URL into chat,
screenshots or source control. **No `KITE_ACCESS_TOKEN` is needed for browser
login.** The token stays in process memory and is not written to disk.

The official `kiteconnect==5.2.2` SDK is already installed on this machine. On a
new machine, install the optional extra using your approved package index:

```powershell
uv pip install --python .venv/Scripts/python.exe --constraint constraints.txt --editable ".[kite]"
```

From the repository root, a fresh download uses:

```powershell
& .\.venv\Scripts\python.exe -m ai_trader.cli.compare_broker_history `
  --browser-login --month 2026-09 --symbol RELIANCE --session-close 15:15 `
  --output data/broker_comparisons/reliance_202609_NEW
```

Complete login and 2FA privately in Zerodha's system-browser page on this
computer. The loopback listener checks a random state value, suppresses request
logging and waits up to five minutes. Leave port 8765 available. Kite access
tokens normally expire at 06:00 the next day; an unsuccessful or expired session
requires a fresh login. The adapter resolves the exact cash-equity instrument
from Kite's master and normalizes aware IST timestamps and Decimal OHLC values.

`--session-close` is required: select the continuous-session boundary applicable
to the instrument and historical period. For September RELIANCE it is **15:15**,
not 15:30, under [NSE's closing-auction rules](https://www.nseindia.com/static/products-services/closing-auction-session).
Non-CAS securities retain a 15:30 close. Do not apply either setting blindly to
all instruments or historical periods; runtime session defaults are unchanged.

### Full-History Quality Result

The follow-up download covers **January 1, 2024 through September 30, 2026**:
254,317 Kite minute candles and 683 daily candles, stored separately from Groww.
The [full quality study](docs/BROKER_DATA_QUALITY.md) recommends **Kite as the
primary historical source**, based on the captured RELIANCE data and 49 official
NSE daily benchmarks, with important limits before a wider migration.

After accounting for special sessions, Kite has 8 absent minutes versus Groww's
858, and zero unknown minute volumes versus 837. Fresh Groww API calls reproduced
four 100-fold price spikes and missing periods. NSE's daily archive rules out
those spikes as genuine prices. Kite daily OHLC matches NSE within INR 0.05 on
all 49 samples, and daily volume matches exactly, after reversing the documented
2024 bonus adjustment. Kite is not flawless: its eight gaps remain unfilled.

Use the [quality report](data/broker_comparisons/reliance_202401_202609/quality/report.json)
and [full Kite CSV](data/broker_comparisons/reliance_202401_202609/combined/kite.csv).
The study includes source links, exact missing dates, adjustment conventions,
resumable download and offline audit commands. Runtime defaults and frozen
models/results are unchanged. Do not splice the two providers into one tape.

### Earlier September RELIANCE Result

Kite returned **7,560 one-minute bars**, all 360 continuous-session minutes on
each of 21 scheduled sessions. The [official NSE holiday calendar](https://www.nseindia.com/resources/exchange-communication-holidays)
excludes September 14. The existing Groww cache has **7,125 corresponding bars**:
September 29 stops at 13:59 and September 30 is absent, leaving 435 missing
continuous-session minutes, 405 of them before the strategy's 15:00 cutoff.
This compares a fresh Kite download with the older Groww cache, not two
simultaneously refreshed feeds. No Groww files were refreshed or overwritten.

Agreement on the 7,125 shared timestamps:

| Field | Exact Matches | Exact Agreement | Largest Absolute Difference |
|---|---:|---:|---:|
| Open | 3,124 | 43.8456% | INR 4.80 |
| High | 6,767 | 94.9754% | INR 2.20 |
| Low | 6,773 | 95.0596% | INR 4.50 |
| Close | 7,069 | 99.2140% | INR 1.00 |
| Volume | 7,113 | 99.8316% | 7,668 shares |

All four prices match on 3,013 bars (42.2877%); all OHLCV values match on 3,010
(42.2456%). Mean absolute close difference is INR 0.001825. Timestamp shifts of
one minute make close agreement much worse, so a simple one-minute offset does
not explain these differences. In 3,947 of the 4,001 differing opens, Kite's open
equals its own previous close; this is an observed pattern, not proof of either
provider's aggregation algorithm or correctness. High/low differences can affect
TP-touch labels, and open differences can affect features and simulated entries.

Use the corrected, session-aware
[data/broker_comparisons/reliance_202609_continuous/report.json](data/broker_comparisons/reliance_202609_continuous/report.json).
Its directory holds normalized provider CSVs, row-level differences and hashes.
The original capture under `data/broker_comparisons/reliance_202609` is retained:
its legacy 375-minute assumption incorrectly counted the auction period as
missing continuous trading. The corrected report preserves the exact provider
snapshot bytes, excludes 259 Groww rows outside the continuous session, and
records its parent report fingerprint. These local artifacts are gitignored.

Recompute from saved snapshots without credentials or broker access:

```powershell
& .\.venv\Scripts\python.exe -m ai_trader.cli.compare_broker_history `
  --saved-comparison data/broker_comparisons/reliance_202609 `
  --month 2026-09 --symbol RELIANCE --session-close 15:15 `
  --output data/broker_comparisons/reliance_202609_RECHECK
```

Single-month and offline runs refuse an existing output directory. Multi-month
downloads can resume only verified checkpoints under an identical range plan.
The generic comparison reports
coverage over observed dates; this study additionally checked those dates against
NSE's holiday calendar. **Feed agreement is not exchange-ground-truth accuracy.**
Keep provider datasets separate and validate OHLC-dependent behavior before a
runtime or model-data migration. No model was retrained and no trade was placed.

## Bulk Research History

The October 3, 2026 download requested January 1, 2024 through September 30,
2026 for the current official Nifty 50 constituents and 16 context instruments.
All 66 instruments returned data: **16,065,134 raw one-minute candles** in
`data/candles`. Frozen `data/research` snapshots were not changed.

This is **not complete end-of-September coverage**. Fresh bounded requests
confirmed that equities/ETFs stop at September 29 14:00 IST and indices at
14:05 IST. ETERNAL begins April 9, 2025, TMPV October 24, 2025, and nine context
indices plus NIFTYBEES begin January 1, 2025. These are observed vendor ranges,
not proof that the instruments did not exist earlier. No renamed-symbol history
or corporate-action adjustment was invented. Current index membership also
introduces survivorship bias; it is not point-in-time 2024 membership.

Local universe and evidence:
- `data/nifty50_context_20261003.json`: immutable source-hashed universe.
- `data/downloads/nifty50_2024_2026.json`: actual dates, row counts, missing-volume
  counts, irregular observed sessions and per-file hashes.
- `data/downloads/nifty50_tail_probe_20261003.json`: fresh final-session requests.
- `data/azure/readiness_20261003.json`: read-only Azure permission/quota checks.

The public source files are the official
[Nifty constituent CSV](https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv)
and [Groww instrument master](https://growwapi-assets.groww.in/instruments/instrument.csv).
For a new dated universe, download both under `data/`, then use a new output path:

```bash
python -m ai_trader.cli.download_history --prepare-only \
  --nifty50-source data/nifty50_constituents_20261003.csv \
  --instrument-source data/groww_instruments_20261003.csv \
  --universe data/nifty50_context_NEW.json \
  --start 2024-01-01 --end 2026-09-30 --report data/downloads/NEW.json
```

Resume the existing download using its existing universe:

```bash
python -m ai_trader.cli.download_history \
  --universe data/nifty50_context_20261003.json \
  --start 2024-01-01 --end 2026-09-30 \
  --cache-dir data/candles --report data/downloads/nifty50_resumed.json
```

Completed 28-day windows survive interruption. A previously empty response is
cached too; use an explicit, bounded refresh to ask the vendor again later:

```bash
python -m ai_trader.cli.download_history --refresh \
  --universe data/nifty50_context_20261003.json \
  --start 2026-09-29 --end 2026-09-30 \
  --cache-dir data/candles --report data/downloads/nifty50_tail_refresh.json
```

`--symbols NSE:RELIANCE NSE:NIFTY NSE:GOLDBEES` restricts a request. Broker error
text and credentials are never printed. `downloaded` means data was returned,
not complete coverage; boundary warnings are not a holiday calendar. Indices
and ETFs are `context_only`, not automatically eligible short-selling targets.
Execution-quality simulation remains deferred, but missing data is not imputed.

The next model experiment will prioritize neural networks for
`P(TP before equal-distance SL and within horizon | features, direction)`.
Both LONG and SHORT remain in scope. The larger dataset has not yet been
converted to a training matrix, joined to market/sector context, labeled or
trained. Use chronological all-symbol splits, training-only preprocessing,
purged label horizons, separate calibration and genuinely forward evaluation.
Select TP/horizon on development data, not on the final evaluation period.

Azure preflight found an enabled subscription, an empty East US resource group,
broad management permissions, a registered ML provider and 100 available
dedicated ML vCPUs. Modern dedicated GPU-family quotas are zero. This is not a
successful deployment or a data-access test: no workspace, storage, compute,
uploads or jobs were created. Confirm authorized use of the selected
subscription and market data, approve a budget, then provision bounded compute
with zero minimum nodes and explicit job limits.

## Check Groww streaming data

Run the bounded RELIANCE LTP stream check manually:

```bash
python -m ai_trader.cli.check_stream
```

The command resolves the NSE RELIANCE exchange token, subscribes only to its
CASH LTP feed, and prints normalized price ticks with IST timestamps. It stops
after five ticks or 30 seconds and always unsubscribes. If no ticks arrive, it
reports that the market may be closed and exits normally.

If the feed cannot be reached at all, the connect is abandoned after 30 seconds
and the command exits 1 with a single line: `Groww live feed unreachable; the
stream connection failed.` Between 2026-09-19 and 2026-09-23 that was the *only*
outcome, in or out of market hours — Groww's tick transport accepted the socket
but never completed its handshake, a server-side fault nothing here could retry
around. **The outage lifted on 2026-09-24 with no change to this client**, and
live windows since have carried real ticks; read the "feed is down" wording in
the older documents as dated history rather than current state. The failure
signature is still worth recognising, because it will recur. See the
market-open addendum in
[docs/LIVE_API_SAMPLES.md](docs/LIVE_API_SAMPLES.md) for what it looks like.

## Check Groww market state

Run the market-state check manually:

```bash
python -m ai_trader.cli.check_market_state
```

This exercises the whole market-state component end to end. It backfills a
recent completed NSE session of one-minute RELIANCE candles, then folds live LTP
ticks into the same state object, aggregating them into further one-minute
candles. Output is a JSON summary of the resulting state: the backfilled trading
date and candle count, the live tick count, the number of retained candles,
late-tick and duplicate-candle counters, the latest price, and the first and
last candles. Outside market hours no ticks arrive and the check still passes,
reporting the backfilled state alone.

Groww's LTP stream carries no volume. It advertises an optional volume field
but does not populate it — across 114 ticks measured during live trading on
2026-09-21 it was absent from every one. The only live volume Groww serves is
the running session total on the REST quote endpoint, so a `VolumePoller` reads
that total on a background thread and stamps it onto each bare tick before the
tick reaches the candle builder, which differences consecutive totals into a
per-minute figure. The check reports the outcome under `volume_source`: how many
ticks were stamped, the latest polled total, and the poll failure and regression
counts.

Volume is an enrichment, never a precondition. A failing quote endpoint costs
volume and nothing else — prices keep flowing, the check still exits 0, and the
failure is reported rather than swallowed. The check's 15-second tick window is
shorter than a minute, so it normally proves that ticks are being stamped
rather than that a stamped candle was emitted.

## Check feature engine

Run the feature-engine check manually:

```bash
python -m ai_trader.cli.check_features
```

Optionally export every candle's features for offline comparison:

```bash
python -m ai_trader.cli.check_features --export-csv features.csv
```

This backfills a recent completed NSE session of one-minute RELIANCE candles and
folds each one through the feature engine, computing 47 derived features:
returns, EMAs and their slopes, RSI, MACD, true range and ATR, the directional
movement index and ADX, rolling extremes and the distances to them, a simple
moving average with its Bollinger band, bandwidth and percent-B, VWAP with its
turnover-weighted deviation, a relative volume ratio, on-balance volume, and the
session frame — open, high, low, range position, elapsed minutes, cumulative
volume and the opening range.
Output is a JSON summary of the latest snapshot alone: the trading date, candle
counts, duplicate and out-of-order candle counters, the candle itself, every
derived value rounded to six decimal places, and a readiness flag per feature.
A feature without enough history behind it is reported as null rather than
guessed. The exported CSV holds one row per candle at full precision, and an
existing file is refused rather than replaced unless `--overwrite` is passed.

Read the per-feature readiness flags, not just `core_ready`. `core_ready`
covers price and momentum only; it says nothing about the volume-derived
fields. Before the volume poller existed, a live run on 2026-09-21 showed
exactly why that separation matters: `core_ready` stayed true across the
historical-to-live seam while `vwap`, `price_vs_vwap` and `volume_ratio_20`
went null and stayed null for want of volume. A later run the same day, with
the poller stamping ticks, carried all three across the seam populated. A live
consumer must still check each flag it depends on rather than trusting
`core_ready` to cover them.

The check exits 0 on success, 1 when the broker, the session lookup or the
export fails, and 2 on a configuration or usage error.

All 47 features have been checked for numerical accuracy against an
independently written reference over a real session, and all 47 agree. Before
comparing the exported CSV against TradingView, read
[`docs/FEATURE_VALIDATION.md`](docs/FEATURE_VALIDATION.md): the conventions
match a charting package almost everywhere, but `volume_ratio_20` deliberately
excludes the current candle from its own baseline and will therefore differ
from TradingView's relative volume on any spike.

## Run the scanner

Scan a real session, minute by minute:

```bash
python -m ai_trader.cli.check_scanner             # last completed session
python -m ai_trader.cli.check_scanner --live      # backfill, then live ticks
python -m ai_trader.cli.check_scanner --export-csv scan.csv
```

By default this backfills the most recent completed NSE session and runs a full
scan cycle on every minute of it, printing how many instruments were considered,
how many were not ready, how many the cost screen ruled unreachable, and which
rules fired. `--live` backfills first — which leaves the indicators warm — then
scans candles as live ticks assemble them.

The cost screen is tunable without editing code: `--clip` sets the notional per
trade, `--gross-target` the gross move being screened for, `--max-atr-multiple`
how many ATRs of room a name must have to be considered reachable, and
`--broker` whose schedule prices the hurdle.

One caution on the numbers it prints. Every cycle is scanned against an empty
portfolio, so nothing is ever suppressed by cooldown or position limit — **the
candidate rate is an upper bound**, not what a running system would emit. And a
candidate means the rule's conditions were met, not that the rule is right: the
thresholds in `scanner/rules.py` are conventional levels chosen so the layer
could be built, and none has been measured on this market yet.

## Run a backtest

Replay the scanner over cached history and measure what its candidates earned:

```bash
python -m ai_trader.cli.backtest \
  --symbols-file backtests/universe.txt \
  --start 2026-07-01 --end 2026-09-24 --offline \
  --latency-seconds 1 --half-spread 0.0002 --slippage 0.0001 \
  --stop-atr 2.0 --label fixed-2.0 \
  --report backtests/report_current.txt \
  --history backtests/current_runs.tsv
```

This is the only command in `cli/` that produces a result rather than a
diagnosis. Its options fall into five groups: universe (`--symbols` or
`--symbols-file`, `--exchange`), window (`--start`, `--end`, `--cache`,
`--offline`), fill model (`--frictionless`, or all three of
`--latency-seconds`, `--half-spread`, `--slippage`; plus `--ambiguous-as`),
strategy (`--clip`, `--gross-target`, `--max-candidates`,
`--max-open-positions`, `--stop-atr`, `--trailing`, `--no-screen`,
`--max-atr-multiple`, `--cooldown-minutes`, `--square-off-minutes`) and output
(`--label`, `--report`, `--history`, `--no-write`, `--json`). Exit codes match
the checks: 0 on success, 1 when the run could not be completed, 2 when the
configuration or the environment was wrong.

Two of those groups are deliberately default-free, for opposite reasons. **The
fill model has no default at all** — pass `--frictionless` or pass all three
friction figures, because a silent zero is the most flattering thing a backtest
can assume and the one a reader is least likely to notice. **The strategy group
states no default either**: every strategy flag parses to `None` and is
forwarded only when actually passed, so `StrategyConfig` stays the one place any
of it is written down and a sweep cannot quietly diverge from what live would
run.

`--offline` never loads credentials and makes no vendor call, which is what
makes a published run reproducible — it reads the cached tape only, and fails
loudly on a gap rather than filling it from whatever the broker serves today.
Populating that cache in the first place needs a broker and no `--offline`.

The first answer this produced was negative. Over 61 sessions of 15 NSE symbols
the default rules lose at every stop multiple swept, and removing friction
entirely leaves a gross hit rate near 47% — the entries are close to a coin
flip. [`backtests/README.md`](backtests/README.md) holds the grid, the
frictionless controls, and the flags that reproduce both.

### Prepare scanner score research

The RELIANCE September 1-28 protocol, shared baseline configuration, timestamp
validation and current data-completeness status are recorded in
[backtests/reliance_sep2026/README.md](backtests/reliance_sep2026/README.md).

Both `backtest` and `check_scanner` accept `--strategy-config PATH`, loading the
same complete versioned JSON. Overrides cannot be combined with that file.
Reports include the resolved strategy SHA256. Replay additionally accepts
`--warmup-start YYYY-MM-DD`; earlier candles seed features without being scored.
New history tables include strategy identity and warm-up counts; use a new TSV
rather than appending to the older published sweep tables.

The [September 1-4 signed-score study](backtests/reliance_sep2026/signed_score_v1/README.md)
audits all 47 features and fits an opt-in, time-discounted target model. Its
schema-v2 profile uses `2 * sigmoid(raw) - 1`, positive LONG and negative SHORT,
ranked by absolute magnitude. It is research-only: forward-day improvement over
zero is 0.40%, with unstable directional quality. Existing default rules and
schema-v1 profiles are unchanged. Offline fitting uses the optional `research`
extra; runtime scoring does not require SciPy.

The follow-up threshold study tests `abs(score) >= score_threshold` with the
direction supplied by the sign. Nonzero cutoffs use schema-v3 profiles and apply
identically in replay and live diagnostics. The best supported tuning accuracy
used 0.03, but its September 4 check was 50% on 18 signals and lost money in
costed replay; it remains unvalidated. Full comparisons are in the study above.

The [TP-precision study](backtests/reliance_sep2026/tp_precision_v1/README.md)
uses the newer objective of actual take-profit exits, permits sparse selections,
and reports the full 0-1 threshold curve plus TP/SL sweeps. A research-only
TP classifier also tests prior-session features. Higher thresholds did not
reliably improve TP precision; no new model or default was promoted.

The [63-session probability study](backtests/tp_probability_2026q3/README.md)
separates fitting, calibration, threshold selection and final evaluation. Its
research score is `+p` for LONG or `-p` for SHORT, so magnitude preserves the
estimated TP-before-SL probability. It reports both directions and all TP/SL
cases; calibration remains imperfect and no default is changed. Curated papers
and their practical limits are in [docs/strategies/README.md](docs/strategies/README.md).

The later [month-split study](backtests/tp_probability_month_split/README.md)
fits base coefficients on July only, validates and calibrates in August, then
evaluates all 19 September 1-28 sessions. It adds actual SL percentages, ranking,
TP-speed diagnostics and exact fitted-function explanations. The August-selected
0.2% TP policy hit 18/36 targets against a 58.1% mean forecast and lost Rs 4,823.48.
All six alternatives lost money. September was already researched, so it is not
an untouched test. The study remains research-only; no scorer is promoted.

The [symmetric horizon study](backtests/tp_horizon_symmetric_v1/README.md)
implements `P(TP before SL and within h | X, direction)` with equal tick-rounded
TP/SL distances from executed entry. It compares logistic regression, shallow
boosted trees and small neural nets, including five-session context, over
0.15-0.40% barriers and 6/15/30/60-minute horizons. Cumulative probabilities are
consistent across horizons. The August-selected logistic model for 0.40% within
six minutes forecast 0.619% on September hypothetical actions, versus 0.642%
observed, but did not find high-probability trades. All 16 combined execution
alternatives lost money. The exact fitted functions, ablations and limitations
are saved; no production scorer or risk default was changed.

### What Git Includes

Tracked: source, tests, scripts (including every research script), dependency
specifications, CI, documentation, `.env.example`, and from `data/` only what
the paper run needs to start: the production scanner
(`data/models/scanner_up1_by_close_v1/`: `model.cbm`, `bundle.json`, the
training source snapshot) and the stock universe
(`data/universe_mcap_20261012/`, plus `data/universe_nifty500_20261012/` as the
industry source for rebuilding it). Files under `scripts/*.py` and `data/` are
stored byte-exact (`-text`), because `bundle.json` and the research reports pin
their SHA-256.

Local only (`.gitignore`): `.env` and other credentials, `.venv/`, `.scratch/`,
`Programs/`, and the rest of `data/`: historical candles (tens of GB), research
outputs, the NSE filings database (rebuilt by `ai_trader.cli.fetch_nse_filings`),
paper-trading state, logs, captures and caches. A Git push is not a backup of
those; copy `data/paper/` to keep the paper portfolios and AI journals. A fresh
clone runs the tests and the paper trader; rerunning research needs the
datasets re-downloaded (see [docs/handover.txt](docs/handover.txt), section 6).

## Check trade costs and sizing

Run the cost check manually:

```bash
python -m ai_trader.cli.check_costs             # a default spread of quotes
python -m ai_trader.cli.check_costs 2450 100    # specific quotes
python -m ai_trader.cli.check_costs --broker zerodha
```

Unlike every other check, this one talks to no broker and needs no market
session — the cost model is arithmetic over a published schedule, so it prints
the same figures at midnight on a Sunday as at 09:20 on a Tuesday. It exits 0
on success and 1 on an unusable price argument or an unknown broker name; it
has no configuration and therefore no exit code 2.

`--broker` takes `groww`, `zerodha` or `kite` and defaults to Groww, which is
the broker this system connects to and not the cheaper of the two. Two
schedules are published and both are modelled, because they differ in exactly
one line — brokerage — and agree on every statutory and exchange charge.

Output is a JSON summary with four parts. The round-trip cost by clip size
shows where the curve bends — brokerage caps at ₹20 *per leg*, so at Groww's
0.1% the cost is a flat 0.2715% of turnover up to ₹20,000 and then falls away,
reaching 0.0827% at ₹1,00,000. The schedule comparison prints both brokers at
every clip size whichever one was selected. The two interpretations block
prices both readings of a stated target. The estimates block sizes the fixed
₹1,00,000 clip at each quote and prints the exits.

**At the configured clip the broker choice costs nothing.** Zerodha charges
0.03% where Groww charges 0.1%, but both cap at ₹20 a leg, so Zerodha's cap
does not bind until ₹66,666.67 and above that figure the two schedules are
identical to the paisa. Below it Zerodha is cheaper and never dearer: a 0.2%
gross capture is a loss at Groww's rates below about ₹28,690 a clip and a gain
at Zerodha's at every size. Since this system trades a one-lakh clip, the swap
named in `AGENTS.md` rule 10 is a configuration change rather than a change of
strategy — which is why the comparison is printed rather than asserted.

Read the estimates with the integral quantity in mind: the clip is a floor, so
a one-lakh clip of a stock at ₹2,450 is forty-one shares and ₹1,00,450 rather
than ₹1,00,000, and every fraction is a fraction of what was actually filled.
Both exits are printed because cost is direction-symmetric and nothing here
picks a side; long exits round up to a tick and short exits round down, always
away from the entry, because rounding to the nearer tick fills a trade that
looks like a win but returns less than the margin that justified it.

Watch `tick_fraction`. At ₹2,450 a five-paisa tick is 0.002% of price and
rounding is a rounding error; at ₹100 it is 0.05% — half of a 0.1% target — and
past that point the exit can only land on the tick grid, so what the trade asks
for depends on where the entry sits in that grid rather than on the target. At
the 0.1% target an entry at ₹100.00 sits exactly on a tick and returns the
₹17.32 it asked for; an entry one tick higher at ₹100.05 must reach ₹100.20 and
returns ₹67.30, roughly four times the target off a five-paisa difference. That
is not a windfall — the trade is asking for a move four times larger and will
fill far less often. Cheap stocks are a coarser instrument for this strategy
than expensive ones.

Every figure carries the same caveat: the rates of both schedules are
transcribed from published tables and neither has been reconciled against a
real contract note, and the spread is not modelled at all. Treat the exit
prices as a floor, not a forecast.

## Sync with GitHub

Set `GITHUB_PAT` in `.env` to a personal access token with `repo` scope, then
pull and push with:

```bash
scripts/sync.sh
```

The token is read from `.env` at invocation time and passed to git through
`GIT_ASKPASS`. It is never written into `.git/config`, a remote URL, or a
credential helper, so running this leaves no token on disk. The script refuses
to run unless `origin` begins with `https://github.com/`, and both commands name
`origin` and the current branch explicitly, so the remote that is checked is the
remote that is contacted even if the branch is configured to track another one.
Any extra arguments are forwarded to `git push` after that pinned remote and
branch.

## Roadmap

Initial milestones:

1. Connect read-only to Groww.
2. Retrieve account/profile information safely.
3. Retrieve historical market data.
4. Retrieve live market data.
5. Build normalized market-state snapshots.
6. Build deterministic strategy/scanner layer.
7. Build the historical replay / strategy evaluation engine.
8. Add shadow trading and journaling.
9. Add the OpenAI decision engine.
10. Add deterministic risk and position management.
11. Evaluate whether the AI adds measurable value, by forward shadow trading
    against the control of taking every candidate.
12. Only after validation, small-capital live; production execution last.

Replay (7) comes before the AI (9) deliberately: it establishes whether the
scanner's candidates have edge at all, which is the control the AI is later
measured against. The ordering here is the same one in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) section 11; keep them in sync.

Items 1–7 are in place. Item 4 was validated against a live market on
2026-09-21: ticks streamed, aggregated into contiguous one-minute candles, and
joined the backfilled history with no late ticks and no duplicates. The same day
the whole chain was run live through the feature engine for the first time,
confirming that price and momentum features cross the historical-to-live seam
intact. The one thing the live stream does not deliver on its own is volume; a
`VolumePoller` now supplies it from the REST quote endpoint, as described under
"Check Groww market state", and a later run the same day carried the
volume-derived features across the seam populated.

The scanner (6) is the exception to all of that. It is tested offline and
deliberately unproven: every threshold in it is a conventional level rather than
a measurement, and item 7 is what turns any of them into evidence. Item 7 has
now done that once, and the verdict on the defaults was negative — see "Run a
backtest" above. That result measures those thresholds over that universe and
that window; it is not a verdict on the layers that computed it, which is why
the work it points at is tuning the rules rather than rebuilding anything
beneath them.

Alongside the scanner sits a transaction cost model (`costs/`), which is not a
roadmap item but a precondition for one. The objective is many small round trips
per session — long or short — each closed as soon as it is a little ahead of
what the round trip cost, so the first question about any candidate is whether a
move that size is available at all. It cannot be answered with a target
percentage: at Groww's rates the same round trip costs 0.2715% of a ₹20,000
clip and 0.0827% of a ₹1,00,000 one, because brokerage is capped per leg, so
0.2% gross is a loss at the first size and a profit at the second. The scanner
therefore computes the hurdle — `round-trip cost at the configured size + the
margin asked for` — rather than storing one. Two schedules are modelled, Groww
and Zerodha/Kite, and the hurdle is computed from whichever one the caller
passes; at the configured clip they agree to the paisa.

The buying model that settles the size is deliberately simple: one fixed clip
of ₹1,00,000 per entry, and a sell is the whole position. Nothing in the code
can express a partial exit — a single quantity serves both legs — so scaling
out would arrive as a visible change rather than quietly as a new code path.
The clip is a *floor* on turnover, not a budget: shares are indivisible, so an
entry buys the fewest whole shares worth at least a lakh — forty-one shares and
₹1,00,450 of a stock at ₹2,450, one share of anything dearer than the clip. The
scanner prices each name's hurdle on the notional that would actually fill,
which is never below the clip and therefore never costlier, as a fraction, than
the clip figure suggests.

The margin follows from a stated target: sell **0.2% above the buy price**.
That is a gross move, so what it keeps is 0.2% minus the round trip — about
0.1173%, or ₹117.32 on a lakh. `STATED_GROSS_TARGET` is the one place the
figure appears and `SizingPolicy.from_gross_target` does the conversion,
refusing a target its own costs would consume rather than clamping it: at
Groww's rates 0.2% at a ₹20,000 clip is a loss, and the constructor raises.
(At Zerodha's it is not — 0.2% clears at every clip size there — which is why
the schedule is an argument to the conversion rather than an import inside it.)
Converting once at the clip is safe in the only direction that matters — the
clip is the smallest permitted fill and therefore the most expensive as a
fraction, so every real name faces a hurdle at or *below* the stated figure and
keeps at or above the implied margin. The screen that applies it stays off
until a caller also states a `max_atr_multiple`, which has no measured value
and is not given one here. Neither fee schedule has been reconciled against a
real contract note, and the spread is not in either of them at all, so the
hurdle is a floor rather than an estimate.

</details>
