# Backtest artefacts

## Retirement Notice: October 5, 2026

The Groww-derived local outputs described below are retired by the authorized
Kite migration. Their tables and conclusions are historical, not Kite results.
Do not rerun these recipes against the replacement cache and claim the same
experiment: provider, data quality, fees and feature gap behavior changed.

The current [Kite data validation](../data/kite_migration_20261005/validation.json)
and [TP dataset](../data/research/tp_only_kite_20261005_final/manifest.json)
replace the research inputs, not the old trained models or performance results.
The [fresh Kite CatBoost study](../docs/TP_ONLY_TRAINING.md) has since completed
eight candidates and one seed repeat, selecting a 462-tree depth-four model.
Its TP-touch evaluation is **not an execution backtest**. No new Kite net-profit
or no-stop TP-by-15:00 portfolio result is available. The user's intended exit
is TP or 15:00 IST without a stop; the older stop-based sweep is not that
contract. The neutral symbol list remains available for diagnostics.

## Current Research Handoff: October 9

The built Kite historical/live paths are validated to the tested scope, and the
next work is [scanner research](../docs/ARCHITECTURE.md#scanner-research-handoff).
The [current probability contract](../docs/strategies/TP_PROBABILITY.md) is TP-only,
not TP-before-SL. Existing replay mechanics remain a useful tested component,
but the historical profiles/results below are not an approved no-stop policy.

The [Kite frequency/return audit](../data/research/tp_frequency_returns_kite_20261006/report.json)
does measure gross TP-or-15:00 price outcomes, including non-hit loss tails;
it does not supply executable fills, transaction costs or portfolio P&L.
Preserve that distinction, write any new study to new paths, and do not rerun
deleted-provider recipes as a way to start the next session. No experiment or
new full-suite check was performed by this documentation update.

The rest of this document is the archived pre-migration report. Refer to the
[retirement plan](../data/kite_migration_20261005/retirement_plan.json) for the
explicit deletion scope; the deletion receipt records actual completion.

Published measurements, not inputs. Nothing here is read by the code; these are
the outputs of `ai_trader.cli.backtest` kept so a claim about the strategy can
be traced to the run that produced it.

## What is here

| File | What it is |
| --- | --- |
| `universe.txt` | The 15 symbols every run below measures on, and why those. |
| `sweep.tsv` | The step 7 grid: 10 rows, stop 1.0-3.0, fixed vs trailing. |
| `report_fixed-*.txt`, `report_trail-*.txt` | One rendered report per grid row. |
| `controls.tsv`, `report_frictionless-*.txt` | Fill model switched off, to separate "no edge" from "edge eaten by my fill assumptions". |
| `baseline_62s.tsv`, `report_baseline_62s.txt` | The earlier 62-session run, kept because the grid supersedes it rather than agreeing with it. |

## Reproducing the grid

Every grid row reads one frozen cache and makes zero vendor calls, which is what
makes the ten rows comparable to each other. Two details are load-bearing:

    --end 2026-09-24    not 09-25
    --offline           no broker, so no row can be moved by a revised bar

The end date is a workaround, not a preference. Four symbols print their last
bar of 2026-09-25 at 15:28 rather than 15:29, and `history/store.py` derives
coverage from the bars themselves, so it reads the missing 15:29 minute as "not
cached" and -- with no broker to ask -- fails the run. Ending a session earlier
sidesteps a gap the vendor will never fill. The CLI pins `end` to 15:30 of the
date given, so 15:29 cannot be requested directly.

    .venv/Scripts/python.exe -m ai_trader.cli.backtest \
      --symbols-file backtests/universe.txt \
      --start 2026-07-01 --end 2026-09-24 --offline \
      --latency-seconds 1 --half-spread 0.0002 --slippage 0.0001 \
      --stop-atr 2.0 [--trailing] \
      --label fixed-2.0 \
      --report backtests/report_fixed-2.0.txt \
    --history backtests/reproduced_sweep.tsv

The report is overwritten per run; the history TSV is appended. Populating the
cache in the first place needs a broker and no `--offline`.

New reports include the complete strategy's SHA256 and pre-period warm-up count.
Their TSV schema differs from these original published tables, which remain
unchanged; reproduce into a new history file rather than append under an old
header. The separate [RELIANCE score study](reliance_sep2026/README.md) records
its frozen-data protocol and any blocking completeness findings.

## What the rows have been re-measured against

All twelve rows were regenerated after `77ffccf` put `ChandelierStop`'s anchor
back on the tick grid, because the `trail-*` rows had been measured with anchors
carrying a residue in the last place. Nothing moved: every P&L, round-trip and
excursion figure came back identical, on the trailing rows as well as the fixed
ones. Instrumenting one cell says why. Of the 57,363 stops the tape asked for at
2.0x, 8,373 anchors really were off the grid, and not one of them reached the
output -- the main path rounds to a boundary regardless, and the clamp that
could have carried an anchor through untouched never bound on an off-grid one.
Priced both ways, all 57,363 stops agree. The correction removes a hazard this
tape does not happen to trigger.

Two things in the reports did move, neither of them a measurement:
`unfilled entries` and `unwound entries` swap one-for-one, because `eecbf90`
re-filed a position abandoned at the square-off cutoff from the second to the
first, and `declined, both ways` is new from `40c30e4`. That counter is
report-only, which is why the TSV columns are unchanged by it.

## Reading the grid

`mean_favourable_fraction` and `mean_adverse_fraction` are excursions, and both
are **conditioned on the exit rule that produced them** -- a tight stop truncates
the adverse side before it finishes, and MFE rises monotonically with stop width
for that reason alone. They are comparable across the fixed/trailing pair at one
multiple; they are not comparable across multiples. The widest stop is the least
truncated and so the most diagnostic.

The frictionless controls exist because the excursion asymmetry in the grid is
partly an artefact: entry price already carries the half spread and slippage, so
MFE is measured from a worse price than the signal saw. Removing the fill model
moves the ratio from 0.74 to 0.98, which is the difference between "the entries
lean the wrong way" and "the entries are a coin flip". They are a coin flip.
