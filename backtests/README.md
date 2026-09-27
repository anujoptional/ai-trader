# Backtest artefacts

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
      --history backtests/sweep.tsv

The report is overwritten per run; the history TSV is appended. Populating the
cache in the first place needs a broker and no `--offline`.

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
