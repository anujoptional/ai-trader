# TP Probability Contract

## Active Kite Contract: October 9, 2026

The signed TP-before-SL contract in the archive below is **not the active model's
event or a selected scanner policy**. The current frozen model and its inputs
are specified in [TP_ONLY_TRAINING.md](../TP_ONLY_TRAINING.md). This page makes
the probability/selection boundaries explicit for the next scanner study.

| Quantity | Active Meaning |
|---|---|
| Reference price | Last completed one-minute close |
| LONG event | At least one of the next 60 complete minutes reaches reference times 1.002 |
| SHORT event | At least one of the next 60 complete minutes reaches reference times 0.998 |
| Adverse movement | Does not end the label; no competing price stop |
| Decision grid | Every five minutes, 09:40-14:15 IST, with a full future window ending by 15:15 |
| Missing future/input window | Unavailable/excluded, not imputed or labeled a failure |
| Model input | 13 finite numerical features plus side; no stock identity, timestamp, labels or excursions |
| Raw output | Sigmoid of the frozen CatBoost margin, a probability in [0, 1] |
| Selected calibrated output | Locked side-specific sigmoid mapping fitted on the earlier calibration period |
| Intended position exit | TP or 15:00 IST; no price stop-loss and no forced 60-minute exit |

LONG and SHORT are paired hypothetical actions, **not complements**. Both can
touch their targets on the same future path. The model does not automatically
choose the higher side, sign a probability, apply a P >= 0.80 threshold or open
a position. The diagnostic returns explicit side and raw/calibrated outputs;
side choice, abstention and duplicate handling belong to a separately specified
scanner protocol. Do not relabel the legacy rule score as calibrated probability.

### Interpretation And Evidence

The validation-selected model is `depth4_l2`, 462 trees, retained in
[best.cbm](../../data/research/tp_catboost_kite_20261005/final/best.cbm) with its
preprocessing, model/selection locks and calibration. All 686,112 frozen later
evaluation forecasts reproduced exactly on October 7. The selected calibration
helped its March selection score but worsened later log loss and Brier versus
the predeclared raw comparator; keep both, without hindsight reselection.

Calibration concerns frequencies over comparable samples, not certainty for
one action. AUC is ranking, not a TP probability or classification accuracy.
Neither establishes trade profitability. The existing band/frequency/return
reports count overlapping paired opportunities, not funded independent trades.
The max-side candle buckets are descriptive only, not an approved selection rule.

Changing the deadline changes the event: next-hour TP, TP by 15:00 and TP by
continuous-session close are distinct labels. Late hourly hits can occur after
the intended 15:00 exit. Do not reuse an hourly probability as a calibrated
cutoff probability merely because the same score ranks both outcomes.

### Next Scanner Study

The built historical/live paths are validated to the tested scope and suitable
for scanner research. Freeze the next comparison and policy choices, retain
separate rule/raw/calibrated controls, measure coverage and loss tails as well
as hit frequency, and label fills/costs/capital assumptions explicitly. Preserve
the no-stop exit contract and all old artifacts. These evaluated dates have
already been inspected; confirmatory claims need separately reserved or
prospective evidence. See [the research handoff](../ARCHITECTURE.md#scanner-research-handoff).
No new study, order capability or automatic retraining is enabled by this page.

## Historical Signed Contract

The remainder is the original pre-Kite TP/SL research contract, retained as
history. Its "Current Implementation" heading and old script/output paths are
not current instructions, source-provider choices or preserved model artifacts.

<details>
<summary>Archived signed TP-before-SL and symmetric-horizon contract</summary>

## Event and Output

For a specified instrument, decision time, direction, TP, SL, cutoff and fill
model, estimate the probability that the position exits at TP before SL or the
intraday cutoff. A cutoff exit is a failure for this event even if profitable.
An ambiguous candle touching both barriers is resolved as a stop in the research
execution model. No position exits on its entry candle.

Let p_LONG and p_SHORT be separate probabilities for hypothetical long and short
actions. They are **not complements**: both actions might fail, and with unequal
barriers both could succeed at different times in their separate hypothetical
paths. Choose the direction with the larger estimated probability and attach its
sign without changing magnitude:

$$S=\begin{cases}+p_{LONG},&p_{LONG}>p_{SHORT}\\-p_{SHORT},&p_{SHORT}>p_{LONG}\\0,&p_{LONG}=p_{SHORT}.\end{cases}$$

Thus -0.80 means an estimated 80% chance for the specified SHORT action. The
sigmoid itself is never negative; the direction supplies the minus sign.
`2*sigmoid(raw)-1` would not preserve probability magnitude and is not this
contract. It remains available only for reproducing the earlier utility studies.

Apply `abs(score) >= entry_threshold` separately. There is no implicit 50% floor.
Zero on a tie means abstention, not proof that neither action can hit TP. A score
is never permission to bypass feasibility, sizing, position limits or other
deterministic risk controls.

## What Calibration Means

Among many comparable, independently evaluated predictions near 0.70, about 70%
should hit the specified TP before SL/cutoff. A single 0.70 forecast can still
lose, and seven trades cannot establish calibration at that probability.

Inspect these separately:

- **Mean forecast versus observed TP rate:** intuitive, but opposing bin errors
  or long/short errors can cancel.
- **Reliability bins:** counts, average forecast and actual TP rate in fixed
  probability ranges, separately by direction and event definition.
- **Brier score and log loss:** proper probability losses; lower is better.
  Compare against a base-rate forecast fixed before the final test.
- **Ranking:** ROC-AUC measures how often a TP action outranks a non-TP action,
  with half credit for ties. It does not establish calibration or profitability.
  Report it separately for LONG and SHORT; overlapping actions are dependent.
- **Actual selected trades:** filtering and position occupancy change the sample.
  Good calibration over all hypothetical entries need not survive selection.
- **Economic results:** a calibrated high TP rate can still lose money when
  losses are larger than gains or costs consume the target.

Do not change forecasts to match outcomes already observed. Calibration mappings
are fitted on a dedicated earlier period and evaluated later. The earlier
cutoff study uses per-side monotone scalar mappings. The symmetric-horizon study
uses per-side joint probability recalibration, which preserves probability mass
and nested horizons, not necessarily cross-sample ranking. A subsequent selection
period chooses mapping versus identity. No final-test recalibration occurs.

## Current Implementation

[research_horizon_probability.py](../../scripts/research_horizon_probability.py)
runs the [symmetric horizon study](../../backtests/tp_horizon_symmetric_v1/README.md).
Its event is $P(TP\text{ before SL and within }h\mid X,d)$, with h in hours from
the executed buy entry for LONG or sell entry for SHORT. Equal tick-rounded
distances place TP and SL on opposite sides of that same entry price. The future
short-cover buy price cannot anchor a decision made before it exists.

The model predicts a joint distribution over TP time bins, SL time bins and
timeout. Summing TP bins up to h gives a cumulative probability that cannot
decrease with h. It compares regularized multinomial logistic regression,
shallow boosted trees and small neural nets, with and without five completed
sessions of context. Scaling and base fits use July only, calibration and all
selection use August, and locked evaluation uses September 1-28.

The first grid uses TP=SL at 0.15/0.20/0.30/0.40% and h=0.1/0.25/0.5/1 hours.
All events share entry times with a complete one-hour window before cutoff;
shorter afternoon windows remain outside that benchmark. An explicit
`ExitReason.HORIZON` is a TP failure, even if the exit is net profitable. A
shorter timeout cannot be labeled as failure at a longer horizon.

This is a coherent probability output, not proof of high-score calibration or
tradability. The validation-selected event was rare and all 16 combined trading
alternatives lost after modeled costs. A high AUC is not a high TP probability.

## Earlier Cutoff Study

[research_tp_probability.py](../../scripts/research_tp_probability.py) runs the
[month-split study](../../backtests/tp_probability_month_split/README.md), with
July base-coefficient fitting, August validation/calibration/selection and
September 1-28 evaluation. Use its explicit `--root`; the default root remains
the [earlier study](../../backtests/tp_probability_2026q3/README.md). The data
snapshot is separately frozen. Model family, regularization, calibration method
and threshold are locked before final-test labels are computed. Final evaluation
reloads that serialized lock rather than relying on a different in-memory model.

Each TP/SL pair has its own fitted model and calibration. Features use the
candidate TP for volatility normalization. The current alternatives are 11
price/session inputs versus 17 including prior-session and candle context,
plus a direction indicator. The prior-session frame is the completed
09:15-15:15 window, not an official 15:30 daily bar.

The output remains **research-only**. The saved probability model is not a
schema-v3 live strategy profile: the production scanner's old signed utility
score has not silently been relabeled as a probability. Execution labels and
non-overlapping scheduling reuse the existing tested replay mechanics.

## Stop Distance and Speed

Earlier studies kept ATR stops and separately reported initial price distance:
`SL_percent = 100 * abs(stop_price / entry_price - 1)`. Report median, mean and
range overall and by direction. This distance excludes fees and is not a
guaranteed maximum loss under gaps or slippage. The symmetric-horizon experiment
explicitly changes its research exit rule to TP=SL, rather than merely changing
the display units. Earlier profiles and production defaults remain unchanged.

End-of-session TP probability does not specify how quickly TP will occur. Report
observed TP-within-5/15/30-minute rates with **all actions/trades** as denominator,
and successful-TP holding times separately. A conditional median among winners
must not hide stops and timeouts. The earlier cutoff model provides timing
diagnostics; the newer model directly estimates the timed event probabilities.

The implemented speed-aware probability explicitly estimates
$P(TP\text{ first and within }h\mid X,d)$, preserving increasing cumulative
probability as the horizon grows. Treat SL as a competing event, not independent
censoring. A time-discounted utility is a different quantity from probability.

## Validation Limits

The horizon study uses one stock, 63 complete sessions and 19 September
evaluation sessions. July 1-7 supply five sessions of warm-up, and base models
use July 8-31 only (18 sessions). The previous month-split cutoff study instead
used July 1 warm-up and July 2-31 fitting. August fits the calibration mappings
and supplies selection data. All of September has already been
examined in earlier studies, so it is not an untouched prospective test.
Many adjacent minutes share features and outcome paths; they are not independent
trials. Source agreement, market shifts and minute-bar fill assumptions remain
limitations. Modest ranking gains do not validate high-precision probabilities.

A useful future probability model needs reliable bins on genuinely new sessions,
both directions represented, and no material degradation against simple base
rates. Further tuning on this final test would invalidate its role as a test.

</details>