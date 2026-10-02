# Strategy Research Notes

Reviewed through 2026-10-01. These are research hypotheses and methods, not proven future
returns. A peer-reviewed empirical result can still fail on another market,
time horizon, execution model or sample. Author backtests and community examples
receive less evidential weight than independent out-of-sample replication.

The current objective is **calibrated TP-before-SL-within-h probability**, with direction
encoded separately. See [the probability contract](TP_PROBABILITY.md) and the
[symmetric-horizon experiment](../../backtests/tp_horizon_symmetric_v1/README.md).
The [month-split cutoff experiment](../../backtests/tp_probability_month_split/README.md)
is the preceding baseline.
The [earlier experiment](../../backtests/tp_probability_2026q3/README.md) is
preserved as dated evidence, not an untouched test for subsequent research.

## Strategy Families

### Opening-Range Breakout With Relative Volume

**Sources:** Zarattini, Barbon and Aziz, 2024,
[A Profitable Day Trading Strategy for the U.S. Equity Market](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284),
[author summary](https://concretumgroup.com/a-profitable-day-trading-strategy-for-the-u-s-equity-market/),
and [QuantConnect implementation](https://www.quantconnect.com/research/18444/opening-range-breakout-for-stocks-in-play/).

The author study concerns a five-minute opening range and unusually active US
stocks over 2016-2023. The implementation selects a liquid universe and measures
opening volume relative to the same opening window over the previous 14 days,
then trades a directional break of the opening range with volatility-based stops.

**Useful hypothesis:** abnormal *same-time-of-day* activity may distinguish a
meaningful breakout from an ordinary extension. Candidate features: opening
relative volume, range width relative to prior volatility, breakout distance,
and price rejection after the break. Use only completed opening-window data.

**Limits:** this is not evidence that every intraday high should be bought.
The author's performance claims are retrospective; the implementation uses a
different universe and a shorter test period. Forum reports are anecdotes, not
verified live evidence. Costs, universe selection, volume warm-up and fill timing
need independent checking. We reviewed the author summary and implementation,
not the full original PDF. No reported return is imported as our expected return.

### Intraday Momentum With Regime Context

**Source:** Gao, Han, Li and Zhou, 2018,
[Market Intraday Momentum](https://ideas.repec.org/a/eee/jfinec/v129y2018i2p394-414.html),
Journal of Financial Economics 129(2), 394-414. Published abstract reviewed.

The paper finds that the first half-hour market return, measured from the prior
close, predicts the final half-hour return in S&P 500 ETF data from 1993-2013,
with stronger effects on high-volume, volatile and major-news days. It also
examines other actively traded ETFs.

**Useful hypothesis:** direction may need market context and interaction with
volatility/activity, not just a stock's EMA stack. Candidate inputs: market and
sector relative strength, opening displacement, and lagged volatility regime.

**Limits:** this is an ETF, half-hour-horizon result, not one-minute RELIANCE TP
prediction. Our 15:15 cutoff excludes part of the exchange's final half hour.
Reusing its timing literally would violate the present intraday contract.

### Short-Term Reversal as Liquidity Provision

**Source:** Stefan Nagel, 2012,
[Evaporating Liquidity](https://www.nber.org/papers/w17653), Review of Financial
Studies 25(7), 2005-2039. NBER abstract and publication record reviewed.

The study links short-term reversal returns to liquidity provision and finds
that those expected returns vary strongly with market conditions, including VIX.

**Useful hypothesis:** a reversal signal should be conditioned on liquidity and
volatility regime rather than triggered by RSI alone. Distinguish a temporary
price displacement from sustained informed order flow.

**Limits:** a reversal premium can compensate substantial risk. This is not a
fixed 0.2% intraday target rule, and high volatility need not make fading a move
safe. Appropriate Indian-market regime data must be timestamp-aligned.

### Order-Flow and Depth Information

**Source:** Cont, Kukanov and Stoikov, 2014,
[The Price Impact of Order Book Events](https://arxiv.org/html/1011.6402v3),
Journal of Financial Econometrics 12(1), 47-88. Full HTML reviewed.

Order-flow imbalance at the best quotes explains short-interval price changes
more robustly than trade volume alone in their 50-stock US sample; depth affects
the sensitivity of price to that flow.

**Useful hypothesis:** lagged bid/ask imbalance, replenishment, depth, spread and
signed order flow may improve very short-horizon TP discrimination. These require
historical quote events, not just candle volume.

**Limits:** the main regressions explain *contemporaneous* price changes. Using
flow from the same future interval as a predictor would leak the answer. Their
fit statistics are not future trading accuracy. We have not fabricated OFI from
OHLCV or validated this strategy locally.

### Stop-Loss Policy and Regime Dependence

**Source:** Kaminski and Lo, 2014,
[When Do Stop-Loss Rules Stop Losses?](https://ideas.repec.org/a/eee/finmar/v18y2014icp234-254.html),
Journal of Financial Markets 18, 234-254. Published abstract reviewed.

The study evaluates how stop policies change return and volatility, using daily
index futures and longer sampling horizons. It does not supply an optimal
one-minute stock stop distance.

**Application:** report TP and SL jointly with payoff sizes and timeouts. A wider
SL can increase TP-hit percentage mechanically without improving the signal.
Do not present a high hit rate obtained with a small TP and large SL as proof of
predictive advantage. Keep every stop finite and every position intraday.

## Probability Methods

**Selective prediction:** Geifman and El-Yaniv, 2017,
[Selective Classification for Deep Neural Networks](https://arxiv.org/html/1705.08500v2).
Full HTML reviewed. A useful selection rule trades coverage for lower error.
Its formal guarantees assume i.i.d. observations; correlated financial paths
and shifting regimes do not inherit those guarantees. Apply the risk-versus-
coverage idea, not the published image-classification success percentages.

**Calibration:** Guo, Pleiss, Sun and Weinberger, ICML 2017,
[On Calibration of Modern Neural Networks](https://arxiv.org/html/1706.04599v2).
Full HTML reviewed. Confidence and correctness can diverge; inspect reliability
bins and proper probability losses on separate data. Temperature scaling can
change confidence without improving class decisions. Mapping a score through
sigmoid is not itself validation.

**Calibration alternatives:** Kull, Silva Filho and Flach, AISTATS 2017,
[Beta Calibration](https://proceedings.mlr.press/v54/kull17a.html),
[paper](https://proceedings.mlr.press/v54/kull17a/kull17a.pdf). Abstract and PDF
methods extraction reviewed. The monotone family is
`sigmoid(a*log(p) - b*log(1-p) + c)`, and includes identity at a=b=1,c=0.
The experiments are on general classification datasets, not trading. Extra
flexibility may overfit a small calibration period. Our current two-parameter
mapping works on logits, already includes identity, and is compared with leaving
the original forecasts unchanged. Full beta calibration is a future comparison,
not an unreported extra trial on the current final test.

**Backtest selection bias:** Bailey, Borwein, Lopez de Prado and Zhu,
[The Probability of Backtest Overfitting](https://scholarworks.wmich.edu/math_pubs/42/),
2017 publication; [author-hosted working paper](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
and its methods extraction reviewed. The paper develops PBO and combinatorially
symmetric cross-validation for assessing selection overfit. Trying many variants
and reporting only the best is biased; a holdout repeatedly consulted for
research decisions is no longer independent. We record the full trial set and
freeze choices before final evaluation. We have **not** computed a PBO statistic
or claimed that a chronological split eliminates all selection bias.

## Scorer Model Choice

**Current result:** the symmetric-horizon study compared 72 model/feature
configurations, including two logistic, two shallow boosted-tree and two small
neural configurations per feature family and target. Logistic won three target
comparisons; a depth-2 tree won one; neither neural configuration won. Prior-week
context helped some August comparisons but not all. The best probability-quality
selection was a rare event, not a high-confidence trade, and all 16 combined
trading alternatives lost money. Preserve those facts rather than declaring a
globally optimal algorithm. Keep logistic, trees and frozen base-rate controls;
improve information and validation before expanding network size.

The current inputs are a small engineered tabular vector. A tree model can
learn interactions such as momentum conditional on volatility, time of day and
market regime that an additive logistic model misses. That flexibility also
increases overfitting risk. Keep the search budget small and predeclared; compare
proper probability losses, per-side reliability, ranking and precision/coverage
at thresholds chosen before evaluation. The tested small networks were useful
challengers, not a cure for unstable labels, weak information or market shifts.

### What the Papers Establish

**Grinsztajn, Oyallon and Varoquaux, NeurIPS 2022:**
[Why do tree-based models still outperform deep learning on typical tabular data?](https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf)
PDF methods/results extraction reviewed. Across 45 tabular datasets, with
training sets capped at about 10,000 samples, tree methods remained strong
against neural models despite substantial tuning. The study investigates
irregular target functions, uninformative features and inappropriate inductive
biases. **Limit:** time-series/stream datasets were excluded, and this is not a
benchmark of financial TP probabilities. It motivates a tree baseline, not a
claim that trees will win here.

**McElfresh and coauthors, NeurIPS 2023:**
[When Do Neural Nets Outperform Boosted Trees on Tabular Data?](https://arxiv.org/html/2305.02997v3)
Methods/results extraction reviewed. The comparison covers 19 algorithms and
176 classification datasets. Dataset properties and tuning matter; simple
baselines or lightly tuned boosted trees often compete with more elaborate
choices. Irregular feature distributions tend to favor boosted trees. **Limit:**
the study is general tabular classification, not chronological financial
forecasting. Neither paper proves that neural networks are always inferior.

**Zhang, Zohren and Roberts, IEEE Transactions on Signal Processing 2019:**
[DeepLOB](https://arxiv.org/html/1808.03668v6). Full HTML reviewed, particularly
data, labels and trading assumptions. CNN/Inception/LSTM layers process the
last 100 order-book updates, each containing prices and sizes at ten levels on
both sides. The LSE study uses one year of data, over 134 million samples,
with six months training, three validation and three testing. This is a strong
reason to consider neural sequence models when rich book data is available.
**Limits:** labels describe smoothed mid-price direction, not executable
TP-before-SL events; its illustrative simulation uses mid-prices and excludes
transaction costs. Its reported accuracy is not an expected TP-hit rate for
our one-minute RELIANCE candles. OHLCV cannot reconstruct its order-book inputs.

**Arroyo, Cartea, Moreno-Pino and Zohren, 2023:**
[Deep Attentive Survival Analysis in Limit Order Books](https://arxiv.org/html/2306.05479v1).
Full HTML reviewed. A convolutional-Transformer encoder and monotone decoder
estimate limit-order fill-time distributions from Nasdaq book sequences. It
uses a right-censored likelihood and emphasizes proper distributional scoring.
**Application:** forecast an event-time distribution, not an eventual hit
probability relabeled as speed. **Limits:** order filling is not TP realization,
and our SL is a competing failure rather than independent censoring. Its results
do not establish the best architecture or profitability for our event.

**Lehmann, Quarterly Journal of Economics 1990:**
[Fads, Martingales, and Market Efficiency](https://www.nber.org/papers/w2533).
NBER abstract and publication metadata reviewed, not the full paper. The study
reports weekly winner/loser reversals in historical equity returns. This supports
testing prior-week context without assuming that its coefficient must favor
continuation. It does not prove a one-week feature predicts the next six minutes
of RELIANCE, nor that the same effect survives modern Indian-market costs.
Our five-session features are explicitly ablated against intraday-only models.

### Next Controlled Comparison

1. Keep the requested July/August/September result as a recorded benchmark.
   September has already been inspected; any new model evaluated there is
   retrospective development, not another independent test. Freeze a subsequent
   choice before new forward sessions and assess several chronological windows.
2. Keep the implemented joint TP/SL/timeout model and consistent 6/15/30/60-minute
   probabilities. Validate its high-score tails, not just average calibration.
   A nearly correct 1% forecast does not provide an attractive entry. Add a
   separate cost-aware action objective, including timeout returns and abstention.
3. Extend the completed logistic/tree/neural and time/volatility-control
   comparison across independent months and instruments. Keep identical labels,
   dates, costs and eligibility across competing models. Test regime stability
   and whether directional inputs add value beyond movement potential.
4. Select with August-only or equivalent earlier validation. Keep sparse trading
   allowed, but use day-block uncertainty and a predeclared evidence requirement
   before promotion; 1/1 selected wins cannot establish high precision. Bootstrap
   sessions rather than treating overlapping minute-actions as independent.
   Purge overlapping outcome windows if a future split occurs within sessions.
5. Add information before model size: investigate same-time relative volume and
   aligned market/sector context with complete historical coverage. Test feature
   ablations on development periods. Use book/quote features only with genuine
   historical depth data; do not fabricate them from candle volume.
6. Expand neural comparisons only after obtaining suitable richer histories or
   quote/book sequences. The two small tabular networks already tested did not
   improve model selection. Retain deterministic inference, safety and sizing
   controls; no research winner is automatically promoted to live execution.

A month is a useful first benchmark, but not evidence that data is abundant.
The symmetric-horizon fit has 2,160 sampled July side-actions at only 1,080
timestamps across 18 sessions after five sessions of prior-week warm-up. Both
sides share the same future path. Collecting more independent sessions and
regimes is more meaningful than counting every overlapping minute as another
independent training example.

## How Research Enters This Repository

1. State the event, time horizon, costs, TP/SL and data needed before fitting.
2. Add causal inputs only; preserve missing-data flags and prior-session boundaries.
3. Compare against a simple base-rate forecast as well as the current model.
4. Keep fitting, calibration, policy selection and evaluation periods disjoint.
5. Report LONG and SHORT separately, along with actual trade counts and probability
   errors. Hypothetical overlapping actions are not independent trades.
6. Record all meaningful trials. Add new hypotheses to a later experiment instead
   of changing a model after inspecting its final-test result.

Priority future tests are same-time relative-volume breakout selection, market-
relative momentum, and lagged quote/depth features if suitable data is obtained.
They are not implemented or validated merely because the supporting papers exist.