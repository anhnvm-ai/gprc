# Results of the second addendum

Addendum: `addendum2_2026-10-27_influential_months.md`, committed and published before these analyses were run. Result files: `stage2/results/qf_influence.json` and `stage2/results/tables/influential_months.csv`. All analyses are exploratory.

## Checks

- Set T recomputed from the regressors: 2026-03 (0.279), 2010-05 (0.081), 2025-04 (0.076), 2021-02 (0.053), 2010-04 (0.038). Share of the five months: 0.528.
- The months and their ranking are those of the addendum. The addendum prints 0.082 for 2010-05; the value is 0.08146, rounded twice when the table was written. The first run stopped at the check of T, which compared the rounded values, before any estimate was computed. The check now compares the months, their order and each value within 0.001 (see the commit history).
- Full-sample OLS: β̂₃ = -0.1708, wild-bootstrap p = 0.140; identical to `stage2/results/qf_a6.json`: True.
- Full-sample time-aware DML by regime: δ̂ = -0.1399 (difference from `qf_a6.json`: 0.0e+00).
- Regime-re-estimating bootstrap with a minimum of 5 stress months: identical to `stage2/results/mc/gr_real.json`: True.

## I1 and I2. OLS without the five months

Regime labels, z-scores, controls and market dummies as in the baseline. p: restricted wild bootstrap clustered by month, Rademacher weights, B = 999, seed 42. Ratio: β̂₃ / β̂₃ of the baseline (r of the decision rule for the last row). Change: (β̂₃ − β̂₃ of the baseline) / |β̂₃ of the baseline|; a positive change moves the estimate towards zero or above.

| Sample | N | Months | Stress months | β̂₃ | Ratio | Change | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 946 | 195 | 37 | -0.171 | 1.00 | | 0.140 |
| Without 2026-03 | 941 | 194 | 36 | -0.143 | 0.84 | +16.5% | 0.371 |
| Without 2010-05 | 942 | 194 | 36 | -0.182 | 1.06 | -6.4% | 0.164 |
| Without 2025-04 | 941 | 194 | 36 | -0.110 | 0.65 | +35.3% | 0.296 |
| Without 2021-02 | 941 | 194 | 36 | -0.159 | 0.93 | +6.8% | 0.202 |
| Without 2010-04 | 942 | 194 | 36 | -0.187 | 1.10 | -9.6% | 0.101 |
| Without all five months | 923 | 190 | 32 | +0.033 | -0.19 | +119.5% | 0.882 |

## I3. Time-aware DML by regime (Lasso)

| Sample | N | δ̂ | p |
|---|---:|---:|---:|
| Full sample | 780 | -0.140 | 0.183 |
| Without 2026-03 | 775 | -0.161 | 0.423 |
| Without 2010-05 | 775 | -0.150 | 0.148 |
| Without 2025-04 | 775 | -0.081 | 0.394 |
| Without 2021-02 | 775 | -0.139 | 0.215 |
| Without 2010-04 | 775 | -0.151 | 0.143 |
| Without all five months | 760 | -0.053 | 0.812 |

## I4. Leave-one-month-out estimates for all months (descriptive)

- Range of the 195 estimates: -0.196 (without 2025-03) to -0.110 (without 2025-04); 195 of 195 are negative.
- The five largest changes: 2025-04: -0.110 (+35.3%); 2026-03: -0.143 (+16.5%); 2025-03: -0.196 (-14.6%); 2011-08: -0.149 (+12.8%); 2015-08: -0.192 (-12.5%).
- Delete-one-month jackknife standard error: 0.102. For comparison, the month-clustered standard error of the baseline is 0.089 (`extra/diag_round3_premise.json`). Not used for inference.
- All months, with their partial leverage: `stage2/results/tables/influential_months.csv`.

## I5. Skipped draws of the regime-re-estimating bootstrap

| Minimum stress months | Draws used | Draws skipped | SE | p |
|---:|---:|---:|---:|---:|
| 3 | 900 | 99 | 0.258 | 0.264 |
| 5 (paper) | 856 | 143 | 0.254 | 0.245 |
| 7 | 787 | 212 | 0.250 | 0.247 |

## Reading under the decision rules of the addendum

- r = β̂₃ without the five months / β̂₃ of the baseline = +0.033 / -0.171 = -0.195. Case C: the association is concentrated in these months; abstract and conclusion reframed.
- In the paper, the abstract, the introduction, Sect. 6.3, Sect. 8 and Sect. 9 describe the association as concentrated in these months rather than spread across stress months.
- I1: the five single-month drops give -0.187 to -0.110. None changes the sign.
- I3: time-aware DML without the five months gives -0.053, reported next to I2.
- I4: the leave-one-month-out estimates range from -0.196 to -0.110.
- I5: with minimums of 3 and 7 stress months, the p-value is 0.264 and 0.247, against 0.245 in the paper.
