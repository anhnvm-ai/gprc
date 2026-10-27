# Results of the addendum analyses

Addendum: `addendum_2026-10-27.md`, committed before these analyses were run. Result files: `stage2/results/qf_a1.json` (key `gr_bootstrap_cell3`), `stage2/results/qf_a1b.json`, `stage2/results/tables/mc_stress.csv` and `stage2/results/mc/`. Rejection rates are in percent, with 95% Wilson intervals in brackets.

## R1. Regime-re-estimating bootstrap, 1,000 replications

- Reproduction check of replications 0-9 against the stored results: identical (`stage2/results/mc/gr_check.json`).
- Replications used: 1,000 of 1,000 (Markov-switching fit failed in 0).
- Rejection rate at 5%: 3.3 [2.4, 4.6] (33 rejections), Monte Carlo standard error 0.6 percentage points. Under the rule of the addendum: below nominal size.
- Driscoll-Kraay (L = 4) on the same replications: 7.7 [6.2, 9.5].
- Bootstrap standard error: mean 0.0845, median 0.0801; standard deviation of the estimate 0.0821. Mean number of skipped draws: 12.5 of 199.
- With the first 200 replications the rejection rate was 3.0% (version 1.0 of the paper).

## H and S. Stress designs for the month-clustered wild bootstrap

Design H: Student-t GPRC innovations. Choice of the degrees of freedom (200 calibration panels): observed top-5 share 0.528; median simulated share 0.309 (nu = 2.5), 0.269 (nu = 3), 0.229 (nu = 4), 0.214 (nu = 5), 0.195 (nu = 8). Chosen: nu = 2.5. Design S: AR(1) common volatility shock with rho = 0.5.

| Procedure | Cell 3 (Gaussian) | H | S |
|---|---|---|---|
| Driscoll-Kraay, L = 4 | 7.7 [6.2, 9.5] | 8.9 [7.3, 10.8] | 6.4 [5.0, 8.1] |
| Driscoll-Kraay fixed-b, L = 4 | 7.3 [5.8, 9.1] | 8.3 [6.7, 10.2] | 5.3 [4.1, 6.9] |
| HC3 | 9.4 [7.7, 11.4] | 8.7 [7.1, 10.6] | 8.4 [6.8, 10.3] |
| Clustering by month | 6.3 [5.0, 8.0] | 6.6 [5.2, 8.3] | 5.2 [4.0, 6.8] |
| Webb wild cluster bootstrap by country | 10.4 [8.7, 12.4] | 11.6 [9.8, 13.7] | 9.7 [8.0, 11.7] |
| Wild bootstrap by month (primary) | 5.0 [3.8, 6.5] | 5.0 [3.8, 6.5] | 4.6 [3.5, 6.1] |
| Moving-block bootstrap, 6 months | 7.0 [5.6, 8.8] | 6.2 [4.9, 7.9] | 5.3 [4.1, 6.9] |

Statistics of the simulated panels (medians over the replications) and of the data (`extra/diag_round3_premise.json`, exploratory):

| Statistic | Data | Cell 3 (Gaussian) | H | S |
|---|---|---|---|---|
| Mean of the estimate (true value 0) | | -0.0009 | +0.0017 | -0.0001 |
| Share of estimates at or below -0.171 | | 2.2% | 2.4% | 0.6% |
| Top-5 share of the identifying variation | 0.528 | 0.178 | 0.324 | 0.115 |
| Excess kurtosis of GPRC | 39.1 | 0.0 | 9.5 | 0.0 |
| Bootstrap SE (1-month blocks) / month-clustered SE | 1.21 | 1.01 | 1.04 | 1.00 |
| Bootstrap SE (6-month blocks) / month-clustered SE | 1.41 | 1.00 | 1.03 | 1.00 |
| Lag-1 autocorrelation of the monthly score | -0.16 | -0.03 | -0.03 | -0.01 |
| Lag-1 autocorrelation of the monthly mean residual | 0.02 | 0.00 | 0.00 | 0.07 |

Validation: the Gaussian design run by `a1b_mc_stress.py` reproduces the main Monte Carlo study for all 1,000 replications (identical estimates: True; identical p-values for every procedure: True).

## Reading under the decision rules of the addendum

- The month-clustered wild bootstrap remains the primary inference procedure.
- R1: the regime-re-estimating bootstrap has a size below the nominal 5%.
- H and S: the month-clustered wild bootstrap rejects in 5.0% (H) and 4.6% (S) of the replications. It is not oversized under either design (7% rule), so its p-value is not described as a lower bound. Procedures above 7%: H: Driscoll-Kraay, L = 4; Driscoll-Kraay fixed-b, L = 4; HC3; Webb wild cluster bootstrap by country. S: HC3; Webb wild cluster bootstrap by country.
- Neither design reproduces the concentration of the identifying variation in the data (top-5 share 0.32 in H and 0.11 in S, against 0.53), nor the gap between bootstrap and clustered standard errors. The size of the primary procedure under this feature of the data therefore remains uncertain.
