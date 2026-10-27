# Second addendum to the analysis plan: influence of the months with the most identifying variation

**Date:** 27 October 2026, the date of the commit that adds this file.

**Approval:** approved by all authors on 27 October 2026.

**Status:**
- Exploratory. The analyses were motivated by a feature of the data found after the analysis plan, in the diagnostics of the first addendum.
- Written after the results of the first addendum were known, and before the analyses below were run.
- The commit that adds this file precedes the commits that add their code and results.

## Purpose

- In the data, five months carry 53% of the identifying variation of β₃. In the simulation designs of the first addendum, the five largest months carry at most 32%.
- This addendum fixes a direct check of how much the empirical estimate of β₃ depends on these months.
- The months, the model, the inference and the wording rules are fixed here, before any of the estimates below is computed.
- The analyses assess robustness. They are not used to select a sample or a specification.

## What was known when this file was written

- **Baseline estimates** (Table 3 of the paper): OLS β₃ = −0.171 with a wild-bootstrap p-value of 0.140; time-aware DML by regime δ = −0.140.
- **The partial leverage of every month** (defined below). It uses the regressors only, not the outcome.
- **Leave-one-stress-episode-out estimates** (`stage2/results/qf_diag.json`, in this repository since version 1.0):
  - The episode April–May 2010 consists of months 2 and 5 of the list below. Without it, the OLS estimate is −0.201.
  - The episodes that contain months 1, 3 and 4 give −0.158, −0.139 and −0.178.
- **Not yet computed:**
  - any regression on a sample without one of the five months, or without all five;
  - any leave-one-month-out estimate.

## Definitions

- **Model:** Eq. (1) of the paper, as estimated in `stage2/analysis/a6_main_tables.py`. It uses:
  - the function `qflib.design` with the baseline ex-post regime (`stress_common`);
  - the persistence control, the six controls and the market dummies.

  N = 946 observations in 195 months, of which 37 are stress months.
- **Month:**
  - Month t is the month of GPRC and of the regime label. The outcome is log RV in month t+1.
  - Dropping month t removes the observations of all markets dated t.
- **Partial leverage of month t:** q_t = Σ_i D̃_it² / Σ_s Σ_i D̃_is².
  - D̃ is the residual from a regression of the GPRC × Stress column on all other columns of the design.
  - It is the metric behind the 53% figure (`extra/diag_round3_premise.py`).
  - It is the partial leverage of MacKinnon, Nielsen and Webb (2023), with months as clusters.
- **The set T of five months,** ranked by q_t and fixed here:

| Rank | Month | q_t | Stress | Market with the largest share of q_t | Its share | GPRC (z) of that market |
|---:|---|---:|---|---|---:|---:|
| 1 | 2026-03 | 0.279 | yes | VNM | 0.59 | 5.76 |
| 2 | 2010-05 | 0.082 | yes | THA | 0.93 | 3.70 |
| 3 | 2025-04 | 0.076 | yes | VNM | 0.81 | 3.25 |
| 4 | 2021-02 | 0.053 | yes | THA | 0.89 | 2.78 |
| 5 | 2010-04 | 0.038 | yes | THA | 0.93 | 2.55 |
| | Total | 0.528 | | | | |

- T is not re-ranked after a month is dropped.
- The code recomputes q_t and stops if the set differs from this table.

## Common settings

As for the baseline:
- **Regime:** the regime labels are kept; the Markov-switching model is not re-estimated.
- **Scaling and controls:** the z-scores keep their full-sample moments; the controls and the market dummies are unchanged.
- **Inference:** restricted wild bootstrap clustered by calendar month, with Rademacher weights, B = 999 and seed 42.

## Analyses

- **Reproduction check:** the full-sample OLS run must reproduce β̂ = −0.171 and p = 0.140.
- **I1. Single-month drops:** five OLS regressions, each without one month of T.
- **I2. Joint drop:** one OLS regression without all five months of T. This is the main result of this addendum.
- **I3. DML:**
  - Settings: time-aware DML by regime with Lasso, five blocks, and the wild bootstrap of the final stage (B = 999, seed 42).
  - Samples: the full sample and the six samples of I1 and I2.
  - The full-sample run must reproduce δ = −0.140.
- **I4. All months:**
  - One OLS regression without each of the 195 months in turn, with point estimates only.
  - The delete-one-month jackknife standard error, sqrt((G − 1)/G · Σ_t (β̂₍₋t₎ − β̂)²) with G = 195.
  - Both are descriptive and are not used for inference.
- **I5. Skipped draws of the regime-re-estimating bootstrap:**
  - Settings of the real-data bootstrap in the paper: 6-month blocks, B = 999, seed 42, 3 starting values, smoothed P > 0.5.
  - It is repeated with minimum stress-month counts of 3 and 7 instead of 5.
  - The upper limit of T/2 stress months is kept.
  - The resampled blocks do not depend on the minimum, so a run with the minimum of 5 must reproduce p = 0.245 and 143 skipped draws.

## Outputs

- `stage2/results/qf_influence.json`: all estimates and p-values of I1–I5, and the check of T.
- `stage2/results/tables/influential_months.csv`: one row per month, with:
  - rank and q_t;
  - stress label;
  - the market with the largest share and its GPRC (z);
  - the OLS estimate without the month and its change from the baseline in percent.
- `docs/analysis_plan/addendum2_results.md`: the results read under the rules below.

## Decision rules for the wording

- **Ratio:** let r = β̂ without T / β̂ of the baseline, from I2.
  - **r ≥ 2/3:** the paper states that most of the estimate remains without the five months. The abstract is unchanged.
  - **1/3 ≤ r < 2/3:** the paper states that the estimate falls substantially but stays negative, so the association is partly concentrated in these months. The abstract says so in one clause.
  - **r < 1/3, including a change of sign:** the paper describes the association as concentrated in a few months of acute stress rather than spread across stress months. The abstract and the conclusion say so.
- **p-values** are reported but are not a criterion. Removing about half of the identifying variation is expected to reduce precision.
- **I1:** the range of the five estimates is reported. A change of sign in any of them is stated in the paper.
- **I3** is reported next to I2. It does not change the wording on its own.
- **I4** is reported as the range of the 195 estimates.
- **I5** is reported as the two p-values, next to the baseline value of 0.245.
- All results are reported in this repository and summarised in the paper, whatever their direction.

## Code and results

- **Code:** `python stage2/analysis/diag_influence.py`, committed after this file.
- **Results:** committed after the runs, in the files listed under Outputs.

## Reference

MacKinnon, J.G., Nielsen, M.Ø., Webb, M.D.: Leverage, influence, and the jackknife in clustered regression models: Reliable inference using summclust. Stata J. 23(4), 942–982 (2023). https://doi.org/10.1177/1536867X231212433
