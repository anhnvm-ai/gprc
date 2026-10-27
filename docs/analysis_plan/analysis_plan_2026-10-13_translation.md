# Pre-specification [E1]

*English translation of the authors' working document, written in Vietnamese and frozen on 13 October 2026.
SHA-256 of the original file: `682dead90c0e4732e233de4fc56d659bb3c615b2242348c478c0bfe232578714`.
The translation follows the original section by section. Edits are marked [E1]-[E4] and a translator's
note [T1]; all are listed at the end. The paper refers to this document as the analysis plan. Its status
is described in `README.md` in this folder.*

**Approval:** approved on 13 October 2026 [E2]. The plan is based on section 9 of a longer internal planning document [E3], together with rule 3.6 of that document and the implementation settings below.

**Freeze:** this file must not be edited after it is frozen. Its SHA-256 hash is recorded in the internal project log (`PROGRESS.md`). Any specification added after seeing results must be labelled *exploratory*.

**Data:** the panel with the corrected exchange-rate series, read from `stage1/rerun/data/processed/asean_panel_regimes.parquet` [E4]; N = 946.

## P1. Main estimand

- β₃ in Eq. (3): log RV_{i,t+1} regressed on GPRC, Stress, GPRC × Stress, log RV_{i,t} (persistence), six controls (VIX, ΔlogFX, S&P 500, US10Y, WTI, domestic return) and country dummies.
- All variables are standardised within each country with full-sample moments.
- **Main specification:** time-aware DML (forward-blocked, K = 5, first block dropped, Lasso).
- **Companion specification:** OLS of Eq. (3).

## P2. Regime

- **Ex post (as in the paper):** two-state Markov-switching (MS) model with a switching mean and a constant variance, estimated on the cross-country average log RV; smoothed P > 0.5.
- **Real time:** the MS model is re-estimated every month τ = 2013:04 … 2026:04 on data up to τ; the filtered P(stress at τ | F_τ) > 0.5 is kept. The stress label goes to the state with the higher mean; 10 starting values. Sensitivity: thresholds 0.4 and 0.6.

## P3. Choice of the primary inference procedure (rule 3.6, applied before the empirical p-values are recomputed)

- **Candidates:**
  - standard Driscoll-Kraay (DK; Bartlett, L = 4);
  - DK fixed-b (L = 4, simulated critical values with T = 195);
  - wild-cluster bootstrap-t by country, Webb weights;
  - wild bootstrap by month;
  - moving-block bootstrap with 6-month blocks.
- **Criterion:** choose the procedure whose largest size over the H0 cells of variants (a)-(c) [T1] is closest to 5% and not above 7%. In a tie, prefer the simpler procedure, in the order above.
- **If no procedure has a size of at most 7%:** report the procedure with the smallest size, together with the simulation-based p-values (P5).

## P4. Monte Carlo

### Calibration

Computed once from the data and saved to `results/mc/calibration.json`:
- p₀₀ and p₁₁ of the ex-post regime;
- ρ calm and ρ stress (AR(1) of the standardised log RV within each regime);
- the difference in means between the regimes;
- the cross-country correlation of log RV (0.51) and of GPRC (0.17);
- φ of GPRC (0.27);
- the correlation of GPRC with the log-RV shock, in the full sample and in stress months;
- β₁ of the baseline OLS.

### Cells (R = 1,000 replications per cell)

| Cell | Variant | δ (reaction of GPRC to the volatility shock) | ρ calm / ρ stress | β₃ |
|---|---|---|---|---|
| 0 | a | 0 | ρ calm / ρ calm | 0 |
| 1 | b | δ̂ | ρ calm / ρ calm | 0 |
| 2 | b | 3δ̂ | ρ calm / ρ calm | 0 |
| 3 | c | δ̂ | ρ calm / ρ stress | 0 |
| 4 | c | 3δ̂ | ρ calm / ρ stress | 0 |
| 5 | c | δ_stress (stress months only) | ρ calm / ρ stress | 0 |
| 6 | a | 0 | ρ calm / ρ calm | −0.10 |
| 7 | a | 0 | ρ calm / ρ calm | −0.17 |
| 8 | c | δ̂ | ρ calm / ρ stress | −0.17 |

### Each replication

- **Sample:** unbalanced as in the data (Vietnam misses the first 29 months; N = 946).
- **MS model:** 3 starting values.
- **Stress flags:** oracle, smoothed > 0.5 (baseline), top share 19.4%, filtered > 0.5, leave-one-country-out top share.
- **OLS for the baseline and oracle flags:**
  - HC3, clustering by month, DK with L = 4, 6 and 12, DK fixed-b with L = 4;
  - wild-cluster bootstrap-t by country with the H0 restriction: Rademacher with all 32 combinations enumerated; Webb with B = 399;
  - wild bootstrap by month (Rademacher, B = 399);
  - moving-block bootstrap with 6-month blocks (B = 399, percentile p-value as in the earlier analysis).
- **Other flags:** β̂₃ and DK with L = 4.
- **DML:** shuffled (K = 5, 3 seeds, median) and forward-blocked (K = 5). Lasso with α fixed at the `LassoCV` value on the real data. iid and DK (L = 4) standard errors on the score summed by month.

### Generated-regressor bootstrap

- Cell 3, first 200 replications.
- Moving-block bootstrap by month (6 months, B = 199), with the MS model re-estimated in each bootstrap draw.

### Seeds

20261013 + 10,000 × cell + replication.

## P5. Mechanical bias

The bias is considered **material** when the mean of β̂₃ under H0 in cell 3 (variant c, δ̂, baseline flag) is at least 0.057 in absolute value (one third of 0.171).

**Simulation-based p-value:** the share of β̂₃ ≤ −0.171 in the H0 distribution of cells 3 and 5.

## P6. Learners and controls (A3)

- **Learners:**
  - Lasso (`LassoCV`, as in the paper);
  - ridge (`RidgeCV`);
  - random forest (200 trees, depth 5, minimum leaf size 5);
  - XGBoost (300 trees, depth 4, learning rate 0.05, subsample 0.8, colsample 0.8);
  - ensemble: the average of Lasso, random forest and XGBoost.
- **Control sets:**
  - C0 = as in the paper.
  - C1 = C0 + 3- and 6-month average log RV + GPRC at t−1 and t−2 + global GPR (z) + one-month lags of VIX, ΔlogFX, WTI and US10Y.
- **Cross-fitting:** shuffled and forward-blocked.
- **Hyperparameters:** fixed, not tuned. Lasso remains the main learner.
- **α path:** 12 values of α, equally spaced on a log scale, from 0.1 to 10 times the `LassoCV` α.

## P7. Cross-fitting schemes (A5)

- Shuffled.
- Forward-blocked.
- Forward-blocked with a gap of g = 1 and 3 months.
- Dropping time neighbours, l = 3 months on each side.
- Reverse cross-fitting following Ciganovic et al. (2026); labelled exploratory if the algorithm is unclear.
- K ∈ {5, 8, 10}.

## P8. Forecasts (A4)

- **Design:** expanding window, one-step-ahead forecasts, market by market. Out of sample from 2020:01, with at least 60 training months; sensitivity from 2016:01.
- **Models:**
  - M0: log RV_t.
  - M1: average log RV over 1, 3 and 6 months.
  - M2: M1 + VIX + ΔlogFX.
  - M3: M2 + GPRC.
  - M4: M2 + GPRC + P^RT + GPRC × P^RT, with P^RT from P2.
- **Loss functions:** MSE(log RV); QLIKE on RV² = exp(2 log RV); MSE(RV).
- **Tests:**
  - one-sided Clark-West test for the nested pairs with M2;
  - two-sided Diebold-Mariano test, HAC Newey-West with L = 4;
  - 90% model confidence set (Hansen et al. 2011): 1,000 bootstrap draws, block length 6.
- **Conditional evaluation:** calm and stress months according to P^RT.

## P9. Tier 2

- **B1 Local projections:** h = 0, …, 6; DK with L = max(4, h + 1); DML version (forward-blocked) for h = 1, 3 and 6.
- **B2 Other volatility measures:**
  - downside semivolatility;
  - mean absolute daily return;
  - GARCH(1,1);
  - Parkinson and Garman-Klass, if daily High/Low prices can be downloaded.
- **B3 Time-varying transition probabilities (TVTP):** ASEAN-average GPRC (z, lagged one month) in the transition probabilities; likelihood-ratio test; with and without VIX in the TVTP.
- **B4 Specification curve:** rerun `ms12_spec_curve.py` on the corrected panel.
- **B5 Economic value:**
  - volatility-managed portfolio with weights proportional to 1/σ̂²_{t+1|t} (M2 against M4), normalised by the full-sample variance; Sharpe ratio and certainty-equivalent gain (γ = 3);
  - 5% value-at-risk backtest (normal distribution; Kupiec and Christoffersen tests).

## P10. Reporting rules

- Report all results of P2-P9, including those that weaken the main result.
- The final conclusion follows the go/no-go gate in section 10 of the internal planning document [E3] and is decided by the authors.

---

## Edits and notes

- **[E1]** The original title also contains an internal phase label and the name of the publication outlet targeted at the time. Both are removed.
- **[E2]** The original quotes, in Vietnamese, the approval message that one of the authors wrote to the AI coding assistant used for the analysis code (see the statement on AI tools in the paper). The quote is omitted.
- **[E3]** The file name of the internal planning document is removed. That document is not published.
- **[E4]** The file path is changed to the folder names of this repository.
- **[T1]** Literally "the H0 cells with flags (a)-(c)". The letters refer to the variants in P4. The implementation (`stage2/analysis/a1_montecarlo.py`, `summarize`) uses the baseline flag in H0 cells 0-5.
- Numbers are written with decimal points; the original uses decimal commas.
