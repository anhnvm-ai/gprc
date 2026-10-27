# Replication files: geopolitical-news salience and equity volatility under estimated stress regimes

Code, derived data and result files for the paper *Geopolitical-News Salience and Equity Volatility under
Estimated Stress Regimes: Double Machine Learning Evidence from ASEAN Markets* by V. M. A. Nguyen, C.-P. Ha
and P.-N. Le (under review). Contact: Phu-Nguyen Le, corresponding author, nguyenlp@vaa.edu.vn.

## 1. Layout

All relative paths in the code assume that commands are run from the root of the repository.

| Folder or file | Content |
|---|---|
| `stage2/analysis/` | Analyses reported in the paper: Monte Carlo study (`a1`), real-time regimes (`a2`), learners (`a3`), forecasts (`a4`), cross-fitting schemes (`a5`), main tables (`a6`), expanding-window standardisation (`a7`, exploratory), further checks (`b_extras`, `diag_extra`), shared library (`qflib.py`) and a reproduction test |
| `stage2/results/` | Result files (`*.json`), Monte Carlo replication shards (`mc/`), real-time regime probabilities and CSV tables |
| `stage1/rerun/data/processed/asean_panel_regimes.parquet` | **Derived monthly panel**: the input of all analyses |
| `stage1/scripts/` | Correction of the exchange-rate series (`cr0_fx_panel.py`) and the rerun of the earlier analysis scripts on the corrected panel (`cr1`-`cr5`) |
| `stage1/rerun/` | Code, configuration and outputs of that rerun (figures, onset logit, permutation test, specification curve) |
| `stage1/results/` | Reference values used by the reproduction test |
| `config/`, `src/`, `scripts/` | Data download, panel construction and the earlier analysis scripts |
| `extra/` | Stress months by cross-fitting block (Sect. 6.2 of the paper) |

## 2. Data

| Series | Source | Included? |
|---|---|---|
| Country-specific geopolitical risk (GPRC) | Caldara and Iacoviello, <https://www.matteoiacoviello.com/gpr.htm> | Monthly values are in the derived panel; `scripts/step1_download_gpr.py` downloads them again |
| Equity indices, exchange rates, VIX, S&P 500, US 10-year yield, WTI | Yahoo Finance (`yfinance`) | **Raw daily prices are not included** (provider terms of use); `scripts/step2_download_asean.py` downloads them again |
| VN-Index | `vnstock` library | As above |
| Monthly panel built from these series | Constructed | **Included** |

Data downloaded at a later date may differ slightly from the vintage used in the paper, because providers
revise their series. The derived panel is the exact input of the reported results.

## 3. Software

The results were produced on Windows 11 with Python 3.12.4 and the packages in `requirements.txt`
(`pip install -r requirements.txt`). All random seeds are fixed in the code.

## 4. Reproducing the results

**Step A, about a minute.** `python stage2/analysis/test_reproduce.py` reproduces the baseline estimates
from the derived panel and compares them with the stored reference values.

**Step B.** All result files are included. Each script recomputes its own result file and skips the parts
already present, so delete a result file to force a full rerun.

| Paper | Command | Result file in `stage2/results/` |
|---|---|---|
| Sect. 5 and Table 2; sizes in Table 4 | `python stage2/analysis/a1_montecarlo.py calibrate`, then `run`, `gr`, `supp`, `supp_summary`, `summarize` (several hours; `--workers 8`) | `qf_a1.json`, `qf_a1_supp.json`, `mc/` |
| Real-time regime (Sects. 4.2 and 6.2) | `python stage2/analysis/a2_realtime.py` | `qf_a2.json`, `qf_a2_rt_probs.parquet` |
| Table 6; time-aware channel estimates in Sect. 6.1 | `python stage2/analysis/a3_learners.py` | `qf_a3.json` |
| Table 8 and Sect. 7 (forecasts) | `python stage2/analysis/a4_forecast.py` | `qf_a4.json` |
| Cross-fitting schemes | `python stage2/analysis/a5_crossfit.py` | `qf_a5.json` |
| Tables 3, 5 and 7; p-values in Table 4 | `python stage2/analysis/a6_main_tables.py` | `qf_a6.json` |
| Expanding-window standardisation (Sect. 6.2, exploratory) | `python stage2/analysis/a7_expanding_z.py` | `qf_a7.json` |
| Time-varying transition probabilities, economic value, specification curve, local projections | `python stage2/analysis/b_extras.py` | `qf_b.json` |
| Further diagnostics | `python stage2/analysis/diag_extra.py` | `qf_diag.json` |
| Stress months by cross-fitting block (Sect. 6.2) | `python extra/diag_balanced_blocks.py stage2` | printed; stored in `extra/diag_balanced_blocks.txt` |

**Step C, optional: from raw downloads.**

```
python scripts/step1_download_gpr.py
python scripts/step2_download_asean.py
python scripts/step3_build_panel.py
python scripts/ms1_regime_identification.py
python stage1/scripts/cr0_fx_panel.py
python stage1/scripts/cr1_shadow_run.py cr
python stage1/scripts/cr2_collect.py cr
python stage1/scripts/cr3_extra.py cr
python stage1/scripts/cr4_experiments.py cr
python stage1/scripts/cr5_figures.py
```

These commands rebuild the derived panel, Figures 1 and 2 (`stage1/rerun/results/figures/`), the channel
ranking of Sect. 6.1, the onset logit of Sect. 7 and the permutation test of Sect. 6.3.

## 5. Analysis plan

The analysis plan of the new analyses was written after the baseline estimates were known and before the
simulations and the new analyses were run. It is available from the corresponding author on request.
Analyses added later are marked as exploratory in the code and in the paper.

## 6. Licence

Code: MIT License (`LICENSE`). Derived data and result files: Creative Commons Attribution 4.0
International (CC BY 4.0). Please also cite Caldara and Iacoviello (2022), *Measuring geopolitical risk*,
American Economic Review 112(4), 1194-1225, for the GPRC series.

## 7. Code assistance

The code was written with the assistance of Claude (Anthropic) through Claude Code, and the authors
reviewed it. This is also stated in the paper.
