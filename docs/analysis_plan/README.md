# Analysis plan and addendum

| File | Content |
|---|---|
| `analysis_plan_2026-10-13_translation.md` | English translation of the analysis plan frozen on 13 October 2026 |
| `addendum_2026-10-27.md` | Plan of the additional analyses in version 1.1 of the paper, fixed before they were run |
| `addendum_results.md` | Results of these analyses, read under the decision rules of the addendum |
| `addendum2_2026-10-27_influential_months.md` | Second addendum: influence of the five months with the most identifying variation (exploratory), fixed before its analyses were run |

## Status of the analysis plan

- The plan was written after the baseline estimates of an earlier version of the paper were known.
- It was frozen on 13 October 2026, before the Monte Carlo study and the new analyses were run.
- It is not a pre-registration. It was not deposited with a registry, and its date cannot be verified independently.
- It was drafted with the AI coding assistant named in the statement on AI tools in the paper, and approved by one of the authors.
- The paper calls this document the analysis plan. The document itself is titled "Pre-specification".

## Original and translation

- The original is a Markdown file in Vietnamese. SHA-256: `682dead90c0e4732e233de4fc56d659bb3c615b2242348c478c0bfe232578714`.
- The hash was recorded in the internal project log when the file was frozen. The file has not changed since.
- The original is available from the corresponding author on request. The hash allows a check that the file sent is the file described here.
- The translation follows the original section by section. Its edits are marked [E1]-[E4] and listed at its end.

## Chronology

From the internal project log and the times of the output files (local time, UTC+7). It cannot be verified independently, except for the dates of the commits in this repository.

| Date | Step |
|---|---|
| Before 13 Oct 2026 | Baseline estimates known from an earlier version of the paper |
| 13 Oct 2026, 14:58 | Analysis plan frozen; SHA-256 recorded |
| 13 Oct 2026, afternoon | New analyses and Monte Carlo study run (output files dated 15:20 to 18:55) |
| 27 Oct 2026 | Diagnostics on tails and serial dependence (`extra/diag_round3_premise.py`, exploratory) |
| 27 Oct 2026 | Analysis plan published in this repository |
| 27 Oct 2026, 12:15 | Addendum approved by all authors and published, before its analyses were run (see the commit history) |
| 27 Oct 2026, 12:27 to 13:48 | Analyses of the addendum run (output files in `stage2/results/mc/`); results in `addendum_results.md` |
| 27 Oct 2026, 15:12 | Second addendum approved by all authors and published, before its analyses were run (see the commit history) |

## Changes after the plan

All are labelled exploratory in the code and in the paper, or stated in the paper.

- **DML design.** The DML estimates in the main tables use the regime-specific design (δ = θ_stress − θ_calm). It was added after the flexible learners were examined, because the planned interaction design (P1, P6) is not identified with flexible learners. The planned design is still reported in `stage2/results/qf_a3.json` (key `interaction`).
- **Skipped bootstrap draws.** Draws with fewer than 5 or more than T/2 stress months are skipped in the regime-re-estimating bootstrap. This rule was added after a first run.
- **Real-time standardisation.** Real-time standardisation of all variables was added on 27 October 2026 (`stage2/analysis/a7_expanding_z.py`).
- **Further checks.** Leave-one-stress-episode-out estimates and size-adjusted DK p-values (`stage2/results/qf_diag.json`).
- **Range-based measures (P9, B2) not computed.** They were not computed because daily High/Low prices were not available.

## Where the planned results are reported

| Plan item | Paper | Result file in `stage2/results/` |
|---|---|---|
| P1 Main estimand, OLS and DML | Sect. 4.1; Tables 3 and 5 | `qf_a6.json`, `qf_a3.json` |
| P2 Ex-post and real-time regimes, thresholds 0.4 and 0.6 | Sects. 4.2 and 6.2; Table 5 | `qf_a2.json`, `qf_a6.json` |
| P3 Choice of the inference procedure | Sects. 4.3 and 5; Tables 2 and 4 | `qf_a1.json` (key `P3`) |
| P4 Monte Carlo cells and procedures | Sect. 5; Tables 2 and 4 | `qf_a1.json`, `mc/` |
| P4 Generated-regressor bootstrap | Sect. 5; Table 4 | `qf_a1.json`, `mc/gr/` |
| P5 Mechanical bias and simulation-based p-value | Sect. 5 | `qf_a1.json` (key `P5`) |
| P6 Learners and control sets | Sect. 6.3; Table 6 | `qf_a3.json` |
| P6 α path | Not in the paper | `qf_a3.json` (key `alpha_path`) |
| P7 Cross-fitting schemes | Shuffled and time-aware schemes in Sects. 4.2 and 6 | `qf_a5.json` (all schemes and K) |
| P8 Forecasts | Sect. 7; Table 8 | `qf_a4.json` (all tests) |
| P9 B1 Local projections | Sect. 8 | `qf_b.json` (key `lp`) |
| P9 B2 Other volatility measures | Sect. 6.3 | `qf_b.json` (key `altvol`) |
| P9 B3 Time-varying transition probabilities | Sect. 7 | `qf_b.json` (key `tvtp`) |
| P9 B4 Specification curve | Sect. 6.3 | `qf_b.json` (key `speccurve`) |
| P9 B5 Economic value | Sect. 7 | `qf_b.json` (key `econ`) |
