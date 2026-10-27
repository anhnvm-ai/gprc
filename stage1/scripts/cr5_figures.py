"""Step A4: redraw Fig. 2 and Fig. 3 from the corrected-data
results, keeping the layout of the submitted figures.

Fig. 2  eight regime definitions, beta_3 with Driscoll-Kraay 95% CI
        (filled marker = p_DK < 0.05)          <- results/phaseB_cr.json
Fig. 3  left : 5,000 stress-label permutations  <- results/perm_draws_cr.npy
        right: return-outcome placebo            <- rerun tableMS9

Writes paper/figures/figMS8_b3_across_regimes.pdf and
paper/figures/figMS9_placebo_distribution.pdf (submitted originals stay in
source_submitted/figures/), plus PNG previews in results/figures_preview/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crutil import CR, RES, load_json  # noqa: E402

FIG = CR / "paper" / "figures"
PREV = RES / "figures_preview"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
                     "pdf.fonttype": 42})

ORDER = ["MS mean, thr=0.4", "MS mean, thr=0.5", "MS mean, thr=0.6",
         "MarkovAR(1) switching-AR", "3-regime MS (crisis)", "Quantile top 20%",
         "Quantile top 25%", "Country-specific MS"]
LABEL = {"MS mean, thr=0.4": "MS mean, $P>0.4$", "MS mean, thr=0.5": "MS mean, $P>0.5$ (baseline)",
         "MS mean, thr=0.6": "MS mean, $P>0.6$", "MarkovAR(1) switching-AR": "MS-AR(1), switching AR",
         "3-regime MS (crisis)": "Three-state MS (crisis)", "Quantile top 20%": "Quantile, top 20%",
         "Quantile top 25%": "Quantile, top 25%", "Country-specific MS": "Country-specific MS"}


def fig2():
    reg8 = load_json(RES / "phaseB_cr.json")["reg8"]
    fig, ax = plt.subplots(figsize=(7.47, 3.39))
    ys = np.arange(len(ORDER))[::-1]
    for y, key in zip(ys, ORDER):
        r = reg8[key]
        lo, hi = r["dk4"]["ci"]
        sig = r["dk4"]["p"] < 0.05
        ax.plot([lo, hi], [y, y], color="black", lw=1.4)
        ax.plot(r["b3"], y, "o", ms=6, mec="black", mfc="black" if sig else "white", zorder=3)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[k] for k in ORDER])
    ax.set_xlabel(r"$\beta_3$ (GPRC $\times$ Stress), 95% CI (Driscoll–Kraay)")
    ax.set_title("Sign robustness across eight alternative regime definitions")
    ax.plot([], [], "o", mec="black", mfc="black", label="significant (5%)")
    ax.plot([], [], "o", mec="black", mfc="white", label="not significant")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "figMS8_b3_across_regimes.pdf")
    fig.savefig(PREV / "fig2.png", dpi=110)
    plt.close(fig)


def fig3():
    ex = load_json(RES / "extra_cr.json")["perm"]
    draws = np.load(RES / "perm_draws_cr.npy")
    ms9 = pd.read_csv(CR / "rerun" / "results" / "tables" / "tableMS9_placebo.csv").set_index("placebo")
    b_ret = ms9.at["Return placebo", "b3_true"]
    se_ret = ms9.at["Return placebo", "placebo_sd"]
    p_ret = ms9.at["Return placebo", "empirical_p"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.54, 3.09))
    a1.hist(draws, bins=30, color="#c8c8c8", edgecolor="black", lw=0.5)
    a1.axvline(ex["b3_true"], color="black", lw=1.4)
    a1.annotate("Observed estimate", xy=(ex["b3_true"], a1.get_ylim()[1] * 0.92),
                xytext=(ex["b3_true"] + 0.04, a1.get_ylim()[1] * 0.88),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    a1.set_title(f"Randomisation placebo (one-sided $p$ = {ex['p_one_sided']:.3f})", fontsize=9)
    a1.set_xlabel(r"$\beta_3$ under permuted stress labels")
    a1.set_ylabel("Frequency")
    x = np.linspace(-3 * se_ret, 3 * se_ret, 400)
    a2.fill_between(x, norm.pdf(x, 0, se_ret), color="#c8c8c8", edgecolor="black", lw=0.6)
    a2.axvline(b_ret, color="black", lw=1.4)
    a2.annotate("Observed estimate\n(returns, not significant)",
                xy=(b_ret, norm.pdf(b_ret, 0, se_ret)), xytext=(-0.13, norm.pdf(0, 0, se_ret) * 0.62),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8, ha="center")
    a2.set_yticks([])
    a2.set_title(f"Return-outcome placebo ($p$ = {p_ret:.2f})", fontsize=9)
    a2.set_xlabel(r"$\beta_3$ with returns as outcome")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "figMS9_placebo_distribution.pdf")
    fig.savefig(PREV / "fig3.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    PREV.mkdir(parents=True, exist_ok=True)
    fig2()
    fig3()
    print("figures written")
