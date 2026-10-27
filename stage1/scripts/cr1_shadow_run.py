"""Step A3: rerun the ORIGINAL analysis scripts, unchanged, inside
an isolated copy of the project ("shadow root").

The original scripts resolve their root from ``src/utils/io.py`` and insert the
working directory into ``sys.path``; running them with the shadow root as the
working directory therefore makes every input and output live inside the
shadow, leaving the real ``data/`` and ``results/`` untouched.

variant raw : processed panels exactly as used for the submission
variant cr  : same panels with dlog_fx_m / fx_vol_m (and derived columns)
              rebuilt from the cleaned daily FX series of cr0

Resumable: each finished script leaves ``.done_<script>`` in the shadow root and
is skipped on the next run. Logs: results/logs/shadow_<variant>_<script>.log

Usage:  python scripts/cr1_shadow_run.py raw|cr [--rebuild]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crutil import CR, ROOT, RES, LOGS  # noqa: E402
from cr0_fx_panel import monthly_all, CC5  # noqa: E402

SCRIPTS = [
    "ms1_regime_identification",   # regimes (re-estimated; must equal the stored ones)
    "new_A_fe_channels",           # Table 3: FE betas, leave-one-channel-out R2
    "new_B_dml_channels",          # Table 3: pooled + country DML (lasso/ridge/rf)
    "ms2_regime_channels",         # split-sample calm/stress estimates
    "ms3_interaction_stress",      # interaction OLS/DML, by-country DML
    "ms_dml_timeaware",            # shuffled vs blocked DML (Table 3 text, Table 4)
    "ms5_regime_forecasting",      # OOS R2 and Clark-West by regime
    "ms7_inference_robust",        # HC3, clustering, Driscoll-Kraay, wild bootstraps
    "ms8_regime_robust",           # eight regime definitions (Fig. 2)
    "ms9_placebo",                 # permutation / return / phase-shift placebos
    "ms11_reconcile_magnitude",    # country interaction slopes
]


def replace_fx(panel: pd.DataFrame, fx_daily_clean: pd.DataFrame) -> pd.DataFrame:
    """Swap the monthly FX columns of the 5 markets for the cleaned versions
    and rebuild the columns derived from them (as in build_asean_panel)."""
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    m = monthly_all(fx_daily_clean)
    p = p.merge(m, on=["date", "country"], how="left", suffixes=("", "_cr"))
    for col in ["dlog_fx_m", "fx_vol_m"]:
        mask = p["country"].isin(CC5) & p[col].notna() & p[col + "_cr"].notna()
        p.loc[mask, col] = p.loc[mask, col + "_cr"]
        p = p.drop(columns=[col + "_cr"])
    p = p.sort_values(["country", "date"])
    if "gpr_x_dfx" in p:
        p["gpr_x_dfx"] = p["gpr_global"] * p["dlog_fx_m"]
    if "gprc_x_dfx" in p:
        p["gprc_x_dfx"] = p["gpr_country"] * p["dlog_fx_m"]
    for col in ["dlog_fx_m", "fx_vol_m"]:
        if f"{col}_lag1" in p:
            p[f"{col}_lag1"] = p.groupby("country")[col].shift(1)
    return p.sort_values(["date", "country"]).reset_index(drop=True)


def build_shadow(variant: str, rebuild: bool) -> Path:
    sh = CR / f"shadow_{variant}"
    if sh.exists() and not rebuild:
        return sh
    if sh.exists():
        shutil.rmtree(sh)
    (sh / "data" / "processed").mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", sh / "src", ignore=ignore)
    shutil.copytree(ROOT / "scripts", sh / "scripts", ignore=ignore)
    shutil.copytree(ROOT / "config", sh / "config", ignore=ignore)
    for name in ["asean_panel_monthly.parquet", "asean_panel_regimes.parquet"]:
        src = ROOT / "data" / "processed" / name
        dst = sh / "data" / "processed" / name
        if variant == "raw":
            shutil.copy2(src, dst)
        else:
            fx_clean = pd.read_parquet(RES / "fx_daily_clean.parquet")
            replace_fx(pd.read_parquet(src), fx_clean).to_parquet(dst, index=False)
    # keep a pristine copy of the input regimes panel for later comparison
    shutil.copy2(sh / "data" / "processed" / "asean_panel_regimes.parquet",
                 sh / "data" / "processed" / "asean_panel_regimes_INPUT.parquet")
    return sh


def run(variant: str, rebuild: bool) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    sh = build_shadow(variant, rebuild)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8",
               PYTHONHASHSEED="0")
    for name in SCRIPTS:
        marker = sh / f".done_{name}"
        if marker.exists():
            print(f"[{variant}] {name}: done earlier -> skip")
            continue
        log = LOGS / f"shadow_{variant}_{name}.log"
        t0 = time.time()
        print(f"[{variant}] {name}: running ...", flush=True)
        with open(log, "w", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, f"scripts/{name}.py"], cwd=sh,
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        dt = time.time() - t0
        if rc != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print(f"[{variant}] {name}: FAILED (rc={rc}, {dt:.0f}s). Log tail:")
            print("\n".join(tail))
            sys.exit(rc)
        marker.write_text(f"ok {time.strftime('%Y-%m-%d %H:%M:%S')} {dt:.0f}s\n")
        print(f"[{variant}] {name}: ok ({dt:.0f}s)", flush=True)
    print(f"[{variant}] all scripts finished")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["raw", "cr"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    run(a.variant, a.rebuild)
