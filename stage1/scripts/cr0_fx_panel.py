"""Step A1-A2: deterministic FX cleaning and corrected panel.

A1  Flag a daily FX print as anomalous when
        |log(P_t / median(P_{t-5..t+5}))| > log(2),
    i.e. when it is more than a factor of two away from its centred 11-day
    rolling median. Flagged prints are set to missing and filled by time
    interpolation. The rule must flag exactly the nine prints documented in
    the submission (seven VND, two IDR); otherwise the script stops.

A2  Recompute the monthly FX variables exactly as in
    ``src/data/build_asean_panel._fx_monthly`` (sum of daily log changes;
    annualised std) from the cleaned series and replace them in a copy of
    ``data/processed/asean_panel_regimes.parquet``. Before replacing, the raw
    recomputation must reproduce the stored panel values.

Outputs (skipped if present unless --force):
    results/fx_daily_clean.parquet
    results/fx_corrections.csv
    results/panel_cr.parquet
    results/logs/cr0_fx_panel.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crutil import md_table  # noqa: E402

CR = Path(__file__).resolve().parents[1]          # stage1/
ROOT = CR.parent                                  # project root
RES = CR / "results"
LOGS = RES / "logs"

FX_DAILY = ROOT / "data" / "interim" / "asean_fx_daily.parquet"
PANEL_RAW = ROOT / "data" / "processed" / "asean_panel_regimes.parquet"

OUT_FX = RES / "fx_daily_clean.parquet"
OUT_CORR = RES / "fx_corrections.csv"
OUT_PANEL = RES / "panel_cr.parquet"
OUT_LOG = LOGS / "cr0_fx_panel.md"

CC5 = ["VNM", "THA", "IDN", "MYS", "PHL"]
EXPECTED_FLAGS = {
    ("VNM", "2010-04-07"), ("VNM", "2011-04-21"), ("VNM", "2013-07-08"),
    ("VNM", "2014-03-10"), ("VNM", "2014-04-30"), ("VNM", "2014-05-02"),
    ("VNM", "2014-07-31"), ("IDN", "2010-11-01"), ("IDN", "2012-02-07"),
}
WINDOW = 11
THRESHOLD = np.log(2.0)


def load_fx_daily() -> pd.DataFrame:
    fx = pd.read_parquet(FX_DAILY)
    if not isinstance(fx.index, pd.DatetimeIndex):
        fx.index = pd.to_datetime(fx.index)
    return fx.sort_index()


def flag_and_clean(fx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = fx.copy()
    rows = []
    for cc in CC5:
        col = f"{cc}_fx"
        s = fx[col].dropna()
        med = s.rolling(WINDOW, center=True, min_periods=5).median()
        dev = np.abs(np.log(s / med))
        bad = dev[dev > THRESHOLD].index
        if len(bad) == 0:
            continue
        s_clean = s.copy()
        s_clean.loc[bad] = np.nan
        s_clean = s_clean.interpolate(method="time")
        clean.loc[s_clean.index, col] = s_clean
        for d in bad:
            rows.append({
                "country": cc,
                "date": d.date().isoformat(),
                "raw_value": float(s.loc[d]),
                "rolling_median_11d": float(med.loc[d]),
                "abs_log_deviation": float(dev.loc[d]),
                "corrected_value": float(s_clean.loc[d]),
            })
    corr = pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)
    return clean, corr


def fx_monthly(fx_daily: pd.DataFrame, cc: str) -> pd.DataFrame:
    """Same construction as src/data/build_asean_panel._fx_monthly."""
    s = fx_daily[f"{cc}_fx"].dropna()
    dlog = np.log(s / s.shift(1))
    return pd.DataFrame({
        "dlog_fx_m": dlog.resample("MS").sum(),
        "fx_vol_m": dlog.resample("MS").std() * np.sqrt(252),
    })


def monthly_all(fx_daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for cc in CC5:
        m = fx_monthly(fx_daily, cc)
        m["country"] = cc
        parts.append(m)
    out = pd.concat(parts)
    out.index.name = "date"
    return out.reset_index()


def main(force: bool = False) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    if OUT_PANEL.exists() and OUT_FX.exists() and OUT_CORR.exists() and not force:
        print("[cr0] outputs already exist -> skip (use --force to rebuild)")
        return

    log = ["# cr0 — FX cleaning and corrected panel", ""]

    # ---------------- A1 ----------------
    fx_raw = load_fx_daily()
    fx_clean, corr = flag_and_clean(fx_raw)
    found = {(r.country, r.date) for r in corr.itertuples()}
    if found != EXPECTED_FLAGS:
        print("[cr0] FLAG MISMATCH")
        print("  unexpected:", sorted(found - EXPECTED_FLAGS))
        print("  missing   :", sorted(EXPECTED_FLAGS - found))
        sys.exit(1)
    log += [f"A1: rule |log(P/median_{WINDOW}d)| > log 2 flags exactly "
            f"{len(corr)} prints (matches the documented nine).", "",
            md_table(corr), ""]

    # ---------------- A2 ----------------
    panel = pd.read_parquet(PANEL_RAW)
    panel["date"] = pd.to_datetime(panel["date"])
    m_raw = monthly_all(fx_raw)
    m_cr = monthly_all(fx_clean)

    # (i) raw recomputation must reproduce the stored panel
    chk = panel.merge(m_raw, on=["date", "country"], how="left", suffixes=("", "_re"))
    chk = chk[chk["country"].isin(CC5)]
    for col in ["dlog_fx_m", "fx_vol_m"]:
        both = chk[[col, col + "_re"]].dropna()
        diff = float(np.max(np.abs(both[col] - both[col + "_re"]))) if len(both) else np.nan
        log.append(f"A2 check: raw recomputation of {col} vs stored panel: "
                   f"max |diff| = {diff:.2e} over {len(both)} rows")
        if not (diff < 1e-10):
            print(f"[cr0] raw recomputation of {col} does not reproduce the panel (max diff {diff})")
            sys.exit(1)

    # (ii) replace with corrected values
    pcr = panel.merge(m_cr, on=["date", "country"], how="left", suffixes=("", "_cr"))
    changed = []
    for col in ["dlog_fx_m", "fx_vol_m"]:
        mask = pcr[col + "_cr"].notna() & pcr[col].notna()
        delta = (pcr.loc[mask, col] - pcr.loc[mask, col + "_cr"]).abs() > 1e-12
        for idx in pcr.loc[mask].index[delta.values]:
            changed.append({"variable": col, "country": pcr.at[idx, "country"],
                            "month": pcr.at[idx, "date"].strftime("%Y-%m"),
                            "raw": pcr.at[idx, col], "corrected": pcr.at[idx, col + "_cr"]})
        pcr.loc[mask, col] = pcr.loc[mask, col + "_cr"]
        pcr = pcr.drop(columns=[col + "_cr"])
    # dependent columns
    pcr = pcr.sort_values(["country", "date"])
    pcr["gpr_x_dfx"] = pcr["gpr_global"] * pcr["dlog_fx_m"]
    pcr["gprc_x_dfx"] = pcr["gpr_country"] * pcr["dlog_fx_m"]
    for col in ["dlog_fx_m", "fx_vol_m"]:
        pcr[f"{col}_lag1"] = pcr.groupby("country")[col].shift(1)
    pcr = pcr.sort_values(["date", "country"]).reset_index(drop=True)

    ch = pd.DataFrame(changed)
    log += ["", f"A2: monthly cells changed by the correction: {len(ch)}", "",
            md_table(ch) if len(ch) else "(none)", ""]

    # (iii) Table 2 style statistics for dlog_fx_m (panel rows, 5 countries)
    def stats(df, label):
        s = df.loc[df["country"].isin(CC5), "dlog_fx_m"].dropna()
        return (f"| {label} | {len(s)} | {s.mean():.4f} | {s.std():.4f} | "
                f"{s.min():.3f} | {s.median():.4f} | {s.max():.3f} |")
    log += ["A2: dlog_fx_m over all panel rows of the 5 markets (Table 2 reports "
            "mean 0.001, std 0.022, min -0.148, median 0.001, max 0.240 after correction)", "",
            "| sample | n | mean | std | min | median | max |",
            "|---|---:|---:|---:|---:|---:|---:|",
            stats(panel, "raw"), stats(pcr, "corrected"), ""]

    # ---------------- save ----------------
    fx_clean.to_parquet(OUT_FX)
    corr.to_csv(OUT_CORR, index=False)
    pcr.to_parquet(OUT_PANEL, index=False)
    OUT_LOG.write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log))
    print(f"\n[cr0] saved {OUT_FX.name}, {OUT_CORR.name}, {OUT_PANEL.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
