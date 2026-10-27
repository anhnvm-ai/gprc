"""Build monthly ASEAN panel: 6 countries × time (2000–2026).

Schema per row:
    date (month-start) | country | ret_m | rv_m | dlog_fx | fx_vol21
    gpr_global | gpr_act | gpr_thr | gpr_country | aigpr_country
    vix | wti | us10y | sp500_ret | dxy
    gpr_x_dfx | ret_sp500_x_gpr
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.io import load_config, ensure_dir, ROOT, save_parquet
from src.data.download_asean_markets import (
    download_asean_indices, download_asean_fx, download_global_macro,
)
from src.data.download_gpr import download_gpr_monthly
from src.data.download_vn_market import download_vn_market

warnings.filterwarnings("ignore")

# Country codes for the 6 ASEAN nations
ASEAN_CODES = ["VNM", "THA", "IDN", "MYS", "PHL", "SGP"]

# Caldara-Iacoviello GPR column names (monthly file)
# Global: GPR, GPRH (threats), GPRA (acts)
# Country: GPRC_VNM, GPRC_THA, GPRC_IDN, GPRC_MYS, GPRC_PHL, GPRC_SGP
GPR_COUNTRY_COLS = {
    "VNM": "GPRC_VNM",
    "THA": "GPRC_THA",
    "IDN": "GPRC_IDN",
    "MYS": "GPRC_MYS",
    "PHL": "GPRC_PHL",
    "SGP": "GPRC_SGP",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_monthly_index(daily: pd.DataFrame, country: str) -> pd.DataFrame:
    """Aggregate daily stock data to monthly frequency.

    Returns DataFrame with columns: ret_m, rv_m indexed by month-start.
    """
    if daily.empty:
        return pd.DataFrame(columns=["ret_m", "rv_m"])

    daily = daily.copy()
    # Monthly return = sum of daily log returns
    ret_m = daily["ret"].resample("MS").sum()
    # Monthly RV = annualised std of daily returns within month
    rv_m = daily["ret"].resample("MS").std() * np.sqrt(252)

    return pd.DataFrame({"ret_m": ret_m, "rv_m": rv_m})


def _fx_monthly(fx_daily: pd.DataFrame, cc: str) -> pd.DataFrame:
    """Monthly FX stats for one country."""
    col = f"{cc}_fx"
    if col not in fx_daily.columns:
        return pd.DataFrame(columns=["dlog_fx_m", "fx_vol_m"])

    s = fx_daily[col].dropna()
    dlog = np.log(s / s.shift(1))
    dlog_m  = dlog.resample("MS").sum()     # cumulative change in month
    fx_vol  = dlog.resample("MS").std() * np.sqrt(252)  # annualised FX vol

    return pd.DataFrame({"dlog_fx_m": dlog_m, "fx_vol_m": fx_vol})


def _global_macro_monthly(macro: pd.DataFrame) -> pd.DataFrame:
    """Aggregate global macro to monthly frequency."""
    out = pd.DataFrame()
    if "VIX" in macro.columns:
        out["vix_m"]     = macro["VIX"].resample("MS").mean()
    if "WTI" in macro.columns:
        out["wti_ret_m"] = np.log(macro["WTI"] / macro["WTI"].shift(1)).resample("MS").sum()
    if "US10Y" in macro.columns:
        out["us10y_m"]   = macro["US10Y"].resample("MS").last()
    if "SP500" in macro.columns:
        out["sp500_ret_m"] = np.log(macro["SP500"] / macro["SP500"].shift(1)).resample("MS").sum()
    if "DXY" in macro.columns:
        out["dxy_ret_m"]   = np.log(macro["DXY"] / macro["DXY"].shift(1)).resample("MS").sum()
    return out


def _gpr_monthly_panel(gpr_m: pd.DataFrame) -> pd.DataFrame:
    """Reindex GPR monthly to month-start (MS) frequency."""
    gpr_m = gpr_m.copy()
    # to_period("M").to_timestamp() gives the FIRST day of each month (month-start)
    gpr_m.index = gpr_m.index.to_period("M").to_timestamp()
    return gpr_m


# ── main build ───────────────────────────────────────────────────────────────

def build_asean_panel(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    start = cfg["data"].get("start_date_asean", "2000-01-01")
    end   = cfg["data"]["end_date"]

    # Override start for the wider ASEAN window
    cfg_asean = dict(cfg)
    cfg_asean["data"] = dict(cfg["data"])
    cfg_asean["data"]["start_date"] = start

    print("\n=== ASEAN Panel: Step 1/5 — Download data ===")
    asean_daily = download_asean_indices(cfg_asean)   # dict: cc -> df
    fx_daily    = download_asean_fx(cfg_asean)
    macro_daily = download_global_macro(cfg_asean)
    gpr_m_raw   = download_gpr_monthly(cfg_asean)

    # Vietnam from vnstock
    print("[ASEAN] Fetching Vietnam via vnstock ...")
    try:
        vn_data = download_vn_market(cfg_asean)
        # Prefer VNINDEX; fallback to first available
        vn_key  = next((k for k in ("VNINDEX", "VN30") if k in vn_data), None)
        if vn_key:
            vn_df = vn_data[vn_key].copy()
            vn_df.columns = [c.replace(f"{vn_key}_", "") for c in vn_df.columns]
            # Ensure 'close' col
            if "close" not in vn_df.columns:
                close_c = [c for c in vn_df.columns if "close" in c.lower()]
                if close_c:
                    vn_df = vn_df.rename(columns={close_c[0]: "close"})
            vn_df["ret"]  = np.log(vn_df["close"] / vn_df["close"].shift(1))
            vn_df["rv21"] = vn_df["ret"].rolling(21).std() * np.sqrt(252)
            asean_daily["VNM"] = vn_df[["close", "ret", "rv21"]]
    except Exception as e:
        print(f"[ASEAN] Vietnam vnstock error: {e}")

    print("\n=== ASEAN Panel: Step 2/5 — Monthly aggregation ===")
    macro_m = _global_macro_monthly(macro_daily)
    gpr_m   = _gpr_monthly_panel(gpr_m_raw)

    print("\n=== ASEAN Panel: Step 3/5 — Build per-country slices ===")
    slices: list[pd.DataFrame] = []
    coverage: dict[str, dict] = {}

    # Global GPR columns (same for all countries)
    gpr_global_cols = {}
    for gcol, alias in [("GPR","gpr_global"), ("GPRH","gpr_thr"), ("GPRA","gpr_act")]:
        if gcol in gpr_m.columns:
            gpr_global_cols[gcol] = alias
    gpr_global = gpr_m[list(gpr_global_cols.keys())].rename(columns=gpr_global_cols) \
                 if gpr_global_cols else pd.DataFrame()

    for cc in ASEAN_CODES:
        # --- equity ---
        if cc in asean_daily and not asean_daily[cc].empty:
            eq_m = _to_monthly_index(asean_daily[cc], cc)
        else:
            print(f"[ASEAN] {cc}: no equity data, filling NaN")
            eq_m = pd.DataFrame(columns=["ret_m", "rv_m"])

        # --- FX ---
        fx_m = _fx_monthly(fx_daily, cc)

        # --- GPR country (GPRC_VNM / GPRC_THA etc.) ---
        gpr_col = GPR_COUNTRY_COLS.get(cc)
        gpr_cc  = pd.DataFrame()
        if gpr_col and gpr_col in gpr_m.columns:
            gpr_cc = gpr_m[[gpr_col]].rename(columns={gpr_col: "gpr_country"})
        # Also try headline country GPR (GPRHC_xxx) as "gpr_country_h"
        gpr_hcol = f"GPRHC_{cc}"
        gpr_hcc  = pd.DataFrame()
        if gpr_hcol in gpr_m.columns:
            gpr_hcc = gpr_m[[gpr_hcol]].rename(columns={gpr_hcol: "gpr_country_h"})

        # --- Combine for this country ---
        frames = [f for f in [eq_m, fx_m, gpr_global, gpr_cc, gpr_hcc, macro_m]
                  if not f.empty]
        df_cc = pd.concat(frames, axis=1).sort_index()

        # Restrict to ASEAN start date to end date
        df_cc = df_cc.loc[start:end]

        # Report earliest actual equity data (may be later than 'start')
        eq_start = eq_m.dropna().index.min() if not eq_m.empty else None
        eq_obs   = int(eq_m["ret_m"].notna().sum()) if "ret_m" in eq_m.columns else 0
        coverage[cc] = {
            "equity_start": eq_start.date() if eq_start else "N/A",
            "n_months_equity": eq_obs,
            "n_months_panel": len(df_cc),
        }
        print(f"[ASEAN] {cc}: equity from {coverage[cc]['equity_start']}, "
              f"{eq_obs} months of returns, panel rows={len(df_cc)}")

        df_cc["country"] = cc
        slices.append(df_cc)

    print("\n[ASEAN] Coverage summary:")
    for cc, cov in coverage.items():
        print(f"  {cc}: equity_start={cov['equity_start']}, "
              f"n_equity={cov['n_months_equity']}, panel_rows={cov['n_months_panel']}")

    print("\n=== ASEAN Panel: Step 4/5 — Concat + interaction terms ===")
    panel = pd.concat(slices).sort_index()
    panel = panel.reset_index().rename(columns={"index": "date"})
    panel["date"] = pd.to_datetime(panel["date"])

    # Interaction terms
    if "gpr_global" in panel.columns and "dlog_fx_m" in panel.columns:
        panel["gpr_x_dfx"]  = panel["gpr_global"] * panel["dlog_fx_m"]
    if "gpr_country" in panel.columns and "dlog_fx_m" in panel.columns:
        panel["gprc_x_dfx"] = panel["gpr_country"] * panel["dlog_fx_m"]
    if "gpr_global" in panel.columns and "sp500_ret_m" in panel.columns:
        panel["gpr_x_sp500"] = panel["gpr_global"] * panel["sp500_ret_m"]

    # Lag controls (t-1) to avoid look-ahead — shift within each country group
    lag_cols = ["gpr_global", "gpr_country", "gpr_thr", "gpr_act",
                "dlog_fx_m", "fx_vol_m", "vix_m", "wti_ret_m", "us10y_m"]
    for col in lag_cols:
        if col in panel.columns:
            panel[f"{col}_lag1"] = panel.groupby("country")[col].shift(1)

    # Drop rows where all numeric cols are NaN
    num_cols = panel.select_dtypes(include=[np.number]).columns
    panel = panel.dropna(subset=num_cols, how="all")
    panel = panel.replace([np.inf, -np.inf], np.nan)

    print("\n=== ASEAN Panel: Step 5/5 — Save ===")
    out = ROOT / cfg["paths"]["processed"] / "asean_panel_monthly.parquet"
    ensure_dir(out.parent)
    panel.to_parquet(out, index=False)
    print(f"Saved -> {out}  shape={panel.shape}")
    print(f"Countries: {panel['country'].value_counts().to_dict()}")
    print(f"Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")
    print(f"Columns: {list(panel.columns)}")
    return panel


if __name__ == "__main__":
    build_asean_panel()
