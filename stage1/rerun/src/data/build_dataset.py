"""Build the merged daily panel = final_dataset.parquet (no look-ahead)."""
from __future__ import annotations
import numpy as np, pandas as pd
from src.utils.io import load_config, save_parquet, ROOT
from src.data.download_gpr import download_gpr_daily, download_gpr_monthly
from src.data.download_ai_gpr import download_ai_gpr_daily, download_ai_gpr_country
from src.data.download_vn_market import download_vn_market
from src.data.download_fx_macro import download_fx_macro

def _bday_index(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)

def build(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]

    print("\n=== 1/4 Download ===")
    gprd = download_gpr_daily(cfg)
    gprm = download_gpr_monthly(cfg)
    aig  = download_ai_gpr_daily(cfg)
    try:
        aigvn = download_ai_gpr_country(cfg)
    except Exception:
        aigvn = pd.DataFrame()
    vn = download_vn_market(cfg)
    macro = download_fx_macro(cfg)

    print("\n=== 2/4 Align to VN business days ===")
    bday = _bday_index(start, end)
    panel = pd.DataFrame(index=bday); panel.index.name = "date"

    # VN indices: returns + realized vol
    for sym, df in vn.items():
        close_col = f"{sym}_close" if f"{sym}_close" in df.columns else df.columns[df.columns.str.endswith("close")][0]
        s = df[close_col].reindex(bday).ffill()
        panel[f"{sym}_close"] = s
        panel[f"{sym}_ret"]   = np.log(s / s.shift(1))
        panel[f"{sym}_rv21"]  = panel[f"{sym}_ret"].rolling(21).std() * np.sqrt(252)

    # Macro
    macro = macro.reindex(bday).ffill()
    panel["dlog_FX"]  = np.log(macro["USDVND"] / macro["USDVND"].shift(1))
    panel["dlog_WTI"] = np.log(macro["WTI"]    / macro["WTI"].shift(1))
    panel["VIX"]      = macro["VIX"]
    panel["US10Y"]    = macro["US10Y"]

    # GPR daily — backward asof to avoid look-ahead
    panel = pd.merge_asof(panel.sort_index(), gprd.sort_index(),
                          left_index=True, right_index=True, direction="backward")

    # AI-GPR daily
    aig = aig.sort_index()
    panel = pd.merge_asof(panel, aig, left_index=True, right_index=True, direction="backward")

    # GPR monthly (Vietnam-specific) — also backward asof
    if not gprm.empty:
        gprm = gprm.add_prefix("GPRm_").sort_index()
        panel = pd.merge_asof(panel, gprm, left_index=True, right_index=True, direction="backward")

    if not aigvn.empty:
        aigvn = aigvn.add_prefix("AIGPRvn_").sort_index()
        panel = pd.merge_asof(panel, aigvn, left_index=True, right_index=True, direction="backward")

    print("\n=== 3/4 Interaction terms & cleaning ===")
    # Robust GPR column picks
    gpr_col = next((c for c in panel.columns if c.upper() == "GPRD"), None) \
              or next((c for c in panel.columns if c.upper().startswith("GPRD")), None)
    aigpr_col = next((c for c in panel.columns if c.upper().startswith("AI_GPR") or c.upper().startswith("AI_")), None)
    if gpr_col:   panel["GPR_x_dFX"]    = panel[gpr_col]   * panel["dlog_FX"]
    if aigpr_col: panel["AIGPR_x_dFX"]  = panel[aigpr_col] * panel["dlog_FX"]

    panel = panel.dropna(subset=[c for c in panel.columns if c.endswith("_ret")], how="all")
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=1)

    # Trim to last actual trading day (drop trailing forward-filled padding where all returns are 0)
    ret_cols = [c for c in panel.columns if c.endswith("_ret")]
    if ret_cols:
        nonzero = (panel[ret_cols].abs().sum(axis=1) > 0)
        last_real = nonzero[nonzero].index.max()
        if last_real is not None:
            panel = panel.loc[:last_real]

    print("\n=== 4/4 Save final_dataset.parquet ===")
    out = ROOT / cfg["paths"]["processed"] / "final_dataset.parquet"
    save_parquet(panel, out)
    print(f"Saved -> {out}  shape={panel.shape}")
    print("Columns:", list(panel.columns))
    return panel

if __name__ == "__main__":
    build()
