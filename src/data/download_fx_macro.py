"""Download FX (USD/VND) and macro controls (WTI, VIX, US10Y) via yfinance."""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from src.utils.io import load_config, ensure_dir, ROOT

def _yf(ticker: str, start: str, end: str, name: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df[["Close"]].rename(columns={"Close": name}).copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out

def download_fx_macro(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    series = [
        _yf(cfg["data"]["fx_ticker"],   start, end, "USDVND"),
        _yf(cfg["data"]["oil_ticker"],  start, end, "WTI"),
        _yf(cfg["data"]["vix_ticker"],  start, end, "VIX"),
        _yf(cfg["data"]["us10y_ticker"],start, end, "US10Y"),
    ]
    df = pd.concat(series, axis=1).sort_index().ffill()
    out = ROOT / cfg["paths"]["interim"] / "fx_macro.parquet"
    ensure_dir(out.parent); df.to_parquet(out)
    print(f"[FX/Macro] shape={df.shape}")
    return df

if __name__ == "__main__":
    download_fx_macro()
