"""Download ASEAN equity indices (ex-Vietnam) via yfinance + all 6 FX pairs."""
from __future__ import annotations
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from src.utils.io import load_config, ensure_dir, ROOT

warnings.filterwarnings("ignore")

# ── Country configs ──────────────────────────────────────────────────────────
# Thailand SET ^SET.BK  | Indonesia JKSE ^JKSE | Malaysia KLCI ^KLSE
# Philippines PSEi PSEI.PS | Singapore STI ^STI
# Vietnam VNM handled by download_vn_market.py (vnstock)
ASEAN_INDEX_CFG = {
    "THA": {"ticker": "^SET.BK", "name": "SET",  "alt": "THD"},
    "IDN": {"ticker": "^JKSE",   "name": "JKSE", "alt": "EIDO"},
    "MYS": {"ticker": "^KLSE",   "name": "KLCI", "alt": "EWM"},
    "PHL": {"ticker": "PSEI.PS", "name": "PSEi", "alt": "EPHE"},
    "SGP": {"ticker": "^STI",    "name": "STI",  "alt": "EWS"},
}

ASEAN_FX_TICKERS = {
    "VNM": "USDVND=X",
    "THA": "USDTHB=X",
    "IDN": "USDIDR=X",
    "MYS": "USDMYR=X",
    "PHL": "USDPHP=X",
    "SGP": "USDSGD=X",
}

GLOBAL_MACRO_TICKERS = {
    "SP500":  "^GSPC",
    "VIX":    "^VIX",
    "WTI":    "CL=F",
    "US10Y":  "^TNX",
    "DXY":    "DX-Y.NYB",
}


def _yf_close(ticker: str, start: str, end: str, col_name: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float, name=col_name)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.index.name = "date"
    s.name = col_name
    return s


def download_asean_indices(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for 5 ASEAN indices (ex-VNM) via yfinance.

    Returns dict: country_code -> DataFrame with columns close, ret, rv21.
    """
    cfg = cfg or load_config()
    start = cfg["data"].get("start_date_asean", cfg["data"]["start_date"])
    end   = cfg["data"]["end_date"]
    out: dict[str, pd.DataFrame] = {}

    for cc, info in ASEAN_INDEX_CFG.items():
        ticker = info["ticker"]
        close = _yf_close(ticker, start, end, "close")

        # Fallback to ETF proxy if primary ticker is empty
        if close.empty or len(close.dropna()) < 10:
            print(f"[ASEAN] {cc} primary ticker {ticker} failed, "
                  f"trying ETF proxy {info['alt']}")
            close = _yf_close(info["alt"], start, end, "close")

        if close.empty or len(close.dropna()) < 10:
            print(f"[ASEAN] WARNING: {cc} ({info['name']}) has no data — skipping")
            continue

        df = pd.DataFrame({"close": close})
        df["ret"]  = np.log(df["close"] / df["close"].shift(1))
        df["rv21"] = df["ret"].rolling(21).std() * np.sqrt(252)
        df.index.name = "date"

        path = ROOT / cfg["paths"]["interim"] / f"asean_{cc.lower()}_daily.parquet"
        ensure_dir(path.parent)
        df.to_parquet(path)
        print(f"[ASEAN] {cc} {info['name']}: shape={df.shape}  "
              f"[{df.index.min().date()} → {df.index.max().date()}]")
        out[cc] = df

    return out


def download_asean_fx(cfg: dict | None = None) -> pd.DataFrame:
    """Download daily USD/XXX FX rates for all 6 ASEAN currencies.

    Returns wide DataFrame: columns = [VNM_fx, THA_fx, IDN_fx, MYS_fx, PHL_fx, SGP_fx].
    """
    cfg = cfg or load_config()
    start = cfg["data"].get("start_date_asean", cfg["data"]["start_date"])
    end   = cfg["data"]["end_date"]
    series: list[pd.Series] = []

    for cc, ticker in ASEAN_FX_TICKERS.items():
        s = _yf_close(ticker, start, end, f"{cc}_fx")
        if s.empty or len(s.dropna()) < 10:
            print(f"[FX] WARNING: {cc} FX ({ticker}) has no data")
            continue
        series.append(s)
        print(f"[FX] {cc} USD/{cc}: {len(s.dropna())} obs")

    df = pd.concat(series, axis=1).sort_index().ffill()
    path = ROOT / cfg["paths"]["interim"] / "asean_fx_daily.parquet"
    ensure_dir(path.parent)
    df.to_parquet(path)
    print(f"[FX] ASEAN FX saved: shape={df.shape}")
    return df


def download_global_macro(cfg: dict | None = None) -> pd.DataFrame:
    """Download SP500, VIX, WTI, US10Y, DXY daily."""
    cfg = cfg or load_config()
    start = cfg["data"].get("start_date_asean", cfg["data"]["start_date"])
    end   = cfg["data"]["end_date"]
    series: list[pd.Series] = []

    for col, ticker in GLOBAL_MACRO_TICKERS.items():
        s = _yf_close(ticker, start, end, col)
        if not s.empty:
            series.append(s)

    df = pd.concat(series, axis=1).sort_index().ffill()
    path = ROOT / cfg["paths"]["interim"] / "global_macro.parquet"
    ensure_dir(path.parent)
    df.to_parquet(path)
    print(f"[MACRO] global macro saved: shape={df.shape}, cols={list(df.columns)}")
    return df


if __name__ == "__main__":
    download_asean_indices()
    download_asean_fx()
    download_global_macro()
