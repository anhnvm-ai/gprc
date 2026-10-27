"""Download VN market data via vnstock v3 — KBS source (only one that works for indices)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.utils.io import load_config, ensure_dir, ROOT

INDEX_SYMBOLS = {"VNINDEX", "HNXINDEX", "VN30", "HNX30", "UPCOMINDEX"}

def _via_vnstock(symbol: str, start: str, end: str, source: str = "KBS") -> pd.DataFrame:
    from vnstock import Vnstock
    obj = Vnstock().stock(symbol=symbol, source=source)
    df = obj.quote.history(start=start, end=end, interval="1D")
    df = df.rename(columns=str.lower)
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    elif "date" not in df.columns:
        df["date"] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index("date").sort_index()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep]
    # KBS returns OHLC scaled (~÷1000). Re-scale OHLC by 1000 so values are in points.
    if source == "KBS" and symbol in INDEX_SYMBOLS:
        for c in ("open", "high", "low", "close"):
            if c in df.columns:
                df[c] = df[c] * 1000.0
    return df.add_prefix(f"{symbol}_")

def download_vn_market(cfg: dict | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or load_config()
    start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
    out: dict[str, pd.DataFrame] = {}
    for sym in cfg["data"]["vn_indices"]:
        try:
            df = _via_vnstock(sym, start, end, source="KBS")
            print(f"[VN] {sym} via vnstock(KBS): {df.shape} "
                  f"[{df.index.min().date()} → {df.index.max().date()}]")
        except Exception as e:
            print(f"[VN] KBS failed for {sym}: {type(e).__name__}: {e}; trying VCI fallback")
            df = _via_vnstock(sym, start, end, source="VCI")
        out[sym] = df
        path = ROOT / cfg["paths"]["interim"] / f"{sym.lower()}.parquet"
        ensure_dir(path.parent); df.to_parquet(path)
    return out

if __name__ == "__main__":
    download_vn_market()
