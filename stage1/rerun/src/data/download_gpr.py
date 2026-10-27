"""Download Caldara & Iacoviello GPR (daily + monthly + country)."""
from __future__ import annotations
from pathlib import Path
import io, requests, pandas as pd
from src.utils.io import load_config, ensure_dir, ROOT

HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact: hybrid-slt-project)"}

def _download(url: str, out: Path) -> Path:
    out = Path(out); ensure_dir(out.parent)
    print(f"[GPR] GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"[GPR] -> {out}  ({len(r.content)/1024:.1f} KB)")
    return out

def download_gpr_daily(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    f = _download(cfg["data"]["gpr_daily_url"], raw_dir / "gpr_daily.xls")
    df = pd.read_excel(f)
    # File contains columns: date, GPRD, GPRD_ACT, GPRD_THREAT (sometimes 'DAY')
    # Prefer real 'date' column; drop conflicting names before rename
    if "date" in df.columns:
        date_col = "date"
    else:
        date_col = next((c for c in df.columns if c.lower() in ("date", "day")), df.columns[0])
    if date_col != "date" and "date" in df.columns:
        df = df.drop(columns=["date"])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    df = df.loc[df.index.notna()]
    interim = ROOT / cfg["paths"]["interim"] / "gpr_daily.parquet"
    ensure_dir(interim.parent); df.to_parquet(interim)
    print(f"[GPR] daily shape={df.shape}, cols={list(df.columns)}")
    return df

def download_gpr_monthly(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]
    f = _download(cfg["data"]["gpr_monthly_url"], raw_dir / "gpr_monthly.xls")
    df = pd.read_excel(f)
    # First column = month
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    # Keep global GPR + all 6 ASEAN country columns
    ASEAN_CODES = ("VNM", "THA", "IDN", "MYS", "PHL", "SGP")
    keep_cols = [
        c for c in df.columns
        if c in ("GPR", "GPRH", "GPRA", "GPRC")
        or any(cc in c.upper() for cc in ASEAN_CODES)
        or "VIETNAM" in c.upper()
    ]
    if keep_cols:
        df = df[keep_cols]
    interim = ROOT / cfg["paths"]["interim"] / "gpr_monthly.parquet"
    ensure_dir(interim.parent); df.to_parquet(interim)
    print(f"[GPR] monthly shape={df.shape}, cols={list(df.columns)}")
    return df

if __name__ == "__main__":
    download_gpr_daily()
    download_gpr_monthly()
