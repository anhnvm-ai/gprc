"""Download AI-GPR (Iacoviello & Tong 2026) — daily + country panel."""
from __future__ import annotations
from pathlib import Path
import requests, pandas as pd
from src.utils.io import load_config, ensure_dir, ROOT

HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}

def _get(url: str, out: Path) -> Path:
    print(f"[AI-GPR] GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out

def download_ai_gpr_daily(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]; ensure_dir(raw_dir)
    f = _get(cfg["data"]["ai_gpr_daily_url"], raw_dir / "ai_gpr_daily.csv")
    df = pd.read_csv(f)
    date_col = next((c for c in df.columns if c.lower() in ("date","day")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"}).set_index("date").sort_index()
    df = df.add_prefix("AI_") if not any(c.startswith("AI_") for c in df.columns) else df
    out = ROOT / cfg["paths"]["interim"] / "ai_gpr_daily.parquet"
    ensure_dir(out.parent); df.to_parquet(out)
    print(f"[AI-GPR] daily shape={df.shape}")
    return df

def download_ai_gpr_country(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw_dir = ROOT / cfg["paths"]["raw"]; ensure_dir(raw_dir)
    f = _get(cfg["data"]["ai_gpr_country_url"], raw_dir / "ai_gpr_country.csv")
    df = pd.read_csv(f)
    # Filter Vietnam
    name_col = next((c for c in df.columns if "country" in c.lower()), None)
    if name_col is not None:
        vn = df[df[name_col].str.contains("Vietnam|VNM", case=False, na=False)].copy()
    else:
        vn = df.copy()
    date_col = next((c for c in vn.columns if c.lower() in ("date","month","day")), vn.columns[0])
    vn[date_col] = pd.to_datetime(vn[date_col])
    vn = vn.rename(columns={date_col: "date"}).set_index("date").sort_index()
    out = ROOT / cfg["paths"]["interim"] / "ai_gpr_vietnam.parquet"
    ensure_dir(out.parent); vn.to_parquet(out)
    print(f"[AI-GPR] Vietnam shape={vn.shape}")
    return vn

if __name__ == "__main__":
    download_ai_gpr_daily()
    try:
        download_ai_gpr_country()
    except Exception as e:
        print(f"[AI-GPR] country file unavailable: {e}")
