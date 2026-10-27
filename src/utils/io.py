"""IO + config helpers."""
from __future__ import annotations
import os, yaml, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_parquet(df: pd.DataFrame, path: str | os.PathLike) -> None:
    path = Path(path); ensure_dir(path.parent)
    df.to_parquet(path, index=True)

def save_json(obj, path: str | os.PathLike) -> None:
    path = Path(path); ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
