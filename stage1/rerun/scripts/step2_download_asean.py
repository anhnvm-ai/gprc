"""Bước 2: Download 5 ASEAN equity indices + 6 FX pairs + global macro."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from src.data.download_asean_markets import (
    download_asean_indices, download_asean_fx, download_global_macro
)
from src.utils.io import load_config

cfg = load_config()

start_asean = cfg["data"].get("start_date_asean", cfg["data"]["start_date"])
print(f"=== Bước 2a: ASEAN Equity Indices (tu {start_asean}) ===")
asean = download_asean_indices(cfg)
print(f"\nTong ket equity:")
for cc, df in asean.items():
    n = df['ret'].notna().sum()
    print(f"  {cc}: {len(df)} ngay, {n} returns, "
          f"{df.index.min().date()} -> {df.index.max().date()}")

print("\n=== Bước 2b: ASEAN FX Rates ===")
fx = download_asean_fx(cfg)
print(f"\nFX shape: {fx.shape}")
for col in fx.columns:
    n = fx[col].notna().sum()
    print(f"  {col}: {n} obs, {fx[col].dropna().index.min().date()} -> {fx[col].dropna().index.max().date()}")

print("\n=== Bước 2c: Global Macro (SP500, VIX, WTI, US10Y, DXY) ===")
macro = download_global_macro(cfg)
print(f"\nMacro shape: {macro.shape}")
for col in macro.columns:
    n = macro[col].notna().sum()
    print(f"  {col}: {n} obs")

print("\nBuoc 2 HOAN THANH.")
