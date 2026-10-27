"""Bước 3: Build monthly ASEAN panel (6 countries, 2000-2026)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from src.data.build_asean_panel import build_asean_panel
from src.utils.io import load_config

cfg = load_config()
start_asean = cfg["data"].get("start_date_asean", "2010-01-01")

print("=== Bước 3: Build ASEAN Monthly Panel ===")
print(f"Su dung start_date_asean={start_asean} tu config")
panel = build_asean_panel()

print("\n=== KIEM TRA PANEL ===")
import pandas as pd
print(f"Shape: {panel.shape}")
print(f"Columns ({len(panel.columns)}): {list(panel.columns)}")
print(f"\nDate range: {panel['date'].min().date()} -> {panel['date'].max().date()}")
print(f"\nCountry counts:")
print(panel['country'].value_counts().to_string())

print("\nMissing values (%) per column:")
miss = panel.isnull().mean() * 100
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) > 0:
    print(miss.round(1).to_string())
else:
    print("  Khong co missing values!")

print("\nMau 5 dong dau:")
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 120)
print(panel.head().to_string())

print("\nBuoc 3 HOAN THANH.")
