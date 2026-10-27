"""Bước 1a: Download GPR daily + monthly (có cột ASEAN)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

from src.data.download_gpr import download_gpr_daily, download_gpr_monthly

print("=== Download GPR Monthly (global + ASEAN country columns) ===")
gpr_m = download_gpr_monthly()
print(f"Monthly GPR: shape={gpr_m.shape}")
print(f"  Columns  : {list(gpr_m.columns)}")
print(f"  Date range: {gpr_m.index.min().date()} -> {gpr_m.index.max().date()}")

asean_cols = [c for c in gpr_m.columns
              if any(cc in c.upper() for cc in ("THA","IDN","MYS","PHL","SGP","VNM"))]
print(f"  ASEAN GPR columns found: {asean_cols}")

print("\n=== Download GPR Daily ===")
gpr_d = download_gpr_daily()
print(f"Daily GPR: shape={gpr_d.shape}, cols={list(gpr_d.columns)}")
print(f"  Date range: {gpr_d.index.min().date()} -> {gpr_d.index.max().date()}")

print("\nBuoc 1a HOAN THANH.")
