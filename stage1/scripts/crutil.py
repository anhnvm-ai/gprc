"""Small shared helpers for the stage-1 scripts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CR = Path(__file__).resolve().parents[1]          # stage1/
ROOT = CR.parent                                  # project root
RES = CR / "results"
LOGS = RES / "logs"


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a DataFrame as a GitHub markdown table without extra deps."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def save_json(obj: dict, path: Path) -> None:
    """Write JSON atomically (temp file + replace) so an interruption never
    leaves a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
