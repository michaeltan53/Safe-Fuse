"""Shared helpers used by every experiment script."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict


# On non-UTF-8 consoles (e.g. Windows GBK) the tables contain Greek letters
# (τ, μ, ε, δ, η, Δ, Ω, χ²) and arrows. Switch stdout to UTF-8 once at
# import time so every experiment can print its table without crashing.
try:                                                  # Python 3.7+
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                     # pragma: no cover
    pass

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def save_results(name: str, payload: Dict[str, Any]) -> str:
    """Write payload to results/<name>.json and return the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def print_table(title: str, header: list[str], rows: list[list[Any]]) -> None:
    """Print a left-aligned plain-ASCII table."""
    cols = [len(h) for h in header]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            cols[i] = max(cols[i], len(cell))

    sep = "  ".join("-" * c for c in cols)
    print()
    print(title)
    print(sep)
    print("  ".join(h.ljust(cols[i]) for i, h in enumerate(header)))
    print(sep)
    for row in str_rows:
        print("  ".join(c.ljust(cols[i]) for i, c in enumerate(row)))
    print(sep)
