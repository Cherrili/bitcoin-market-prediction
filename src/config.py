"""
Global constants shared across all modules.
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = str(BASE_DIR / "archive")
OUTPUT_DIR = str(BASE_DIR / "output")

# ── Label scheme ──────────────────────────────────────────────────────────────
CLASSES     = [-1, 0, 1]
CLASS_NAMES = ["Bear(-1)", "Sideways(0)", "Bull(1)"]

# XGBoost / LightGBM require labels in {0, 1, 2}
LBL_MAP     = {-1: 0, 0: 1, 1: 2}
LBL_MAP_INV = {0: -1, 1: 0, 2: 1}

# ── Plot colours (one per model) ──────────────────────────────────────────────
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
