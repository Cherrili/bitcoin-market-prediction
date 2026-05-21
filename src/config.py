"""
Global constants shared across all modules.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = r"C:\Users\LIYIYI\Desktop\blockchain\SC6122"
DATA_DIR   = os.path.join(BASE_DIR, "archive")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ── Label scheme ──────────────────────────────────────────────────────────────
CLASSES     = [-1, 1]
CLASS_NAMES = ["Bear(-1)", "Bull(1)"]

# XGBoost / LightGBM require labels in {0, 1}
LBL_MAP     = {-1: 0, 1: 1}
LBL_MAP_INV = {0: -1, 1: 1}

# ── Plot colours (one per model) ──────────────────────────────────────────────
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
