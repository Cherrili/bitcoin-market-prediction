"""
Step 8 — Feature importance plots and EDA charts.

Public API
----------
plot_feature_importances(trained, feature_cols, output_dir)
plot_eda(df, output_dir)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import CLASSES, CLASS_NAMES


def plot_feature_importances(trained: dict, feature_cols: list,
                             output_dir: str) -> None:
    """
    Save Top-20 feature importance plots for RF, XGBoost, LightGBM,
    Logistic Regression coefficients, and a combined normalised average.
    """
    os.makedirs(output_dir, exist_ok=True)

    rf_imp  = trained["Random Forest"]["model"].feature_importances_
    xgb_imp = trained["XGBoost"]["model"].feature_importances_
    lgb_imp = trained["LightGBM"]["model"].feature_importances_
    lr_imp  = np.abs(trained["Logistic Regression"]["model"].coef_).mean(axis=0)

    _bar(rf_imp,  feature_cols,
         "Random Forest — Top 20 Feature Importances",
         "feature_importance_rf.png",  "#1f77b4", output_dir)

    _bar(xgb_imp, feature_cols,
         "XGBoost — Top 20 Feature Importances",
         "feature_importance_xgb.png", "#ff7f0e", output_dir)

    _bar(lgb_imp, feature_cols,
         "LightGBM — Top 20 Feature Importances",
         "feature_importance_lgbm.png","#2ca02c", output_dir)

    _bar(lr_imp,  feature_cols,
         "Logistic Regression — Top 20 |Coefficient| (mean across classes)",
         "feature_importance_lr.png",  "#9467bd", output_dir)

    # Normalised average of tree-model importances
    combined = (_norm(rf_imp) + _norm(xgb_imp) + _norm(lgb_imp)) / 3.0
    _bar(combined, feature_cols,
         "Combined Feature Importance (RF + XGBoost + LightGBM, avg)",
         "feature_importance_combined.png", "#d62728", output_dir)

    print(f"   Feature importance plots saved to: {output_dir}")


def plot_eda(df: pd.DataFrame, output_dir: str) -> None:
    """
    Save a two-panel EDA chart:
      left  — label distribution bar chart
      right — BTC price timeline coloured by market state
    """
    os.makedirs(output_dir, exist_ok=True)
    y = df["label"].values

    fig, (ax_bar, ax_price) = plt.subplots(1, 2, figsize=(14, 4))

    # Label counts
    counts = [int((y == c).sum()) for c in CLASSES]
    ax_bar.bar(CLASS_NAMES, counts,
               color=["#d62728", "#ff7f0e", "#2ca02c"])
    ax_bar.set_title("Label Distribution")
    ax_bar.set_ylabel("Count")

    # Price timeline
    color_map = {-1: "#d62728", 0: "#ff7f0e", 1: "#2ca02c"}
    for lbl in CLASSES:
        mask = df["label"] == lbl
        ax_price.scatter(df.loc[mask, "datetime"],
                         df.loc[mask, "price"],
                         s=2, c=color_map[lbl],
                         label=CLASS_NAMES[lbl + 1])
    ax_price.set_yscale("log")
    ax_price.set_title("BTC Price (log scale) coloured by Market State")
    ax_price.set_ylabel("Price (USD, log)")
    ax_price.legend(markerscale=4)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/eda_label_distribution.png")
    plt.close(fig)
    print(f"   EDA chart saved to: {output_dir}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(arr: np.ndarray) -> np.ndarray:
    s = arr.sum()
    return arr / s if s > 0 else arr


def _bar(importances: np.ndarray, feature_names: list,
         title: str, filename: str, color: str,
         output_dir: str, top_n: int = 20) -> None:
    ser = pd.Series(importances, index=feature_names).nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ser.plot(kind="barh", ax=ax, color=color)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{filename}")
    plt.close(fig)
