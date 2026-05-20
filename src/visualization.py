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


def plot_distribution_shift(df: pd.DataFrame, split_idx: int,
                             output_dir: str) -> None:
    """
    Two-panel chart quantifying train/test distribution shift:
      left  — class distribution (%) in train vs test
      right — rolling 90-day Bull/Bear/Sideways fraction over time
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    def _pct(d):
        counts = {c: int((d["label"] == c).sum()) for c in CLASSES}
        total  = sum(counts.values())
        return {k: v / total * 100 for k, v in counts.items()}

    tr_pct = _pct(train_df)
    te_pct = _pct(test_df)

    fig, (ax_bar, ax_roll) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: grouped bar chart ───────────────────────────────────────────────
    x       = np.arange(3)
    width   = 0.35
    colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
    tr_vals = [tr_pct[c] for c in CLASSES]
    te_vals = [te_pct[c] for c in CLASSES]

    bars_tr = ax_bar.bar(x - width/2, tr_vals, width,
                         label="Train (2010–2021)", color=colors, alpha=0.85)
    bars_te = ax_bar.bar(x + width/2, te_vals, width,
                         label="Test  (2021–2023)", color=colors, alpha=0.45,
                         edgecolor=colors, linewidth=1.5)

    for bar, v in zip(bars_tr, tr_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar, v in zip(bars_te, te_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    ax_bar.set_xticks(x); ax_bar.set_xticklabels(CLASS_NAMES)
    ax_bar.set_ylabel("Percentage of Samples (%)")
    ax_bar.set_title("Class Distribution: Train vs Test\n(Distribution Shift)")
    ax_bar.legend(); ax_bar.set_ylim(0, 75)

    # Print numbers to terminal
    print("\n   Distribution shift analysis:")
    for c, cn in zip(CLASSES, CLASS_NAMES):
        print(f"     {cn:<14}  train={tr_pct[c]:.1f}%  test={te_pct[c]:.1f}%"
              f"  Δ={te_pct[c]-tr_pct[c]:+.1f}%")

    # ── right: rolling 90-day label fraction ─────────────────────────────────
    df2 = df.copy()
    df2 = df2.sort_values("datetime").reset_index(drop=True)
    df2["is_bear"] = (df2["label"] == -1).astype(float)
    df2["is_bull"] = (df2["label"] ==  1).astype(float)
    df2["is_side"] = (df2["label"] ==  0).astype(float)

    roll = 90
    for col, color, label in [
        ("is_bear", "#d62728", "Bear"),
        ("is_side", "#ff7f0e", "Sideways"),
        ("is_bull", "#2ca02c", "Bull"),
    ]:
        ax_roll.plot(df2["datetime"],
                     df2[col].rolling(roll, min_periods=1).mean() * 100,
                     color=color, lw=1.5, label=label)

    split_date = df2["datetime"].iloc[split_idx] if split_idx < len(df2) else None
    if split_date is not None:
        ax_roll.axvline(split_date, color="black", ls="--", lw=1.5,
                        label="Train/Test split")

    ax_roll.set_ylabel(f"Rolling {roll}-day fraction (%)")
    ax_roll.set_title("Market State Distribution Over Time")
    ax_roll.legend(fontsize=9); ax_roll.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/distribution_shift.png", bbox_inches="tight")
    plt.close(fig)
    print(f"   Distribution shift chart saved.")


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
