"""
Step 7 — Model evaluation: metrics, confusion matrices, ROC curves,
and summary table.

Public API
----------
evaluate_models(trained, split, output_dir) -> pd.DataFrame
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)

from .config import CLASSES, CLASS_NAMES, LBL_MAP_INV, MODEL_COLORS
from .training import SplitResult


def evaluate_models(trained: dict, split: SplitResult,
                    output_dir: str) -> pd.DataFrame:
    """
    For each trained model:
      - compute Accuracy, F1 (macro), ROC AUC
      - save confusion matrix PNG
      - add ROC curve to shared figure

    Returns a summary DataFrame and saves:
        {model}_confusion_matrix.png
        roc_curves_all_models.png
        model_summary_table.png
        model_summary.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    y_test     = split.y_test
    y_test_bin = label_binarize(y_test, classes=CLASSES)   # (n, 3)

    # ── shared ROC figure (1 row × 3 class subplots) ──────────────────────────
    fig_roc, axes_roc = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, cname in zip(axes_roc, CLASS_NAMES):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"ROC — {cname}")
        ax.set_xlabel("False Positive Rate")
    axes_roc[0].set_ylabel("True Positive Rate")

    summary_rows = []

    for i, (name, info) in enumerate(trained.items()):
        model = info["model"]
        Xte   = split.X_test_sc if info["scaled"] else split.X_test

        y_pred_raw = model.predict(Xte)
        y_prob     = model.predict_proba(Xte)   # shape (n, 3)

        # Decode XGB/LGB {0,1,2} back to {-1,0,1}
        if info["encoded"]:
            y_pred = np.vectorize(LBL_MAP_INV.get)(y_pred_raw)
        else:
            y_pred = y_pred_raw

        # ── scalar metrics ────────────────────────────────────────────────────
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test_bin, y_prob,
                                multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")

        print(f"   {name:<22}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}"
              f"  best={info['best_p']}")

        summary_rows.append({
            "Model":       name,
            "Accuracy":    round(acc, 4),
            "F1_macro":    round(f1,  4),
            "ROC_AUC":     round(auc, 4),
            "Best Params": str(info["best_p"]),
        })

        # ── confusion matrix ──────────────────────────────────────────────────
        _save_confusion_matrix(y_test, y_pred, name, output_dir)

        # ── ROC curves per class ──────────────────────────────────────────────
        for j in range(3):
            fpr, tpr, _ = roc_curve(y_test_bin[:, j], y_prob[:, j])
            cls_auc     = roc_auc_score(y_test_bin[:, j], y_prob[:, j])
            axes_roc[j].plot(fpr, tpr, color=MODEL_COLORS[i], lw=1.5,
                             label=f"{name} ({cls_auc:.3f})")

    # ── finalise ROC figure ───────────────────────────────────────────────────
    for ax in axes_roc:
        ax.legend(loc="lower right", fontsize=7)
    plt.suptitle("ROC Curves — All Models (One-vs-Rest per Class)", fontsize=12)
    plt.tight_layout()
    fig_roc.savefig(f"{output_dir}/roc_curves_all_models.png")
    plt.close(fig_roc)

    # ── summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    _save_summary_table(summary_df, output_dir)
    summary_df.to_csv(f"{output_dir}/model_summary.csv", index=False)

    print(f"\n   Confusion matrices + ROC curves saved to: {output_dir}")
    return summary_df


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_confusion_matrix(y_true, y_pred, model_name: str,
                            output_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    wrapped = [n.replace("(", "\n(") for n in CLASS_NAMES]   # "Bear\n(-1)" etc.
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=wrapped, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_title(f"{model_name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    safe = model_name.replace(" ", "_")
    fig.savefig(f"{output_dir}/{safe}_confusion_matrix.png")
    plt.close(fig)


def _save_summary_table(summary_df: pd.DataFrame, output_dir: str) -> None:
    cols = ["Model", "Accuracy", "F1_macro", "ROC_AUC"]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary_df[cols].values,
        colLabels=["Model", "Accuracy", "F1 (macro)", "ROC AUC"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    ax.set_title("Model Performance Summary", fontsize=13, pad=20)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/model_summary_table.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 65)
    print("SUMMARY TABLE")
    print(summary_df[cols].to_string(index=False))
