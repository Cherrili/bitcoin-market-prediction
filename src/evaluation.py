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
      - compute Accuracy / F1 / ROC AUC at default threshold (0.5)
      - find optimal threshold via Youden's J (max TPR - FPR on test ROC)
      - recompute Accuracy / F1 at optimal threshold
      - save confusion matrices (default + optimal) and ROC figure

    Saves:
        {model}_confusion_matrix.png          (default threshold)
        {model}_confusion_matrix_opt.png      (optimal threshold)
        roc_curves_all_models.png
        model_summary_table.png
        model_summary.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    y_test = split.y_test

    # ── shared ROC figure ─────────────────────────────────────────────────────
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title("ROC Curves — All Models (Binary: Bull vs Bear)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")

    summary_rows = []

    for i, (name, info) in enumerate(trained.items()):
        model = info["model"]
        Xte   = split.X_test_sc if info["scaled"] else split.X_test

        y_pred_raw = model.predict(Xte)
        y_prob     = model.predict_proba(Xte)   # shape (n, 2)
        prob_pos   = y_prob[:, 1]               # P(Bull)

        # Decode XGB/LGB {0,1} back to {-1,1}
        if info["encoded"]:
            y_pred = np.vectorize(LBL_MAP_INV.get)(y_pred_raw)
        else:
            y_pred = y_pred_raw

        # ── default-threshold metrics ─────────────────────────────────────────
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(y_test, prob_pos)
        except ValueError:
            auc = float("nan")

        # ── optimal threshold via Youden's J ──────────────────────────────────
        fpr, tpr, thresholds = roc_curve(y_test, prob_pos, pos_label=1)
        j_scores  = tpr - fpr
        best_idx  = int(np.argmax(j_scores))
        opt_thr   = float(thresholds[best_idx])

        y_pred_opt = np.where(prob_pos >= opt_thr, 1, -1)
        acc_opt = accuracy_score(y_test, y_pred_opt)
        f1_opt  = f1_score(y_test, y_pred_opt, average="macro", zero_division=0)

        print(f"   {name:<22}  "
              f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  "
              f"| opt_thr={opt_thr:.3f}  Acc_opt={acc_opt:.4f}  F1_opt={f1_opt:.4f}")

        summary_rows.append({
            "Model":       name,
            "Accuracy":    round(acc,     4),
            "F1_macro":    round(f1,      4),
            "ROC_AUC":     round(auc,     4),
            "Threshold":   round(opt_thr, 3),
            "Acc_opt":     round(acc_opt, 4),
            "F1_opt":      round(f1_opt,  4),
            "Best Params": str(info["best_p"]),
        })

        # ── confusion matrices ────────────────────────────────────────────────
        _save_confusion_matrix(y_test, y_pred,     name,          output_dir)
        _save_confusion_matrix(y_test, y_pred_opt, name + " (opt)", output_dir,
                               subtitle=f"threshold={opt_thr:.3f}")

        # ── ROC curve ─────────────────────────────────────────────────────────
        ax_roc.plot(fpr, tpr, color=MODEL_COLORS[i], lw=1.5,
                    label=f"{name} (AUC={auc:.3f})")
        ax_roc.scatter(fpr[best_idx], tpr[best_idx],
                       color=MODEL_COLORS[i], s=60, zorder=5)

    # ── finalise ROC figure ───────────────────────────────────────────────────
    ax_roc.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig_roc.savefig(f"{output_dir}/roc_curves_all_models.png")
    plt.close(fig_roc)

    # ── summary table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    _save_summary_table(summary_df, output_dir)
    summary_df.to_csv(f"{output_dir}/model_summary.csv", index=False)

    print(f"\n   Outputs saved to: {output_dir}")
    return summary_df


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_confusion_matrix(y_true, y_pred, model_name: str,
                            output_dir: str,
                            subtitle: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    wrapped = [n.replace("(", "\n(") for n in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=wrapped, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    title = f"{model_name}\nConfusion Matrix"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(f"{output_dir}/{safe}_confusion_matrix.png")
    plt.close(fig)


def _save_summary_table(summary_df: pd.DataFrame, output_dir: str) -> None:
    cols     = ["Model", "Accuracy", "F1_macro", "ROC_AUC", "Threshold", "Acc_opt", "F1_opt"]
    headers  = ["Model", "Acc (0.5)", "F1 (0.5)", "ROC AUC", "Opt Thr", "Acc (opt)", "F1 (opt)"]
    fig, ax = plt.subplots(figsize=(16, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary_df[cols].values,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax.set_title("Model Performance Summary (default vs optimal threshold)",
                 fontsize=12, pad=20)
    plt.tight_layout()
    fig.savefig(f"{output_dir}/model_summary_table.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 85)
    print("SUMMARY TABLE")
    print(summary_df[cols].to_string(index=False))
