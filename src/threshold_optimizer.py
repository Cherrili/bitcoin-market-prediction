"""
Per-class threshold optimisation for probability-calibrated models.

For imbalanced multi-class problems the default argmax(prob) decision
rule is suboptimal.  Instead we find a threshold vector t = [t0, t1, t2]
such that prediction = argmax(prob / t) maximises macro F1.

Optimisation is done on out-of-fold training-set probabilities so the
test set is never touched during tuning.

Public API
----------
optimize_thresholds(model, X_train, y_train_enc, X_test, y_test_enc,
                    output_dir, model_name) -> dict
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import label_binarize


def optimize_thresholds(
    model,
    X_train:      np.ndarray,
    y_train_enc:  np.ndarray,
    X_test:       np.ndarray,
    y_test_enc:   np.ndarray,
    output_dir:   str,
    model_name:   str = "LightGBM",
) -> dict:
    """
    1. Generate out-of-fold predicted probabilities on the training set
       using TimeSeriesSplit (same CV as GridSearchCV) — no test leakage.
    2. Grid-search per-class thresholds on those OOF probabilities.
    3. Apply best thresholds to the test set and compute metrics.
    4. Save confusion matrix and return results dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: OOF probabilities on training set ─────────────────────────────
    print(f"   [{model_name}] generating OOF probabilities …")
    oof_probs = np.zeros((len(X_train), 3))
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
        model.fit(X_train[tr_idx], y_train_enc[tr_idx])
        oof_probs[val_idx] = model.predict_proba(X_train[val_idx])

    # Re-fit on full training set for test inference
    model.fit(X_train, y_train_enc)

    # ── Step 2: grid-search thresholds on OOF probs ───────────────────────────
    print(f"   [{model_name}] searching optimal thresholds …")
    best_f1     = 0.0
    best_thresh = np.array([1/3, 1/3, 1/3])

    grid = np.arange(0.05, 0.70, 0.05)
    for t0 in grid:
        for t1 in grid:
            for t2 in grid:
                t = np.array([t0, t1, t2])
                preds = (oof_probs / t).argmax(axis=1)
                f1 = f1_score(y_train_enc, preds,
                              average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1     = f1
                    best_thresh = t.copy()

    print(f"   [{model_name}] best thresholds: "
          f"Bear={best_thresh[0]:.2f}  "
          f"Side={best_thresh[1]:.2f}  "
          f"Bull={best_thresh[2]:.2f}  "
          f"(OOF F1={best_f1:.4f})")

    # ── Step 3: apply to test set ─────────────────────────────────────────────
    y_test_prob = model.predict_proba(X_test)
    y_pred      = (y_test_prob / best_thresh).argmax(axis=1)

    acc = accuracy_score(y_test_enc, y_pred)
    f1  = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
    try:
        y_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, y_test_prob,
                              multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"   {model_name} + threshold opt   "
          f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # ── Step 4: confusion matrix ──────────────────────────────────────────────
    CLASS_NAMES = ["Bear(0)", "Sideways(1)", "Bull(2)"]
    cm = confusion_matrix(y_test_enc, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    safe = model_name.replace(" ", "_")
    ax.set_title(f"{model_name} + Threshold Opt\nConfusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{safe}_threshold_opt_confusion_matrix.png")
    plt.close(fig)

    return {
        "accuracy":    acc,
        "f1_macro":    f1,
        "roc_auc":     auc,
        "thresholds":  best_thresh.tolist(),
        "oof_f1":      best_f1,
    }
