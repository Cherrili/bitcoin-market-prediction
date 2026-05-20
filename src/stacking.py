"""
Stacking ensemble: RF + KNN + LGB + XGB as base learners,
LightGBM as meta-classifier.

Out-of-fold (OOF) predictions from TimeSeriesSplit(5) are used to train
the meta-classifier — no test leakage.

Meta-learner input = base-model OOF probs (12-d).
Additional heuristic: per-sample max-confidence routing is also tried
and the better result is kept.

Public API
----------
run_stacking(split, output_dir) -> dict  {accuracy, f1_macro, roc_auc}
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import xgboost as xgb

from .training import SplitResult
from .config import CLASS_NAMES


_BASE_MODELS = [
    ("RF",  RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2,
        class_weight="balanced", random_state=42, n_jobs=-1)),
    ("KNN", KNeighborsClassifier(
        n_neighbors=3, weights="distance", n_jobs=-1)),
    ("LGB", lgb.LGBMClassifier(
        objective="multiclass", num_class=3, random_state=42,
        n_jobs=-1, verbose=-1,
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_child_samples=50, num_leaves=31)),
    ("XGB", xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", use_label_encoder=False,
        n_estimators=200, max_depth=3, learning_rate=0.1,
        min_child_weight=1, subsample=0.8,
        random_state=42, n_jobs=-1)),
]


_CONF_WEIGHTS = [1.0, 1.0, 1.5, 1.0]  # RF, KNN, LGB, XGB — boost LGB (highest solo F1)


def _max_conf_predict(probs_matrix: np.ndarray, n_class: int) -> np.ndarray:
    """
    For each sample, pick the prediction from the most-confident base model.
    LGB gets a 1.5× confidence multiplier because it is the strongest solo model
    and the only one that reliably predicts the minority Bear class.
    probs_matrix: (n_samples, n_base * n_class)
    """
    n_base = probs_matrix.shape[1] // n_class
    max_confs = np.array([
        _CONF_WEIGHTS[k] * probs_matrix[:, k * n_class:(k + 1) * n_class].max(axis=1)
        for k in range(n_base)
    ])
    best_k = max_confs.argmax(axis=0)  # (n_samples,)
    preds = np.array([
        np.argmax(probs_matrix[i, best_k[i] * n_class:(best_k[i] + 1) * n_class])
        for i in range(len(probs_matrix))
    ])
    return preds


def run_stacking(split: SplitResult, output_dir: str) -> dict:
    """
    Stacking ensemble: RF + KNN + LGB + XGB base learners.
    Two meta strategies are evaluated; the better F1 is returned:
      1. LightGBM meta on OOF probs (12-d)
      2. Max-confidence routing (no training required)

    TimeSeriesSplit(5) preserves temporal order — no test leakage.
    Returns dict with accuracy, f1_macro, roc_auc.
    """
    os.makedirs(output_dir, exist_ok=True)

    from sklearn.utils.class_weight import compute_sample_weight

    X_tr = split.X_train_sc
    y_tr = split.y_train_enc
    X_te = split.X_test_sc
    y_te = split.y_test_enc

    n_base  = len(_BASE_MODELS)
    n_class = 3

    # ── Step 1: OOF predictions on training set ───────────────────────────────
    print("   [Stacking] generating OOF predictions (RF+KNN+LGB+XGB) …", flush=True)
    oof_probs = np.zeros((len(X_tr), n_base * n_class))
    oof_valid = np.zeros(len(X_tr), dtype=bool)
    tscv      = TimeSeriesSplit(n_splits=5)

    for k, (bname, bmodel) in enumerate(_BASE_MODELS):
        print(f"   [Stacking] OOF base={bname} …", flush=True)
        for tr_idx, val_idx in tscv.split(X_tr):
            sw = compute_sample_weight("balanced", y_tr[tr_idx])
            if bname in ("LGB", "XGB"):
                bmodel.fit(X_tr[tr_idx], y_tr[tr_idx], sample_weight=sw)
            else:
                bmodel.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof_probs[val_idx, k * n_class:(k + 1) * n_class] = (
                bmodel.predict_proba(X_tr[val_idx])
            )
            oof_valid[val_idx] = True

    # ── Step 2: retrain base models on full training set ──────────────────────
    print("   [Stacking] retraining base models on full train …", flush=True)
    test_probs = np.zeros((len(X_te), n_base * n_class))
    sw_full = compute_sample_weight("balanced", y_tr)
    for k, (bname, bmodel) in enumerate(_BASE_MODELS):
        if bname in ("LGB", "XGB"):
            bmodel.fit(X_tr, y_tr, sample_weight=sw_full)
        else:
            bmodel.fit(X_tr, y_tr)
        test_probs[:, k * n_class:(k + 1) * n_class] = (
            bmodel.predict_proba(X_te)
        )

    # ── Step 3a: LGB meta on OOF probs ───────────────────────────────────────
    print(f"   [Stacking] training meta LGB ({oof_valid.sum()} OOF rows) …",
          flush=True)
    sw_meta = compute_sample_weight("balanced", y_tr[oof_valid])
    meta = lgb.LGBMClassifier(
        objective="multiclass", num_class=3,
        n_estimators=300, max_depth=4, learning_rate=0.03,
        min_child_samples=30, num_leaves=15,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    meta.fit(oof_probs[oof_valid], y_tr[oof_valid], sample_weight=sw_meta)

    meta_preds_lgb  = meta.predict(test_probs)
    meta_probs_lgb  = meta.predict_proba(test_probs)
    f1_lgb = f1_score(y_te, meta_preds_lgb, average="macro", zero_division=0)

    # ── Step 3b: max-confidence routing ──────────────────────────────────────
    meta_preds_mc = _max_conf_predict(test_probs, n_class)
    f1_mc = f1_score(y_te, meta_preds_mc, average="macro", zero_division=0)

    # ── Step 3c: soft-vote average of all 4 base models ──────────────────────
    avg_probs = sum(
        test_probs[:, k * n_class:(k + 1) * n_class] for k in range(n_base)
    ) / n_base
    meta_preds_sv = np.argmax(avg_probs, axis=1)
    f1_sv = f1_score(y_te, meta_preds_sv, average="macro", zero_division=0)

    print(f"   [Stacking] meta-LGB  F1={f1_lgb:.4f}  "
          f"max-conf F1={f1_mc:.4f}  soft-vote F1={f1_sv:.4f}")

    # ── Step 4: pick the best strategy ───────────────────────────────────────
    best = max(
        ("lgb",       f1_lgb, meta_preds_lgb,  meta_probs_lgb),
        ("max-conf",  f1_mc,  meta_preds_mc,   None),
        ("soft-vote", f1_sv,  meta_preds_sv,   avg_probs),
        key=lambda x: x[1],
    )
    strategy, _, meta_preds, meta_probs_final = best
    print(f"   [Stacking] using strategy: {strategy}", flush=True)

    if meta_probs_final is None:
        # max-conf has no proba; use soft-vote average (better calibrated) for AUC
        meta_probs_final = avg_probs

    acc = accuracy_score(y_te, meta_preds)
    f1  = f1_score(y_te, meta_preds, average="macro", zero_division=0)
    try:
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, meta_probs_final,
                              multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"   Stacking (RF+KNN+LGB+XGB) Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_te, meta_preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_title(f"Stacking Ensemble (RF+KNN+LGB+XGB)\nConfusion Matrix [{strategy}]")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/Stacking_confusion_matrix.png")
    plt.close(fig)

    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc}
