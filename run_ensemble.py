"""
Soft-voting ensemble: LSTM + RF + KNN.

Loads the cached split (output/split_cache.npz), trains all three
models from scratch, averages their class probabilities, then reports
Accuracy, F1 macro, and ROC-AUC.

Usage:
    python run_ensemble.py
"""

import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from src.config import OUTPUT_DIR
from src.lstm_model import train_lstm, _predict_proba_lstm

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CACHE  = os.path.join(OUTPUT_DIR, "split_cache.npz")
WINDOW = 30


def main():
    if not os.path.exists(CACHE):
        print(f"Cache not found at {CACHE}")
        print("Run the full pipeline first: python -m src.main")
        return

    print("Loading cached split …")
    data        = np.load(CACHE, allow_pickle=True)
    X_train_sc  = data["X_train_sc"]
    X_test_sc   = data["X_test_sc"]
    y_train_enc = data["y_train_enc"]
    y_test_enc  = data["y_test_enc"]
    print(f"   Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")

    # ── RF ────────────────────────────────────────────────────────────────────
    print("\n[1/3] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_sc, y_train_enc)
    rf_probs = rf.predict_proba(X_test_sc)           # (n_test, 3)
    rf_f1    = f1_score(y_test_enc, rf_probs.argmax(1), average="macro", zero_division=0)
    print(f"   RF  F1={rf_f1:.4f}")

    # ── KNN ───────────────────────────────────────────────────────────────────
    print("\n[2/3] Training KNN …")
    knn = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)
    knn.fit(X_train_sc, y_train_enc)
    knn_probs = knn.predict_proba(X_test_sc)
    knn_f1    = f1_score(y_test_enc, knn_probs.argmax(1), average="macro", zero_division=0)
    print(f"   KNN F1={knn_f1:.4f}")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n[3/3] Training LSTM …")
    lstm_model, _ = train_lstm(
        X_train_sc, y_train_enc,
        window=WINDOW, n_epochs=60, verbose=True,
    )
    lstm_probs_full = _predict_proba_lstm(lstm_model, X_test_sc, WINDOW)
    lstm_f1         = f1_score(y_test_enc, lstm_probs_full.argmax(1),
                               average="macro", zero_division=0)
    print(f"   LSTM F1={lstm_f1:.4f}")

    # ── Ensemble: average probs ───────────────────────────────────────────────
    print("\n[Ensemble] Soft-voting (equal weights) …")
    ensemble_probs = (rf_probs + knn_probs + lstm_probs_full) / 3.0
    preds          = ensemble_probs.argmax(1)

    acc = accuracy_score(y_test_enc, preds)
    f1  = f1_score(y_test_enc, preds, average="macro", zero_division=0)
    try:
        y_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, ensemble_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"\n{'='*50}")
    print(f"  Ensemble (RF+KNN+LSTM)  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    print(f"  RF alone                F1={rf_f1:.4f}")
    print(f"  KNN alone               F1={knn_f1:.4f}")
    print(f"  LSTM alone              F1={lstm_f1:.4f}")
    print(f"{'='*50}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    CLASS_NAMES = ["Bear(0)", "Sideways(1)", "Bull(2)"]
    cm = confusion_matrix(y_test_enc, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_title("Ensemble (RF+KNN+LSTM)\nConfusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Ensemble_confusion_matrix.png"))
    plt.close(fig)

    # ── update model_summary.csv ──────────────────────────────────────────────
    import pandas as pd
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        summary = summary[~summary["Model"].str.startswith("Ensemble")]
        row = pd.DataFrame([{
            "Model":       "Ensemble (RF+KNN+LSTM)",
            "Accuracy":    round(acc, 4),
            "F1_macro":    round(f1,  4),
            "ROC_AUC":     round(auc, 4) if not np.isnan(auc) else float("nan"),
            "Best Params": f"soft-vote: RF+KNN+LSTM (window={WINDOW})",
        }])
        summary = pd.concat([summary, row], ignore_index=True)
        summary.to_csv(summary_path, index=False)
        print("model_summary.csv updated")


if __name__ == "__main__":
    main()
