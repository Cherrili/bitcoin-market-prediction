"""
Binary direction classification experiment.

Labels:
    Up   (+1) : 30-day return > +5%
    Down (-1) : 30-day return < -5%
    Neutral zone (−5% to +5%) is dropped — too noisy for direction signal.

Run from project root:
    python run_binary.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import lightgbm as lgb

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import build_features

os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.05   # ±5% neutral zone boundary


def create_binary_labels(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    df = df.copy()
    df["future_price_30d"] = df["price"].shift(-30)
    df["return_30d"] = (df["future_price_30d"] - df["price"]) / df["price"]
    df.dropna(subset=["future_price_30d"], inplace=True)

    # Keep only clear Up / Down samples; drop neutral zone
    df = df[(df["return_30d"] > threshold) | (df["return_30d"] < -threshold)].copy()
    df["label_bin"] = (df["return_30d"] > threshold).astype(int)  # 1=Up, 0=Down

    counts = df["label_bin"].value_counts().sort_index()
    print(f"   Binary labels  : Down={counts.get(0,0)}, Up={counts.get(1,0)}  "
          f"(dropped {len(df)} → kept after neutral removal)")
    return df


def run_binary_experiment():
    print("=" * 65)
    print("Binary Direction Classification (±5% neutral zone dropped)")
    print("=" * 65)

    print("\n1. Loading data …")
    df = load_and_clean(DATA_DIR)

    print("\n2. Creating binary labels …")
    # Need return_30d first — reuse raw price
    df["future_price_30d"] = df["price"].shift(-30)
    df["return_30d"] = (df["future_price_30d"] - df["price"]) / df["price"]
    df.dropna(subset=["future_price_30d"], inplace=True)

    # Temporarily set label so build_features doesn't complain
    df["label"] = 0
    total_before = len(df)
    df_bin = df[(df["return_30d"] > THRESHOLD) | (df["return_30d"] < -THRESHOLD)].copy()
    df_bin["label_bin"] = (df_bin["return_30d"] > THRESHOLD).astype(int)
    print(f"   Kept {len(df_bin)}/{total_before} rows after dropping ±{THRESHOLD*100:.0f}% neutral zone")
    print(f"   Down: {(df_bin['label_bin']==0).sum()}  Up: {(df_bin['label_bin']==1).sum()}")

    print("\n3. Feature engineering …")
    # Set label for build_features, but drop label_bin and return cols
    # to prevent any leakage into feature_cols
    df_bin["label"] = df_bin["label_bin"]
    df_bin = df_bin.drop(columns=["label_bin", "return_30d", "future_price_30d"],
                         errors="ignore")
    df_feat, feature_cols = build_features(df_bin)

    print("\n4. Time-ordered 80/20 split …")
    X = df_feat[feature_cols].values
    y = df_feat["label"].values  # label_bin already stored in label column
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"   Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"   Test Up: {y_test.sum()}  Test Down: {(y_test==0).sum()}")

    # Feature selection: top 40 by RF importance
    from sklearn.ensemble import RandomForestClassifier as RFC
    rf_sel = RFC(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train, y_train)
    top_idx = sorted(
        range(len(feature_cols)),
        key=lambda i: rf_sel.feature_importances_[i], reverse=True
    )[:40]
    top_idx = sorted(top_idx)
    X_train = X_train[:, top_idx]
    X_test  = X_test[:, top_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    tscv = TimeSeriesSplit(n_splits=5)

    models = [
        ("Logistic Regression", LogisticRegression(solver="lbfgs", max_iter=2000,
             random_state=42, class_weight="balanced"),
         {"C": [0.01, 0.1, 1, 10]}, True, False),
        ("Random Forest", RFC(random_state=42, n_jobs=-1, class_weight="balanced"),
         {"n_estimators": [100, 200], "max_depth": [5, 10], "min_samples_split": [2, 5]},
         False, False),
        ("XGBoost", xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
             use_label_encoder=False, random_state=42, n_jobs=-1),
         {"n_estimators": [100, 200], "max_depth": [3, 5],
          "learning_rate": [0.05, 0.1], "subsample": [0.8, 1.0]},
         False, True),
        ("LightGBM", lgb.LGBMClassifier(objective="binary", random_state=42,
             n_jobs=-1, verbose=-1),
         {"n_estimators": [100, 200], "max_depth": [3, 5],
          "learning_rate": [0.05, 0.1], "min_child_samples": [20, 50]},
         False, True),
        ("SVM", SVC(probability=True, random_state=42, class_weight="balanced"),
         {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]}, True, False),
        ("KNN", KNeighborsClassifier(n_jobs=-1),
         {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
         True, False),
    ]

    results = []
    print("\n5. Training models …\n")

    for name, est, params, scaled, sw in models:
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test
        fit_params = {}
        if sw:
            fit_params["sample_weight"] = compute_sample_weight("balanced", y_train)

        gs = GridSearchCV(est, params, cv=tscv, scoring="f1_macro",
                          n_jobs=-1, refit=True)
        gs.fit(Xtr, y_train, **fit_params)

        best = gs.best_estimator_
        preds = best.predict(Xte)
        proba = best.predict_proba(Xte)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro")
        auc = roc_auc_score(y_test, proba)

        print(f"   {name:22s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        results.append({
            "Model": name, "Accuracy": round(acc, 4),
            "F1_macro": round(f1, 4), "ROC_AUC": round(auc, 4),
            "Best Params": str(gs.best_params_),
        })

    # ── Baselines ──────────────────────────────────────────────────────────────
    n_up   = y_test.sum()
    n_down = len(y_test) - n_up
    maj_class = 1 if n_up >= n_down else 0
    maj_preds = np.full(len(y_test), maj_class)
    rand_preds = np.random.RandomState(42).randint(0, 2, len(y_test))

    def _baseline(name, preds):
        return {
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "F1_macro": round(f1_score(y_test, preds, average="macro", zero_division=0), 4),
            "ROC_AUC": "N/A",
            "Best Params": "—",
        }
    results.append(_baseline("Always-Up (majority)", maj_preds))
    results.append(_baseline("Random Guess", rand_preds))

    results_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "binary_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n   Saved: {out_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    model_results = results_df[~results_df["Model"].str.startswith("Always")
                                & ~results_df["Model"].str.startswith("Random")].copy()
    model_results["ROC_AUC_num"] = pd.to_numeric(model_results["ROC_AUC"], errors="coerce")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [("Accuracy", "Accuracy"), ("F1_macro", "Macro F1"), ("ROC_AUC_num", "ROC-AUC")]
    for ax, (col, label) in zip(axes, metrics):
        vals = model_results[col].values.astype(float)
        ax.barh(model_results["Model"], vals, color="#2c7bb6")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="0.5")
        ax.set_xlabel(label)
        ax.set_title(f"Binary: {label}")
        ax.set_xlim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "binary_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("   Saved: binary_model_comparison.png")

    # ── Comparison table: 3-class vs binary ───────────────────────────────────
    three_class_f1 = {
        "Logistic Regression": 0.1843,
        "Random Forest":       0.3105,
        "XGBoost":             0.1644,
        "LightGBM":            0.3384,
        "SVM":                 0.1223,
        "KNN":                 0.3119,
    }
    print("\n   3-class vs Binary F1 Comparison:")
    print(f"   {'Model':22s}  {'3-class F1':>12}  {'Binary F1':>10}  {'Δ':>8}")
    print("   " + "-" * 58)
    for row in results[:6]:
        n  = row["Model"]
        b  = row["F1_macro"]
        t  = three_class_f1.get(n, "—")
        delta = f"+{b - t:.4f}" if isinstance(t, float) else "—"
        print(f"   {n:22s}  {t:>12}  {b:>10}  {delta:>8}")

    print("\n" + "=" * 65)
    print("Binary experiment complete.")
    return results_df


if __name__ == "__main__":
    run_binary_experiment()
