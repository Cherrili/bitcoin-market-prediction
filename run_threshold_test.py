"""
Quick ablation: compare ±10% vs ±15% label thresholds.
Runs RF, LGB, XGB, KNN with max_depth=10 (fixed, no grid search) for speed.
"""
import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import label_binarize
import lightgbm as lgb
import xgboost as xgb

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import build_features

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_labels(df, threshold):
    df = df.copy()
    df["future_price_30d"] = df["price"].shift(-30)
    df["return_30d"] = (df["future_price_30d"] - df["price"]) / df["price"]
    df.dropna(subset=["future_price_30d"], inplace=True)
    df["label"] = df["return_30d"].apply(
        lambda r: 1 if r > threshold else (-1 if r < -threshold else 0)
    )
    return df


def run_experiment(threshold):
    df_raw = load_and_clean(DATA_DIR)
    df = make_labels(df_raw, threshold)
    counts = df["label"].value_counts().sort_index().to_dict()
    print(f"\n  Labels (thr=±{threshold*100:.0f}%): Bear={counts.get(-1,0)} "
          f"Side={counts.get(0,0)} Bull={counts.get(1,0)}")

    df, feature_cols = build_features(df)

    X = df[feature_cols].values
    y = df["label"].values
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Feature selection on train only
    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_sel.fit(X_train, y_train)
    top_idx = sorted(sorted(range(len(feature_cols)),
                     key=lambda i: rf_sel.feature_importances_[i], reverse=True)[:40])
    X_train, X_test = X_train[:, top_idx], X_test[:, top_idx]

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    # LBL_MAP for XGB/LGB: {-1:0, 0:1, 1:2}
    lbl_map = {-1: 0, 0: 1, 1: 2}
    y_tr_enc = np.array([lbl_map[v] for v in y_train])
    y_te_enc = np.array([lbl_map[v] for v in y_test])

    sw = compute_sample_weight("balanced", y_tr_enc)

    models = [
        ("RF",  RandomForestClassifier(n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1),
         X_train, X_test, y_train, y_test, None),
        ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1),
         X_train_sc, X_test_sc, y_train, y_test, None),
        ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=200, max_depth=10, learning_rate=0.1,
                min_child_samples=20, num_leaves=63,
                random_state=42, n_jobs=-1, verbose=-1),
         X_train, X_test, y_tr_enc, y_te_enc, sw),
        ("XGB", xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", use_label_encoder=False,
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1),
         X_train, X_test, y_tr_enc, y_te_enc, sw),
    ]

    results = []
    for name, est, Xtr, Xte, ytr, yte, sample_w in models:
        if sample_w is not None:
            est.fit(Xtr, ytr, sample_weight=sample_w)
        else:
            est.fit(Xtr, ytr)
        preds = est.predict(Xte)
        proba = est.predict_proba(Xte)

        # Map encoded preds back to {-1,0,1} for RF/KNN
        if name in ("RF", "KNN"):
            preds_eval, yte_eval = preds, yte
        else:
            inv = {0:-1, 1:0, 2:1}
            preds_eval = np.array([inv[p] for p in preds])
            yte_eval   = np.array([inv[p] for p in yte])

        acc = accuracy_score(yte_eval, preds_eval)
        f1  = f1_score(yte_eval, preds_eval, average="macro", zero_division=0)
        try:
            y_bin = label_binarize(yte_eval, classes=[-1,0,1])
            auc   = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        except Exception:
            auc = float("nan")

        print(f"    {name:4s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        results.append({"Model": name, "Threshold": f"±{threshold*100:.0f}%",
                        "Acc": round(acc,4), "F1": round(f1,4), "AUC": round(auc,4)})
    return results


print("=" * 55)
print("Threshold ablation: ±15% vs ±10%  (max_depth=10)")
print("=" * 55)

print("\n[±15% — baseline]")
r15 = run_experiment(0.15)

print("\n[±10% — test]")
r10 = run_experiment(0.10)

print("\n" + "=" * 55)
print("Summary — F1 delta (10% minus 15%)")
print(f"  {'Model':6s}  {'F1@15%':>8}  {'F1@10%':>8}  {'Delta':>8}")
print("  " + "-" * 38)
for a, b in zip(r15, r10):
    delta = b["F1"] - a["F1"]
    arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "~")
    print(f"  {a['Model']:6s}  {a['F1']:>8.4f}  {b['F1']:>8.4f}  {arrow}{abs(delta):>7.4f}")
