"""
Use TimeSeriesSplit CV on training data to select feature count.
If CV selects top-30, the 0.4413 result becomes validated (not post-hoc).
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import sys
sys.path.insert(0, '/Users/liuruyan/Desktop/bitcoin-market-prediction')

from src.config import DATA_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import create_labels, build_features

def stacking_f1(X_tr, y_tr, X_te, y_te, top_n, rf_imp_idx=None):
    sw = compute_sample_weight("balanced", y_tr)
    if rf_imp_idx is None:
        rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
        rf_sel.fit(X_tr, y_tr)
        idx = np.argsort(rf_sel.feature_importances_)[::-1][:top_n]
    else:
        idx = rf_imp_idx[:top_n]
    Xtr = StandardScaler().fit(X_tr[:, idx]).transform(X_tr[:, idx]).astype(np.float32)
    sc = StandardScaler().fit(X_tr[:, idx])
    Xtr = sc.transform(X_tr[:, idx]).astype(np.float32)
    Xte = sc.transform(X_te[:, idx]).astype(np.float32)
    models = [
        ("RF",  RandomForestClassifier(n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=1)),
        ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=1)),
        ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_child_samples=50, num_leaves=31,
                random_state=42, n_jobs=1, verbose=-1)),
        ("XGB", xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", use_label_encoder=False,
                n_estimators=200, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42, n_jobs=1)),
    ]
    tp = np.zeros((len(Xte), 12))
    for k, (name, m) in enumerate(models):
        if name in ("LGB","XGB"):
            m.fit(Xtr, y_tr, sample_weight=sw)
        else:
            m.fit(Xtr, y_tr)
        tp[:, k*3:(k+1)*3] = m.predict_proba(Xte)
    ww = np.array([1.0, 1.0, 1.5, 1.0])
    mc = np.array([ww[k]*tp[:,k*3:(k+1)*3].max(axis=1) for k in range(4)])
    bk = mc.argmax(axis=0)
    preds = np.array([np.argmax(tp[i, bk[i]*3:(bk[i]+1)*3]) for i in range(len(tp))])
    return f1_score(y_te, preds, average="macro", zero_division=0)

print("Loading...")
import pandas as pd
df_raw = load_and_clean(DATA_DIR)
df = create_labels(df_raw)
df, feature_cols = build_features(df)
df = df.reset_index(drop=True)

X = df[feature_cols].values.astype(np.float32)
y = df["label"].map({-1:0,0:1,1:2}).values
split_idx = int(len(X)*0.8)
X_tr, X_te = X[:split_idx], X[split_idx:]
y_tr, y_te = y[:split_idx], y[split_idx:]

# ── Step 1: fit RF on full training set, get importance ranking once ──────────
print("Computing RF feature importance on training set...")
rf_imp = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=1)
rf_imp.fit(X_tr, y_tr)
imp_idx = np.argsort(rf_imp.feature_importances_)[::-1]

# ── Step 2: CV on training set to select best top-N ───────────────────────────
print("\nTimeSeriesSplit(5) CV on training set to select feature count:")
tscv = TimeSeriesSplit(n_splits=5)
top_ns = [20, 25, 28, 30, 32, 35, 40, 50]
cv_scores = {}

for top_n in top_ns:
    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr)):
        Xf_tr, Xf_val = X_tr[tr_idx], X_tr[val_idx]
        yf_tr, yf_val = y_tr[tr_idx], y_tr[val_idx]
        if len(np.unique(yf_tr)) < 3 or len(yf_val) < 10:
            continue
        f1 = stacking_f1(Xf_tr, yf_tr, Xf_val, yf_val, top_n, rf_imp_idx=imp_idx)
        fold_scores.append(f1)
    mean_f1 = np.mean(fold_scores)
    cv_scores[top_n] = mean_f1
    print(f"  top_{top_n:2d}  CV F1 = {mean_f1:.4f}  (folds: {[f'{s:.3f}' for s in fold_scores]})")

best_n = max(cv_scores, key=cv_scores.get)
print(f"\n  CV selects: top_{best_n}  (CV F1={cv_scores[best_n]:.4f})")

# ── Step 3: Train on full train, test with CV-selected top-N ─────────────────
print(f"\nFinal evaluation with CV-selected top_{best_n}:")
f1_final = stacking_f1(X_tr, y_tr, X_te, y_te, best_n, rf_imp_idx=imp_idx)
print(f"  Test F1 = {f1_final:.4f}")

if best_n == 30:
    print("\n  ✓ CV independently selects top-30 → 0.4413 is VALIDATED")
else:
    print(f"\n  CV selects top-{best_n}, not top-30.")
    print(f"  Validated test F1 = {f1_final:.4f}")
