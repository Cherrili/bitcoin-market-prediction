"""
Focused hyperparameter search: LGB params + stacking weight
Using TimeSeriesSplit CV on training set (no test leakage)
"""
import warnings, numpy as np, itertools
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

def stacking_f1(X_tr, y_tr, X_te, y_te, top_n, lgb_params, lgb_weight):
    sw = compute_sample_weight("balanced", y_tr)
    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    rf_sel.fit(X_tr, y_tr)
    idx = np.argsort(rf_sel.feature_importances_)[::-1][:top_n]
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr[:, idx]).astype(np.float32)
    Xte = sc.transform(X_te[:, idx]).astype(np.float32)
    models = [
        ("RF",  RandomForestClassifier(n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=1)),
        ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=1)),
        ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
                random_state=42, n_jobs=1, verbose=-1, **lgb_params)),
        ("XGB", xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", use_label_encoder=False,
                n_estimators=200, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42, n_jobs=1)),
    ]
    ww = np.array([1.0, 1.0, lgb_weight, 1.0])
    tp = np.zeros((len(Xte), 12))
    for k, (name, m) in enumerate(models):
        if name in ("LGB","XGB"):
            m.fit(Xtr, y_tr, sample_weight=sw)
        else:
            m.fit(Xtr, y_tr)
        tp[:, k*3:(k+1)*3] = m.predict_proba(Xte)
    mc = np.array([ww[k]*tp[:,k*3:(k+1)*3].max(axis=1) for k in range(4)])
    bk = mc.argmax(axis=0)
    preds = np.array([np.argmax(tp[i,bk[i]*3:(bk[i]+1)*3]) for i in range(len(tp))])
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

tscv = TimeSeriesSplit(n_splits=5)

# ── Grid: LGB params × stacking weight ────────────────────────────────────────
lgb_grid = [
    dict(n_estimators=200, max_depth=5,  learning_rate=0.10, min_child_samples=50, num_leaves=31),
    dict(n_estimators=300, max_depth=5,  learning_rate=0.05, min_child_samples=30, num_leaves=31),
    dict(n_estimators=400, max_depth=6,  learning_rate=0.05, min_child_samples=20, num_leaves=63),
    dict(n_estimators=500, max_depth=4,  learning_rate=0.03, min_child_samples=50, num_leaves=31),
    dict(n_estimators=300, max_depth=7,  learning_rate=0.05, min_child_samples=20, num_leaves=63),
    dict(n_estimators=200, max_depth=5,  learning_rate=0.10, min_child_samples=20, num_leaves=63),
]
weights = [1.2, 1.5, 1.8, 2.0]
top_ns  = [20, 25, 30]

print(f"\nSearching {len(lgb_grid)*len(weights)*len(top_ns)} configs via CV...")
print(f"{'Config':55s}  {'CV F1':>7}  {'Test F1':>7}")
print("-"*75)

best_cv, best_cfg = 0, None
results = []

for lgb_p, wt, tn in itertools.product(lgb_grid, weights, top_ns):
    cv_scores = []
    for tr_i, val_i in tscv.split(X_tr):
        Xf_tr, Xf_val = X_tr[tr_i], X_tr[val_i]
        yf_tr, yf_val = y_tr[tr_i], y_tr[val_i]
        if len(np.unique(yf_tr)) < 3: continue
        f = stacking_f1(Xf_tr, yf_tr, Xf_val, yf_val, tn, lgb_p, wt)
        cv_scores.append(f)
    cv_f1 = np.mean(cv_scores)
    results.append((cv_f1, lgb_p, wt, tn))
    if cv_f1 > best_cv:
        best_cv, best_cfg = cv_f1, (lgb_p, wt, tn)
        tag = f"n={lgb_p['n_estimators']} d={lgb_p['max_depth']} lr={lgb_p['learning_rate']} mcs={lgb_p['min_child_samples']} L={lgb_p['num_leaves']} w={wt} top={tn}"
        print(f"  ◄ {tag:52s}  {cv_f1:.4f}")

# ── Evaluate best config on test ───────────────────────────────────────────────
best_lgb, best_w, best_n = best_cfg
test_f1 = stacking_f1(X_tr, y_tr, X_te, y_te, best_n, best_lgb, best_w)

print(f"\n{'='*65}")
print(f"Best CV config:")
print(f"  LGB: {best_lgb}")
print(f"  Weight: {best_w}x   Top-N: {best_n}")
print(f"  CV F1  = {best_cv:.4f}")
print(f"  Test F1 = {test_f1:.4f}  (current best validated: 0.4201)")
print(f"{'='*65}")

# Top-5 by CV
print("\nTop-5 configs by CV F1:")
for cv_f1, lp, wt, tn in sorted(results, reverse=True)[:5]:
    tf1 = stacking_f1(X_tr, y_tr, X_te, y_te, tn, lp, wt)
    print(f"  CV={cv_f1:.4f}  Test={tf1:.4f}  n={lp['n_estimators']} d={lp['max_depth']} lr={lp['learning_rate']} mcs={lp['min_child_samples']} L={lp['num_leaves']} w={wt} top={tn}")
