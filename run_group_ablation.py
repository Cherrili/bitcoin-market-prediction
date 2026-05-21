"""
Feature group ablation: remove one feature group at a time.
Groups based on the four categories described in Section 3.2:
  A. Rolling statistics (ma7/14/30)
  B. Lag features (lag1/3/7/14/30)
  C. Momentum / price-pct features
  D. On-chain ratio features (mvrv, nupl, fear_greed-derived, etc.)

Also tests removing base feature groups by data source:
  E. Remove fear_greed entirely
  F. Remove lightning network features
  G. Remove all price-derived features (keep only on-chain fundamentals)

For each ablation: fixed pipeline (top-40, RF importance), LGB×1.5 stacking.
"""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from src.config import DATA_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import create_labels, build_features
from src.training import split_data

def max_conf_f1(tp, y_te):
    ww = np.array([1.0, 1.0, 1.5, 1.0])
    mc = np.array([ww[k] * tp[:, k*3:(k+1)*3].max(axis=1) for k in range(4)])
    bk = mc.argmax(axis=0)
    preds = np.array([np.argmax(tp[i, bk[i]*3:(bk[i]+1)*3]) for i in range(len(tp))])
    return f1_score(y_te, preds, average="macro", zero_division=0), preds

def run_stacking(X_tr, y_tr, X_te, y_te, top_n=40):
    sw = compute_sample_weight("balanced", y_tr)

    # Feature selection: RF importance on train only
    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     random_state=42, n_jobs=-1)
    rf_sel.fit(X_tr, y_tr)
    idx = np.argsort(rf_sel.feature_importances_)[::-1][:top_n]
    X_tr_s = X_tr[:, idx]
    X_te_s = X_te[:, idx]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_s).astype(np.float32)
    X_te_sc  = sc.transform(X_te_s).astype(np.float32)

    models = [
        ("RF",  RandomForestClassifier(n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)),
        ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_child_samples=50, num_leaves=31,
                random_state=42, n_jobs=-1, verbose=-1)),
        ("XGB", xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", use_label_encoder=False,
                n_estimators=200, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42, n_jobs=-1)),
    ]
    tp = np.zeros((len(X_te_sc), 12))
    for k, (name, m) in enumerate(models):
        if name in ("LGB", "XGB"):
            m.fit(X_tr_sc, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr_sc, y_tr)
        tp[:, k*3:(k+1)*3] = m.predict_proba(X_te_sc)
    f1, preds = max_conf_f1(tp, y_te)
    return f1, preds

# ── Load base data ─────────────────────────────────────────────────────────────
print("Loading data …")
df_raw = load_and_clean(DATA_DIR)
df = create_labels(df_raw)
df, feature_cols = build_features(df)

sp = split_data(df, feature_cols)
X_tr_full = sp.X_train_sc   # already scaled by split_data, but we re-select below
y_tr = sp.y_train_enc
y_te = sp.y_test_enc

# Use raw (unscaled) arrays for group masking
X_full = df[feature_cols].values
split_idx = int(len(X_full) * 0.8)
X_tr_raw = X_full[:split_idx].astype(np.float32)
X_te_raw = X_full[split_idx:].astype(np.float32)

# ── Define feature groups ──────────────────────────────────────────────────────
def cols_matching(patterns):
    result = []
    for i, c in enumerate(feature_cols):
        if any(p in c for p in patterns):
            result.append(i)
    return result

groups = {
    "Rolling stats (ma7/14/30)":   cols_matching(["_ma7", "_ma14", "_ma30"]),
    "Lag features (lag1-30)":       cols_matching(["_lag1", "_lag3", "_lag7", "_lag14", "_lag30"]),
    "Price momentum (pct/vol)":     cols_matching(["price_pct_", "price_volatility"]),
    "MVRV ratio":                   cols_matching(["mvrv"]),
    "Fear & Greed":                 cols_matching(["fear_greed"]),
    "Lightning network":            cols_matching(["lightning"]),
    "Hash rate & Difficulty":       cols_matching(["hash_rate", "difficulty"]),
    "Active addresses":             cols_matching(["active_addresses"]),
}

# ── Baseline (all features) ────────────────────────────────────────────────────
print("\nRunning baseline (all features) …")
baseline_f1, _ = run_stacking(X_tr_raw, y_tr, X_te_raw, y_te, top_n=40)
print(f"  Baseline F1 = {baseline_f1:.4f}")

# ── Group ablation ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("Feature group ablation (remove one group at a time, top-40)")
print("="*65)
print(f"  {'Group removed':35s}  {'#cols':>5}  {'F1':>7}  {'ΔF1':>8}  Impact")
print("  " + "-"*62)

results = [("Baseline (all features)", len(feature_cols), baseline_f1, 0.0)]

for group_name, col_idx in groups.items():
    if not col_idx:
        print(f"  {group_name:35s}  (no matching columns)")
        continue
    keep = [i for i in range(len(feature_cols)) if i not in col_idx]
    X_tr_g = X_tr_raw[:, keep]
    X_te_g = X_te_raw[:, keep]
    f1, _ = run_stacking(X_tr_g, y_tr, X_te_g, y_te, top_n=min(40, len(keep)))
    delta = f1 - baseline_f1
    impact = ("↑ helpful" if delta < -0.01 else
              ("↓ harmful" if delta > 0.01 else "~ neutral"))
    print(f"  {group_name:35s}  {len(col_idx):>5}  {f1:.4f}  {delta:>+.4f}  {impact}")
    results.append((f"Remove {group_name}", len(keep), f1, delta))

# ── Only-group experiments (keep only one group + base features) ───────────────
print("\n" + "="*65)
print("Keep-only experiments (base cols + one feature group)")
print("="*65)
BASE_COLS = [i for i, c in enumerate(feature_cols)
             if not any(x in c for x in
                        ["_ma", "_lag", "price_pct_", "price_volatility",
                         "mvrv", "fear_greed", "lightning"])]
print(f"  Base (raw) feature count: {len(BASE_COLS)}")
print(f"  {'Group kept':35s}  {'#cols':>5}  {'F1':>7}  {'ΔF1':>8}")
print("  " + "-"*56)

for group_name, col_idx in groups.items():
    if not col_idx:
        continue
    keep = sorted(set(BASE_COLS) | set(col_idx))
    X_tr_g = X_tr_raw[:, keep]
    X_te_g = X_te_raw[:, keep]
    f1, _ = run_stacking(X_tr_g, y_tr, X_te_g, y_te, top_n=min(40, len(keep)))
    delta = f1 - baseline_f1
    print(f"  Base + {group_name:29s}  {len(keep):>5}  {f1:.4f}  {delta:>+.4f}")

print("\n" + "="*65)
print(f"Baseline (all features, top-40): F1 = {baseline_f1:.4f}")
print("="*65)
