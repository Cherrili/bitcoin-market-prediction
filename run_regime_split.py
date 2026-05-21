"""
Regime-aware training experiments.
  A. Trim to 2017+ (all features available, modern market)
  B. Time-decay sample weights (recent data weighted more)
  C. Manual regime segments: train on bear segments only -> predict bear-heavy test
  D. Walk-forward: train up to year Y, test year Y+1, rolling
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import sys
sys.path.insert(0, '/Users/liuruyan/Desktop/bitcoin-market-prediction')

from src.config import DATA_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import create_labels, build_features

def run_stacking(X_tr, y_tr, X_te, y_te, top_n=40, sample_weight=None):
    from sklearn.utils.class_weight import compute_sample_weight
    sw = sample_weight if sample_weight is not None else compute_sample_weight("balanced", y_tr)

    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    rf_sel.fit(X_tr, y_tr)
    idx = np.argsort(rf_sel.feature_importances_)[::-1][:top_n]
    X_tr_s, X_te_s = X_tr[:, idx], X_te[:, idx]

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_s).astype(np.float32)
    X_te_sc  = sc.transform(X_te_s).astype(np.float32)

    models = [
        ("RF",  RandomForestClassifier(n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=1)),
        ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=1)),
        ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=200, max_depth=5, learning_rate=0.1,
                min_child_samples=50, num_leaves=31,
                random_state=42, n_jobs=1, verbose=-1)),
    ]
    tp = np.zeros((len(X_te_sc), 9))
    for k, (name, m) in enumerate(models):
        if name == "LGB":
            m.fit(X_tr_sc, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr_sc, y_tr)
        tp[:, k*3:(k+1)*3] = m.predict_proba(X_te_sc)

    ww = np.array([1.0, 1.0, 1.5])
    mc = np.array([ww[k] * tp[:, k*3:(k+1)*3].max(axis=1) for k in range(3)])
    bk = mc.argmax(axis=0)
    preds = np.array([np.argmax(tp[i, bk[i]*3:(bk[i]+1)*3]) for i in range(len(tp))])
    return f1_score(y_te, preds, average="macro", zero_division=0)

# Load
print("Loading...")
df_raw = load_and_clean(DATA_DIR)
df = create_labels(df_raw)
df, feature_cols = build_features(df)
df = df.reset_index(drop=True)

X_full = df[feature_cols].values.astype(np.float32)
y_full = df["label_enc"].values if "label_enc" in df.columns else (df["label"].map({-1:0,0:1,1:2}).values)
dates  = pd.to_datetime(df["datetime"].values)

split_idx = int(len(X_full) * 0.8)
X_tr_base, X_te = X_full[:split_idx], X_full[split_idx:]
y_tr_base, y_te = y_full[:split_idx], y_full[split_idx:]
dates_tr = dates[:split_idx]

print(f"Baseline train={len(X_tr_base)}, test={len(X_te)}")

# ── Baseline ───────────────────────────────────────────────────────────────────
print("\nBaseline (all data, 80/20)...")
baseline_f1 = run_stacking(X_tr_base, y_tr_base, X_te, y_te, top_n=40)
print(f"  Baseline F1 = {baseline_f1:.4f}")

# ══ A. Trim to 2017+ ══════════════════════════════════════════════════════════
print("\n" + "="*55)
print("A. Trim training data to 2017+ (modern market, all features)")
print("="*55)
for year in [2015, 2016, 2017, 2018]:
    mask_tr = dates_tr.year >= year
    X_tr_y = X_tr_base[mask_tr]
    y_tr_y = y_tr_base[mask_tr]
    f1 = run_stacking(X_tr_y, y_tr_y, X_te, y_te, top_n=min(40, X_tr_y.shape[1]))
    delta = f1 - baseline_f1
    print(f"  Train from {year}+  n={len(X_tr_y):4d}  F1={f1:.4f}  Δ={delta:+.4f}")

# ══ B. Time-decay sample weights ═════════════════════════════════════════════
print("\n" + "="*55)
print("B. Time-decay sample weights (exponential, recent = heavier)")
print("="*55)
years_tr = dates_tr.year.values
max_year = years_tr.max()
for half_life in [2, 3, 5, 8]:
    age = max_year - years_tr
    decay_w = np.exp(-age * np.log(2) / half_life)
    # combine with class balance
    class_w = compute_sample_weight("balanced", y_tr_base)
    sw = decay_w * class_w
    sw = sw / sw.mean()
    f1 = run_stacking(X_tr_base, y_tr_base, X_te, y_te, top_n=40, sample_weight=sw)
    delta = f1 - baseline_f1
    print(f"  Half-life={half_life}yr  F1={f1:.4f}  Δ={delta:+.4f}")

# ══ C. Bear-segment augmented training ═══════════════════════════════════════
print("\n" + "="*55)
print("C. Oversample historical bear periods in training (2014-15, 2018-19)")
print("="*55)
# Bear years in training: 2014, 2015, 2018, 2019
bear_years = [2014, 2015, 2018, 2019]
bear_mask = np.isin(dates_tr.year.values, bear_years)
for repeat in [2, 3, 5]:
    bear_idx = np.where(bear_mask)[0]
    extra_idx = np.tile(bear_idx, repeat - 1)
    aug_idx = np.concatenate([np.arange(len(X_tr_base)), extra_idx])
    X_aug = X_tr_base[aug_idx]
    y_aug = y_tr_base[aug_idx]
    f1 = run_stacking(X_aug, y_aug, X_te, y_te, top_n=40)
    delta = f1 - baseline_f1
    print(f"  Bear-repeat={repeat}x  train_n={len(X_aug)}  F1={f1:.4f}  Δ={delta:+.4f}")

# ══ D. Walk-forward (train up to Y, test Y+1) ════════════════════════════════
print("\n" + "="*55)
print("D. Walk-forward yearly evaluation")
print("="*55)
wf_results = []
all_years = sorted(dates.year.unique())
for test_year in [2019, 2020, 2021, 2022]:
    train_mask = dates.year < test_year
    test_mask  = dates.year == test_year
    if train_mask.sum() < 200 or test_mask.sum() < 30:
        continue
    X_tr_wf = X_full[train_mask]
    y_tr_wf = y_full[train_mask]
    X_te_wf = X_full[test_mask]
    y_te_wf = y_full[test_mask]
    f1 = run_stacking(X_tr_wf, y_tr_wf, X_te_wf, y_te_wf, top_n=min(40, X_tr_wf.shape[1]))
    wf_results.append((test_year, len(X_tr_wf), len(X_te_wf), f1))
    print(f"  Test {test_year}  train_n={len(X_tr_wf):4d}  test_n={len(X_te_wf):3d}  F1={f1:.4f}")

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"  Baseline (full 80/20):  F1={baseline_f1:.4f}")
if wf_results:
    avg_wf = np.mean([r[3] for r in wf_results])
    print(f"  Walk-forward avg F1:    {avg_wf:.4f}  (train→test year pairs)")
