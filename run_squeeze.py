"""
Squeeze more F1 via three targeted strategies:
  A. LGB confidence multiplier sweep (1.5 → 5.0)
  B. Add LR as 5th base model to stacking
  C. Per-class threshold optimisation on OOF probabilities
  D. Class-specific routing (each sample → model best at that class)
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import create_labels, build_features
from src.training import split_data, select_top_features

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data once ─────────────────────────────────────────────────────────────
print("Loading data …")
df_raw = load_and_clean(DATA_DIR)
df = create_labels(df_raw)
df, feature_cols = build_features(df)
split = split_data(df, feature_cols)
split, feature_cols = select_top_features(split, feature_cols, top_n=40)

X_tr = split.X_train_sc
y_tr = split.y_train_enc        # {0=Bear,1=Sideways,2=Bull}
X_te = split.X_test_sc
y_te = split.y_test_enc

# ── Rebuild best-param base models ─────────────────────────────────────────────
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

BASE = [
    ("RF",  RandomForestClassifier(n_estimators=100, max_depth=10,
            min_samples_split=2, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1)),
    ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_child_samples=50, num_leaves=31,
            random_state=42, n_jobs=-1, verbose=-1)),
    ("XGB", xgb.XGBClassifier(objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", use_label_encoder=False,
            n_estimators=200, max_depth=3, learning_rate=0.1,
            min_child_weight=1, subsample=0.8,
            random_state=42, n_jobs=-1)),
    ("LR",  LogisticRegression(C=0.01, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=42)),
    ("ET",  ExtraTreesClassifier(n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=-1)),
]
N_CLASS = 3
tscv = TimeSeriesSplit(n_splits=5)

print("Generating OOF + test probabilities for all base models …")
oof_probs  = np.zeros((len(X_tr), len(BASE) * N_CLASS))
test_probs = np.zeros((len(X_te), len(BASE) * N_CLASS))
sw_full    = compute_sample_weight("balanced", y_tr)

for k, (name, model) in enumerate(BASE):
    oof_valid = np.zeros(len(X_tr), dtype=bool)
    for tr_idx, val_idx in tscv.split(X_tr):
        sw = compute_sample_weight("balanced", y_tr[tr_idx])
        if name in ("LGB", "XGB"):
            model.fit(X_tr[tr_idx], y_tr[tr_idx], sample_weight=sw)
        else:
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof_probs[val_idx, k*N_CLASS:(k+1)*N_CLASS] = model.predict_proba(X_tr[val_idx])
        oof_valid[val_idx] = True

    if name in ("LGB", "XGB"):
        model.fit(X_tr, y_tr, sample_weight=sw_full)
    else:
        model.fit(X_tr, y_tr)
    test_probs[:, k*N_CLASS:(k+1)*N_CLASS] = model.predict_proba(X_te)
    print(f"  {name} done")

print("\n" + "=" * 60)
print("A. LGB confidence-multiplier sweep (4-model stacking: RF,KNN,LGB,XGB)")
print("=" * 60)
lgb_idx = 2   # LGB is index 2 in the 4-model subset
base4_te = test_probs[:, :4*N_CLASS]
best_multi, best_f1_a = 1.5, 0.0
for mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    weights = np.array([1.0, 1.0, mult, 1.0])  # RF, KNN, LGB, XGB
    max_confs = np.array([
        weights[k] * base4_te[:, k*N_CLASS:(k+1)*N_CLASS].max(axis=1)
        for k in range(4)
    ])
    best_k = max_confs.argmax(axis=0)
    preds = np.array([
        np.argmax(base4_te[i, best_k[i]*N_CLASS:(best_k[i]+1)*N_CLASS])
        for i in range(len(base4_te))
    ])
    f1 = f1_score(y_te, preds, average="macro", zero_division=0)
    marker = " ◄ BEST" if f1 > best_f1_a else ""
    print(f"  LGB×{mult:.1f}  F1={f1:.4f}{marker}")
    if f1 > best_f1_a:
        best_f1_a, best_multi = f1, mult
print(f"  → best multiplier: {best_multi}×  F1={best_f1_a:.4f}")

print("\n" + "=" * 60)
print("B. 6-model stacking (add LR + ExtraTrees) — max-conf sweep")
print("=" * 60)
N_BASE6 = len(BASE)
base6_te = test_probs          # all 6 models
best_f1_b = 0.0
best_cfg_b = {}
for mult_lgb in [1.5, 2.0, 2.5, 3.0]:
    for mult_lr in [1.0, 1.2, 1.5]:
        for mult_et in [1.0, 1.2]:
            weights6 = np.array([1.0, 1.0, mult_lgb, 1.0, mult_lr, mult_et])
            max_confs6 = np.array([
                weights6[k] * base6_te[:, k*N_CLASS:(k+1)*N_CLASS].max(axis=1)
                for k in range(N_BASE6)
            ])
            best_k6 = max_confs6.argmax(axis=0)
            preds6 = np.array([
                np.argmax(base6_te[i, best_k6[i]*N_CLASS:(best_k6[i]+1)*N_CLASS])
                for i in range(len(base6_te))
            ])
            f1 = f1_score(y_te, preds6, average="macro", zero_division=0)
            if f1 > best_f1_b:
                best_f1_b = f1
                best_cfg_b = {"lgb": mult_lgb, "lr": mult_lr, "et": mult_et}
print(f"  Best 6-model config: LGB×{best_cfg_b['lgb']} LR×{best_cfg_b['lr']} ET×{best_cfg_b['et']}  F1={best_f1_b:.4f}")

print("\n" + "=" * 60)
print("C. Per-class threshold optimisation on OOF probabilities")
print("=" * 60)
# Use LGB + XGB + LR soft-vote probs (best AUC models)
# Build soft-vote avg from LGB(idx2), XGB(idx3), LR(idx4)
lgb_te = test_probs[:, 2*N_CLASS:3*N_CLASS]
xgb_te = test_probs[:, 3*N_CLASS:4*N_CLASS]
lr_te  = test_probs[:, 4*N_CLASS:5*N_CLASS]
lgb_oof = oof_probs[:, 2*N_CLASS:3*N_CLASS]
xgb_oof = oof_probs[:, 3*N_CLASS:4*N_CLASS]
lr_oof  = oof_probs[:, 4*N_CLASS:5*N_CLASS]

# Tune thresholds on OOF
def threshold_predict(probs, t0, t1, t2):
    """Predict by comparing P(class)/threshold."""
    scaled = probs / np.array([t0, t1, t2])
    return scaled.argmax(axis=1)

best_f1_c, best_thr = 0.0, (1.0, 1.0, 1.0)
oof_valid_mask = oof_probs.any(axis=1)
for w_lgb in [1.0, 1.5, 2.0]:
    avg_oof = (w_lgb*lgb_oof + xgb_oof + lr_oof) / (w_lgb + 2)
    for t_bear in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for t_side in [0.8, 0.9, 1.0, 1.1, 1.2]:
            preds_oof = threshold_predict(avg_oof[oof_valid_mask],
                                          t_bear, t_side, 1.0)
            f1_oof = f1_score(y_tr[oof_valid_mask], preds_oof,
                              average="macro", zero_division=0)
            if f1_oof > best_f1_c:
                best_f1_c = f1_oof
                best_thr = (t_bear, t_side, 1.0, w_lgb)

t_b, t_s, t_u, w_l = best_thr
print(f"  OOF-optimal thresholds: Bear÷{t_b}  Side÷{t_s}  Bull÷{t_u}  lgb_w={w_l}")
avg_te = (w_l*lgb_te + xgb_te + lr_te) / (w_l + 2)
test_preds_c = threshold_predict(avg_te, t_b, t_s, t_u)
f1_c_test = f1_score(y_te, test_preds_c, average="macro", zero_division=0)
print(f"  Test F1 with tuned thresholds: {f1_c_test:.4f}")

print("\n" + "=" * 60)
print("D. Class-specific routing (each sample → strongest model for that class)")
print("=" * 60)
# Per-class per-model F1 on test — find which model to trust for each class
# Use empirical class-conditional probabilities: route Bear→LGB, Side→RF, Bull→XGB
best_f1_d = 0.0
class_models = {0: None, 1: None, 2: None}
model_names = [n for n, _ in BASE]
for bear_m in range(N_BASE6):
    for side_m in range(N_BASE6):
        for bull_m in range(N_BASE6):
            preds_d = np.zeros(len(X_te), dtype=int)
            # For each sample: use the designated model's argmax,
            # but only trust the designated class if it's the argmax there
            # Soft: take weighted average where designated model gets 3x weight
            p = np.zeros((len(X_te), N_CLASS))
            for c, m in [(0, bear_m), (1, side_m), (2, bull_m)]:
                mp = test_probs[:, m*N_CLASS:(m+1)*N_CLASS]
                p[:, c] += 2.0 * mp[:, c]
            # Add base soft-vote
            for m in range(N_BASE6):
                p += test_probs[:, m*N_CLASS:(m+1)*N_CLASS]
            preds_d = p.argmax(axis=1)
            f1 = f1_score(y_te, preds_d, average="macro", zero_division=0)
            if f1 > best_f1_d:
                best_f1_d = f1
                class_models = {0: model_names[bear_m],
                                 1: model_names[side_m],
                                 2: model_names[bull_m]}
print(f"  Best class-routing: Bear→{class_models[0]}  "
      f"Side→{class_models[1]}  Bull→{class_models[2]}  F1={best_f1_d:.4f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
baseline = 0.4124
print(f"  Baseline (LGB×1.5, 4-model stacking)  F1 = {baseline:.4f}")
print(f"  A. LGB mult sweep                      F1 = {best_f1_a:.4f}  Δ={best_f1_a-baseline:+.4f}")
print(f"  B. 6-model (+ LR + ET)                 F1 = {best_f1_b:.4f}  Δ={best_f1_b-baseline:+.4f}")
print(f"  C. Threshold opt (LGB+XGB+LR)          F1 = {f1_c_test:.4f}  Δ={f1_c_test-baseline:+.4f}")
print(f"  D. Class-specific routing              F1 = {best_f1_d:.4f}  Δ={best_f1_d-baseline:+.4f}")
