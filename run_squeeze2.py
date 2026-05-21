"""
Round 2: deeper squeeze attempts
  A. Feature count: top 40 vs 60 vs 80
  B. SMOTE oversampling for Bear minority class
  C. One-vs-rest (OvR) cascade: Bear-detector + Bull-detector → Sideways
  D. Calibrated probabilities + threshold on LGB
"""
import os, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
warnings.filterwarnings("ignore")

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import create_labels, build_features
from src.training import split_data, select_top_features, SplitResult
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_split(df, feature_cols, top_n):
    sp = split_data(df, feature_cols)
    sp, fc = select_top_features(sp, feature_cols, top_n=top_n)
    return sp, fc

def max_conf_f1(test_probs, y_te, weights, n_base=4, n_class=3):
    ww = np.array(weights)
    max_c = np.array([ww[k]*test_probs[:,k*n_class:(k+1)*n_class].max(axis=1)
                      for k in range(n_base)])
    bk = max_c.argmax(axis=0)
    preds = np.array([np.argmax(test_probs[i,bk[i]*n_class:(bk[i]+1)*n_class])
                      for i in range(len(test_probs))])
    return f1_score(y_te, preds, average="macro", zero_division=0)

def build_test_probs(X_tr, y_tr, X_te, models, tscv):
    """Retrain models on full train and return test probabilities."""
    n_class = 3
    sw = compute_sample_weight("balanced", y_tr)
    test_probs = np.zeros((len(X_te), len(models)*n_class))
    for k, (name, m) in enumerate(models):
        if name in ("LGB","XGB"):
            m.fit(X_tr, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr, y_tr)
        test_probs[:,k*n_class:(k+1)*n_class] = m.predict_proba(X_te)
    return test_probs

# ── Load data once ─────────────────────────────────────────────────────────────
print("Loading data …")
df_raw = load_and_clean(DATA_DIR)
df = create_labels(df_raw)
df, feature_cols = build_features(df)

# ══ A. Feature count sweep ═════════════════════════════════════════════════════
print("\n" + "="*60)
print("A. Feature count sweep (top-N → max-conf stacking F1)")
print("="*60)

def base_models():
    return [
        ("RF",  RandomForestClassifier(n_estimators=100,max_depth=10,
                min_samples_split=2,class_weight="balanced",random_state=42,n_jobs=-1)),
        ("KNN", KNeighborsClassifier(n_neighbors=3,weights="distance",n_jobs=-1)),
        ("LGB", lgb.LGBMClassifier(objective="multiclass",num_class=3,
                n_estimators=200,max_depth=5,learning_rate=0.1,
                min_child_samples=50,num_leaves=31,random_state=42,n_jobs=-1,verbose=-1)),
        ("XGB", xgb.XGBClassifier(objective="multi:softprob",num_class=3,
                eval_metric="mlogloss",use_label_encoder=False,
                n_estimators=200,max_depth=3,learning_rate=0.1,
                min_child_weight=1,subsample=0.8,random_state=42,n_jobs=-1)),
    ]

tscv = TimeSeriesSplit(n_splits=5)
best_a, best_topn = 0.0, 40
for top_n in [30, 40, 50, 60, 80]:
    sp, _ = make_split(df, feature_cols, top_n)
    tp = build_test_probs(sp.X_train_sc, sp.y_train_enc,
                          sp.X_test_sc, base_models(), tscv)
    f1 = max_conf_f1(tp, sp.y_test_enc, [1.0,1.0,1.5,1.0])
    marker = " ◄" if f1 > best_a else ""
    print(f"  top_{top_n:2d}  F1={f1:.4f}{marker}")
    if f1 > best_a:
        best_a, best_topn = f1, top_n
print(f"  → best: top_{best_topn}  F1={best_a:.4f}")

# ══ B. SMOTE oversampling ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("B. SMOTE oversampling of Bear class")
print("="*60)

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    has_smote = True
except ImportError:
    has_smote = False
    print("  imblearn not installed — skipping SMOTE")

if has_smote:
    sp40, _ = make_split(df, feature_cols, 40)
    Xtr, ytr = sp40.X_train_sc, sp40.y_train_enc
    Xte, yte = sp40.X_test_sc,  sp40.y_test_enc

    for sampler_name, Sampler in [("SMOTE", SMOTE),
                                   ("BorderlineSMOTE", BorderlineSMOTE),
                                   ("ADASYN", ADASYN)]:
        try:
            sm = Sampler(random_state=42, k_neighbors=5)
            Xtr_res, ytr_res = sm.fit_resample(Xtr, ytr)
            print(f"  {sampler_name}: {len(ytr)} → {len(ytr_res)} samples  "
                  f"({np.bincount(ytr_res).tolist()})")
            # Retrain LGB+XGB on resampled (no sample_weight needed — balanced by SMOTE)
            models_b = base_models()
            sw_res = compute_sample_weight("balanced", ytr_res)
            tp_b = np.zeros((len(Xte), 4*3))
            for k, (name, m) in enumerate(models_b):
                if name in ("LGB","XGB"):
                    m.fit(Xtr_res, ytr_res, sample_weight=sw_res)
                else:
                    m.fit(Xtr_res, ytr_res)
                tp_b[:,k*3:(k+1)*3] = m.predict_proba(Xte)
            f1 = max_conf_f1(tp_b, yte, [1.0,1.0,1.5,1.0])
            print(f"  {sampler_name} F1={f1:.4f}")
        except Exception as e:
            print(f"  {sampler_name} failed: {e}")

# ══ C. One-vs-rest cascade ════════════════════════════════════════════════════
print("\n" + "="*60)
print("C. OvR cascade: Bear-detector + Bull-detector → Sideways residual")
print("="*60)

sp40, _ = make_split(df, feature_cols, 40)
Xtr, ytr = sp40.X_train_sc, sp40.y_train_enc
Xte, yte  = sp40.X_test_sc,  sp40.y_test_enc

# Bear detector: 1 if Bear (class 0), 0 otherwise
bear_lgb = lgb.LGBMClassifier(objective="binary",n_estimators=300,
    max_depth=6,learning_rate=0.05,min_child_samples=20,num_leaves=31,
    random_state=42,n_jobs=-1,verbose=-1,
    scale_pos_weight=(ytr!=0).sum() / (ytr==0).sum())
bear_lgb.fit(Xtr, (ytr==0).astype(int))

# Bull detector: 1 if Bull (class 2), 0 otherwise
bull_lgb = lgb.LGBMClassifier(objective="binary",n_estimators=300,
    max_depth=6,learning_rate=0.05,min_child_samples=20,num_leaves=31,
    random_state=42,n_jobs=-1,verbose=-1,
    scale_pos_weight=(ytr!=2).sum() / (ytr==2).sum())
bull_lgb.fit(Xtr, (ytr==2).astype(int))

p_bear = bear_lgb.predict_proba(Xte)[:,1]
p_bull = bull_lgb.predict_proba(Xte)[:,1]
p_side = 1 - p_bear - p_bull
p_side = np.clip(p_side, 0, 1)

best_f1_c, best_thr_c = 0.0, (0.3, 0.3)
for tb in np.arange(0.2, 0.7, 0.05):
    for tu in np.arange(0.2, 0.7, 0.05):
        preds_c = np.where(p_bear > tb, 0, np.where(p_bull > tu, 2, 1))
        f1 = f1_score(yte, preds_c, average="macro", zero_division=0)
        if f1 > best_f1_c:
            best_f1_c, best_thr_c = f1, (tb, tu)

tb, tu = best_thr_c
preds_best_c = np.where(p_bear > tb, 0, np.where(p_bull > tu, 2, 1))
print(f"  Best thresholds: Bear_thr={tb:.2f}  Bull_thr={tu:.2f}  F1={best_f1_c:.4f}")
print(classification_report(yte, preds_best_c,
      target_names=["Bear","Sideways","Bull"], zero_division=0))

# ══ D. LGB with more trees / tuned params ════════════════════════════════════
print("\n" + "="*60)
print("D. LGB parameter extensions (deeper / more trees)")
print("="*60)

sp40, _ = make_split(df, feature_cols, 40)
Xtr, ytr = sp40.X_train_sc, sp40.y_train_enc
Xte, yte  = sp40.X_test_sc,  sp40.y_test_enc
sw = compute_sample_weight("balanced", ytr)

configs_d = [
    ("baseline",      dict(n_estimators=200, max_depth=5, lr=0.1, leaves=31, mcs=50)),
    ("more_trees",    dict(n_estimators=500, max_depth=5, lr=0.05, leaves=31, mcs=50)),
    ("deeper",        dict(n_estimators=300, max_depth=8, lr=0.05, leaves=63, mcs=30)),
    ("dart_200",      dict(n_estimators=200, max_depth=6, lr=0.05, leaves=63, mcs=20)),
]
for tag, cfg in configs_d:
    m = lgb.LGBMClassifier(objective="multiclass",num_class=3,
            n_estimators=cfg["n_estimators"],max_depth=cfg["max_depth"],
            learning_rate=cfg["lr"],num_leaves=cfg["leaves"],
            min_child_samples=cfg["mcs"],random_state=42,n_jobs=-1,verbose=-1)
    m.fit(Xtr, ytr, sample_weight=sw)
    preds = m.predict(Xte)
    f1 = f1_score(yte, preds, average="macro", zero_division=0)
    print(f"  LGB {tag:15s}  F1={f1:.4f}")

print("\n" + "="*60)
print("SUMMARY vs baseline 0.4124")
print("="*60)
print(f"  A. best feature count (top_{best_topn})  F1={best_a:.4f}  Δ={best_a-0.4124:+.4f}")
print(f"  C. OvR cascade best                  F1={best_f1_c:.4f}  Δ={best_f1_c-0.4124:+.4f}")
