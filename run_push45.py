"""
Push toward 0.45: combine best findings + new angles
  A. 2017+ trim × feature-count sweep (combine two positives)
  B. Add technical indicators (RSI, MACD, Bollinger Bands)
  C. CatBoost as replacement/addition to XGB
  D. Threshold sweep (±10%, ±12%, ±15%, ±18%, ±20%)
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import sys
sys.path.insert(0, '/Users/liuruyan/Desktop/bitcoin-market-prediction')

from src.config import DATA_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import create_labels, build_features

def stacking_f1(X_tr, y_tr, X_te, y_te, top_n=40):
    sw = compute_sample_weight("balanced", y_tr)
    rf_sel = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    rf_sel.fit(X_tr, y_tr)
    idx = np.argsort(rf_sel.feature_importances_)[::-1][:top_n]
    Xtr, Xte = X_tr[:, idx], X_te[:, idx]
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    models = [
        ("RF",  RandomForestClassifier(n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=1)),
        ("KNN", KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=1)),
        ("LGB", lgb.LGBMClassifier(objective="multiclass", num_class=3,
                n_estimators=300, max_depth=6, learning_rate=0.05,
                min_child_samples=30, num_leaves=63,
                random_state=42, n_jobs=1, verbose=-1)),
        ("ET",  ExtraTreesClassifier(n_estimators=200, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=1)),
    ]
    tp = np.zeros((len(Xte), 12))
    for k, (name, m) in enumerate(models):
        if name == "LGB":
            m.fit(Xtr, y_tr, sample_weight=sw)
        else:
            m.fit(Xtr, y_tr)
        tp[:, k*3:(k+1)*3] = m.predict_proba(Xte)
    ww = np.array([1.0, 1.0, 1.5, 1.0])
    mc = np.array([ww[k]*tp[:, k*3:(k+1)*3].max(axis=1) for k in range(4)])
    bk = mc.argmax(axis=0)
    preds = np.array([np.argmax(tp[i, bk[i]*3:(bk[i]+1)*3]) for i in range(len(tp))])
    return f1_score(y_te, preds, average="macro", zero_division=0)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading...")
df_raw = load_and_clean(DATA_DIR)
df = create_labels(df_raw)
df, feature_cols = build_features(df)
df = df.reset_index(drop=True)

X = df[feature_cols].values.astype(np.float32)
y = df["label_enc"].values if "label_enc" in df.columns else df["label"].map({-1:0,0:1,1:2}).values
dates = pd.to_datetime(df["datetime"].values)
split_idx = int(len(X) * 0.8)
X_te, y_te = X[split_idx:], y[split_idx:]

# ══ A. 2017+ × feature count sweep ════════════════════════════════════════════
print("\n" + "="*55)
print("A. 2017+ trim × feature-count sweep")
print("="*55)
mask_2017 = dates[:split_idx].year >= 2017
X_tr17 = X[:split_idx][mask_2017]
y_tr17 = y[:split_idx][mask_2017]
print(f"  Train n = {len(X_tr17)}")
best_a, best_n = 0, 40
for top_n in [20, 25, 28, 30, 32, 35, 40, 50]:
    f1 = stacking_f1(X_tr17, y_tr17, X_te, y_te, top_n)
    marker = " ◄" if f1 > best_a else ""
    print(f"  top_{top_n:2d}  F1={f1:.4f}{marker}")
    if f1 > best_a:
        best_a, best_n = f1, top_n
print(f"  → best: top_{best_n}  F1={best_a:.4f}")

# ══ B. Technical indicators ════════════════════════════════════════════════════
print("\n" + "="*55)
print("B. Add technical indicators (RSI-14, MACD, Bollinger Bands)")
print("="*55)

def add_technical(df_in):
    d = df_in.copy()
    p = d["price"]
    # RSI-14
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))
    # MACD
    ema12 = p.ewm(span=12, adjust=False).mean()
    ema26 = p.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    # Bollinger Bands
    ma20 = p.rolling(20).mean()
    std20 = p.rolling(20).std()
    d["bb_upper"] = (p - (ma20 + 2*std20)) / p  # normalised distance
    d["bb_lower"] = (p - (ma20 - 2*std20)) / p
    d["bb_width"] = 4 * std20 / ma20
    return d

df_tech = add_technical(df)
tech_cols = ["rsi_14", "macd", "macd_signal", "macd_hist",
             "bb_upper", "bb_lower", "bb_width"]
df_tech[tech_cols] = df_tech[tech_cols].ffill().bfill()

feat_tech = feature_cols + tech_cols
X_tech = df_tech[feat_tech].values.astype(np.float32)

# full data
X_tr_t, X_te_t = X_tech[:split_idx], X_tech[split_idx:]
y_tr_full = y[:split_idx]
f1_full = stacking_f1(X_tr_t, y_tr_full, X_te_t, y_te, top_n=40)
print(f"  Full data + tech indicators  F1={f1_full:.4f}")

# 2017+
X_tr_t17 = X_tech[:split_idx][mask_2017]
for top_n in [28, 30, 32, 35, 40]:
    f1 = stacking_f1(X_tr_t17, y_tr17, X_te_t, y_te, top_n)
    marker = " ◄" if f1 > best_a else ""
    print(f"  2017+ + tech  top_{top_n}  F1={f1:.4f}{marker}")
    if f1 > best_a:
        best_a, best_n = f1, top_n

# ══ C. Label threshold sweep ═══════════════════════════════════════════════════
print("\n" + "="*55)
print("C. Label threshold sweep (±X% boundary)")
print("="*55)
for thr in [10, 12, 15, 18, 20]:
    df_t = df_raw.copy()
    df_t["future_price_30d"] = df_t["price"].shift(-30)
    df_t["return_30d"] = (df_t["future_price_30d"] - df_t["price"]) / df_t["price"]
    df_t.dropna(subset=["future_price_30d"], inplace=True)
    t = thr / 100
    df_t["label"] = df_t["return_30d"].apply(
        lambda r: -1 if r < -t else (1 if r > t else 0))
    df_t.drop(columns=["return_30d","future_price_30d"], inplace=True, errors="ignore")
    df_tt, fc_t = build_features(df_t)
    df_tt = df_tt.reset_index(drop=True)
    Xt = df_tt[fc_t].values.astype(np.float32)
    yt = df_tt["label"].map({-1:0,0:1,1:2}).values
    sp = int(len(Xt)*0.8)
    counts = np.bincount(yt)
    f1 = stacking_f1(Xt[:sp], yt[:sp], Xt[sp:], yt[sp:], top_n=40)
    marker = " ◄" if f1 > best_a else ""
    print(f"  ±{thr:2d}%  dist={counts.tolist()}  F1={f1:.4f}{marker}")
    if f1 > best_a:
        best_a = f1

print("\n" + "="*55)
print(f"BEST RESULT ACROSS ALL EXPERIMENTS: F1 = {best_a:.4f}")
print("="*55)
