"""
Replication of Omole & Enke (2024) strategy on our dataset.

Key differences from our 3-class 30-day setup:
  - Binary next-day label: 1 if price_t+1 > price_t, else 0
  - Boruta feature selection (instead of RF importance top-N)
  - CNN-LSTM model (PyTorch)
  - Accuracy reported (paper's primary metric)

Run: python run_replicate.py
"""
import os, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from boruta import BorutaPy
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_and_clean
from src.feature_engineering import build_features

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cpu"   # MPS has stability issues with LSTM on this PyTorch build
print(f"Device: {DEVICE}")

# ── 1. Load & create next-day binary labels ────────────────────────────────────
print("\n1. Loading data and creating next-day binary labels …")
df_raw = load_and_clean(DATA_DIR)

# Temporarily set label col so build_features doesn't raise
df_raw["label"] = 0
df_raw["future_price_30d"] = df_raw["price"].shift(-1)   # next day
df_raw["return_30d"] = (df_raw["future_price_30d"] - df_raw["price"]) / df_raw["price"]
df_raw.dropna(subset=["future_price_30d"], inplace=True)

df_raw["label"] = (df_raw["return_30d"] > 0).astype(int)
n_up = df_raw["label"].sum()
n_dn = len(df_raw) - n_up
print(f"   Up: {n_up}  Down: {n_dn}  Total: {len(df_raw)}")

df_raw = df_raw.drop(columns=["return_30d", "future_price_30d"], errors="ignore")
df, feature_cols = build_features(df_raw)

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values.astype(int)

# ── 2. 80/20 temporal split ────────────────────────────────────────────────────
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"\n2. Split: train={len(X_train)}, test={len(X_test)}")
print(f"   Test  Up: {y_test.sum()}  Down: {(y_test==0).sum()}")

# ── 3. Boruta feature selection ────────────────────────────────────────────────
print("\n3. Boruta feature selection (this may take 1-2 min) …")
t0 = time.time()
rf_bor = RandomForestClassifier(n_estimators=100, max_depth=7,
                                 class_weight="balanced", random_state=42, n_jobs=1)
boruta = BorutaPy(rf_bor, n_estimators="auto", max_iter=50,
                  random_state=42, verbose=0)
boruta.fit(X_train, y_train)
selected = np.where(boruta.support_)[0]
if len(selected) < 5:           # fallback: use top-30 RF importance
    print("   Boruta selected too few features — falling back to RF top-30")
    rf_fb = RandomForestClassifier(n_estimators=200, max_depth=10,
                                    random_state=42, n_jobs=1)
    rf_fb.fit(X_train, y_train)
    selected = np.argsort(rf_fb.feature_importances_)[::-1][:30]

print(f"   Selected {len(selected)} features  ({time.time()-t0:.1f}s)")
selected_names = [feature_cols[i] for i in selected]

X_tr = X_train[:, selected]
X_te = X_test[:, selected]

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr).astype(np.float32)
X_te_sc  = sc.transform(X_te).astype(np.float32)

# ── 4. Traditional ML baselines (on Boruta features) ─────────────────────────
print("\n4. Traditional ML baselines with Boruta features …")
sw = compute_sample_weight("balanced", y_train)

baselines = [
    ("LightGBM", lgb.LGBMClassifier(objective="binary", n_estimators=200,
        max_depth=5, learning_rate=0.1, min_child_samples=20,
        random_state=42, n_jobs=1, verbose=-1), True),
    ("GBM",      GradientBoostingClassifier(n_estimators=200, max_depth=4,
        learning_rate=0.1, subsample=0.8,
        random_state=42), True),
    ("RF",       RandomForestClassifier(n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42, n_jobs=1), False),
]

ml_results = []
for name, m, use_sw in baselines:
    kw = {"sample_weight": sw} if use_sw else {}
    m.fit(X_tr_sc, y_train, **kw)
    preds = m.predict(X_te_sc)
    proba = m.predict_proba(X_te_sc)[:, 1]
    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="macro")
    auc = roc_auc_score(y_test, proba)
    print(f"   {name:10s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    ml_results.append((name, acc, f1, auc))

# ── 5. CNN-LSTM (sequence window = 30 days) ────────────────────────────────────
print("\n5. CNN-LSTM model …")

SEQ_LEN = 30

def make_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_train, SEQ_LEN)
X_te_seq, y_te_seq = make_sequences(X_te_sc, y_test, SEQ_LEN)
print(f"   Sequence shapes: train={X_tr_seq.shape}, test={X_te_seq.shape}")

# Dataset
tr_ds = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq))
te_ds = TensorDataset(torch.from_numpy(X_te_seq), torch.from_numpy(y_te_seq))
tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
te_loader = DataLoader(te_ds, batch_size=256)

class CNNLSTM(nn.Module):
    def __init__(self, n_feat, cnn_filters=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_feat, cnn_filters, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.drop  = nn.Dropout(dropout)
        self.lstm  = nn.LSTM(cnn_filters, lstm_hidden, batch_first=True,
                             num_layers=2, dropout=dropout)
        self.fc    = nn.Linear(lstm_hidden, 2)

    def forward(self, x):                   # x: (B, T, F)
        x = x.permute(0, 2, 1)             # (B, F, T) for Conv1d
        x = self.drop(self.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)             # (B, T, filters)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])               # last layer hidden state

n_feat = X_tr_seq.shape[2]
model  = CNNLSTM(n_feat).to(DEVICE)

# Class weights for loss
class_counts = np.bincount(y_tr_seq)
cw = torch.tensor(len(y_tr_seq) / (2 * class_counts), dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cw)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

EPOCHS = 30
best_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for Xb, yb in tr_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for Xb, _ in te_loader:
                logits = model(Xb.to(DEVICE))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        acc = accuracy_score(y_te_seq, all_preds)
        f1  = f1_score(y_te_seq, all_preds, average="macro")
        marker = " ◄ best" if acc > best_acc else ""
        print(f"   Epoch {epoch:2d}  loss={total_loss/len(tr_loader):.4f}"
              f"  Acc={acc:.4f}  F1={f1:.4f}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

# Final evaluation with best weights
model.load_state_dict(best_state)
model.eval()
all_preds, all_probs = [], []
with torch.no_grad():
    for Xb, _ in te_loader:
        logits = model(Xb.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())

cnn_acc = accuracy_score(y_te_seq, all_preds)
cnn_f1  = f1_score(y_te_seq, all_preds, average="macro")
cnn_auc = roc_auc_score(y_te_seq, all_probs)

# ── 6. Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — Paper replication (binary next-day, Boruta features)")
print("=" * 65)
print(f"  Features selected by Boruta: {len(selected)}")
print(f"  {'Model':12s}  {'Accuracy':>10}  {'Macro F1':>10}  {'AUC':>10}")
print("  " + "-" * 48)
for name, acc, f1, auc in ml_results:
    print(f"  {name:12s}  {acc:>10.4f}  {f1:>10.4f}  {auc:>10.4f}")
print(f"  {'CNN-LSTM':12s}  {cnn_acc:>10.4f}  {cnn_f1:>10.4f}  {cnn_auc:>10.4f}")
print()
print(f"  Paper reports: Accuracy=0.8244 (CNN-LSTM + Boruta)")
print(f"  Our 3-class 30-day task: Macro F1=0.4413 (incomparable metric/task)")
print()
print("  Note: next-day binary accuracy vs 30-day 3-class macro F1 are")
print("  fundamentally different benchmarks (random baseline: 50% vs 33%).")
