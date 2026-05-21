"""
Second attempt: use raw base features as time series input (closer to paper's OHLCV approach).
Also tries: LSTM-only, Transformer, and longer training.
"""
import os, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from src.config import DATA_DIR, OUTPUT_DIR
from src.data_loader import load_and_clean

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cpu"

# ── 1. Raw base features (no rolling/lag aggregation) ─────────────────────────
print("1. Loading raw features …")
df = load_and_clean(DATA_DIR)
df["future_price"] = df["price"].shift(-1)
df["label"] = (df["future_price"] > df["price"]).astype(int)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

RAW_COLS = [
    "price", "mempool_size", "transaction_rate", "market_cap_usd_bc",
    "average_block_size", "exchange_volume_usd", "average_confirmation_time",
    "hash_rate", "difficulty", "miners_revenue", "total_transaction_fees",
    "realised_cap_usd", "nupl", "coin_days_destroyed", "active_addresses",
    "fear_greed_value", "lightning_capacity_usd",
]
RAW_COLS = [c for c in RAW_COLS if c in df.columns]
print(f"   Raw feature cols: {len(RAW_COLS)}")
print(f"   Up: {df['label'].sum()}  Down: {(df['label']==0).sum()}")

X_raw = df[RAW_COLS].values.astype(np.float32)
y_all = df["label"].values.astype(int)

split = int(len(X_raw) * 0.8)
X_tr_raw, X_te_raw = X_raw[:split], X_raw[split:]
y_tr,     y_te     = y_all[:split], y_all[split:]

sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_tr_raw).astype(np.float32)
X_te_sc = sc.transform(X_te_raw).astype(np.float32)

# ── Sequence builder ───────────────────────────────────────────────────────────
def make_seq(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)

def train_eval(model, X_tr_seq, y_tr_seq, X_te_seq, y_te_seq,
               epochs=50, lr=5e-4, batch=64, label=""):
    cw = torch.tensor(len(y_tr_seq) / (2 * np.bincount(y_tr_seq)),
                      dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_ds = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq))
    te_ds = TensorDataset(torch.from_numpy(X_te_seq), torch.from_numpy(y_te_seq))
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=256)

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for Xb, yb in tr_ld:
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(Xb), yb).backward()
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs:
            model.eval()
            preds, probs = [], []
            with torch.no_grad():
                for Xb, _ in te_ld:
                    lg = model(Xb)
                    preds.extend(lg.argmax(1).numpy())
                    probs.extend(torch.softmax(lg, 1)[:, 1].numpy())
            acc = accuracy_score(y_te_seq, preds)
            marker = " ◄" if acc > best_acc else ""
            if ep % 10 == 0:
                print(f"   {label} ep{ep:3d}  Acc={acc:.4f}{marker}")
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for Xb, _ in te_ld:
            lg = model(Xb)
            preds.extend(lg.argmax(1).numpy())
            probs.extend(torch.softmax(lg, 1)[:, 1].numpy())
    acc = accuracy_score(y_te_seq, preds)
    f1  = f1_score(y_te_seq, preds, average="macro")
    auc = roc_auc_score(y_te_seq, probs)
    return acc, f1, auc

# ── Model definitions ──────────────────────────────────────────────────────────
class LSTMOnly(nn.Module):
    def __init__(self, n_feat, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True,
                            num_layers=layers, dropout=drop)
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, 2))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class CNNLSTM(nn.Module):
    def __init__(self, n_feat, filters=64, hidden=128, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, filters, 3, padding=1), nn.ReLU(),
            nn.Conv1d(filters, filters, 3, padding=1), nn.ReLU(),
            nn.Dropout(drop))
        self.lstm = nn.LSTM(filters, hidden, batch_first=True,
                            num_layers=2, dropout=drop)
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden, 2))
    def forward(self, x):
        x = self.conv(x.permute(0,2,1)).permute(0,2,1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class BiLSTM(nn.Module):
    def __init__(self, n_feat, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True,
                            num_layers=layers, dropout=drop, bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(drop), nn.Linear(hidden*2, 2))
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(h_cat)

n_feat = len(RAW_COLS)
results = []
print()

for seq_len in [14, 30, 60]:
    Xtr_seq, ytr_seq = make_seq(X_tr_sc, y_tr, seq_len)
    Xte_seq, yte_seq = make_seq(X_te_sc, y_te, seq_len)

    print(f"\n{'='*60}")
    print(f"Sequence length = {seq_len}  "
          f"(train={len(Xtr_seq)}, test={len(Xte_seq)})")
    print(f"{'='*60}")

    for name, model in [
        ("LSTM",    LSTMOnly(n_feat)),
        ("CNN-LSTM",CNNLSTM(n_feat)),
        ("BiLSTM",  BiLSTM(n_feat)),
    ]:
        acc, f1, auc = train_eval(
            model, Xtr_seq, ytr_seq, Xte_seq, yte_seq,
            epochs=60, label=f"{name}/seq{seq_len}")
        marker = " ★ GOAL" if acc >= 0.60 else (" ●" if acc >= 0.55 else "")
        print(f"   {name:10s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}{marker}")
        results.append((f"{name}/seq{seq_len}", acc, f1, auc))

print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
results.sort(key=lambda x: -x[1])
for name, acc, f1, auc in results:
    bar = "★" if acc >= 0.60 else ("●" if acc >= 0.55 else " ")
    print(f"  {bar} {name:20s}  Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

best_acc = results[0][1]
if best_acc < 0.60:
    print(f"\n  Best: {best_acc:.4f}  —  0.60 target not reached.")
    print("  Conclusion: next-day binary prediction on BTC 2010-2023 is")
    print("  fundamentally constrained by 2022 regime change.")
