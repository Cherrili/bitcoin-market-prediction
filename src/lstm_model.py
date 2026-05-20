"""
LSTM-based Bitcoin market state classifier.

Converts the tabular feature matrix into sliding-window sequences so the
model can capture temporal dependencies that the tree/linear models miss.

Architecture
------------
    Input  : (batch, window=30, features=40)
    LSTM   : 2 layers, hidden=128, dropout=0.3
    FC     : Linear(128 → 3)
    Loss   : CrossEntropy with class weights

Public API
----------
train_lstm(X_train_sc, y_train_enc, *, window, n_epochs, verbose)
    -> (LSTMClassifier, history_dict)

predict_lstm(model, X_sc, window)
    -> np.ndarray  (encoded labels {0,1,2})

evaluate_lstm(model, X_test_sc, y_test_enc, y_test, output_dir, window)
    -> dict  {accuracy, f1_macro, roc_auc}
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize
import seaborn as sns


# ── Dataset ───────────────────────────────────────────────────────────────────

class _SeqDataset(Dataset):
    """Sliding-window sequences over a pre-scaled feature matrix."""

    def __init__(self, X: np.ndarray, y: np.ndarray, window: int) -> None:
        self.X      = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y      = torch.tensor(np.ascontiguousarray(y), dtype=torch.long)
        self.window = window
        self.indices = list(range(window, len(X)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        i   = self.indices[idx]
        seq = self.X[i - self.window : i]   # (window, features)
        lbl = self.y[i]
        return seq, lbl


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Two-layer bidirectional LSTM with residual connection on the
    hidden state, followed by a classification head.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout:    float = 0.3,
        n_classes:  int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        lstm_out_dim = hidden_dim
        self.norm  = nn.LayerNorm(lstm_out_dim)
        self.drop  = nn.Dropout(dropout)
        self.fc    = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        out, _ = self.lstm(x)           # (batch, window, hidden*2)
        last   = out[:, -1, :]          # take last time-step
        last   = self.norm(last)
        last   = self.drop(last)
        return self.fc(last)            # (batch, 3)


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(
    X_train_sc:  np.ndarray,
    y_train_enc: np.ndarray,
    *,
    window:      int   = 30,
    hidden_dim:  int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    lr:          float = 5e-4,
    batch_size:  int   = 64,
    n_epochs:    int   = 60,
    patience:    int   = 10,
    val_frac:    float = 0.15,
    seed:        int   = None,
    verbose:     bool  = True,
) -> tuple:
    """
    Train the LSTM classifier on scaled training data.

    A chronological validation split (last val_frac of the training set)
    is used for early stopping — no random shuffling.

    Returns
    -------
    model   : best LSTMClassifier (lowest val-loss checkpoint)
    history : dict with lists 'train_loss', 'val_loss', 'val_f1'
    """
    # MPS has known issues with LSTM — use CPU for stability
    # torch.set_num_threads(1) prevents deadlock after joblib n_jobs=-1 forks
    torch.set_num_threads(1)
    _seed = seed if seed is not None else 6
    torch.manual_seed(_seed)
    np.random.seed(_seed)
    device = torch.device("cpu")
    print(f"   LSTM using device: {device}", flush=True)

    n_features = X_train_sc.shape[1]

    # Chronological val split
    split = int(len(X_train_sc) * (1 - val_frac))
    X_tr, X_val = X_train_sc[:split], X_train_sc[split:]
    y_tr, y_val = y_train_enc[:split], y_train_enc[split:]

    train_ds = _SeqDataset(X_tr, y_tr, window)
    val_ds   = _SeqDataset(X_val, y_val, window)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # Class weights from training set (excluding first `window` rows)
    labels_for_weight = y_tr[window:]
    counts = np.bincount(labels_for_weight, minlength=3).astype(float)
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32).to(device)

    model     = LSTMClassifier(n_features, hidden_dim, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [], "val_f1": []
    }
    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0

    for epoch in range(1, n_epochs + 1):
        # ── train ──────────────────────────────────────────────────────────
        print(f"   LSTM epoch {epoch} starting ({len(train_ds)} samples) …", flush=True)
        model.train()
        tr_loss = 0.0
        for seqs, lbls in train_dl:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss   = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(lbls)
        tr_loss /= len(train_ds)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for seqs, lbls in val_dl:
                seqs, lbls = seqs.to(device), lbls.to(device)
                logits    = model(seqs)
                val_loss += criterion(logits, lbls).item() * len(lbls)
                val_preds.extend(logits.argmax(1).tolist())
                val_true.extend(lbls.tolist())
        val_loss /= len(val_ds)
        val_f1    = f1_score(val_true, val_preds,
                             average="macro", zero_division=0)

        scheduler.step(val_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"   LSTM ep {epoch:3d}/{n_epochs}  "
                  f"train_loss={tr_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_F1={val_f1:.4f}", flush=True)

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"   Early stopping at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_lstm(
    model:  LSTMClassifier,
    X_sc:   np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """
    Return encoded predictions {0,1,2} for rows window..n-1.
    The first `window` rows cannot form a complete sequence and are
    filled with the most-common prediction for alignment purposes.
    """
    device = next(model.parameters()).device
    model.eval()
    preds: list[int] = []

    X_t = torch.tensor(np.ascontiguousarray(X_sc), dtype=torch.float32).to(device)
    with torch.no_grad():
        for i in range(window, len(X_sc)):
            seq    = X_t[i - window : i].unsqueeze(0)
            logits = model(seq)
            preds.append(int(logits.argmax(1).item()))

    # Pad the first `window` positions
    if preds:
        pad_val = int(np.bincount(preds).argmax())
    else:
        pad_val = 1   # sideways
    full_preds = [pad_val] * window + preds
    return np.array(full_preds, dtype=int)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_lstm(
    model:       LSTMClassifier,
    X_test_sc:   np.ndarray,
    y_test_enc:  np.ndarray,   # {0,1,2}
    y_test:      np.ndarray,   # original {-1,0,1} for label names
    history:     dict,
    output_dir:  str,
    window:      int = 30,
) -> dict:
    """
    Compute metrics, save confusion matrix and loss curve.

    Returns dict with accuracy, f1_macro, roc_auc.
    """
    os.makedirs(output_dir, exist_ok=True)

    preds = predict_lstm(model, X_test_sc, window)
    probs = _predict_proba_lstm(model, X_test_sc, window)

    acc = accuracy_score(y_test_enc, preds)
    f1  = f1_score(y_test_enc, preds, average="macro", zero_division=0)

    try:
        y_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
        auc   = roc_auc_score(y_bin, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print(f"   LSTM            Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    CLASS_NAMES = ["Bear(0)", "Sideways(1)", "Bull(2)"]
    cm  = confusion_matrix(y_test_enc, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_title("LSTM\nConfusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(f"{output_dir}/LSTM_confusion_matrix.png")
    plt.close(fig)

    # ── training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train loss", color="#1f77b4")
    axes[0].plot(history["val_loss"],   label="Val loss",   color="#ff7f0e")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("LSTM Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(history["val_f1"], color="#2ca02c")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 macro")
    axes[1].set_title("LSTM Validation F1 (macro)")

    plt.tight_layout()
    fig.savefig(f"{output_dir}/LSTM_training_curves.png")
    plt.close(fig)

    return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc}


def _predict_proba_lstm(
    model:  LSTMClassifier,
    X_sc:   np.ndarray,
    window: int,
) -> np.ndarray:
    """Return softmax probabilities, shape (n, 3), padded for first `window` rows."""
    device = next(model.parameters()).device
    model.eval()
    probs: list = []
    X_t = torch.tensor(np.ascontiguousarray(X_sc), dtype=torch.float32).to(device)
    with torch.no_grad():
        for i in range(window, len(X_sc)):
            seq  = X_t[i - window : i].unsqueeze(0)
            prob = torch.softmax(model(seq), dim=1).squeeze(0).cpu().tolist()
            probs.append(prob)

    pad = [[1/3, 1/3, 1/3]] * window
    return np.array(pad + probs, dtype=float)
