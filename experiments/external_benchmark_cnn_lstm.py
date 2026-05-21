"""
External benchmark replication: Boruta-like feature selection + CNN-LSTM.

Inspired by Omole & Enke (2024), Financial Innovation — evaluated under this
project's strict chronological 80/20 split and existing data pipeline.

Does NOT modify the main 3-class market-state task.

Run from project root:
    python experiments/external_benchmark_cnn_lstm.py
    python experiments/external_benchmark_cnn_lstm.py --label-mode zero
    python experiments/external_benchmark_cnn_lstm.py --label-mode threshold5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings

# Avoid macOS OpenMP mutex issues after multiprocessing (sklearn joblib)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# ── project imports ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.data_loader import load_and_clean  # noqa: E402
from src.feature_engineering import build_features  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "external_benchmark"
TRAIN_RATIO = 0.8
WINDOW_SIZE = 30
RANDOM_SEED = 42
THRESHOLD_5PCT = 0.05
PAPER_REFERENCE_ACCURACY = "~90%+ (binary direction; different split/labels/data)"

LabelMode = Literal["zero", "threshold5"]


@dataclass
class PreparedData:
    df: pd.DataFrame
    feature_cols: list[str]
    split_idx: int
    label_mode: LabelMode
    label_description: str


# ── labels & features (no leakage into feature_cols) ──────────────────────────

def _attach_forward_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["future_price_30d"] = df["price"].shift(-30)
    df["return_30d"] = (df["future_price_30d"] - df["price"]) / df["price"]
    df.dropna(subset=["future_price_30d"], inplace=True)
    return df


def prepare_data(label_mode: LabelMode) -> PreparedData:
    """Same load/merge pipeline as main project; binary labels only here."""
    df = load_and_clean(DATA_DIR)
    df = _attach_forward_return(df)

    if label_mode == "zero":
        df["label"] = (df["return_30d"] > 0).astype(int)
        desc = "Up=1 if 30-day forward return > 0; Down=0 otherwise"
        df = df.drop(columns=["return_30d", "future_price_30d"], errors="ignore")
    else:
        total = len(df)
        df = df[
            (df["return_30d"] > THRESHOLD_5PCT) | (df["return_30d"] < -THRESHOLD_5PCT)
        ].copy()
        df["label"] = (df["return_30d"] > THRESHOLD_5PCT).astype(int)
        desc = (
            f"Up=1 if return > +{THRESHOLD_5PCT:.0%}; Down=0 if return < -{THRESHOLD_5PCT:.0%}; "
            f"neutral zone dropped ({len(df)}/{total} rows kept)"
        )
        df = df.drop(columns=["return_30d", "future_price_30d"], errors="ignore")

    df_feat, feature_cols = build_features(df)
    split_idx = int(len(df_feat) * TRAIN_RATIO)
    dates = df_feat["datetime"]
    print(
        f"   [{label_mode}] rows={len(df_feat)}  "
        f"train {dates.iloc[0].date()}→{dates.iloc[split_idx - 1].date()}  "
        f"test {dates.iloc[split_idx].date()}→{dates.iloc[-1].date()}"
    )
    print(f"   Label dist (all): {df_feat['label'].value_counts().to_dict()}")
    return PreparedData(df_feat, feature_cols, split_idx, label_mode, desc)


# ── Boruta-like selection (train only) ────────────────────────────────────────

def boruta_like_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    max_iter: int = 60,
    top_k_fallback: int = 40,
    random_state: int = RANDOM_SEED,
) -> tuple[list[str], str, pd.DataFrame]:
    """
    Fit feature selection on training rows only.
    Prefer BorutaPy; else shadow-feature RF approximation (documented fallback).
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        n_jobs=1,
        class_weight="balanced",
    )

    try:
        from boruta import BorutaPy  # type: ignore

        selector = BorutaPy(
            rf,
            n_estimators="auto",
            max_iter=max_iter,
            random_state=random_state,
            verbose=0,
        )
        selector.fit(X_train, y_train)
        mask = selector.support_
        method = "BorutaPy"
        ranking = selector.ranking_
    except ImportError:
        method = "shadow_rf_approximation (BorutaPy not installed)"
        n_feat = X_train.shape[1]
        hits = np.zeros(n_feat, dtype=int)
        rng = np.random.RandomState(random_state)

        for _ in range(max_iter):
            X_shadow = X_train.copy()
            for j in range(n_feat):
                X_shadow[:, j] = rng.permutation(X_shadow[:, j])
            X_aug = np.hstack([X_train, X_shadow])
            rf.fit(X_aug, y_train)
            imp = rf.feature_importances_
            real_imp = imp[:n_feat]
            shadow_max = imp[n_feat:].max()
            hits += (real_imp > shadow_max).astype(int)

        mask = hits >= (max_iter // 2)
        ranking = np.where(mask, 1, 2)
        if not mask.any():
            rf.fit(X_train, y_train)
            top_idx = np.argsort(rf.feature_importances_)[::-1][:top_k_fallback]
            mask = np.zeros(n_feat, dtype=bool)
            mask[top_idx] = True
            ranking = np.where(mask, 1, 2)
            method += f"; no confirmed features — top-{top_k_fallback} RF fallback"

    selected = [feature_names[i] for i, m in enumerate(mask) if m]
    if len(selected) < 5:
        rf.fit(X_train, y_train)
        top_idx = np.argsort(rf.feature_importances_)[::-1][:top_k_fallback]
        selected = [feature_names[i] for i in top_idx]
        method += f"; expanded to top-{top_k_fallback} RF features"

    feat_df = pd.DataFrame(
        {
            "feature": feature_names,
            "selected": mask,
            "ranking": ranking if isinstance(ranking, np.ndarray) else ranking,
        }
    )
    feat_df = feat_df.sort_values(["selected", "ranking"], ascending=[False, True])
    print(f"   Feature selection: {method}")
    print(f"   Selected {len(selected)} / {len(feature_names)} features")
    return selected, method, feat_df


# ── strict-boundary sequence datasets ───────────────────────────────────────

class StrictSeqDataset(Dataset):
    """
    Sliding windows that do not cross the train/test boundary.

    Train: label index i in [window, split_idx) — window rows all from train.
    Test:  label index i in [split_idx + window, n) — window rows all from test.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window: int,
        region_start: int,
        region_end: int,
    ) -> None:
        self.X = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32)
        self.window = window
        # region_* define the partition [region_start, region_end) for labels/features
        lo = region_start + window
        hi = region_end
        self.indices = list(range(lo, hi))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        seq = self.X[i - self.window : i]
        lbl = self.y[i]
        return seq, lbl


class CNNLSTMClassifier(nn.Module):
    """Conv1D → optional MaxPool → LSTM → Dropout → sigmoid (binary)."""

    def __init__(
        self,
        n_features: int,
        window: int,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        use_pool: bool = True,
    ) -> None:
        super().__init__()
        self.use_pool = use_pool
        self.conv = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2) if use_pool else nn.Identity()
        conv_len = window // 2 if use_pool else window
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features) → (batch, features, window) for Conv1d
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = self.drop(out[:, -1, :])
        return self.fc(last).squeeze(-1)


def _class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor | None:
    classes = np.unique(y)
    if len(classes) < 2:
        return None
    weights = compute_class_weight("balanced", classes=classes, y=y)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w


def train_cnn_lstm(
    X_train_sc: np.ndarray,
    y_train: np.ndarray,
    train_len: int,
    n_features: int,
    *,
    window: int = WINDOW_SIZE,
    n_epochs: int = 80,
    patience: int = 12,
    val_frac: float = 0.15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = RANDOM_SEED,
) -> tuple[CNNLSTMClassifier, dict]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # Chronological validation inside train partition only
    train_end = train_len
    val_rows = int((train_end - window) * val_frac)
    val_start = train_end - val_rows
    if val_rows < window + 50:
        val_start = int(train_end * (1 - val_frac))

    train_ds = StrictSeqDataset(
        X_train_sc, y_train, window, region_start=0, region_end=val_start
    )
    val_ds = StrictSeqDataset(
        X_train_sc, y_train, window, region_start=val_start, region_end=train_end
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    y_tr = y_train[window:val_start]
    pos_weight = None
    cw = _class_weights(y_tr, device)
    if cw is not None:
        pos_weight = cw[1] / cw[0]

    model = CNNLSTMClassifier(n_features, window).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss = 0.0
        for seqs, lbls in train_dl:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(lbls)
        tr_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for seqs, lbls in val_dl:
                seqs, lbls = seqs.to(device), lbls.to(device)
                logits = model(seqs)
                val_loss += criterion(logits, lbls).item() * len(lbls)
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_true.extend(lbls.cpu().numpy())
        val_loss /= max(len(val_ds), 1)
        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        scheduler.step(val_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"      CNN-LSTM ep {epoch:3d}/{n_epochs}  "
                f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_F1={val_f1:.4f}"
            )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"      Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_cnn_lstm(
    model: CNNLSTMClassifier,
    X_sc: np.ndarray,
    y: np.ndarray,
    split_idx: int,
    window: int,
    *,
    eval_train: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict on strict-boundary sequences; returns y_true, y_pred, y_prob."""
    device = next(model.parameters()).device
    model.eval()
    n = len(X_sc)

    if eval_train:
        region_start, region_end = 0, split_idx
    else:
        region_start, region_end = split_idx, n

    ds = StrictSeqDataset(X_sc, y, window, region_start, region_end)
    dl = DataLoader(ds, batch_size=128, shuffle=False)

    y_true, y_pred, y_prob = [], [], []
    for seqs, lbls in dl:
        seqs = seqs.to(device)
        logits = model(seqs)
        prob = torch.sigmoid(logits).cpu().numpy()
        pred = (prob > 0.5).astype(int)
        y_true.extend(lbls.numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def train_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Logistic Regression and XGBoost on same train-only selected/scaled features."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    results: list[dict] = []

    lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
        class_weight="balanced",
    )
    lr.fit(X_tr, y_train)
    lr_pred = lr.predict(X_te)
    lr_prob = lr.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_test, lr_pred, lr_prob)
    results.append({"model": "Logistic Regression", "split": "test", **m})

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=seed,
        n_jobs=1,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
    )
    sw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sample_weight = np.array([sw[int(lbl)] for lbl in y_train])
    xgb_clf.fit(X_tr, y_train, sample_weight=sample_weight)
    xgb_pred = xgb_clf.predict(X_te)
    xgb_prob = xgb_clf.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_test, xgb_pred, xgb_prob)
    results.append({"model": "XGBoost", "split": "test", **m})

    maj = int(np.bincount(y_test.astype(int)).argmax())
    maj_pred = np.full(len(y_test), maj)
    m = compute_metrics(y_test, maj_pred, np.full(len(y_test), y_test.mean()))
    results.append({"model": "Majority class", "split": "test", **m})

    rng = np.random.RandomState(seed)
    rand_pred = rng.randint(0, 2, len(y_test))
    rand_prob = rng.random(len(y_test))
    m = compute_metrics(y_test, rand_pred, rand_prob)
    results.append({"model": "Random baseline", "split": "test", **m})

    return results


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Down(0)", "Up(1)"],
        yticklabels=["Down(0)", "Up(1)"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curve(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train loss")
    axes[0].plot(history["val_loss"], label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("CNN-LSTM training / validation loss")
    axes[0].legend()

    axes[1].plot(history["val_f1"], color="#2ca02c")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation F1")
    axes[1].set_title("CNN-LSTM validation F1")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_prior_binary_results() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "output" / "binary_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def write_summary(
    out_dir: Path,
    prepared: PreparedData,
    selection_method: str,
    selected_features: list[str],
    cnn_metrics: dict,
    baseline_rows: list[dict],
    clf_report: str,
) -> None:
    prior = load_prior_binary_results()
    prior_block = ""
    if prior is not None:
        prior_block = (
            "\n### Comparison with existing `run_binary.py` (±5% neutral dropped)\n\n"
            "```\n"
            + prior.to_string(index=False)
            + "\n```\n"
        )

    high_perf = cnn_metrics["accuracy"] >= 0.65 or cnn_metrics["f1"] >= 0.65
    if high_perf:
        interp = (
            "The gap between binary direction results and the main three-class "
            "market-state task (best macro-F1 ≈ 0.41) supports the claim that "
            "the three-class formulation is substantially harder."
        )
    else:
        interp = (
            "Moderate test performance under strict chronological splitting suggests "
            "that distribution shift (train ≈ 2010–2021, test ≈ 2022–2023) and "
            "evaluation protocol differences substantially reduce accuracy relative "
            "to previously reported binary direction-prediction results."
        )

    baselines_md = "```\n" + pd.DataFrame(baseline_rows).to_string(index=False, float_format="%.4f") + "\n```"

    body = f"""# External Benchmark: Boruta-like Selection + CNN-LSTM

## Disclaimer

This experiment is inspired by prior CNN-LSTM Bitcoin direction-prediction work
(Omole & Enke, 2024, *Financial Innovation*), but serves as a **controlled reference**
under our strict chronological evaluation. **Do not claim direct comparability** with
the paper unless task, split, labels, and data are identical.

Paper-reported direction accuracy ({PAPER_REFERENCE_ACCURACY}) used different
methodology and data; our replication uses the project's merged on-chain + sentiment
features and an 80/20 time-ordered split.

## Task definition

- **Label mode:** `{prepared.label_mode}`
- {prepared.label_description}
- **Horizon:** 30 calendar days (same forward window as the main project)

## Split protocol

- Strict chronological **80% train / 20% test** (no shuffle).
- `split_idx = {prepared.split_idx}` (train rows `[0, split_idx)`, test `[split_idx, n)`).

## Leakage prevention

1. **Labels / targets:** `return_30d` and `future_price_30d` are dropped before `build_features`.
2. **Feature selection:** Boruta-like selector fit on **train rows only** (`X[:split_idx]`).
3. **Scaler:** `StandardScaler` fit on train only, applied to test.
4. **Sequences:** Sliding windows **do not cross** the train/test boundary — train windows
   stay in the train partition; test windows stay in the test partition (first `{WINDOW_SIZE}`
   test dates are skipped).
5. **Validation:** Chronological hold-out from the end of the train partition for early stopping.

## Feature selection

- **Method:** {selection_method}
- **Count:** {len(selected_features)} features (see `selected_features.csv`)

## CNN-LSTM architecture

- Input: `(batch, window={WINDOW_SIZE}, n_features)`
- `Conv1D(filters=64, kernel_size=3, activation=relu)`
- `MaxPool1d(kernel_size=2)`
- `LSTM(hidden=64, batch_first=True)`
- `Dropout(0.3)`
- `Dense(1)` + sigmoid (BCEWithLogitsLoss)
- Optimizer: AdamW; early stopping on validation loss; class weights if imbalanced

## Final metrics (CNN-LSTM, strict test windows)

| Metric | Value |
|--------|------:|
| Accuracy | {cnn_metrics['accuracy']:.4f} |
| F1 | {cnn_metrics['f1']:.4f} |
| Precision | {cnn_metrics['precision']:.4f} |
| Recall | {cnn_metrics['recall']:.4f} |
| ROC-AUC | {cnn_metrics['roc_auc']:.4f} |

### Classification report

```
{clf_report.strip()}
```

## Baselines (same selected features, test split)

{baselines_md}

{prior_block}

## Main project reference

- Three-class market state (±15% bull/bear): best **Stacking Ensemble macro-F1 = 0.4124**
  (`output/model_summary.csv`).

## Interpretation

{interp}

---
*Generated by `experiments/external_benchmark_cnn_lstm.py`*
"""
    (out_dir / "summary.md").write_text(body, encoding="utf-8")


def run_experiment(label_mode: LabelMode, args: argparse.Namespace) -> pd.DataFrame:
    print(f"\n{'=' * 65}\nLabel mode: {label_mode}\n{'=' * 65}")
    out_dir = OUTPUT_DIR / label_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(label_mode)
    df = prepared.df
    split_idx = prepared.split_idx

    X = df[prepared.feature_cols].values
    y = df["label"].values.astype(int)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    selected_cols, selection_method, feat_df = boruta_like_select(
        X_train,
        y_train,
        prepared.feature_cols,
        max_iter=args.boruta_iter,
        random_state=args.seed,
    )
    feat_df.to_csv(out_dir / "selected_features.csv", index=False)

    col_idx = [prepared.feature_cols.index(c) for c in selected_cols]
    X_train = X_train[:, col_idx]
    X_test = X_test[:, col_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\n   Training CNN-LSTM …")
    model, history = train_cnn_lstm(
        X_train_sc,
        y_train,
        len(X_train_sc),
        n_features=len(selected_cols),
        window=args.window,
        n_epochs=args.epochs,
        seed=args.seed,
    )

    # Use full scaled matrix so split_idx aligns with strict test windows
    X_full_sc = np.vstack([X_train_sc, X_test_sc])
    y_true, y_pred, y_prob = predict_cnn_lstm(
        model, X_full_sc, y, split_idx, args.window, eval_train=False
    )
    cnn_metrics = compute_metrics(y_true, y_pred, y_prob)
    print(
        f"   CNN-LSTM test: Acc={cnn_metrics['accuracy']:.4f}  "
        f"F1={cnn_metrics['f1']:.4f}  AUC={cnn_metrics['roc_auc']:.4f}"
    )

    clf_report = classification_report(y_true, y_pred, target_names=["Down", "Up"])
    plot_confusion_matrix(
        y_true,
        y_pred,
        out_dir / "confusion_matrix.png",
        f"CNN-LSTM ({label_mode}, test)",
    )
    plot_training_curve(history, out_dir / "training_curve.png")

    # Align tabular baselines with CNN test rows (skip first `window` test days)
    X_test_aligned = X_test[args.window :]
    y_test_aligned = y_test[args.window :]

    print("\n   Training tabular baselines (aligned test rows) …")
    baseline_rows = train_baselines(
        X_train, X_test_aligned, y_train, y_test_aligned, seed=args.seed
    )

    rows = [
        {
            "label_mode": label_mode,
            "model": "CNN-LSTM",
            "split": "test_strict_windows",
            **cnn_metrics,
            "selection_method": selection_method,
            "n_features": len(selected_cols),
            "n_test_sequences": len(y_true),
        }
    ]
    for b in baseline_rows:
        rows.append(
            {
                "label_mode": label_mode,
                "model": b["model"],
                "split": b["split"],
                "accuracy": b["accuracy"],
                "f1": b["f1"],
                "precision": b["precision"],
                "recall": b["recall"],
                "roc_auc": b["roc_auc"],
                "selection_method": selection_method,
                "n_features": len(selected_cols),
                "n_test_sequences": len(y_test_aligned),
            }
        )

    metrics_df = pd.DataFrame(rows)
    write_summary(
        out_dir,
        prepared,
        selection_method,
        selected_cols,
        cnn_metrics,
        baseline_rows,
        clf_report,
    )

    # Also expose per-mode artifacts at label_mode subfolder (already saved above)
    return metrics_df, {
        "label_mode": label_mode,
        "cnn_metrics": cnn_metrics,
        "selection_method": selection_method,
        "selected_cols": selected_cols,
        "feat_df": feat_df,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "prepared": prepared,
        "clf_report": clf_report,
        "baseline_rows": baseline_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External benchmark CNN-LSTM experiment")
    parser.add_argument(
        "--label-mode",
        choices=["zero", "threshold5", "both"],
        default="both",
        help="Binary label definition (default: run both)",
    )
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--boruta-iter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("External Benchmark: Boruta-like + CNN-LSTM")
    print("=" * 65)

    modes: list[LabelMode] = (
        ["zero", "threshold5"] if args.label_mode == "both" else [args.label_mode]  # type: ignore
    )

    all_metrics: list[pd.DataFrame] = []
    artifacts: dict[str, dict] = {}
    for mode in modes:
        metrics_df, art = run_experiment(mode, args)
        all_metrics.append(metrics_df)
        artifacts[mode] = art

    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"\nSaved combined metrics → {OUTPUT_DIR / 'metrics.csv'}")

    # Root-level deliverables (primary = zero-threshold; suffix for threshold5 when both)
    primary = "zero" if "zero" in artifacts else modes[0]
    for mode, art in artifacts.items():
        suffix = "" if (mode == primary and len(artifacts) == 1) else f"_{mode}"
        art["feat_df"].to_csv(OUTPUT_DIR / f"selected_features{suffix}.csv", index=False)
        plot_confusion_matrix(
            art["y_true"],
            art["y_pred"],
            OUTPUT_DIR / f"confusion_matrix{suffix}.png",
            f"CNN-LSTM test ({mode})",
        )
        plot_training_curve(art["history"], OUTPUT_DIR / f"training_curve{suffix}.png")

    _write_combined_summary(OUTPUT_DIR, artifacts, primary)
    print("Done.")


def _write_combined_summary(out_dir: Path, artifacts: dict, primary: str) -> None:
    """Merge per-mode summaries into one root summary.md."""
    parts = [
        "# External Benchmark: Boruta-like Selection + CNN-LSTM\n",
        "## Disclaimer\n\n",
        "This experiment is inspired by prior CNN-LSTM Bitcoin direction-prediction work "
        "(Omole & Enke, 2024, *Financial Innovation*), but serves as a **controlled reference** "
        "under our strict chronological evaluation. **Do not claim direct comparability** "
        "with the paper unless task, split, labels, and data are identical.\n\n",
    ]
    for mode, art in artifacts.items():
        sub_summary = (out_dir / mode / "summary.md").read_text(encoding="utf-8")
        parts.append(f"\n---\n\n## Run: `{mode}`\n\n")
        # Strip duplicate top-level title from sub-summary
        lines = sub_summary.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        parts.append("\n".join(lines).strip())
        parts.append("\n")

    if primary in artifacts:
        src = out_dir / f"confusion_matrix{'_' + primary if len(artifacts) > 1 else ''}.png"
        dst = out_dir / "confusion_matrix.png"
        if src.exists() and src != dst:
            shutil.copy(src, dst)
        src = out_dir / f"training_curve{'_' + primary if len(artifacts) > 1 else ''}.png"
        dst = out_dir / "training_curve.png"
        if src.exists() and src != dst:
            shutil.copy(src, dst)
        feat_src = out_dir / f"selected_features{'_' + primary if len(artifacts) > 1 else ''}.csv"
        feat_dst = out_dir / "selected_features.csv"
        if feat_src.exists() and feat_src != feat_dst:
            shutil.copy(feat_src, feat_dst)

    (out_dir / "summary.md").write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
