"""
Paper-protocol benchmark (Omole & Enke 2024) — closer replication settings.

Differences from `external_benchmark_cnn_lstm.py`:
  - **Next-day** binary labels: Up if price(t+1) > price(t) (paper Eq. class 1/0)
  - Test windows may use pre-split history (standard seq. forecasting; not strict-partition)
  - Window-size grid {3, 5, 7, 14, 30} as in the paper
  - Optional multi-seed runs; reports honest val-selected test + paper-style grid max

Still uses this project's merged CSV pipeline (not Glassnode export).

Does NOT modify the main 3-class task.

Run from project root:
    python experiments/external_benchmark_paper_protocol.py
    python experiments/external_benchmark_paper_protocol.py --seeds 42,0,1,7,21
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.data_loader import load_and_clean  # noqa: E402
from src.feature_engineering import build_features  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "external_benchmark_paper"
TRAIN_RATIO = 0.8
PAPER_WINDOWS = [3, 5, 7, 14, 30]
PAPER_MAX_ACC = 0.8244
RANDOM_SEED = 42
# Paper sample: 2013-02-06 → 2023-02-18 (Glassnode); filter when our data allows
PAPER_DATE_START = "2013-02-06"
PAPER_DATE_END = "2023-02-18"

# Reuse Boruta + metrics + CNN from the 30-day benchmark script
_EXT_PATH = PROJECT_ROOT / "experiments" / "external_benchmark_cnn_lstm.py"
_spec = importlib.util.spec_from_file_location("ext_bench", _EXT_PATH)
_ext = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["ext_bench"] = _ext
_spec.loader.exec_module(_ext)

boruta_like_select = _ext.boruta_like_select
CNNLSTMClassifier = _ext.CNNLSTMClassifier
compute_metrics = _ext.compute_metrics


# ── next-day labels (paper) ───────────────────────────────────────────────────

def prepare_paper_data(
    *,
    use_paper_dates: bool = True,
) -> tuple[pd.DataFrame, list[str], int, str]:
    """
    Label at day t: 1 if price(t+1) > price(t), else 0.
    Drop last row (no t+1). No future columns enter feature_cols.
    """
    df = load_and_clean(DATA_DIR)

    if use_paper_dates:
        df = df[
            (df["datetime"] >= PAPER_DATE_START) & (df["datetime"] <= PAPER_DATE_END)
        ].copy()
        print(f"   Date filter (paper range): {PAPER_DATE_START} → {PAPER_DATE_END}")

    df["label"] = (df["price"].shift(-1) > df["price"]).astype(int)
    df.dropna(subset=["price"], inplace=True)
    df = df.iloc[:-1].copy()  # last row has no t+1 label

    counts = df["label"].value_counts().sort_index()
    print(f"   Next-day labels: Down(0)={counts.get(0, 0)}, Up(1)={counts.get(1, 0)}")

    df_feat, feature_cols = build_features(df)
    split_idx = int(len(df_feat) * TRAIN_RATIO)
    dates = df_feat["datetime"]
    desc = "Class 1 if price(t+1) > price(t); Class 0 otherwise (Omole & Enke 2024)"
    print(
        f"   rows={len(df_feat)}  "
        f"train {dates.iloc[0].date()}→{dates.iloc[split_idx - 1].date()}  "
        f"test {dates.iloc[split_idx].date()}→{dates.iloc[-1].date()}"
    )
    return df_feat, feature_cols, split_idx, desc


# ── sequences: test windows may include train-period lags (standard protocol) ─

class StandardSeqDataset(Dataset):
    """Predict at index i using features [i-window, i). No label leakage."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        window: int,
        start: int,
        end: int,
    ) -> None:
        self.X = torch.tensor(np.ascontiguousarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32)
        self.window = window
        lo = max(start, window)
        self.indices = list(range(lo, end))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        return self.X[i - self.window : i], self.y[i]


def train_cnn_lstm_paper(
    X_train_sc: np.ndarray,
    y_train: np.ndarray,
    *,
    window: int,
    n_features: int,
    n_epochs: int = 60,
    patience: int = 10,
    val_frac: float = 0.15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = RANDOM_SEED,
    verbose: bool = False,
) -> tuple[CNNLSTMClassifier, dict]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    n = len(X_train_sc)
    val_rows = max(int((n - window) * val_frac), window + 20)
    val_start = n - val_rows

    train_ds = StandardSeqDataset(X_train_sc, y_train, window, 0, val_start)
    val_ds = StandardSeqDataset(X_train_sc, y_train, window, val_start, n)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    y_tr = y_train[window:val_start]
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    pos_weight = torch.tensor(cw[1] / cw[0], dtype=torch.float32, device=device)

    model = CNNLSTMClassifier(n_features, window).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
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
            loss = criterion(model(seqs), lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(lbls)
        tr_loss /= max(len(train_ds), 1)

        model.eval()
        val_loss, val_preds, val_true = 0.0, [], []
        with torch.no_grad():
            for seqs, lbls in val_dl:
                seqs, lbls = seqs.to(device), lbls.to(device)
                logits = model(seqs)
                val_loss += criterion(logits, lbls).item() * len(lbls)
                val_preds.extend((torch.sigmoid(logits) > 0.5).cpu().numpy())
                val_true.extend(lbls.cpu().numpy())
        val_loss /= max(len(val_ds), 1)
        from sklearn.metrics import f1_score

        val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

        scheduler.step(val_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"      window={window} seed={seed} stopped ep={len(history['train_loss'])}")
    return model, history


@torch.no_grad()
def predict_sequences(
    model: CNNLSTMClassifier,
    X_sc: np.ndarray,
    y: np.ndarray,
    window: int,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    ds = StandardSeqDataset(X_sc, y, window, start, end)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    y_true, y_pred, y_prob = [], [], []
    for seqs, lbls in dl:
        seqs = seqs.to(device)
        logits = model(seqs)
        prob = torch.sigmoid(logits).cpu().numpy()
        y_true.extend(lbls.numpy())
        y_pred.extend((prob > 0.5).astype(int))
        y_prob.extend(prob)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def eval_on_split(
    model: CNNLSTMClassifier,
    X_sc: np.ndarray,
    y: np.ndarray,
    split_idx: int,
    window: int,
    *,
    region: str,
) -> dict:
    if region == "val":
        n = split_idx
        start, end = int(n * 0.7), n
    else:
        start, end = split_idx, len(y)
    y_true, y_pred, y_prob = predict_sequences(model, X_sc, y, window, start, end)
    m = compute_metrics(y_true, y_pred, y_prob)
    m["n_samples"] = len(y_true)
    return m


def run_lr_baseline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    lr = LogisticRegression(
        max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed
    )
    lr.fit(X_tr, y_train)
    pred = lr.predict(X_te)
    prob = lr.predict_proba(X_te)[:, 1]
    m = compute_metrics(y_test, pred, prob)
    m["model"] = "Logistic Regression"
    return m


def plot_cm(y_true, y_pred, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"],
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    *,
    label_desc: str,
    selection_method: str,
    n_selected: int,
    metrics_df: pd.DataFrame,
    best_honest: pd.Series,
    paper_style_max: float,
    paper_style_window: int,
) -> None:
    honest_acc = best_honest["accuracy"]
    gap = PAPER_MAX_ACC - honest_acc
    text = f"""# Paper-Protocol Benchmark (Next-Day Direction)

## Disclaimer

Inspired by Omole & Enke (2024) (*Financial Innovation*, Boruta + CNN-LSTM, reported
**82.44%** max accuracy). This run uses **our merged CSV features** (not Glassnode) and
the same **80/20 chronological split** spirit, but is **not** a byte-for-byte replication.

## Task (paper-aligned)

- {label_desc}
- Date span: optional filter `{PAPER_DATE_START}` – `{PAPER_DATE_END}`

## Protocol vs. our 30-day benchmark

| Item | 30-day benchmark | This script (paper protocol) |
|------|------------------|------------------------------|
| Label | 30-day forward return | **Next-day** price direction |
| Test window | Strict partition only | **Standard**: lags may use train history |
| Window sizes | 30 only | **3, 5, 7, 14, 30** |

## Leakage controls

1. Next-day label computed then **removed** from features; only `label` kept for training.
2. Boruta / scaler fit on **train rows only**.
3. **Honest** row: window chosen by **validation F1** on train tail (no test peeking).
4. **Paper-style max** row: best test accuracy over window grid — matches how paper
   reports **overall max** across configurations (optimistic; not a single pre-registered test).

## Feature selection

- Method: {selection_method}
- Selected features: {n_selected}

## Results snapshot

```
{metrics_df.to_string(index=False)}
```

### Honest estimate (val-selected window, seed={int(best_honest.get('seed', RANDOM_SEED))})

- Window = **{int(best_honest['window'])}**
- Test accuracy = **{honest_acc:.4f}** (paper max = {PAPER_MAX_ACC:.4f}, gap = {gap:.4f})

### Paper-style grid maximum (exploratory)

- Best test accuracy over windows on same split = **{paper_style_max:.4f}** (window={paper_style_window})
- Still **below 82.44%** with our data pipeline unless future work imports Glassnode features
  and full paper hyperparameter search.

## Interpretation

If honest accuracy remains near 50–60%, the gap to 82.44% is driven by **data source**,
**feature set**, and **reporting** (multi-seed/window max), not only model code.
Next-day labels are easier than 30-day horizons but do not automatically reproduce paper
numbers on a different feature matrix.

---
*Generated by `experiments/external_benchmark_paper_protocol.py`*
"""
    (out_dir / "summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=str,
        default=str(RANDOM_SEED),
        help="Comma-separated seeds (paper varies seeds)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--boruta-iter", type=int, default=60)
    parser.add_argument("--no-date-filter", action="store_true")
    parser.add_argument("--max-features", type=int, default=50,
                        help="Cap Boruta selection count (paper uses few features)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Paper-protocol benchmark (next-day + Boruta + CNN-LSTM)")
    print("=" * 65)

    df, feature_cols, split_idx, label_desc = prepare_paper_data(
        use_paper_dates=not args.no_date_filter
    )

    X = df[feature_cols].values
    y = df["label"].values.astype(int)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    selected_cols, selection_method, feat_df = boruta_like_select(
        X_train,
        y_train,
        feature_cols,
        max_iter=args.boruta_iter,
        top_k_fallback=min(40, args.max_features),
        random_state=seeds[0],
    )
    if len(selected_cols) > args.max_features:
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=seeds[0], n_jobs=1
        )
        idx = [feature_cols.index(c) for c in selected_cols]
        rf.fit(X_train[:, idx], y_train)
        imp = rf.feature_importances_
        top = np.argsort(imp)[::-1][: args.max_features]
        selected_cols = [selected_cols[i] for i in top]
        selection_method += f"; capped to top {args.max_features} by RF importance"

    feat_df["selected_final"] = feat_df["feature"].isin(selected_cols)
    feat_df.to_csv(OUTPUT_DIR / "selected_features.csv", index=False)
    print(f"   Final features: {len(selected_cols)}")

    col_idx = [feature_cols.index(c) for c in selected_cols]
    X_train = X_train[:, col_idx]
    X_test = X_test[:, col_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_full_sc = np.vstack([X_train_sc, X_test_sc])

    rows: list[dict] = []
    grid_best = {"accuracy": -1.0, "window": None, "seed": None}

    for seed in seeds:
        for window in PAPER_WINDOWS:
            if window >= split_idx - 100:
                continue
            model, _ = train_cnn_lstm_paper(
                X_train_sc,
                y_train,
                window=window,
                n_features=len(selected_cols),
                n_epochs=args.epochs,
                seed=seed,
                verbose=True,
            )
            val_m = eval_on_split(
                model, X_full_sc, y, split_idx, window, region="val"
            )
            test_m = eval_on_split(
                model, X_full_sc, y, split_idx, window, region="test"
            )
            print(
                f"   seed={seed} w={window:2d}  "
                f"val_acc={val_m['accuracy']:.4f}  test_acc={test_m['accuracy']:.4f}"
            )
            rows.append(
                {
                    "protocol": "grid_per_window",
                    "seed": seed,
                    "window": window,
                    "split": "val_tail",
                    **{k: val_m[k] for k in ("accuracy", "f1", "precision", "recall", "roc_auc", "n_samples")},
                }
            )
            rows.append(
                {
                    "protocol": "grid_per_window",
                    "seed": seed,
                    "window": window,
                    "split": "test",
                    **{k: test_m[k] for k in ("accuracy", "f1", "precision", "recall", "roc_auc", "n_samples")},
                }
            )
            if test_m["accuracy"] > grid_best["accuracy"]:
                grid_best = {
                    "accuracy": test_m["accuracy"],
                    "window": window,
                    "seed": seed,
                    "model": model,
                    "y_true": None,
                    "y_pred": None,
                }

    # Honest: pick window with best val accuracy (seed=seeds[0] only)
    val_by_w = {}
    for window in PAPER_WINDOWS:
        model, hist = train_cnn_lstm_paper(
            X_train_sc,
            y_train,
            window=window,
            n_features=len(selected_cols),
            n_epochs=args.epochs,
            seed=seeds[0],
        )
        val_m = eval_on_split(model, X_full_sc, y, split_idx, window, region="val")
        val_by_w[window] = (val_m["f1"], model, hist)

    best_w = max(val_by_w, key=lambda w: val_by_w[w][0])
    best_model, best_hist = val_by_w[best_w][1], val_by_w[best_w][2]
    y_true, y_pred, y_prob = predict_sequences(
        best_model, X_full_sc, y, best_w, split_idx, len(y)
    )
    honest_m = compute_metrics(y_true, y_pred, y_prob)
    print(
        f"\n   Honest (val-picked window={best_w}): test_acc={honest_m['accuracy']:.4f}"
    )

    rows.append(
        {
            "protocol": "honest_val_selected",
            "seed": seeds[0],
            "window": best_w,
            "split": "test",
            "n_samples": len(y_true),
            **honest_m,
        }
    )
    rows.append(
        {
            "protocol": "paper_style_max_on_test",
            "seed": grid_best["seed"],
            "window": grid_best["window"],
            "split": "test",
            "n_samples": np.nan,
            "accuracy": grid_best["accuracy"],
            "f1": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "roc_auc": np.nan,
        }
    )

    lr_m = run_lr_baseline(X_train, X_test, y_train, y_test, seeds[0])
    rows.append({"protocol": "baseline_lr", "seed": seeds[0], "window": np.nan, "split": "test", "n_samples": len(y_test), **lr_m})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    plot_cm(
        y_true,
        y_pred,
        OUTPUT_DIR / "confusion_matrix.png",
        f"Paper protocol CNN-LSTM (w={best_w}, honest)",
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    test_grid = metrics_df[
        (metrics_df["protocol"] == "grid_per_window") & (metrics_df["split"] == "test")
    ]
    for seed in seeds:
        sub = test_grid[test_grid["seed"] == seed]
        ax.plot(sub["window"], sub["accuracy"], marker="o", label=f"seed={seed}")
    ax.axhline(PAPER_MAX_ACC, color="red", linestyle="--", label="Paper max 82.44%")
    ax.axhline(0.5, color="gray", linestyle=":")
    ax.set_xlabel("Window size")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Paper protocol: test accuracy vs window")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "window_grid_accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(best_hist["train_loss"], label="Train")
    ax.plot(best_hist["val_loss"], label="Val")
    ax.set_title(f"Training curve (window={best_w})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curve.png", dpi=150)
    plt.close(fig)

    best_honest = pd.Series(
        {"accuracy": honest_m["accuracy"], "window": best_w, "seed": seeds[0]}
    )
    write_summary(
        OUTPUT_DIR,
        label_desc=label_desc,
        selection_method=selection_method,
        n_selected=len(selected_cols),
        metrics_df=metrics_df,
        best_honest=best_honest,
        paper_style_max=grid_best["accuracy"],
        paper_style_window=grid_best["window"],
    )

    print(f"\n   Paper max (reported): {PAPER_MAX_ACC:.4f}")
    print(f"   Our honest test acc:  {honest_m['accuracy']:.4f}")
    print(f"   Our grid max test:    {grid_best['accuracy']:.4f}")
    print(f"   Saved → {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
