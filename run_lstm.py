"""
Run LSTM training independently — skips supervised model training.

Usage:
    # First run the full pipeline once to generate split cache:
    python -m src.main

    # Then iterate on LSTM quickly:
    python run_lstm.py
"""

import os
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from src.config import OUTPUT_DIR
from src.lstm_model import train_lstm, evaluate_lstm

CACHE = os.path.join(OUTPUT_DIR, "split_cache.npz")


def main():
    if not os.path.exists(CACHE):
        print(f"Cache not found at {CACHE}")
        print("Run the full pipeline first: python -m src.main")
        return

    print("Loading cached split data …")
    data = np.load(CACHE, allow_pickle=True)
    X_train_sc  = data["X_train_sc"]
    X_test_sc   = data["X_test_sc"]
    y_train_enc = data["y_train_enc"]
    y_test_enc  = data["y_test_enc"]
    y_test      = data["y_test"]
    print(f"   Train: {X_train_sc.shape}  Test: {X_test_sc.shape}")

    print("\nTraining LSTM …\n")
    model, history = train_lstm(
        X_train_sc, y_train_enc,
        window=30,
        n_epochs=60,
        verbose=True,
    )

    print("\nEvaluating LSTM …")
    results = evaluate_lstm(
        model, X_test_sc, y_test_enc, y_test,
        history, OUTPUT_DIR, window=30,
    )
    print(f"\nLSTM  Acc={results['accuracy']:.4f}  "
          f"F1={results['f1_macro']:.4f}  "
          f"AUC={results['roc_auc']:.4f}")

    # Update model_summary.csv
    import pandas as pd
    summary_path = os.path.join(OUTPUT_DIR, "model_summary.csv")
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        summary = summary[summary["Model"] != "LSTM"]   # remove old row
        lstm_row = pd.DataFrame([{
            "Model":       "LSTM",
            "Accuracy":    round(results["accuracy"], 4),
            "F1_macro":    round(results["f1_macro"],  4),
            "ROC_AUC":     round(results["roc_auc"],   4),
            "Best Params": "hidden=128, layers=2, window=30",
        }])
        summary = pd.concat([summary, lstm_row], ignore_index=True)
        summary.to_csv(summary_path, index=False)
        print("model_summary.csv updated")


if __name__ == "__main__":
    main()
