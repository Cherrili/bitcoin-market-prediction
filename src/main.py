"""
Bitcoin Market State Prediction — Main Entry Point

Run from the project root:
    python -m src.main
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"]  = 10

from .config              import DATA_DIR, OUTPUT_DIR
from .data_loader         import load_and_clean
from .feature_engineering import create_labels, build_features
from .models              import get_model_configs
from .training            import split_data, select_top_features, train_models
from .evaluation          import evaluate_models
from .visualization       import plot_feature_importances, plot_eda, plot_distribution_shift

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 65)

    print("1. Loading & merging data …")
    df = load_and_clean(DATA_DIR)

    print("\n2. Creating labels (30-day forward return) …")
    df = create_labels(df)

    print("\n3. Feature engineering …")
    df, feature_cols = build_features(df)

    print("\n4. EDA chart …")
    plot_eda(df, OUTPUT_DIR)

    print("\n4.5. Distribution shift analysis …")
    _split_idx_for_vis = int(len(df) * 0.8)
    plot_distribution_shift(df, _split_idx_for_vis, OUTPUT_DIR)

    print("\n5. Splitting train / test (time-ordered 80/20) …")
    split = split_data(df, feature_cols, train_ratio=0.8)

    print("\n5.5. Feature selection (Top 40 by RF importance) …")
    split, feature_cols = select_top_features(split, feature_cols, top_n=40)

    # Save split for fast LSTM iteration
    import numpy as np
    cache_path = os.path.join(OUTPUT_DIR, "split_cache.npz")
    np.savez(cache_path,
             X_train_sc=split.X_train_sc, X_test_sc=split.X_test_sc,
             y_train_enc=split.y_train_enc, y_test_enc=split.y_test_enc,
             y_test=split.y_test)
    print(f"   Split cached to {cache_path}")

    print("\n6. Training supervised models with GridSearchCV (TimeSeriesSplit=5) …\n")
    configs = get_model_configs()
    trained = train_models(configs, split)

    print("\n7. Evaluating supervised models …\n")
    evaluate_models(trained, split, OUTPUT_DIR)

    print("\n8. Feature importance plots …")
    plot_feature_importances(trained, feature_cols, OUTPUT_DIR)

    # ── LSTM ──────────────────────────────────────────────────────────────────
    print("\n9. Training LSTM classifier …\n")
    _run_lstm(split, OUTPUT_DIR)

    # ── RL agent ──────────────────────────────────────────────────────────────
    print("\n10. Training DQN Reinforcement Learning agent …\n")
    _run_rl_agent(df, split, OUTPUT_DIR)

    # ── Stacking ensemble ─────────────────────────────────────────────────────
    print("\n11. Training Stacking Ensemble (RF+KNN+LSTM → LR meta) …\n")
    _run_stacking(split, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   {f}")
    print("Done!")


def _run_stacking(split, output_dir: str) -> None:
    """Stacking ensemble: RF + KNN + XGB base → LR meta."""
    try:
        import pandas as pd
        from .stacking import run_stacking

        results = run_stacking(split, output_dir)

        summary_path = os.path.join(output_dir, "model_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            summary = summary[~summary["Model"].str.startswith("Stacking")]
            row = pd.DataFrame([{
                "Model":       "Stacking (RF+KNN+LGB+XGB)",
                "Accuracy":    round(results["accuracy"], 4),
                "F1_macro":    round(results["f1_macro"],  4),
                "ROC_AUC":     round(results["roc_auc"],   4),
                "Best Params": "OOF RF+KNN+LGB+XGB; max-conf routing, LGB weight=1.5x",
            }])
            summary = pd.concat([summary, row], ignore_index=True)
            summary.to_csv(summary_path, index=False)
            print("   Stacking row appended to model_summary.csv")

    except Exception as e:
        print(f"   [ERROR] Stacking failed: {e}")
        import traceback; traceback.print_exc()


def _run_threshold_opt(trained: dict, split, output_dir: str) -> None:
    """Run threshold optimisation on LightGBM (highest AUC model)."""
    try:
        import pandas as pd
        from .threshold_optimizer import optimize_thresholds

        lgb_info = trained.get("LightGBM")
        if lgb_info is None:
            print("   [SKIP] LightGBM not found in trained models")
            return

        Xtr = split.X_train   # LightGBM uses unscaled features
        Xte = split.X_test

        results = optimize_thresholds(
            lgb_info["model"],
            Xtr, split.y_train_enc,
            Xte, split.y_test_enc,
            output_dir,
            model_name="LightGBM",
        )

        # Append to model_summary.csv
        summary_path = os.path.join(output_dir, "model_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            row = pd.DataFrame([{
                "Model":       "LightGBM + Threshold Opt",
                "Accuracy":    round(results["accuracy"], 4),
                "F1_macro":    round(results["f1_macro"],  4),
                "ROC_AUC":     round(results["roc_auc"],   4),
                "Best Params": f"thresholds={[round(t,2) for t in results['thresholds']]}",
            }])
            summary = pd.concat([summary, row], ignore_index=True)
            summary.to_csv(summary_path, index=False)

    except Exception as e:
        print(f"   [ERROR] Threshold optimisation failed: {e}")
        import traceback; traceback.print_exc()


def _run_lstm(split, output_dir: str) -> None:
    """Train LSTM on scaled train features, evaluate on test set."""
    try:
        import pandas as pd
        from .lstm_model import train_lstm, evaluate_lstm

        model, history = train_lstm(
            split.X_train_sc,
            split.y_train_enc,
            window=30,
            n_epochs=60,
            verbose=True,
        )

        results = evaluate_lstm(
            model,
            split.X_test_sc,
            split.y_test_enc,
            split.y_test,
            history,
            output_dir,
            window=30,
        )

        # Append LSTM row to model_summary.csv
        summary_path = os.path.join(output_dir, "model_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            lstm_row = pd.DataFrame([{
                "Model":       "LSTM",
                "Accuracy":    round(results["accuracy"], 4),
                "F1_macro":    round(results["f1_macro"],  4),
                "ROC_AUC":     round(results["roc_auc"],   4),
                "Best Params": "hidden=128, layers=2, window=30, bidir",
            }])
            summary = pd.concat([summary, lstm_row], ignore_index=True)
            summary.to_csv(summary_path, index=False)
            print("   LSTM row appended to model_summary.csv")

    except ImportError as e:
        print(f"   [SKIP] LSTM requires PyTorch: {e}")
    except Exception as e:
        print(f"   [ERROR] LSTM training failed: {e}")
        import traceback
        traceback.print_exc()


def _run_rl_agent(df, split, output_dir: str) -> None:
    """
    Train the DQN agent on the training split (using the actual 30-day
    returns as rewards) and evaluate on the test split.
    """
    try:
        import numpy as np
        from .rl_agent import train_dqn, evaluate_dqn

        # Align returns with the split indices.
        # split.X_train / X_test are already feature-selected;
        # the matching rows in df give us the raw return_30d column.
        train_end = split.split_idx
        returns_train = df["return_30d"].values[:train_end]
        returns_test  = df["return_30d"].values[train_end : train_end + len(split.X_test)]

        # Use scaled features (same as LR/SVM) for the RL agent.
        agent, rewards_hist = train_dqn(
            split.X_train_sc,
            returns_train,
            n_episodes=60,
            verbose=True,
        )

        results = evaluate_dqn(
            agent,
            split.X_test_sc,
            split.y_test_enc,      # {0,1,2}
            returns_test,
            rewards_hist,
            output_dir,
        )

        print(f"\n   DQN cumulative return (test) : "
              f"{results['cumulative_return_dqn']:+.4f}")
        print(f"   Buy-and-hold  (test)         : "
              f"{results['cumulative_return_bh']:+.4f}")

        # Append DQN row to model_summary.csv
        import pandas as pd
        summary_path = os.path.join(output_dir, "model_summary.csv")
        if os.path.exists(summary_path):
            summary = pd.read_csv(summary_path)
            dqn_row = pd.DataFrame([{
                "Model":       "DQN (RL)",
                "Accuracy":    round(results["accuracy"], 4),
                "F1_macro":    round(results["f1_macro"],  4),
                "ROC_AUC":     float("nan"),
                "Best Params": "DQN 60 episodes",
            }])
            summary = pd.concat([summary, dqn_row], ignore_index=True)
            summary.to_csv(summary_path, index=False)
            print("   DQN row appended to model_summary.csv")

    except ImportError as e:
        print(f"   [SKIP] RL agent requires PyTorch: {e}")
    except Exception as e:
        print(f"   [ERROR] RL training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
