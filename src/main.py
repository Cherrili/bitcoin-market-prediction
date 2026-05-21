"""
Bitcoin Market State Prediction — Main Entry Point

Run from the SC6122 directory:
    python -m src.main
"""

import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, no tkinter needed
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
from .visualization       import plot_feature_importances, plot_eda

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

    print("\n5. Splitting train / test (time-ordered 80/20) …")
    split = split_data(df, feature_cols, train_ratio=0.8)

    print("   Label dist (full) :", pd.Series(df["label"]).value_counts().sort_index().to_dict())
    print("   Label dist (train):", pd.Series(split.y_train).value_counts().sort_index().to_dict())
    print("   Label dist (test) :", pd.Series(split.y_test).value_counts().sort_index().to_dict())

    print("\n5.5. Feature selection (Top 40 by RF importance) …")
    split, feature_cols = select_top_features(split, feature_cols, top_n=40)

    print("\n6. Training models with GridSearchCV (TimeSeriesSplit=5) …\n")
    configs = get_model_configs()
    trained = train_models(configs, split)

    print("\n7. Evaluating models …\n")
    evaluate_models(trained, split, OUTPUT_DIR)

    print("\n8. Feature importance plots …")
    plot_feature_importances(trained, feature_cols, OUTPUT_DIR)

    print("\n" + "=" * 65)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"   {f}")
    print("Done!")


if __name__ == "__main__":
    main()
