# Bitcoin Market State Prediction

Predicting Bitcoin market regimes (**Bull / Sideways / Bear**) using on-chain blockchain indicators and classical machine learning models.

## Overview

| Item | Detail |
|------|--------|
| **Dataset** | Bitcoin Network On-Chain Blockchain Data (Kaggle) |
| **Period** | Aug 2010 — Sep 2023 (~4,700 daily records) |
| **Task** | 3-class classification |
| **Label** | 30-day forward price return: >+15% Bull, <−15% Bear, else Sideways |
| **Features** | 176 engineered → Top 40 selected by RF importance |
| **Split** | Time-ordered 80/20 (no shuffle, no data leakage) |

## Models & Results

| Model | Accuracy | F1 (macro) | ROC AUC |
|-------|----------|------------|---------|
| Logistic Regression | 0.1977 | 0.1843 | 0.5611 |
| **Random Forest** | **0.5781** | **0.3105** | 0.5229 |
| XGBoost | 0.2200 | 0.1575 | 0.5392 |
| LightGBM | 0.2232 | 0.1216 | 0.5764 |
| SVM | 0.2168 | 0.1218 | 0.3870 |
| **KNN** | 0.5770 | **0.3119** | 0.4974 |

All models tuned with `GridSearchCV` + `TimeSeriesSplit(n_splits=5)`. Class imbalance handled via `class_weight="balanced"` (or `sample_weight` for XGBoost).

## Project Structure

```
├── src/
│   ├── main.py                # Entry point
│   ├── config.py              # Paths and constants
│   ├── data_loader.py         # Data loading and cleaning
│   ├── feature_engineering.py # Label creation and feature engineering
│   ├── models.py              # Model definitions and hyperparameter grids
│   ├── training.py            # Train/test split, feature selection, GridSearchCV
│   ├── evaluation.py          # Metrics, confusion matrices, ROC curves
│   └── visualization.py      # Feature importance plots
├── output/                    # Generated charts and results
├── app.py                     # Streamlit dashboard
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Train models** (generates all charts to `output/`):
```bash
python -m src.main
```

**Launch dashboard**:
```bash
python -m streamlit run app.py
```

## Output Charts

| File | Description |
|------|-------------|
| `eda_label_distribution.png` | Label distribution & BTC price timeline |
| `*_confusion_matrix.png` | Confusion matrix per model (6 files) |
| `roc_curves_all_models.png` | ROC curves, all models, one-vs-rest |
| `feature_importance_*.png` | Top 20 feature importance per model |
| `model_summary_table.png` | Performance comparison table |
| `model_summary.csv` | Metrics in CSV format |
