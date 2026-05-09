"""
Step 6 — Time-ordered train/test split and GridSearchCV training.

Public API
----------
split_data(df, feature_cols)                    -> SplitResult (dataclass)
select_top_features(split, feature_cols, top_n) -> (SplitResult, list[str])
train_models(configs, split)                    -> dict[name -> {model, best_p, scaled, encoded}]
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as _RFC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import LBL_MAP


@dataclass
class SplitResult:
    X_train:     np.ndarray
    X_test:      np.ndarray
    y_train:     np.ndarray
    y_test:      np.ndarray
    X_train_sc:  np.ndarray   # StandardScaler-transformed
    X_test_sc:   np.ndarray
    y_train_enc: np.ndarray   # labels mapped to {0,1,2} for XGB/LGB
    y_test_enc:  np.ndarray
    scaler:      StandardScaler
    dates:       pd.Series
    split_idx:   int


def split_data(df: pd.DataFrame, feature_cols: list,
               train_ratio: float = 0.8) -> SplitResult:
    """
    Strict time-ordered split — no shuffling.
    Fit StandardScaler on training data only.
    """
    X      = df[feature_cols].values
    y      = df["label"].values
    dates  = df["datetime"]

    split_idx = int(len(df) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train : {len(X_train)} rows  "
          f"({dates.iloc[0].date()} → {dates.iloc[split_idx-1].date()})")
    print(f"   Test  : {len(X_test)} rows  "
          f"({dates.iloc[split_idx].date()} → {dates.iloc[-1].date()})")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    encode     = np.vectorize(LBL_MAP.get)
    y_train_enc = encode(y_train)
    y_test_enc  = encode(y_test)

    return SplitResult(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        X_train_sc=X_train_sc, X_test_sc=X_test_sc,
        y_train_enc=y_train_enc, y_test_enc=y_test_enc,
        scaler=scaler, dates=dates, split_idx=split_idx,
    )


def select_top_features(split: SplitResult, feature_cols: list,
                        top_n: int = 40) -> tuple:
    """
    Fit a quick RF on training data only, keep the top_n features by
    importance. Re-fits StandardScaler on the reduced feature set.
    Returns an updated SplitResult and the filtered feature name list.
    """
    rf = _RFC(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(split.X_train, split.y_train)

    top_idx = sorted(
        range(len(feature_cols)),
        key=lambda i: rf.feature_importances_[i],
        reverse=True,
    )[:top_n]
    top_idx = sorted(top_idx)   # restore column order

    selected_cols = [feature_cols[i] for i in top_idx]
    print(f"   Reduced {len(feature_cols)} → {len(selected_cols)} features")

    X_train_sel = split.X_train[:, top_idx]
    X_test_sel  = split.X_test[:, top_idx]

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train_sel)
    X_test_sc   = scaler.transform(X_test_sel)

    new_split = SplitResult(
        X_train=X_train_sel,    X_test=X_test_sel,
        y_train=split.y_train,  y_test=split.y_test,
        X_train_sc=X_train_sc,  X_test_sc=X_test_sc,
        y_train_enc=split.y_train_enc, y_test_enc=split.y_test_enc,
        scaler=scaler, dates=split.dates, split_idx=split.split_idx,
    )
    return new_split, selected_cols


def train_models(configs: list, split: SplitResult,
                 n_cv_splits: int = 5) -> dict:
    """
    Run GridSearchCV with TimeSeriesSplit for every model config.
    Returns a dict keyed by model name.
    """
    tscv    = TimeSeriesSplit(n_splits=n_cv_splits)
    trained = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"   [{name}]  grid={cfg['params']} …", end=" ", flush=True)

        Xtr = split.X_train_sc if cfg["scaled"] else split.X_train
        ytr = split.y_train_enc if cfg["encoded"] else split.y_train

        gs = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["params"],
            cv=tscv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        fit_params = {}
        if cfg.get("use_sample_weight"):
            fit_params["sample_weight"] = compute_sample_weight("balanced", ytr)
        gs.fit(Xtr, ytr, **fit_params)

        print(f"best={gs.best_params_}")
        trained[name] = {
            "model":   gs.best_estimator_,
            "best_p":  gs.best_params_,
            "scaled":  cfg["scaled"],
            "encoded": cfg["encoded"],
        }

    return trained
