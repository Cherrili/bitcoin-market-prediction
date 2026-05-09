"""
Step 3 & 4 — Label creation and feature engineering.

Public API
----------
create_labels(df)                    -> df  (adds label, return_30d columns)
build_features(df) -> (df, list[str])       (adds rolling / lag / momentum features)
"""

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "mempool_size", "transaction_rate", "market_cap_usd_bc",
    "average_block_size", "exchange_volume_usd", "average_confirmation_time",
    "hash_rate", "difficulty", "miners_revenue", "total_transaction_fees",
    "total_supply", "realised_cap_usd", "nupl",
    "coin_days_destroyed", "active_addresses", "fear_greed_value",
    "lightning_nodes", "lightning_capacity_usd", "price",
]

_EXCLUDE_COLS = {"datetime", "label", "future_price_30d", "return_30d"}


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach 30-day forward-return labels:
        Bull  ( 1) : return > +15 %
        Bear  (-1) : return < -15 %
        Side  ( 0) : otherwise
    Drops the last 30 rows where the future price is unknown.
    """
    df = df.copy()
    df["future_price_30d"] = df["price"].shift(-30)
    df["return_30d"] = (df["future_price_30d"] - df["price"]) / df["price"]

    def _label(ret: float) -> int:
        if   ret >  0.15: return  1
        elif ret < -0.15: return -1
        else:             return  0

    df["label"] = df["return_30d"].apply(_label)
    df.dropna(subset=["future_price_30d"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    counts = df["label"].value_counts().sort_index()
    print(f"   Label distribution : {counts.to_dict()}")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Add rolling-mean (7/14/30 d), lag (1/3/7/14/30 d), momentum, and
    MVRV features. Returns the enriched DataFrame and the final feature
    column list (excludes date / label / target columns).
    """
    df = df.copy()
    base = [c for c in BASE_FEATURES if c in df.columns]

    # Rolling means
    for col in base:
        for w in [7, 14, 30]:
            df[f"{col}_ma{w}"] = df[col].rolling(w, min_periods=1).mean()

    # Lag features
    for col in base:
        for lag in [1, 3, 7, 14, 30]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Price momentum
    df["price_pct_7d"]         = df["price"].pct_change(7)
    df["price_pct_14d"]        = df["price"].pct_change(14)
    df["price_pct_30d"]        = df["price"].pct_change(30)
    df["price_volatility_30d"] = (df["price"].rolling(30).std()
                                  / df["price"].rolling(30).mean())

    # MVRV ratio
    df["mvrv"] = (df["market_cap_usd_bc"]
                  / df["realised_cap_usd"].replace(0, np.nan))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

    print(f"   Features      : {len(feature_cols)}")
    print(f"   Dataset shape : {df.shape}")
    print(f"   Date range    : {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    return df, feature_cols
