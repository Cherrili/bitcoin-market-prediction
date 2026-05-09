"""
Step 1 & 2 — Data loading, merging, and cleaning.

Public API
----------
load_and_clean(data_dir) -> pd.DataFrame
"""

import numpy as np
import pandas as pd


def load_and_clean(data_dir: str) -> pd.DataFrame:
    """
    Load both daily CSV files, merge on date, clean, and return a
    single tidy DataFrame with a unified `price` column.
    """
    df1 = pd.read_csv(f"{data_dir}/blockchain_dot_com_daily_data.csv",
                      parse_dates=["datetime"])
    df2 = pd.read_csv(f"{data_dir}/look_into_bitcoin_daily_data.csv",
                      parse_dates=["datetime"])

    df = (pd.merge(df1, df2, on="datetime", how="inner",
                   suffixes=("_bc", "_lib"))
            .sort_values("datetime")
            .reset_index(drop=True))

    print(f"   Merged shape : {df.shape}")
    print(f"   Date range   : {df['datetime'].min().date()} → {df['datetime'].max().date()}")

    # ── drop non-numeric ───────────────────────────────────────────────────────
    df.drop(columns=["fear_greed_category"], inplace=True, errors="ignore")

    # ── unified price column ───────────────────────────────────────────────────
    df["price"] = df["market_price_usd_lib"].replace(0, np.nan)
    df["price"].fillna(df["market_price_usd_bc"].replace(0, np.nan), inplace=True)
    df = df[df["price"] > 0].reset_index(drop=True)

    # ── drop redundant duplicates ──────────────────────────────────────────────
    df.drop(columns=["market_price_usd_bc", "market_price_usd_lib",
                     "market_cap_usd_lib"], inplace=True, errors="ignore")

    # ── fill remaining NaNs ────────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    print(f"   Rows after cleaning : {len(df)}")
    print(f"   Remaining NaNs      : {df[numeric_cols].isna().sum().sum()}")
    return df
