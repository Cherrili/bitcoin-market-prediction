"""
Step 1 & 2 — Data loading, merging, and cleaning.

Primary sources (daily frequency):
  1. blockchain_dot_com_daily_data      — transaction / mempool metrics
  2. look_into_bitcoin_daily_data       — NUPL, MVRV, fear/greed, on-chain

Additional sources explored but excluded after ablation:
  - look_into_bitcoin_hodl_waves_data          — HODL wave coin-age bands
  - look_into_bitcoin_realised_cap_hodl_waves  — realised-cap HODL waves
  - look_into_bitcoin_address_balances_data    — whale address counts
  These files exhibit structural non-stationarity over 2010-2023 (Bitcoin
  maturation effect): long-duration bands are zero until ~2015 and grow
  monotonically thereafter. Including them degraded F1 by ~30% across all
  models in ablation experiments; details are reported in Section 5.

Public API
----------
load_and_clean(data_dir) -> pd.DataFrame
"""

import numpy as np
import pandas as pd


def load_and_clean(data_dir: str) -> pd.DataFrame:
    """
    Load both primary daily CSV files, merge on date, clean, and return a
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
    df["price"] = (df["market_price_usd_lib"].replace(0, np.nan)
                   .fillna(df["market_price_usd_bc"].replace(0, np.nan)))
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
