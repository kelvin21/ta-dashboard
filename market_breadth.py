"""
Fast vectorized market breadth helpers.

This module implements a speedy pattern for computing breadth statistics such as
“% of stocks above SMA(50)” using pandas groupby/transform and pandas_ta.

Typical input is a tidy DataFrame with columns:
    - Date   (datetime-like)
    - Ticker (string)
    - Adj Close (or Close)

Example
-------
from market_breadth import calculate_breadth_speedy

breadth = calculate_breadth_speedy(df_prices)
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd
import pandas_ta as ta  # type: ignore


def _ensure_date_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure explicit Date / Ticker columns exist for grouping."""
    if isinstance(df.index, pd.MultiIndex):
        # Bring MultiIndex levels out as columns if needed
        if not {"Date", "Ticker"}.issubset(df.columns):
            df = df.reset_index()
    return df


def calculate_breadth_speedy(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    sma_period: int = 50,
) -> pd.Series:
    """
    Calculate % of stocks above SMA(N) for each date in a fully vectorized way.

    Args:
        df: DataFrame with at least [Date, Ticker, price_col]
        price_col: column name containing adjusted/close prices
        sma_period: SMA window length (e.g. 50)

    Returns:
        pandas Series indexed by Date, values = % above SMA(period)
    """
    if df.empty:
        return pd.Series(dtype="float64")

    df = _ensure_date_ticker_columns(df).copy()

    if price_col not in df.columns:
        raise KeyError(f"price_col '{price_col}' not in DataFrame columns: {list(df.columns)}")

    # 1) Per‑ticker SMA, aligned with original rows
    df[f"SMA{sma_period}"] = (
        df.groupby("Ticker")[price_col]
        .transform(lambda x: x.ta.sma(sma_period))
    )

    # 2) Boolean mask: stock above its own SMA on each day
    df["AboveSMA"] = df[price_col] > df[f"SMA{sma_period}"]

    # 3) Breadth by date: mean of boolean → fraction of True
    breadth = df.groupby("Date")["AboveSMA"].mean() * 100.0
    breadth.name = f"above_sma{sma_period}_pct"

    return breadth


def calculate_multi_sma_breadth(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
    periods: Iterable[int] = (20, 50, 200),
) -> pd.DataFrame:
    """
    Compute breadth for multiple SMA windows in one pass.

    Returns a DataFrame indexed by Date with one column per SMA period:
        ['above_sma20_pct', 'above_sma50_pct', ...]
    """
    if df.empty:
        return pd.DataFrame()

    df = _ensure_date_ticker_columns(df).copy()

    if price_col not in df.columns:
        raise KeyError(f"price_col '{price_col}' not in DataFrame columns: {list(df.columns)}")

    # Compute all SMAs per ticker
    for period in periods:
        col = f"SMA{period}"
        df[col] = df.groupby("Ticker")[price_col].transform(
            lambda x, p=period: x.ta.sma(p)
        )
        mask_col = f"AboveSMA{period}"
        df[mask_col] = df[price_col] > df[col]

    results: Mapping[int, pd.Series] = {}
    for period in periods:
        mask_col = f"AboveSMA{period}"
        series = df.groupby("Date")[mask_col].mean() * 100.0
        series.name = f"above_sma{period}_pct"
        results[period] = series

    # Align into a single DataFrame
    out = pd.concat(results.values(), axis=1)
    out.index.name = "Date"
    return out



