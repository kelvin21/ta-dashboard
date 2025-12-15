# Market Breadth – Fast Vectorized Implementation

## Overview

This document describes a **fast, vectorized** implementation of market breadth
for the MACD Reversal Dashboard, using `pandas_ta` and grouped operations on a
multi-ticker DataFrame. The goal is to avoid Python loops and compute breadth
indicators (e.g. "% of stocks above SMA(50)") in a single pass over the data.

The same pattern can be extended to EMA breadth, RSI breadth, MACD stage
distributions, etc., and the results can be stored in MongoDB via
`db_adapter.py` / `utils/db_async.py`.

---

## Core Idea: Vectorized Breadth with `pandas_ta`

- **Input**: a tidy/multi-index-friendly DataFrame with:
  - columns: at minimum `Date`, `Ticker`, `Adj Close`
  - index: either simple RangeIndex, or a MultiIndex `(Date, Ticker)`
- **Per‑ticker indicators**: computed with `groupby('Ticker')` +
  `.transform(...)` so they remain aligned with the original rows.
- **Breadth**: computed with `groupby('Date')` on boolean masks and taking the
  mean, which is the fraction of tickers satisfying a condition on that date.

### Example: % of Stocks Above SMA(50)

```python
import pandas as pd
import pandas_ta as ta

# df columns required: 'Date', 'Ticker', 'Adj Close'
# It can be index=RangeIndex, or a MultiIndex (Date, Ticker).

def calculate_breadth_speedy(df: pd.DataFrame) -> pd.Series:
    # Ensure we have explicit Date / Ticker columns for grouping
    if isinstance(df.index, pd.MultiIndex):
        if "Date" not in df.columns:
            df = df.reset_index()  # brings MultiIndex levels into columns

    # 1) Per‑ticker SMA(50), fully vectorized
    df["SMA50"] = (
        df.groupby("Ticker")["Adj Close"]
          .transform(lambda x: x.ta.sma(50))
    )

    # 2) Boolean mask: stock above its own SMA(50) on each day
    df["AboveSMA50"] = df["Adj Close"] > df["SMA50"]

    # 3) Breadth: percentage of stocks above SMA(50) by date
    #    mean() on a boolean Series → fraction of True values
    breadth_indicator = df.groupby("Date")["AboveSMA50"].mean() * 100.0

    return breadth_indicator  # pd.Series indexed by Date
```

**Key properties:**

- **No Python loops over tickers or dates** – everything is done with
  `groupby().transform()` and `groupby().mean()` in C-optimized Pandas code.
- **O(N)** over number of rows, independent of the number of tickers.
- Naturally handles:
  - missing data for some tickers on some dates,
  - adding/removing tickers (e.g. IPOs, delistings).

---

## Integrating with the Existing Architecture

### Data Flow

- **Source price data**:
  - Loaded from MongoDB `price_data` (via `db_adapter.py` or `utils/db_async.py`)
    into a DataFrame with columns like:
    - `date` (datetime), `ticker`, `close` / `adj_close`, `volume`, etc.
  - For the snippet above, you map to:
    - `Date`  ← `date`
    - `Ticker` ← `ticker`
    - `Adj Close` ← `close` or `adj_close` (depending on your schema)

- **Breadth calculation**:
  - Build a single DataFrame for the full universe and date range.
  - Call `calculate_breadth_speedy(df_mapped)` to get a `pd.Series`:
    - index: `Date`
    - value: `% of stocks above SMA(50)` on that date.

- **Storage**:
  - Either keep this Series in memory for the current Streamlit session, or
  - Persist into a `market_breadth` collection, e.g.:

```javascript
{
  date: ISODate("2025-12-08"),
  above_sma50_pct: 56.7,
  total_tickers: 150,
  updated_at: ISODate("2025-12-08T10:30:00Z")
}
```

---

## Extending the Pattern to Other Breadth Metrics

- **EMA breadth**: replace `x.ta.sma(50)` with `x.ta.ema(length)` for any period.
- **Multiple windows**: compute several SMAs/EMAs in one function:
  - `SMA20`, `SMA50`, `SMA200`, then produce 3 breadth series.
- **RSI breadth**:
  - `df["RSI14"] = df.groupby("Ticker")["Adj Close"].transform(lambda x: x.ta.rsi(14))`
  - then `df["RSI_Oversold"] = df["RSI14"] < 30`, and group by `Date` as above.
- **MACD stage counts**:
  - compute MACD/Signal/Hist per ticker (via `pandas_ta.macd` or `utils/macd_stage.py`)
  - map each row to a discrete stage (e.g. `"confirmed_trough"`)
  - group by `Date` and `stage` and normalize to get distributions.

All of these can be implemented with the same pattern:

- per‑ticker indicator: `groupby("Ticker").transform(...)`
- daily breadth: `groupby("Date")` on boolean or categorical columns.

---

## Usage in `1_📊_Market_Breadth.py`

- When recalculating historical breadth:
  - Load the required window of `price_data` from MongoDB.
  - Build a tidy DataFrame with `Date`, `Ticker`, `Adj Close`.
  - Call the fast vectorized function(s) to compute breadth series.
  - Optionally save results back to MongoDB for future sessions.

- For interactive charts:
  - Convert the breadth Series to a DataFrame:

```python
breadth = calculate_breadth_speedy(df_prices)
breadth_df = breadth.reset_index().rename(columns={"AboveSMA50": "above_sma50_pct"})
```

  - Plot with Plotly as a time series, and overlay with VNINDEX.

---

## Dependencies

- **Required**:
  - `pandas`
  - `pandas_ta`
- **Recommended** (existing project stack):
  - `streamlit`, `plotly`, `pymongo`, `python-dotenv`

```bash
pip install pandas pandas_ta
```

This approach plugs into the existing MACD Reversal Dashboard without changing
the UI, while making breadth calculations **significantly faster and more
scalable** as the number of tickers and dates grows.