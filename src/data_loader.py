"""Data loading and basic cleaning functions extracted from the notebook."""
from typing import Optional
import pandas as pd
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV and standardize a `date` index and a numeric `close` column.

    Behavior mirrors the notebook: detect a date-like column, coerce close-like column to numeric,
    drop rows without close and sort by date.
    """
    df = pd.read_csv(path)
    data = df.copy()

    # detect date column
    date_cols = [c for c in data.columns if 'date' in c.lower() or 'time' in c.lower() or 'timestamp' in c.lower()]
    if len(date_cols) > 0:
        date_col = date_cols[0]
        data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
        data = data.sort_values(by=date_col).set_index(date_col)
    else:
        try:
            data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], errors='coerce')
            data = data.set_index(data.columns[0])
        except Exception:
            pass

    # detect close column
    close_cols = [c for c in data.columns if c.lower() in ('close', 'close_price', 'close_usd', 'price')]
    if len(close_cols) == 0:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        close_col = numeric_cols[-1] if numeric_cols else None
    else:
        close_col = close_cols[0]

    if close_col is None:
        raise ValueError('No numeric column found to use as close price')

    data['close'] = pd.to_numeric(data[close_col], errors='coerce')
    data = data[~data['close'].isna()].copy()
    data.index.name = 'date'
    data = data.sort_index()
    return data


def load_aave_default() -> pd.DataFrame:
    """Convenience loader for the provided aave csv if present in ../crypto-trend-prediction/data/"""
    # try current workspace common path
    import os
    candidates = [
        os.path.join(os.getcwd(), 'data', 'aave_usd_day.csv'),
        os.path.join(os.getcwd(), '..', 'crypto-trend-prediction', 'data', 'aave_usd_day.csv'),
        os.path.join(os.getcwd(), '..', 'data', 'aave_usd_day.csv'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return load_csv(p)
    raise FileNotFoundError('aave_usd_day.csv not found in expected locations; pass path to load_csv')
