"""Exploratory Data Analysis utilities extracted from the notebook."""
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


sns.set(style='darkgrid')


def plot_close_and_rms(data: pd.DataFrame, windows=(7, 30, 90)) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(data.index, data['close'], label='Close', alpha=0.8)
    for w in windows:
        data[f'rm_{w}'] = data['close'].rolling(window=w).mean()
        plt.plot(data.index, data[f'rm_{w}'], label=f'RM {w}d')
    plt.title('Close Price and Rolling Means')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def plot_log_return_dist(data: pd.DataFrame) -> None:
    data['log_return'] = np.log(data['close']).diff()
    plt.figure(figsize=(10, 4))
    sns.histplot(data['log_return'].dropna(), bins=80, kde=True)
    plt.title('Log Return Distribution')
    plt.show()


def correlation_heatmap(data: pd.DataFrame) -> None:
    numeric = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt='.2f', cmap='vlag')
    plt.title('Numeric Feature Correlations')
    plt.show()


def volatility_and_range(data: pd.DataFrame) -> None:
    cols_map = {c.lower(): c for c in data.columns}
    if 'high' in cols_map and 'low' in cols_map:
        high_col, low_col = cols_map['high'], cols_map['low']
        data['range'] = data[high_col] - data[low_col]
    else:
        data['range'] = data['close'].rolling(window=2).apply(lambda x: x.max() - x.min(), raw=True)
    data['range_rm_14'] = data['range'].rolling(14).mean()
    data['range_rm_30'] = data['range'].rolling(30).mean()
    plt.figure(figsize=(12, 4))
    plt.plot(data.index, data['range'], label='Daily Range', alpha=0.6)
    plt.plot(data.index, data['range_rm_14'], label='RM 14')
    plt.plot(data.index, data['range_rm_30'], label='RM 30')
    plt.title('Price Range (High-Low) and Rolling Means')
    plt.legend()
    plt.show()
    data['log_return'] = np.log(data['close']).diff()
    data['vol_30'] = data['log_return'].rolling(30).std()
    plt.figure(figsize=(12, 3))
    plt.plot(data.index, data['vol_30'], label='30-day vol (log returns)')
    plt.title('30-day Volatility (std of log returns)')
    plt.legend()
    plt.show()


def acf_pacf_plots(data: pd.DataFrame, max_lags: Optional[int] = None) -> None:
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        lags = min(60, int(len(data) / 2)) if len(data) > 10 else 10
        if max_lags is not None:
            lags = min(lags, max_lags)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plot_acf(data['close'].dropna(), lags=lags, ax=plt.gca(), title='ACF of Close')
        plt.subplot(2, 1, 2)
        plot_pacf(data['close'].dropna(), lags=lags, ax=plt.gca(), title='PACF of Close', method='ywm')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('statsmodels tsaplots not available or failed:', e)
