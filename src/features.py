"""Feature engineering extracted from the notebook: lags, rolling means/stds and target."""
import pandas as pd
from typing import List


def build_features_close(df: pd.DataFrame, lags: List[int] = [1,2,3,7,14], windows: List[int] = [7,14,30]) -> pd.DataFrame:
    df_feat = df[['close']].copy()
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat['close'].shift(lag)
    for w in windows:
        df_feat[f'roll_mean_{w}'] = df_feat['close'].rolling(window=w).mean().shift(1)
        df_feat[f'roll_std_{w}'] = df_feat['close'].rolling(window=w).std().shift(1)
    df_feat['target'] = df_feat['close'].shift(-1)
    df_feat = df_feat.dropna().copy()
    return df_feat
