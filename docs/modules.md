# Module Reference

This document briefly describes the main modules inside `crypto_trend_prediction/src`.

- `data_loader.py`: Functions to load CSVs and standardize time index and close column.
- `eda.py`: Exploratory plots (rolling means, distribution, ACF/PACF).
- `features.py`: Feature engineering — lags, rolling windows, target creation.
- `train.py`: Model training helpers (`train_random_forest`, `train_xgboost`, `train_lstm_sequence`) and model saving/loading.

See in-code docstrings for parameter details.
