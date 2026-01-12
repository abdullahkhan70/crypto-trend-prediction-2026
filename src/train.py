"""Model training utilities: RandomForest, optional XGBoost and LSTM (conditional)."""
from typing import Tuple
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, save_dir: str = './crypto_trend_prediction/models', X_test_raw=None) -> Tuple[float, float, object, float, float]:
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    # directional accuracy: requires the current close for each test row
    dir_acc = None
    try:
        if X_test_raw is not None:
            # X_test_raw may be a DataFrame-like or numpy array with 'close' as first column
            if hasattr(X_test_raw, 'get') or hasattr(X_test_raw, 'iloc'):
                current = X_test_raw['close'].values
            else:
                current = X_test_raw[:, 0]
            actual_dir = np.sign(y_test - current)
            pred_dir = np.sign(pred - current)
            dir_acc = float((actual_dir == pred_dir).mean())
    except Exception:
        dir_acc = None
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(save_dir, 'rf_aave.joblib'))
    return mae, rmse, rf, r2, dir_acc


def train_xgboost(X_train, y_train, X_test, y_test, save_dir: str = './crypto_trend_prediction/models', X_test_raw=None):
    if XGBRegressor is None:
        raise RuntimeError('XGBoost not installed')
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    dir_acc = None
    try:
        if X_test_raw is not None:
            if hasattr(X_test_raw, 'get') or hasattr(X_test_raw, 'iloc'):
                current = X_test_raw['close'].values
            else:
                current = X_test_raw[:, 0]
            actual_dir = np.sign(y_test - current)
            pred_dir = np.sign(pred - current)
            dir_acc = float((actual_dir == pred_dir).mean())
    except Exception:
        dir_acc = None
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(xgb, os.path.join(save_dir, 'xgb_aave.joblib'))
    return mae, rmse, xgb, r2, dir_acc


def train_lstm_sequence(arr: np.ndarray, seq_len: int = 14, save_dir: str = './crypto_trend_prediction/models'):
    if keras is None:
        raise RuntimeError('TensorFlow/Keras not installed')
    X_seq, y_seq = [], []
    for i in range(len(arr) - seq_len):
        X_seq.append(arr[i:i+seq_len])
        y_seq.append(arr[i+seq_len])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    split_idx = int(len(X_seq) * 0.8)
    Xtr, Xte = X_seq[:split_idx], X_seq[split_idx:]
    ytr, yte = y_seq[:split_idx], y_seq[split_idx:]
    scaler_lstm = MinMaxScaler()
    Xtr_flat = Xtr.reshape(-1,1)
    scaler_lstm.fit(Xtr_flat)
    Xtr_s = scaler_lstm.transform(Xtr_flat).reshape(Xtr.shape[0], seq_len, 1)
    Xte_s = scaler_lstm.transform(Xte.reshape(-1,1)).reshape(Xte.shape[0], seq_len, 1)
    ytr_s = scaler_lstm.transform(ytr.reshape(-1,1)).reshape(-1,1)
    yte_s = scaler_lstm.transform(yte.reshape(-1,1)).reshape(-1,1)
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len,1)),
        keras.layers.LSTM(64),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(Xtr_s, ytr_s, validation_split=0.1, epochs=30, batch_size=32, verbose=2)
    pred_s = model.predict(Xte_s).reshape(-1,1)
    pred = scaler_lstm.inverse_transform(pred_s).reshape(-1)
    yte_inv = scaler_lstm.inverse_transform(yte_s).reshape(-1)
    mae = mean_absolute_error(yte_inv, pred)
    rmse = np.sqrt(mean_squared_error(yte_inv, pred))
    # R^2 and directional accuracy for sequence model
    try:
        r2 = r2_score(yte_inv, pred)
    except Exception:
        r2 = None
    try:
        prev_close = Xte[:, -1]
        actual_dir = np.sign(yte_inv - prev_close)
        pred_dir = np.sign(pred - prev_close)
        dir_acc = float((actual_dir == pred_dir).mean())
    except Exception:
        dir_acc = None
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'lstm_aave.h5'))
    joblib.dump(scaler_lstm, os.path.join(save_dir, 'scaler_lstm_aave.joblib'))
    return mae, rmse, model, r2, dir_acc
