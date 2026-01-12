"""Small orchestration script to run the pipeline end-to-end using the modules."""
import argparse
import os
import numpy as np

from crypto_trend_prediction.src.data_loader import load_csv, load_aave_default
from crypto_trend_prediction.src.eda import plot_close_and_rms, plot_log_return_dist, correlation_heatmap
from crypto_trend_prediction.src.features import build_features_close
from crypto_trend_prediction.src.train import train_random_forest, train_xgboost, train_lstm_sequence
from sklearn.preprocessing import StandardScaler


def main(csv: str = None):
    if csv is None:
        data = load_aave_default()
    else:
        data = load_csv(csv)

    print('Data loaded, rows:', len(data))
    # Quick EDA (plots will show when run interactively)
    try:
        plot_close_and_rms(data)
        plot_log_return_dist(data)
        correlation_heatmap(data)
    except Exception as e:
        print('EDA plots failed (maybe running headless):', e)

    df_feat = build_features_close(data)
    split_frac = 0.8
    split_index = int(len(df_feat) * split_frac)
    train = df_feat.iloc[:split_index].copy()
    test = df_feat.iloc[split_index:].copy()
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    X_test = test.drop(columns=['target'])
    y_test = test['target']
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    save_dir = './crypto_trend_prediction/models'
    print('Training RandomForest baseline...')
    mae, rmse, model, r2, dir_acc = train_random_forest(X_train_s, y_train.values, X_test_s, y_test.values, save_dir=save_dir, X_test_raw=X_test)
    print(f'RF MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}' + (f", DirAcc: {dir_acc:.3%}" if dir_acc is not None else ", DirAcc: N/A"))

    # Optional: XGBoost (if installed)
    try:
        print('Attempting to train XGBoost (if available)...')
        mae_xgb, rmse_xgb, xgb_model, r2_xgb, dir_acc_xgb = train_xgboost(X_train_s, y_train.values, X_test_s, y_test.values, save_dir=save_dir, X_test_raw=X_test)
        print(f'XGB MAE: {mae_xgb:.4f}, RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}' + (f", DirAcc: {dir_acc_xgb:.3%}" if dir_acc_xgb is not None else ", DirAcc: N/A"))
    except Exception as e:
        print('XGBoost training skipped or failed:', e)

    # Optional: LSTM (if TensorFlow available) — train on the close series directly
    try:
        print('Attempting to train LSTM (if TensorFlow available)...')
        arr = data['close'].values
        mae_lstm, rmse_lstm, lstm_model, r2_lstm, dir_acc_lstm = train_lstm_sequence(arr, seq_len=14, save_dir=save_dir)
        print(f'LSTM MAE: {mae_lstm:.4f}, RMSE: {rmse_lstm:.4f}, R2: {r2_lstm:.4f}' + (f", DirAcc: {dir_acc_lstm:.3%}" if dir_acc_lstm is not None else ", DirAcc: N/A"))
    except Exception as e:
        print('LSTM training skipped or failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=None, help='Path to aave csv')
    args = parser.parse_args()
    main(args.csv)
