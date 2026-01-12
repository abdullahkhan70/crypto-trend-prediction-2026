# crypto_trend_prediction

This small package contains code extracted from the AAVE notebook for: loading data, EDA, feature engineering and baseline model training.

Usage (quick):

1. Put your CSV somewhere accessible (e.g. `data/aave_usd_day.csv`).
2. Run the pipeline: `python -m crypto_trend_prediction.scripts.run_pipeline --csv PATH_TO_CSV` or `cd "F:\Necessary Data\Kaggle\own-datatset\free-crypto-api"; python -m crypto_trend_prediction.scripts.run_pipeline --csv "F:\Necessary Data\Kaggle\own-datatset\free-crypto-api\crypto_trend_prediction\data\aave_usd_day.csv"`

See `requirements.txt` for dependencies.
