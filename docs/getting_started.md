# Getting Started

Prerequisites
- Python 3.8+ recommended
- Install required packages:

```bash
pip install -r requirements.txt
```

Data
- Place the `aave_usd_day.csv` file in the `data/` folder. The repository already includes example CSVs under `data/`.

Run the pipeline

From the repository root run as a module so Python can find the package:

```powershell
cd "F:\Necessary Data\Kaggle\own-datatset\free-crypto-api"
python -m crypto_trend_prediction.scripts.run_pipeline --csv "crypto_trend_prediction/data/aave_usd_day.csv"
```

Notes
- If you run the script directly from inside `scripts/`, set `PYTHONPATH` to the project root or run the module form above. The runner will attempt XGBoost and TensorFlow only if those packages are installed.
