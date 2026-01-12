# Overview

Purpose
- Provide a reproducible, modular pipeline to explore and build baseline predictive models for cryptocurrency daily prices (example: AAVE / USD).

What this project contains
- Data loading and cleaning utilities that detect date/close columns and standardize the time index.
- EDA helpers to visualize price, rolling means, distributions, volatility and ACF/PACF.
- Feature engineering that builds lag features, rolling statistics and next-day target.
- Training utilities for a baseline RandomForest, optional XGBoost, and optional LSTM sequence model.
- A small CLI runner `scripts/run_pipeline.py` to orchestrate load → EDA → features → training and save models into `models/`.

Design goals
- Small, focused Python modules for easy testing and reuse.
- Notebook-to-code translation: the original analyses and model experiments were converted into importable functions.
- Minimal dependencies by default; optional heavy packages (`xgboost`, `tensorflow`) are used only if installed.
