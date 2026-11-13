# Store Sales – Time Series Forecasting

Forecasting 16 days of daily sales for 1,782 store × product-family series of Corporación Favorita (Ecuador), from Kaggle's [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition. The focus is on time series analysis and regression done properly: honest time-based backtesting, no leakage, and a written reason for every decision.

## Results

Backtest: 4 consecutive 16-day windows ending 2017-08-15, expanding training window, metric RMSLE (lower is better). Mean ± std across folds.

| Model | Backtest RMSLE | Kaggle public score |
|---|---|---|
| *(filled in from Phase 3 onwards)* | | |

Full per-fold table: [`results.csv`](results.csv).

## Project structure

```text
data/raw/            Kaggle files, read-only (not in git)
data/processed/      backtest predictions of every model, cached features (not in git)
notebooks/           00_setup_smoke_test, 01_eda ... 08_ensemble_submission
src/
  cv.py              seed, paths, backtest folds, rmsle/evaluate, log_result, show_results
  features.py        calendar, holiday, lag and rolling features (Phase 2)
  models.py          model wrappers (Phase 3+)
reports/figures/     saved plots, prefixed with the phase number
submissions/         named with the date and backtest score (not in git)
results.csv          one row per model, per-fold scores
```

## How to reproduce

```bash
git clone <repo-url> && cd store-sales
bash setup.sh                       # venv (Python 3.12), packages, Jupyter kernel, folders
export KAGGLE_API_TOKEN=...         # from kaggle.com/settings/api, after joining the competition
bash download_data.sh               # downloads and unzips into data/raw, makes the files read-only
```

Then run the notebooks in order with the `Python 3.12 (store-sales)` kernel.

## Decision log

### Phase 0 – Setup

- **Folds are defined once, in `src/cv.py`, before any EDA.** Four consecutive, non-overlapping 16-day validation windows ending on the last training day (2017-06-13 → 2017-08-15), each with an expanding training window that stops the day before. 16 days mirrors the test horizon, and there is no gap between train and validation because the test period also starts the day after the training data ends. Every model is scored on exactly these folds.
- **Metric is computed on the sales scale with predictions clipped at 0**, exactly as Kaggle scores the submission. Models will predict log1p(sales); the conversion back with expm1 happens before scoring.
- **`evaluate()` refuses incomplete predictions.** It checks that every series × day of every window is predicted exactly once, so a model can't look better by dropping hard rows.
- **One results table.** `results.csv` has a fixed column layout (mean, sample std, one column per fold, params, notes). Re-running a model replaces its row; an empty file is treated as missing. Each model's backtest predictions are saved to `data/processed/backtest_preds/` for blending in Phase 8.
- **Raw data stays out of git and is read-only on disk**, so no notebook can overwrite it by accident.
- **Pinned library versions** in `requirements.txt` (pandas 3.1, scikit-learn 1.9, LightGBM 4.7, statsmodels 0.15), so results can be reproduced.
