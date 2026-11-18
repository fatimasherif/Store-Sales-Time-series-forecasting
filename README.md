# Store Sales – Time Series Forecasting

Forecasting 16 days of daily sales for 1,782 store × product-family series of Corporación Favorita (Ecuador), from Kaggle's [Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition. The focus is on time series analysis and regression done properly: honest time-based backtesting, no leakage, and a written reason for every decision.

## Results

Backtest: 4 consecutive 16-day windows ending 2017-08-15, expanding training window, metric RMSLE (lower is better). Mean ± std across folds; fold 4 (Jul 31 – Aug 15) is shown separately because it is the window closest to the test period and the hardest for every model so far.

| Model | Phase | Backtest RMSLE | Fold 4 | Kaggle public score |
|---|---|---|---|---|
| Recent level × weekday shape (`ma14_x_dow8w`) | 3 | **0.4403 ± 0.0563** | 0.5244 | *pending* |
| Weekday log-mean, 8 weeks (`dow_8w`) | 3 | 0.4478 ± 0.0557 | 0.5285 | |
| Log-mean, 28 days (`ma_28`) | 3 | 0.4722 ± 0.0439 | 0.5378 | |
| Arithmetic mean, 28 days (`ma_28_raw`) | 3 | 0.4817 ± 0.0279 | 0.5213 | |
| Seasonal naive, same weekday last week | 3 | 0.5441 ± 0.0484 | 0.6166 | |
| Last value | 3 | 0.6080 ± 0.0330 | 0.6565 | |
| Predict zero | 3 | 4.4093 ± 0.0207 | 4.4195 | |

Full per-fold table, including every variant: [`results.csv`](results.csv).

**Key findings so far**

- Ten product families were not recorded in 2013 and switch on and off until mid-2015, so training starts on **2015-06-01** (detected from the data). 8.1% of all rows are days a store was closed; both kinds of zero are treated as missing, not as demand.
- Sales correlate by **weekday, not by recency**: per-series autocorrelation is 0.32 at lag 7, 0.03 at lag 16, 0.20 at lag 21. Lag features use same-weekday lags of at least 16 days, and a truncation test proves no feature leaks.
- Fold 4 is about 0.10 worse for every model, and SCHOOL AND OFFICE SUPPLIES (0.03% of sales) causes 17% of the best baseline's error, which points to an August back-to-school surge right where the test period sits (being checked in Phase 4).

## Project structure

```text
data/raw/            Kaggle files, read-only (not in git)
data/processed/      cached feature table and backtest predictions of every model (not in git)
notebooks/           00_setup_smoke_test, 01_eda, 02_features, 03_baselines, ... 08_ensemble_submission
src/
  cv.py              seed, paths, backtest and tuning folds, rmsle/evaluate, log_result, show_results
  features.py        clean() and the feature table: calendar, holidays, promotions, lags and rolling stats
  models.py          baselines and model wrappers, backtest(), submission()
reports/figures/     saved plots, prefixed with the phase number
submissions/         named with the date and backtest score (not in git)
results.csv          one row per model, per-fold scores
```

## How to reproduce

```bash
git clone https://github.com/fatimasherif/Store-Sales-Time-series-forecasting.git && cd Store-Sales-Time-series-forecasting
bash setup.sh                       # venv (Python 3.12), packages, Jupyter kernel, folders
export KAGGLE_API_TOKEN=...         # from kaggle.com/settings/api, after joining the competition
bash download_data.sh               # downloads and unzips into data/raw, makes the files read-only
```

Then run the notebooks in order with the `Python 3.12 (store-sales)` kernel. Notebook 02 builds `data/processed/features.parquet`, which later notebooks read.

## Decision log

### Phase 0 – Setup

- **Folds are defined once, in `src/cv.py`, before any EDA.** Four consecutive, non-overlapping 16-day validation windows ending on the last training day (2017-06-13 → 2017-08-15), each with an expanding training window that stops the day before. 16 days mirrors the test horizon, and there is no gap between train and validation because the test period also starts the day after the training data ends. Every model is scored on exactly these folds.
- **Metric is computed on the sales scale with predictions clipped at 0**, exactly as Kaggle scores the submission. Models will predict log1p(sales); the conversion back with expm1 happens before scoring.
- **`evaluate()` refuses incomplete predictions.** It checks that every series × day of every window is predicted exactly once, so a model can't look better by dropping hard rows.
- **One results table.** `results.csv` has a fixed column layout (mean, sample std, one column per fold, params, notes). Re-running a model replaces its row; an empty file is treated as missing. Each model's backtest predictions are saved to `data/processed/backtest_preds/` for blending in Phase 8.
- **Raw data stays out of git and is read-only on disk**, so no notebook can overwrite it by accident.
- **Pinned library versions** in `requirements.txt` (pandas 3.1, scikit-learn 1.9, LightGBM 4.7, statsmodels 0.15), so results can be reproduced.

### Phase 1 – EDA

- **Training starts in mid-2015, not 2013.** Ten families have no records in 2013 and switch on and off in 2014 – early 2015. Their zeros are missing data, not demand, and they create the fake "+49% growth" and the jumps in total sales.
- **Closed store-days are missing, not zero.** Days before 8 late openings, 6 long closures and Jan 1 explain exactly 8.1% of all rows (the zero baseline shared by most families).
- **Lags by weekday, not by recency.** Median per-series autocorrelation: lag 7 = 0.32, lag 16 = 0.03, lag 21 = 0.20.
- **Calendar features from the data, not assumptions.** The weekly pattern (Sun ≈ 1.30×, Thu ≈ 0.79×) varies by family → day-of-week × family. The pay cycle is mostly the start of the month (1st ≈ +26%, fading to ≈ −7% by the 25th–28th), with a half-month cycle in the ACF → day-of-month + payday flags. December peak (Dec 23); the Jan 1 closure is a calendar rule (2017 shows the holidays file can't be trusted for it); Jan 2 rebounds.
- **Holiday effects are estimated jointly, not one by one.** Effects measured one at a time are confounded with the month start and the December ramp (Dia del Trabajo is May 1). Local holidays ≈ −3% in the matching stores.
- **Earthquake period flagged** (2016-04-16 to about mid-May 2016): +11% total and one-off bulk purchases (124,717 in one day).
- **Promotions kept, oil and transactions dropped.** Promotions are known in advance (+31% within series-month, an upper bound). Oil correlates −0.79 with sales in levels but +0.05 in changes (spurious regression). Transactions aren't available for the test period.
- **SARIMA starting point:** log total is non-stationary with breaks; one difference is enough. After a weekly difference: AR(1) decay and a seasonal MA spike at lag 7 → SARIMA(1,0,0)(0,1,1)[7] with calendar regressors.

### Phase 2 – Features without leakage

- **One rule removes leakage: every sales-history feature is at least 16 days old.** For a cutoff c, a target day t ≤ c + 16 only sees sales up to t − 16 ≤ c, so one feature table serves every fold and the test set. Proven by a truncation test: hide every sale after the cutoff, rebuild everything, and no feature changes on any row up to c + 16 (fold 1 and fold 4 cutoffs), with a positive control showing each lag does change from c + lag + 1.
- **Lags follow the weekday finding:** 21, 28, 35, 42, 49, 56, 364 days (same weekday), rolling means over 7–112 days ending 16 days back, same-weekday means over 4 and 8 weeks, zero share, days since the last sale. Lag 16 itself is not used.
- **Known-in-advance features use the target day itself:** calendar (day of week, day of month, payday flags, yearly Fourier terms, December and Jan 1/2 flags, earthquake flag), holidays (national, plus regional/local matched to each store's state or city; transfers moved to the day actually celebrated; Work Days are normal days; collapsed per date so merges can't duplicate rows), promotions (from test.csv). Oil and transactions are not used.
- **`clean()` holds the row-by-row rules, outside any pipeline**, because nothing in it is fitted to a validation window: closed store-days (8.1% of cells) → missing; the clean start is detected as the day after the last 7+ day run with 3+ families at zero everywhere (**2015-06-01**); 12 gap families (zero or under 1% of their normal level during the gaps — PRODUCE was partly recorded, not absent) are fully masked before that date.
- **Spike rule, past-only:** a value is removed if it is > 10× the series' 28-day median AND > 2× its maximum over the previous year (with ~10 months of history required). It took three versions: median-only removed 1,276 values including weekend LIQUOR peaks and Christmas surges; ignoring past spikes made Decembers cascade. Final: 118 removed (0.004%), including MEATS 89,576 and the earthquake bulk purchases. Known cost: ~24 December rows (FROZEN FOODS Christmas peaks more than 2× the previous year).
- **Missing values stay NaN.** lag_364 is missing on 20% of training rows (the first year after the start date); LightGBM handles NaN natively, linear models impute with a missing indicator inside their pipelines.

### Phase 3 – Validation protocol and baselines

- **Baselines forecast at the cutoff.** Each gets cleaned history up to the fold's last training day and predicts the next 16 days for all series, so it can't leak (and, unlike the Phase 2 lag features, it can use last week).
- **Best baseline: recent level × weekday shape** (log-mean of the last 14 days + weekday offsets from the last 8 weeks): **0.4403 RMSLE** (folds 0.412 / 0.405 / 0.420 / 0.524). Every model must beat it fold by fold, not just on the mean.
- **Averages on the log scale.** Under squared log error the optimal constant is expm1(mean(log1p y)); the 28-day log-mean beats the arithmetic mean on 3 of 4 folds (0.472 vs 0.482 overall). Averaging 8 same weekdays beats one (0.448 vs 0.544).
- **Settings are chosen on tuning folds** (two 16-day windows before fold 1), never on the backtest folds, and the optimism is measured. Zero-rule lookback: optimism 0.00004, and the rule adds nothing to log-mean baselines (a log-mean of zeros is already 0). To revisit for LightGBM.
- **Fold 4 is ~0.10 worse for every model, and SCHOOL AND OFFICE SUPPLIES (0.03% of sales) causes 17% of the error.** Hypothesis: the August back-to-school surge, invisible to recent-weeks methods, and the test period (Aug 16–31) sits in it. Checked at the start of Phase 4; points to per-family yearly memory (lag_364).