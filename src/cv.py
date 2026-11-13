"""Shared project constants, backtest folds, the metric and results logging.

Every notebook imports from here, so the folds and the metric are defined once.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed and paths
# ---------------------------------------------------------------------------
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PREDS_DIR = DATA_PROCESSED / "backtest_preds"
FIGURES = ROOT / "reports" / "figures"
SUBMISSIONS = ROOT / "submissions"
RESULTS_PATH = ROOT / "results.csv"

# ---------------------------------------------------------------------------
# Competition constants
# ---------------------------------------------------------------------------
TRAIN_START = pd.Timestamp("2013-01-01")
TRAIN_END = pd.Timestamp("2017-08-15")
TEST_START = pd.Timestamp("2017-08-16")
TEST_END = pd.Timestamp("2017-08-31")
HORIZON = 16                      # days to forecast, same as the test period
N_FOLDS = 4                       # number of backtest windows
KEYS = ["store_nbr", "family"]    # one series = one store x one product family

RAW_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays": "holidays_events.csv",
    "transactions": "transactions.csv",
    "sample_submission": "sample_submission.csv",
}


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy. Models also get random_state=SEED explicitly."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    """Create the output folders if they are missing."""
    for d in (DATA_RAW, DATA_PROCESSED, PREDS_DIR, FIGURES, SUBMISSIONS):
        d.mkdir(parents=True, exist_ok=True)


def read_raw(name: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Read one raw Kaggle file with compact dtypes and parsed dates.

    `family` stays a string column here. Converting it to category is done
    later, once, so train and test share the same categories.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_RAW
    path = data_dir / RAW_FILES[name]
    dtypes = {
        # sales stays float64: summing 3M float32 values loses precision
        "train": {"id": "int32", "store_nbr": "int8", "sales": "float64", "onpromotion": "int32"},
        "test": {"id": "int32", "store_nbr": "int8", "onpromotion": "int32"},
        "stores": {"store_nbr": "int8", "cluster": "int8"},
        "oil": {"dcoilwtico": "float32"},
        "holidays": {"transferred": "bool"},
        "transactions": {"store_nbr": "int8", "transactions": "int32"},
        "sample_submission": {"id": "int32", "sales": "float32"},
    }[name]
    parse_dates = [] if name in ("stores", "sample_submission") else ["date"]
    return pd.read_csv(path, dtype=dtypes, parse_dates=parse_dates)


# ---------------------------------------------------------------------------
# Backtest folds
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    fold: int                   # 1 = oldest window, N_FOLDS = most recent
    train_end: pd.Timestamp     # last day the model may see
    valid_start: pd.Timestamp   # first forecast day (horizon day 1)
    valid_end: pd.Timestamp     # last forecast day (horizon day 16)


def make_folds(n_folds: int = N_FOLDS, horizon: int = HORIZON,
               last_date: pd.Timestamp = TRAIN_END) -> list[Fold]:
    """Consecutive, non-overlapping validation windows of `horizon` days.

    The last window ends on the last training date. Each window's training
    data is everything before it (expanding window), with no gap, because the
    test period also starts the day after the training data ends.
    """
    folds = []
    for k in range(n_folds):  # k = 0 is the most recent window
        valid_end = last_date - pd.Timedelta(days=k * horizon)
        valid_start = valid_end - pd.Timedelta(days=horizon - 1)
        folds.append((valid_start - pd.Timedelta(days=1), valid_start, valid_end))
    folds.sort(key=lambda t: t[1])
    return [Fold(i + 1, *t) for i, t in enumerate(folds)]


FOLDS = make_folds()


def folds_table(folds: list[Fold] = FOLDS) -> pd.DataFrame:
    """Readable summary of the folds, with weekdays."""
    rows = []
    for f in folds:
        rows.append({
            "fold": f.fold,
            "train_end": f.train_end.date(),
            "valid_start": f.valid_start.date(),
            "valid_start_day": f.valid_start.day_name(),
            "valid_end": f.valid_end.date(),
            "valid_end_day": f.valid_end.day_name(),
        })
    return pd.DataFrame(rows)


def split(df: pd.DataFrame, fold: Fold, date_col: str = "date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, valid) for one fold. Train is everything up to train_end."""
    d = df[date_col]
    train = df[d <= fold.train_end]
    valid = df[(d >= fold.valid_start) & (d <= fold.valid_end)]
    return train, valid


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error on the sales scale (the Kaggle metric).

    Predictions are clipped at 0 first, like the submission. Equivalent to
    RMSE between log1p(y_true) and log1p(y_pred).
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if np.isnan(y_pred).any():
        raise ValueError("predictions contain NaN")
    if (y_true < 0).any():
        raise ValueError("y_true has negative values")
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def evaluate(preds: pd.DataFrame, actuals: pd.DataFrame,
             folds: list[Fold] = FOLDS) -> dict[int, float]:
    """Score backtest predictions fold by fold.

    preds:   columns fold, date, store_nbr, family, y_pred (sales scale)
    actuals: the training data (date, store_nbr, family, sales)

    Checks that every (series, date) of every validation window is predicted
    exactly once, so a model cannot look better by silently dropping rows.
    """
    need = {"fold", "date", *KEYS, "y_pred"}
    missing_cols = need - set(preds.columns)
    if missing_cols:
        raise ValueError(f"preds is missing columns: {sorted(missing_cols)}")
    keys = ["date", *KEYS]
    scores = {}
    for f in folds:
        p = preds.loc[preds["fold"] == f.fold, [*keys, "y_pred"]]
        _, truth = split(actuals, f)
        truth = truth[[*keys, "sales"]]
        if p.duplicated(keys).any():
            raise ValueError(f"fold {f.fold}: duplicated predictions")
        m = truth.merge(p, on=keys, how="left", validate="one_to_one")
        if len(p) != len(truth) or m["y_pred"].isna().any():
            raise ValueError(
                f"fold {f.fold}: expected {len(truth)} predictions, got {len(p)}, "
                f"{int(m['y_pred'].isna().sum())} validation rows unmatched"
            )
        scores[f.fold] = rmsle(m["sales"], m["y_pred"])
    return scores


# ---------------------------------------------------------------------------
# Results table and saved predictions
# ---------------------------------------------------------------------------
FOLD_COLS = [f"fold_{i}" for i in range(1, N_FOLDS + 1)]
RESULT_COLS = ["model", "mean_rmsle", "std_rmsle", *FOLD_COLS, "params", "notes", "logged_at"]


def _read_results(path: Path) -> pd.DataFrame:
    """Read results.csv. A missing or empty file is an empty table."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=RESULT_COLS)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=RESULT_COLS)
    if list(df.columns) != RESULT_COLS:
        raise ValueError(f"{path} has columns {list(df.columns)}, expected {RESULT_COLS}")
    # empty text fields come back as NaN from the CSV
    df[["params", "notes"]] = df[["params", "notes"]].fillna("").astype(str)
    return df


def log_result(model: str, fold_scores: dict[int, float], params: str = "",
               notes: str = "", path: Path = RESULTS_PATH) -> pd.DataFrame:
    """Add or replace one model's row in results.csv and return the table.

    fold_scores must contain every fold (1..N_FOLDS). std is the sample
    standard deviation across folds (ddof=1).
    """
    if sorted(fold_scores) != list(range(1, N_FOLDS + 1)):
        raise ValueError(f"need scores for folds 1..{N_FOLDS}, got {sorted(fold_scores)}")
    vals = np.array([fold_scores[i] for i in range(1, N_FOLDS + 1)], dtype="float64")
    row = {
        "model": model,
        "mean_rmsle": round(float(vals.mean()), 5),
        "std_rmsle": round(float(vals.std(ddof=1)), 5),
        **{c: round(float(v), 5) for c, v in zip(FOLD_COLS, vals)},
        "params": params,
        "notes": notes,
        "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    df = _read_results(path)
    df = df[df["model"] != model]
    new = pd.DataFrame([row], columns=RESULT_COLS)
    df = new if df.empty else pd.concat([df, new], ignore_index=True)
    df = df.sort_values("mean_rmsle", kind="stable").reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def show_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """The results table sorted by mean RMSLE (best first)."""
    return _read_results(path).sort_values("mean_rmsle", kind="stable").reset_index(drop=True)


def save_backtest_preds(model: str, preds: pd.DataFrame, preds_dir: Path = PREDS_DIR) -> Path:
    """Save one model's backtest predictions for blending later. Overwrites."""
    cols = ["fold", "date", *KEYS, "y_pred"]
    preds_dir = Path(preds_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)
    out = preds_dir / f"{model}.parquet"
    preds[cols].reset_index(drop=True).to_parquet(out, index=False)
    return out


def load_backtest_preds(model: str, preds_dir: Path = PREDS_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(preds_dir) / f"{model}.parquet")
