"""Model wrappers.

Phase 3: baselines. A baseline is a function `f(history, future) -> predictions`:

- history: cleaned sales, dates x series (full daily calendar), ending ON the cutoff day
- future:  the forecast dates (16 days after the cutoff)
- returns: dates x series predictions on the sales scale (>= 0)

Everything a baseline knows comes from `history`, and `backtest()` only ever passes
history up to the cutoff, so a baseline cannot see past it. NaN in history means
"not observed" (closed store, removed spike) and is skipped, never read as zero.

Averages are taken on the log1p scale: the constant c that minimises the squared
log error sum((log1p(y) - log1p(c))^2) is c = expm1(mean(log1p(y))), not the
arithmetic mean. `moving_average(..., log=False)` exists to show the difference.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from . import cv

Baseline = Callable[[pd.DataFrame, pd.DatetimeIndex], pd.DataFrame]


def _frame(values, future: pd.DatetimeIndex, columns) -> pd.DataFrame:
    return pd.DataFrame(values, index=future, columns=columns, dtype="float64")


def _repeat(row: pd.Series, future: pd.DatetimeIndex) -> pd.DataFrame:
    """The same prediction for every future day."""
    return _frame(np.tile(row.to_numpy(dtype="float64"), (len(future), 1)), future, row.index)


def _finish(p: pd.DataFrame) -> pd.DataFrame:
    """Series with no usable history get 0; no negative predictions."""
    return p.fillna(0.0).clip(lower=0.0)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def zero(history: pd.DataFrame, future: pd.DatetimeIndex) -> pd.DataFrame:
    """Predict 0 everywhere."""
    return _frame(0.0, future, history.columns)


def naive_last(history: pd.DataFrame, future: pd.DatetimeIndex) -> pd.DataFrame:
    """Last observed value, repeated for all 16 days."""
    return _finish(_repeat(history.ffill().iloc[-1], future))


def seasonal_naive(history: pd.DataFrame, future: pd.DatetimeIndex) -> pd.DataFrame:
    """Last observed value on the same weekday as the forecast day."""
    out = _frame(np.nan, future, history.columns)
    for wd in range(7):
        same = history[history.index.day_of_week == wd]
        if len(same):
            out.loc[future.day_of_week == wd] = same.ffill().iloc[-1].to_numpy()
    return _finish(out)


def moving_average(days: int, log: bool = True) -> Baseline:
    """Mean of the last `days` days (on the log1p scale by default), repeated."""
    def f(history, future):
        win = history.iloc[-days:]
        level = np.expm1(np.log1p(win).mean()) if log else win.mean()
        return _finish(_repeat(level, future))
    f.__name__ = f"ma_{days}" + ("" if log else "_raw")
    return f


def dow_profile(weeks: int) -> Baseline:
    """For each weekday, the log1p-mean of that weekday over the last `weeks` weeks."""
    def f(history, future):
        win = np.log1p(history.iloc[-7 * weeks:])
        by_wd = win.groupby(win.index.day_of_week).mean()
        return _finish(np.expm1(by_wd.reindex(future.day_of_week).set_axis(future)))
    f.__name__ = f"dow_{weeks}w"
    return f


def level_x_dow(level_days: int = 14, weeks: int = 8) -> Baseline:
    """Recent level times a weekday profile (additive on the log scale).

    log forecast = mean(last `level_days` days) + (weekday mean - overall mean) over the last `weeks` weeks.
    The level reacts to recent changes; the weekday shape uses more weeks, so it's less noisy.
    """
    def f(history, future):
        level = np.log1p(history.iloc[-level_days:]).mean()
        win = np.log1p(history.iloc[-7 * weeks:])
        offset = win.groupby(win.index.day_of_week).mean() - win.mean()
        offset = offset.fillna(0.0).reindex(future.day_of_week).set_axis(future)
        return _finish(np.expm1(offset.add(level, axis=1)))
    f.__name__ = f"ma{level_days}_x_dow{weeks}w"
    return f


def with_zero_rule(base: Baseline, lookback: int) -> Baseline:
    """Predict exactly 0 for series that sold nothing on every observed day of the last `lookback` days.

    At least half of those days must be observed (a store closed for the whole
    period is not "dead", it's closed).
    """
    def f(history, future):
        p = base(history, future)
        win = history.iloc[-lookback:]
        observed = win.notna().sum()
        dead = win.fillna(0).eq(0).all() & (observed >= lookback / 2)
        p.loc[:, dead.to_numpy()] = 0.0
        return p
    f.__name__ = f"{base.__name__}+zr{lookback}"
    return f


BASELINES: list[Baseline] = [
    zero, naive_last, seasonal_naive,
    moving_average(7), moving_average(14), moving_average(28), moving_average(28, log=False),
    dow_profile(4), dow_profile(8), level_x_dow(14, 8),
]


# ---------------------------------------------------------------------------
# Running a baseline on the folds
# ---------------------------------------------------------------------------
def to_long(p: pd.DataFrame, fold: int | None = None) -> pd.DataFrame:
    """dates x series predictions -> rows of (fold, date, store_nbr, family, y_pred)."""
    n = p.shape[1]
    out = pd.DataFrame({
        "date": np.repeat(p.index.to_numpy(), n),
        "store_nbr": np.tile(p.columns.get_level_values("store_nbr").to_numpy(), len(p)),
        "family": np.tile(p.columns.get_level_values("family").to_numpy(), len(p)),
        "y_pred": p.to_numpy(dtype="float64").ravel(),
    })
    if fold is not None:
        out.insert(0, "fold", fold)
    return out


def forecast(fn: Baseline, sales: pd.DataFrame, cutoff: pd.Timestamp, horizon: int = cv.HORIZON) -> pd.DataFrame:
    """Forecast the `horizon` days after `cutoff` from history up to and including the cutoff."""
    history = sales.loc[:cutoff]
    assert history.index.max() == cutoff, "history must end on the cutoff"
    future = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon, freq="D")
    p = fn(history, future)
    assert p.index.equals(future) and p.columns.equals(sales.columns)
    assert np.isfinite(p.to_numpy()).all() and (p.to_numpy() >= 0).all()
    return p


def backtest(fn: Baseline, sales: pd.DataFrame, folds=cv.FOLDS) -> pd.DataFrame:
    """Predictions for every fold window, in the long format `cv.evaluate` expects."""
    return pd.concat([to_long(forecast(fn, sales, f.train_end), f.fold) for f in folds], ignore_index=True)


def submission(p: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Test-period predictions (dates x series) -> Kaggle submission (id, sales)."""
    long = to_long(p)
    sub = test[["id", "date", *cv.KEYS]].merge(long, on=["date", *cv.KEYS], how="left", validate="one_to_one")
    assert len(sub) == len(test) and sub["y_pred"].notna().all()
    return sub[["id"]].assign(sales=sub["y_pred"].clip(lower=0).to_numpy())