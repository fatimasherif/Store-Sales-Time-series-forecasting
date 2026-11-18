"""Cleaning and feature engineering.

Two rules keep this file free of leakage:

1. `clean()` only applies row-by-row rules that look at the past (or at data far
   before any backtest window). Nothing in it learns from the target of a
   validation window.
2. Every feature built from sales uses data at least `cv.HORIZON` (16) days
   old. So one feature table is valid for every fold and for the test period:
   for a forecast made at cutoff c, a target day t <= c + 16 only sees sales up
   to t - 16 <= c. Calendar, holiday and promotion features are known in
   advance (the test file contains onpromotion) and may use the target day.

The truncation test in notebook 02 checks rule 2 directly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import cv

FEATURES_PATH = cv.DATA_PROCESSED / "features.parquet"
META_PATH = cv.DATA_PROCESSED / "features_meta.json"

QUAKE_START = pd.Timestamp("2016-04-16")
QUAKE_END = pd.Timestamp("2016-05-15")

MIN_LAG = cv.HORIZON                                  # 16: the shortest lag any sales feature may use
LAGS = (21, 28, 35, 42, 49, 56, 364)                  # same weekday as the target, all >= 16 days back
ROLL_WINDOWS = (7, 14, 28, 56, 112)                   # windows ending MIN_LAG days before the target
DOW_LAGS = {"dow_mean_4w": (21, 28, 35, 42), "dow_mean_8w": (21, 28, 35, 42, 49, 56, 63, 70)}
OFF_TYPES = ("Holiday", "Transfer", "Additional", "Bridge")   # days off / celebrated days


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def full_calendar(start: pd.Timestamp = cv.TRAIN_START, end: pd.Timestamp = cv.TRAIN_END) -> pd.DatetimeIndex:
    """Every calendar day between start and end, including the missing Dec 25s."""
    return pd.date_range(start, end, freq="D")


def wide_sales(train: pd.DataFrame, col: str = "sales") -> pd.DataFrame:
    """A column as a dates x series matrix on the full daily calendar.

    Columns are a (store_nbr, family) MultiIndex. Days with no rows in the data
    (Dec 25) are NaN, so shifting by k rows is always shifting by k days.
    """
    w = train.pivot(index="date", columns=cv.KEYS, values=col)
    return w.reindex(full_calendar(train["date"].min(), train["date"].max()))


def daily_oil(oil: pd.DataFrame, end: pd.Timestamp = cv.TEST_END) -> pd.Series:
    """Oil price on every calendar day, forward filled over weekends and gaps."""
    s = oil.set_index("date")["dcoilwtico"].astype("float64")
    s = s.reindex(pd.date_range(s.index.min(), end, freq="D")).ffill()
    s.name = "oil"
    return s


def days_since_payday(dates: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """Days since the last public-sector payday (the 15th or the last day of the month)."""
    idx = pd.DatetimeIndex(dates)
    day = idx.day.to_numpy()
    return np.where(idx.is_month_end, 0, np.where(day >= 15, day - 15, day)).astype("int16")


def _by_level(w: pd.DataFrame, level: str) -> pd.DataFrame:
    """Sum the series columns of a wide frame by store or by family (NaN if all NaN)."""
    return w.T.groupby(level=level).sum(min_count=1).T


def _broadcast(w: pd.DataFrame, columns: pd.MultiIndex, level: str) -> pd.DataFrame:
    """Expand a dates x stores (or x families) frame to dates x series."""
    out = w.reindex(columns=columns.get_level_values(level))
    out.columns = columns
    return out


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
@dataclass
class Cleaned:
    raw: pd.DataFrame            # dates x series, sales as given
    sales: pd.DataFrame          # dates x series, cleaned: NaN = not usable as target or as lag input
    train_from: pd.Timestamp     # first day of the clean period
    closed: pd.DataFrame         # dates x stores, True = store closed (no sales at all that day)
    gap_runs: pd.DataFrame       # periods when several families were unrecorded at once
    gap_families: list           # families masked entirely before train_from
    spikes: pd.DataFrame         # one-day spikes removed
    families_at_zero: pd.Series  # per day: families with zero sales across every open store
    masked: dict = field(default_factory=dict)   # cells masked, per reason


def _runs(flag: pd.Series) -> pd.DataFrame:
    """Start, end and length of every run of True values."""
    if not flag.any():
        return pd.DataFrame({"start": pd.DatetimeIndex([]), "end": pd.DatetimeIndex([]), "days": pd.Series([], dtype="int64")})
    rid = (flag != flag.shift()).cumsum()
    g = flag[flag].groupby(rid[flag])
    return pd.DataFrame({"start": g.apply(lambda s: s.index.min()),
                         "end": g.apply(lambda s: s.index.max()),
                         "days": g.size()}).reset_index(drop=True)


def clean(train: pd.DataFrame, train_from: pd.Timestamp | None = None, *,
          min_families: int = 3, min_run_days: int = 7, min_open_stores: int = 10,
          gap_family_share: float = 0.5, near_zero: float = 0.01,
          spike_ratio: float = 10.0, spike_min_median: float = 20.0, spike_max_mult: float = 2.0,
          max_passes: int = 5) -> Cleaned:
    """Row-by-row cleaning rules. Nothing here is fitted on a validation window.

    1. Closed store-days: a store with zero total sales on a day was not open.
       All its series are set to NaN that day (pre-opening, long closures, Jan 1).
    2. Recording gaps: a "systemic gap day" is a normal day on which at least
       `min_families` families have zero sales across every open store. Runs of
       at least `min_run_days` such days are data gaps; the clean period starts
       the day after the last one. A family that is missing on at least
       `gap_family_share` of the days inside those runs was affected by the
       gaps (a family that is only zero on Sundays is not). "Missing" means zero
       or below `near_zero` x the family's median daily total after the clean
       start: PRODUCE, for one, was partly recorded during the gaps (about 0.3%
       of its real level) rather than absent. All values of an affected family
       before the clean start are set to NaN.
    3. One-day spikes: a value is removed when it is more than `spike_ratio`
       times the series' median of the previous 28 days (median at least
       `spike_min_median`) AND more than `spike_max_mult` times the series'
       maximum of the previous 365 days. The second condition keeps peaks that
       have happened before (Christmas, weekly peaks of LIQUOR), and it only
       applies once a series has most of a year of usable history (a new store
       or a family's first season can't be judged "unprecedented").
       The rule only looks at past days. It repeats so that a second spike
       within 28 days of a first one is compared without the first; older
       values always count, so last year's Christmas is never dropped from
       the comparison (which would make every later Christmas look new).

    Pass `train_from` to fix the start date instead of detecting it (the
    truncation test does this, so both runs use the same date).
    """
    raw = wide_sales(train)
    dates = raw.index
    masked = {}

    # 1. closed store-days
    store_total = _by_level(raw, "store_nbr")
    closed = store_total.eq(0)
    closed_cells = _broadcast(closed, raw.columns, "store_nbr").fillna(False).astype(bool)
    s = raw.mask(closed_cells)
    masked["closed store-day"] = int((closed_cells & raw.notna()).to_numpy().sum())

    # 2. systemic recording gaps
    fam_total = _by_level(s, "family")
    open_stores = (store_total.gt(0)).sum(axis=1)
    special = ((dates.month == 12) & (dates.day == 25)) | ((dates.month == 1) & (dates.day == 1))
    normal = (open_stores >= min_open_stores) & ~special
    n_zero_fam = fam_total.eq(0).sum(axis=1)
    systemic = (n_zero_fam >= min_families)[normal]
    runs = _runs(systemic)
    if len(runs):
        runs = runs[runs["days"] >= min_run_days].reset_index(drop=True)
        runs["max_families_at_zero"] = [int(n_zero_fam[r.start:r.end].max()) for r in runs.itertuples()]
    if train_from is None:
        train_from = (runs["end"].max() + pd.Timedelta(days=1)) if len(runs) else dates.min()
    train_from = pd.Timestamp(train_from)
    before = np.asarray(dates < train_from)
    in_runs = np.zeros(len(dates), dtype=bool)
    for r in runs.itertuples():
        in_runs |= np.asarray((dates >= r.start) & (dates <= r.end))
    in_runs &= before & np.asarray(normal)
    level = fam_total[~before & np.asarray(normal)].median()           # typical daily total in the clean period
    missing = fam_total.le(near_zero * level, axis=1)                  # zero or near zero (partly recorded)
    share = missing[in_runs].mean() if in_runs.any() else pd.Series(0.0, index=fam_total.columns)
    gap_families = sorted(share[share >= gap_family_share].index)
    fam_cols = np.asarray(raw.columns.get_level_values("family").isin(gap_families))
    gap_cells = pd.DataFrame(before[:, None] & fam_cols[None, :], index=dates, columns=raw.columns) & s.notna()
    s = s.mask(gap_cells)
    masked["gap family before train_from"] = int(gap_cells.to_numpy().sum())

    # 3. one-day spikes, judged against the past only
    med = s.shift(1).rolling(28, min_periods=14).median()
    candidate = s.gt(spike_ratio * med) & med.ge(spike_min_median)
    # days t-365 .. t-29, never masked; needs ~10 months of usable history
    old_max = s.shift(29).rolling(337, min_periods=300 - 28).max()
    spike = pd.DataFrame(False, index=dates, columns=raw.columns)
    for _ in range(max_passes):
        recent_max = s.mask(spike).shift(1).rolling(28, min_periods=1).max()   # days t-28 .. t-1
        past_max = np.fmax(old_max, recent_max).where(old_max.notna())
        new = candidate & s.gt(spike_max_mult * past_max)
        if new.equals(spike):
            break
        spike = new
    r, c = np.nonzero(spike.to_numpy())
    sp = pd.DataFrame({"date": dates[r],
                       "store_nbr": raw.columns.get_level_values("store_nbr")[c],
                       "family": raw.columns.get_level_values("family")[c],
                       "sales": raw.to_numpy()[r, c],
                       "median_prev_28d": med.to_numpy()[r, c],
                       "max_prev_365d": past_max.to_numpy()[r, c]}).sort_values("date", ignore_index=True)
    s = s.mask(spike)
    masked["one-day spike"] = int(spike.to_numpy().sum())

    return Cleaned(raw=raw, sales=s, train_from=train_from, closed=closed, gap_runs=runs,
                   gap_families=gap_families, spikes=sp, families_at_zero=n_zero_fam.where(normal),
                   masked=masked)


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------
def holiday_features(hol: pd.DataFrame, stores: pd.DataFrame, dates: pd.DatetimeIndex):
    """Holiday flags per date (national) and per date x store (regional, local).

    - transferred=True rows are dropped: the holiday was not celebrated that day.
    - Transfer, Bridge, Additional and Holiday rows count as days off.
    - Work Day rows (weekend days made into working days) get their own flag.
    - Events (Mother's Day, Black Friday, World Cup...) get their own flag; the
      earthquake events are left out because the quake flag covers them.
    - Several rows on one date collapse into one flag, so merging can't duplicate rows.
    """
    h = hol[~hol["transferred"]]
    off = h["type"].isin(OFF_TYPES)
    nat = h["locale"].eq("National")
    ext = pd.date_range(dates.min() - pd.Timedelta(days=1), dates.max() + pd.Timedelta(days=1))
    nat_off = pd.Series(ext.isin(h.loc[nat & off, "date"]), index=ext)
    event = h["type"].eq("Event") & ~h["description"].str.contains("Terremoto", case=False)
    date_level = pd.DataFrame({
        "nat_holiday": nat_off.reindex(dates).to_numpy(),
        "nat_holiday_eve": nat_off.shift(-1, fill_value=False).reindex(dates).to_numpy(),
        "nat_holiday_after": nat_off.shift(1, fill_value=False).reindex(dates).to_numpy(),
        "nat_event": dates.isin(h.loc[nat & event, "date"]),
        "work_day": dates.isin(h.loc[h["type"].eq("Work Day"), "date"]),
    }, index=dates).astype("int8")

    st = stores.set_index("store_nbr").sort_index()
    def per_store(locale, attr):
        rows = h[off & h["locale"].eq(locale)]
        out = {}
        for sn, name in st[attr].items():
            out[sn] = dates.isin(rows.loc[rows["locale_name"].eq(name), "date"])
        return pd.DataFrame(out, index=dates).astype("int8")
    return date_level, per_store("Regional", "state"), per_store("Local", "city")


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
CALENDAR = ["dow", "dom", "month", "year", "doy", "t_index", "is_month_start", "is_15", "is_16",
            "is_month_end", "days_since_payday", "days_to_month_end",
            "fy_sin1", "fy_cos1", "fy_sin2", "fy_cos2", "fy_sin3", "fy_cos3", "fm_sin1", "fm_cos1",
            "dec_to_xmas", "is_dec_24", "is_dec_26_31", "is_jan_1", "is_jan_2", "quake"]
HOLIDAY = ["nat_holiday", "nat_holiday_eve", "nat_holiday_after", "nat_event", "work_day",
           "reg_holiday", "loc_holiday", "any_holiday"]
PROMO = ["promo", "promo_log", "promo_any", "promo_lag1", "promo_roll7"]
STATIC = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
LAG = ([f"lag_{k}" for k in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
       + ["roll_std_28", "zero_share_28", "zero_share_112", *DOW_LAGS, "days_since_sale"])
# shortest lag (days back from the target day) each sales feature uses
LAG_MIN = {**{f"lag_{k}": k for k in LAGS}, **{f"roll_mean_{w}": MIN_LAG for w in ROLL_WINDOWS},
           "roll_std_28": MIN_LAG, "zero_share_28": MIN_LAG, "zero_share_112": MIN_LAG,
           **{k: min(v) for k, v in DOW_LAGS.items()}, "days_since_sale": MIN_LAG}
FEATURES = STATIC + CALENDAR + HOLIDAY + PROMO + LAG


def feature_spec() -> pd.DataFrame:
    """One row per feature: its group, whether it's known in advance, and its shortest lag."""
    rows = []
    for group, names in [("static", STATIC), ("calendar", CALENDAR), ("holiday", HOLIDAY),
                         ("promo", PROMO), ("lag", LAG)]:
        for n in names:
            rows.append({"feature": n, "group": group, "known_in_advance": group != "lag",
                         "min_lag_days": LAG_MIN.get(n, 0)})
    return pd.DataFrame(rows)


def _nanmean_of_shifts(y: pd.DataFrame, lags) -> np.ndarray:
    """Mean of y shifted by each lag, ignoring NaN (NaN if all are NaN)."""
    tot = np.zeros(y.shape)
    cnt = np.zeros(y.shape)
    for k in lags:
        v = y.shift(k).to_numpy()
        ok = ~np.isnan(v)
        tot += np.where(ok, v, 0.0)
        cnt += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / cnt, np.nan)


def build_features(cleaned: Cleaned, train: pd.DataFrame, test: pd.DataFrame,
                   stores: pd.DataFrame, hol: pd.DataFrame, start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Long feature table, one row per (date, store, family), from `start` to the end of the test period.

    Columns: identifiers, sales (raw), sales_clean, y = log1p(sales_clean), is_test,
    train_ok (usable training row), then every name in FEATURES.
    """
    D = full_calendar(cv.TRAIN_START, cv.TEST_END)
    S = cleaned.sales.reindex(D)                        # test days are NaN
    Y = np.log1p(S)
    series = S.columns
    n = len(series)
    start = pd.Timestamp(start if start is not None else cleaned.train_from)
    keep = np.asarray(D >= start)
    Dk = D[keep]
    nk = len(Dk)
    cols = {}

    def put(name, wide, dtype="float32"):
        cols[name] = np.asarray(wide, dtype="float64")[keep].ravel().astype(dtype)

    def per_date(name, values, dtype="float32"):
        cols[name] = np.repeat(np.asarray(values)[keep], n).astype(dtype)

    # identifiers and target
    cols["date"] = np.repeat(Dk.to_numpy(), n)
    st_vals = series.get_level_values("store_nbr").to_numpy()
    fa_vals = series.get_level_values("family").to_numpy()
    raw = cleaned.raw.reindex(D)
    put("sales", raw, "float64")
    put("sales_clean", S, "float64")
    put("y", Y)
    is_test = np.repeat(Dk >= cv.TEST_START, n)

    # static
    st = stores.set_index("store_nbr").reindex(st_vals)
    static = {"store_nbr": st_vals, "family": fa_vals, "city": st["city"].to_numpy(),
              "state": st["state"].to_numpy(), "store_type": st["type"].to_numpy(),
              "cluster": st["cluster"].to_numpy()}
    for k, v in static.items():
        cols[k] = np.tile(v, nk)

    # calendar (known in advance)
    dom = D.day.to_numpy()
    month = D.month.to_numpy()
    dim = D.days_in_month.to_numpy()
    doy = D.day_of_year.to_numpy()
    cal = {
        "dow": D.day_of_week.to_numpy(), "dom": dom, "month": month, "year": D.year.to_numpy(),
        "doy": doy, "t_index": (D - cv.TRAIN_START).days.to_numpy(),
        "is_month_start": dom == 1, "is_15": dom == 15, "is_16": dom == 16,
        "is_month_end": D.is_month_end, "days_since_payday": days_since_payday(D),
        "days_to_month_end": dim - dom,
        "fm_sin1": np.sin(2 * np.pi * (dom - 1) / dim), "fm_cos1": np.cos(2 * np.pi * (dom - 1) / dim),
        "dec_to_xmas": np.where((month == 12) & (dom <= 24), 25 - dom, 0),
        "is_dec_24": (month == 12) & (dom == 24), "is_dec_26_31": (month == 12) & (dom >= 26),
        "is_jan_1": (month == 1) & (dom == 1), "is_jan_2": (month == 1) & (dom == 2),
        "quake": (D >= QUAKE_START) & (D <= QUAKE_END),
    }
    for k in (1, 2, 3):
        cal[f"fy_sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        cal[f"fy_cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for name in CALENDAR:
        per_date(name, cal[name], "float32" if name.startswith(("fy_", "fm_")) else "int16")

    # holidays (known in advance)
    date_level, reg, loc = holiday_features(hol, stores, D)
    for c in date_level.columns:
        per_date(c, date_level[c].to_numpy(), "int8")
    put("reg_holiday", _broadcast(reg, series, "store_nbr"), "int8")
    put("loc_holiday", _broadcast(loc, series, "store_nbr"), "int8")
    cols["any_holiday"] = (cols["nat_holiday"] | cols["reg_holiday"] | cols["loc_holiday"]).astype("int8")

    # promotions (known in advance for every forecast day)
    both = pd.concat([train[["date", *cv.KEYS, "onpromotion"]], test[["date", *cv.KEYS, "onpromotion"]]])
    P = both.pivot(index="date", columns=cv.KEYS, values="onpromotion").reindex(D).reindex(columns=series)
    P = P.fillna(0.0)                                   # Dec 25: closed, no promotion
    put("promo", P)
    put("promo_log", np.log1p(P))
    put("promo_any", P.gt(0), "int8")
    put("promo_lag1", P.shift(1))
    put("promo_roll7", P.rolling(7, min_periods=1).mean())

    # sales history: everything at least MIN_LAG days old
    for k in LAGS:
        put(f"lag_{k}", Y.shift(k))
    Y16 = Y.shift(MIN_LAG)
    for w in ROLL_WINDOWS:
        put(f"roll_mean_{w}", Y16.rolling(w, min_periods=max(1, w // 2)).mean())
    put("roll_std_28", Y16.rolling(28, min_periods=14).std())
    Z16 = S.eq(0).astype("float64").where(S.notna()).shift(MIN_LAG)
    put("zero_share_28", Z16.rolling(28, min_periods=14).mean())
    put("zero_share_112", Z16.rolling(112, min_periods=28).mean())
    for name, lags in DOW_LAGS.items():
        put(name, _nanmean_of_shifts(Y, lags))
    idx = np.arange(len(D), dtype="float64")[:, None]
    last_sale = pd.DataFrame(np.where(S.to_numpy() > 0, idx, np.nan)).ffill().shift(MIN_LAG).to_numpy()
    put("days_since_sale", idx - last_sale)

    df = pd.DataFrame(cols)
    for c in ["family", "city", "state", "store_type"]:
        df[c] = df[c].astype("category")
    df["is_test"] = is_test
    df["train_ok"] = (~df["is_test"]) & df["y"].notna()
    lead = ["date", "store_nbr", "family", "sales", "sales_clean", "y", "is_test", "train_ok"]
    return df[lead + [c for c in FEATURES if c not in lead]]