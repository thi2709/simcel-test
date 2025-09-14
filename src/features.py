# feature.py
# import argparse
import sys
from typing import List
import numpy as np
import pandas as pd


DAY_ORDER: List[str] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]
MONTH_ORDER: List[str] = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
]
SEASON_ORDER: List[str] = ["spring", "summer", "autumn", "winter"]  # year progression


def _season_from_month(month_num: int) -> str:
    # Northern-hemisphere seasons
    if month_num in (3, 4, 5):
        return "spring"
    if month_num in (6, 7, 8):
        return "summer"
    if month_num in (9, 10, 11):
        return "autumn"
    return "winter"  # months 12, 1, 2


def generate_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features on the provided dataframe.

    Assumptions:
      - df has columns: date (datetime), store_id (string), sku_id (string),
        final_price (float), competitor_price (float), units_sold (int/float).
      - Types are already correct as stated by the user.

    Fallbacks:
      - Any numeric missing value at the end is filled with -1 (except sold_last_week,
        which we impute specially below).
    """
    out = df.copy()

    # --- Basic date bits ---
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out["day_of_week"] = out["date"].dt.day_name().str.lower()
    out["month_of_year"] = out["date"].dt.month_name().str.lower()
    out["season"] = out["date"].dt.month.map(_season_from_month)

    # --- Time-based demand lags (robust to gaps via self-merge on shifted dates) ---
    key_cols = ["store_id", "sku_id", "date"]
    base = out[key_cols + ["units_sold"]].copy()

    # yesterday
    y = base.copy()
    y["date"] = y["date"] + pd.Timedelta(days=1)
    y = y.rename(columns={"units_sold": "sold_yesterday"})
    out = out.merge(y[key_cols[:2] + ["date", "sold_yesterday"]], on=key_cols, how="left")

    # last week (same weekday)
    w = base.copy()
    w["date"] = w["date"] + pd.Timedelta(days=7)
    w = w.rename(columns={"units_sold": "sold_last_week"})
    out = out.merge(w[key_cols[:2] + ["date", "sold_last_week"]], on=key_cols, how="left")

    # Ensure lag cols are float so forward averages (float) can be assigned safely
    out["sold_yesterday"] = pd.to_numeric(out["sold_yesterday"], errors="coerce").astype(float)
    out["sold_last_week"] = pd.to_numeric(out["sold_last_week"], errors="coerce").astype(float)

    # ---------- NEW: moving averages on units_sold ----------
    out = out.sort_values(["store_id", "sku_id", "date"])
    grp = out.groupby(["store_id", "sku_id"], as_index=False, sort=False)

    # past MA7 / MA30 (exclude today via shift)
    out["units_sold_ma7"] = (
        grp["units_sold"].apply(lambda s: s.shift(1).rolling(window=7, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    out["units_sold_ma30"] = (
        grp["units_sold"].apply(lambda s: s.shift(1).rolling(window=30, min_periods=1).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # Forward-looking averages (including today) used only for first-day fallbacks
    def _forward_avg(s: pd.Series, win: int) -> pd.Series:
        return s.iloc[::-1].rolling(window=win, min_periods=1).mean().iloc[::-1]

    out["_fwd7_incl_today"] = grp["units_sold"].transform(lambda s: _forward_avg(s, 7))
    out["_fwd30_incl_today"] = grp["units_sold"].transform(lambda s: _forward_avg(s, 30))

    # First chronological row in each (store, sku)
    first_mask = grp["date"].transform("min").eq(out["date"])

    # First-day rules
    out.loc[first_mask, "units_sold_ma7"] = out.loc[first_mask, "_fwd7_incl_today"]
    out.loc[first_mask, "units_sold_ma30"] = out.loc[first_mask, "_fwd30_incl_today"]
    out.loc[first_mask, "sold_yesterday"] = out.loc[first_mask, "_fwd7_incl_today"]
    out.loc[first_mask, "sold_last_week"] = out.loc[first_mask, "_fwd7_incl_today"]

    # ---------- NEW: sold_last_week special imputation ----------
    # Build lag_k columns k=1..6 (units sold k days ago)
    for k in range(1, 7):
        lagk = base.copy()
        lagk["date"] = lagk["date"] + pd.Timedelta(days=k)  # fetch units from k days ago
        lagk = lagk.rename(columns={"units_sold": f"_lag_{k}d"})
        out = out.merge(lagk[key_cols[:2] + ["date", f"_lag_{k}d"]], on=key_cols, how="left")

    # Farthest-to-nearest fill: 6d -> ... -> 1d
    fill_series = pd.Series(np.nan, index=out.index)
    for k in range(6, 0, -1):
        fill_series = fill_series.combine_first(pd.to_numeric(out[f"_lag_{k}d"], errors="coerce").astype(float))

    need_fill = out["sold_last_week"].isna()
    out.loc[need_fill, "sold_last_week"] = fill_series[need_fill]

    # If still NA, use forward 7-day average including today
    still_na = out["sold_last_week"].isna()
    out.loc[still_na, "sold_last_week"] = out.loc[still_na, "_fwd7_incl_today"]

    # Clean up helpers
    helper_cols = [c for c in out.columns if c.startswith("_lag_")] + ["_fwd7_incl_today", "_fwd30_incl_today"]
    out.drop(columns=helper_cols, inplace=True, errors="ignore")

    # --- Promo sequence feature: first day of consecutive promo run ---
    out = out.sort_values(["store_id", "sku_id", "date"])
    grp = out.groupby(["store_id", "sku_id"], sort=False)

    # Previous day's promo flag
    out["promo_flag_prevday"] = grp["promo_flag"].shift(1)

    # Identify first promo day
    out["flag_promo_1stday"] = np.where(
        (out["promo_flag"] == 1) & ((out["promo_flag_prevday"] != 1) | (out["promo_flag_prevday"].isna())),
        1,
        0
    )

    # Drop helper column
    out.drop(columns=["promo_flag_prevday"], inplace=True)

    # --- Price features ---
    fp = out.get("final_price")
    out["final_price_ln"] = np.where((fp.notna()) & (fp > 0), np.log(fp), np.nan)

    cp = out.get("competitor_price")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["competitor_price_diff"] = (cp - fp) / fp

    # --- Categorical typing & ordering ---
    out["store_id"] = out["store_id"].astype("category")
    out["sku_id"] = out["sku_id"].astype("category")

    out["day_of_week"] = pd.Categorical(out["day_of_week"], categories=DAY_ORDER, ordered=True)
    out["month_of_year"] = pd.Categorical(out["month_of_year"], categories=MONTH_ORDER, ordered=True)
    out["season"] = pd.Categorical(out["season"], categories=SEASON_ORDER, ordered=True)

    # --- One-hot encode the ordered categoricals (keep originals) ---
    dummies = pd.get_dummies(
        out[["day_of_week", "month_of_year", "season"]],
        prefix="onehot",
        prefix_sep="_",
        dtype=int
    )
    out = pd.concat([out, dummies], axis=1)

    # --- Final numeric NA handling (exclude sold_last_week which was imputed) ---
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if "sold_last_week" in num_cols:
        num_cols.remove("sold_last_week")
    out[num_cols] = out[num_cols].fillna(-1)

    return out
