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
      - Any numeric missing value at the end is filled with -1.
    """
    out = df.copy()

    # --- Basic date bits ---
    # Ensure datetime
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Lowercase day/month names for consistency
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
    
    # --- Price features ---
    # natural log of final_price -> invalid / non-positive -> -1
    fp = out.get("final_price")
    out["final_price_ln"] = np.where(
        (fp.notna()) & (fp > 0), np.log(fp), np.nan
    )

    # competitor_price_diff = (competitor_price - final_price) / final_price
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

    # --- Final numeric NA handling: fill with -1 as requested ---
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(-1)

    return out
