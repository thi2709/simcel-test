# features.py
"""
Feature engineering for demand forecasting & price optimization.

Creates:
- Time encodings: day_of_week, month_of_year, season (ordered categoricals)
- Robust lags (by self-merge on shifted dates):
    * sold_yesterday
    * sold_last_week
- Trailing moving averages (grouped by store_id × sku_id):
    * units_sold_ma7
    * units_sold_ma30
- Price-derived features:
    * final_price_ln
    * competitor_price_diff
- Promo flag for first day of a promo streak:
    * flag_promo_1stday
- Optional one-hot for time encodings (numeric; safe for modeling)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List

from utils.constants import DAY_ORDER, MONTH_ORDER, SEASON_ORDER


def _season_from_month(m: int) -> str:
    # Northern hemisphere convention; adjust if needed
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"


def generate_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features on top of a raw frame that includes at least:
    - date, store_id, sku_id, units_sold, final_price, competitor_price,
      base_price, promo_flag, promo_depth, holiday_flag, weather_index
    """
    out = df.copy()

    # ---- Basic hygiene ----
    if "date" not in out.columns:
        raise KeyError("Input must include a 'date' column.")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # IDs as category (important for XGBoost enable_categorical=True)
    for c in ("store_id", "sku_id"):
        if c in out.columns:
            out[c] = out[c].astype("category")

    # ---- Time encodings ----
    out["day_of_week"] = pd.Categorical(
        out["date"].dt.day_name().str.lower(),
        categories=DAY_ORDER,
        ordered=True,
    )
    out["month_of_year"] = pd.Categorical(
        out["date"].dt.month_name().str.lower(),
        categories=MONTH_ORDER,
        ordered=True,
    )
    out["season"] = pd.Categorical(
        out["date"].dt.month.map(_season_from_month),
        categories=SEASON_ORDER,
        ordered=True,
    )

    # ---- Robust lags via self-merge on shifted dates (handles gaps) ----
    # Base key and column
    key_cols = ["store_id", "sku_id", "date"]

    # sold_yesterday
    base_units = out[key_cols + ["units_sold"]].copy()
    y = base_units.copy()
    y["date"] = y["date"] + pd.Timedelta(days=1)  # so it aligns to "today"
    y = y.rename(columns={"units_sold": "sold_yesterday"})
    out = out.merge(y[key_cols[:2] + ["date", "sold_yesterday"]], on=key_cols, how="left")
    out["sold_yesterday"] = out["sold_yesterday"].fillna(-1)

    # sold_last_week
    w = base_units.copy()
    w["date"] = w["date"] + pd.Timedelta(days=7)
    w = w.rename(columns={"units_sold": "sold_last_week"})
    out = out.merge(w[key_cols[:2] + ["date", "sold_last_week"]], on=key_cols, how="left")
    out["sold_last_week"] = out["sold_last_week"].fillna(-1)

    # ---- Trailing moving averages (grouped) ----
    # Sort then groupby to ensure correct temporal order
    out = out.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
    g = out.groupby(["store_id", "sku_id"], observed=False, group_keys=False)
    out["units_sold_ma7"] = g["units_sold"].apply(lambda s: s.rolling(window=7, min_periods=1).mean()).values
    out["units_sold_ma30"] = g["units_sold"].apply(lambda s: s.rolling(window=30, min_periods=1).mean()).values

    # ---- Price-derived features ----
    out["final_price_ln"] = np.log(out["final_price"].clip(lower=1.0))
    out["competitor_price_diff"] = (
        (out["competitor_price"] - out["final_price"]) / out["final_price"]
    )

    # ---- First day of promo streak (prev day's promo == 0) ----
    # Robust to gaps: merge previous day's promo_flag
    if "promo_flag" in out.columns:
        prev = out[["store_id", "sku_id", "date", "promo_flag"]].copy()
        prev["date"] = prev["date"] + pd.Timedelta(days=1)
        prev = prev.rename(columns={"promo_flag": "prev_promo_flag"})
        out = out.merge(prev[["store_id", "sku_id", "date", "prev_promo_flag"]], on=key_cols, how="left")
        out["prev_promo_flag"] = out["prev_promo_flag"].fillna(0).astype(int)
        out["flag_promo_1stday"] = ((out["promo_flag"] == 1) & (out["prev_promo_flag"] == 0)).astype(int)
        out.drop(columns=["prev_promo_flag"], inplace=True)
    else:
        out["flag_promo_1stday"] = 0

    # ---- Optional one-hot for time encodings (purely numeric; harmless for modeling) ----
    dummies = pd.get_dummies(
        out[["day_of_week", "month_of_year", "season"]],
        prefix="onehot",
        prefix_sep="_",
        dtype=int,
    )
    out = pd.concat([out, dummies], axis=1)

    # ---- Final numeric NA handling (keep sold_last_week as already imputed) ----
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if "sold_last_week" in num_cols:
        num_cols.remove("sold_last_week")
    out[num_cols] = out[num_cols].fillna(-1)

    return out
