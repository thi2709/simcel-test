#!/usr/bin/env python3
"""
feature.py
Simple, clear feature engineering for demand data.

Functions:
- generate_feature(df): returns a new DataFrame with engineered features.
CLI:
  python feature.py --input data.csv --output features.csv
"""

from __future__ import annotations
import sys
import argparse
from typing import List
import numpy as np
import pandas as pd


DAY_ORDER: List[str] = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
MONTH_ORDER: List[str] = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
]
SEASON_ORDER: List[str] = ["spring","summer","autumn","winter"]  # Northern hemisphere


def _season_from_month(month: int) -> str:
    """Map month (1-12) to a season name (lowercase)."""
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "winter"  # months 12,1,2


def generate_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from an input DataFrame.
    Expected columns (already correctly typed): 
        date (datetime64), store_id (str), sku_id (str), final_price (float),
        competitor_price (float), units_sold (int)
    Returns a new DataFrame with added features and one-hot encodings.
    """
    required_cols = {"date", "store_id", "sku_id", "final_price", "competitor_price", "units_sold"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Work on a copy to avoid mutating caller's DataFrame
    data = df.copy()

    # Ensure date is datetime64[ns]
    if not np.issubdtype(data["date"].dtype, np.datetime64):
        try:
            data["date"] = pd.to_datetime(data["date"], errors="raise", utc=False)
        except Exception as e:
            raise ValueError("Column 'date' must be datetime. Failed to convert.") from e

    # Sort for stable group operations
    data = data.sort_values(["store_id", "sku_id", "date"], kind="mergesort").reset_index(drop=True)

    # Lags by calendar offsets within each (store_id, sku_id)
    grp = data.groupby(["store_id", "sku_id"], sort=False)

    # sold_yesterday: previous calendar day within each SKU-store series
    data["sold_yesterday"] = grp["units_sold"].shift(1)

    # sold_last_week: same weekday last week (7-day lag)
    data["sold_last_week"] = grp["units_sold"].shift(7)

    # sold_last_month: value at same calendar date last month (via self-merge on date-1month)
    prev_month_date = data["date"] - pd.DateOffset(months=1)
    key_cols = ["store_id", "sku_id"]
    # Build a lookup of units_sold by exact date
    lookup = data[key_cols + ["date", "units_sold"]].rename(columns={"units_sold": "sold_last_month", "date": "date_key"})
    data = data.merge(
        lookup,
        left_on=key_cols + [prev_month_date.rename("date_key")],
        right_on=key_cols + ["date_key"],
        how="left",
        validate="m:1"
    ).drop(columns=["date_key"])

    # Calendar features
    data["day_of_week"] = data["date"].dt.day_name().str.lower()      # monday .. sunday
    data["month_of_year"] = data["date"].dt.month_name().str.lower()  # january .. december
    data["season"] = data["date"].dt.month.map(_season_from_month)

    # Price transforms
    # final_price_ln: log(final_price) if > 0 else NaN
    data["final_price_ln"] = np.where(data["final_price"] > 0, np.log(data["final_price"]), np.nan)

    # competitor_price_diff = (competitor_price - final_price) / final_price
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = data["final_price"].replace(0, np.nan)
        data["competitor_price_diff"] = (data["competitor_price"] - data["final_price"]) / denom

    # Categoricals
    for col in ["store_id", "sku_id"]:
        data[col] = data[col].astype("category")

    data["day_of_week"] = pd.Categorical(data["day_of_week"], categories=DAY_ORDER, ordered=True)
    data["month_of_year"] = pd.Categorical(data["month_of_year"], categories=MONTH_ORDER, ordered=True)
    data["season"] = pd.Categorical(data["season"], categories=SEASON_ORDER, ordered=True)

    # One-hot encode the ordered categoricals (keep original columns as well)
    to_onehot = ["day_of_week", "month_of_year", "season"]
    onehot = pd.get_dummies(data[to_onehot], prefix="onehot", prefix_sep="_", dtype=np.uint8)
    data = pd.concat([data, onehot], axis=1)

    return data


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate features for demand dataset.")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", "-o", required=False, help="Path to write output CSV. If omitted, prints to stdout.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = _parse_args(argv)
        df = pd.read_csv(args.input, parse_dates=["date"])
        out = generate_feature(df)

        if args.output:
            out.to_csv(args.output, index=False)
        else:
            # Print to stdout without index
            out.to_csv(sys.stdout, index=False)
        return 0
    except Exception as e:
        print(f"[feature.py] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
