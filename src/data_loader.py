"""
data_loader.py
Simple CSV loader with schema checks.

"""

# from __future__ import annotations
# import argparse
import sys
import os
import pandas as pd


class DataLoaderError(Exception):
    """Raised when rows are dropped due to validation errors."""


# ---- Schema specification (column groups) ----
DATE_COL = "date"
ID_COLS = ["store_id", "sku_id"]
FLOAT_COLS = ["base_price", "final_price", "competitor_price", "revenue", "margin"]
FLAG_COLS = ["promo_flag", "holiday_flag", "stockout_flag"]
RATIO_COLS = ["promo_depth", "weather_index"]  # must be in [0, 1]
INT_COLS = ["week_of_year", "units_sold", "stock_cap"]
SET_COL = "set"

# _DEFAULT_PATH = 'ai-ml-challenge-main/retail_pricing_demand_2024.csv'

REQUIRED_COLUMNS = (
    [DATE_COL] + ID_COLS + FLOAT_COLS + FLAG_COLS + RATIO_COLS + INT_COLS + [SET_COL]
)

def _default_dataset_path() -> str:
    # project_root = parent of the directory that holds this file (i.e., .../submission)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "ai-ml-challenge-main", "retail_pricing_demand_2024.csv")

def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_data(path: str = _default_dataset_path()) -> pd.DataFrame:
    """
    Read CSV and return a validated, cleaned DataFrame.
    - Invalid rows are removed.
    - If any rows are removed, a DataLoaderError is raised (caller decides what to do).
    - Returns a deduplicated, typed DataFrame
    """
    
    # Read raw (let pandas parse dates only for `date`)
    df = pd.read_csv(
        path,
        dtype="string",                 # read everything as string first (predictable)
        keep_default_na=True,
        na_values=["", "NA", "N/A", "na", "n/a", "NULL", "null", "-", "--"],
        parse_dates=[DATE_COL],
        infer_datetime_format=True,
    )

    _ensure_required_columns(df)

    original_len = len(df)

    # ---------- Coerce types ----------
    # non-null date
    invalid_mask = df[DATE_COL].isna()

    # string ids + set, non-null
    for c in ID_COLS + [SET_COL]:
        df[c] = df[c].astype("string")
        invalid_mask |= df[c].isna()

    # floats (allow NaN, but coerce to float; your rule didn’t require non-null here)
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # no extra rule beyond type

    # flags must be 0 or 1
    for c in FLAG_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        invalid_mask |= ~df[c].isin([0, 1])

    # ratios in [0, 1]
    for c in RATIO_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        invalid_mask |= df[c].isna() | (df[c] < 0) | (df[c] > 1)

    # ints (coerce; allow NaN? your spec says int, so treat NaN as invalid)
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        invalid_mask |= df[c].isna()
        # finally cast to pandas nullable Int64
        df[c] = df[c].astype("Int64")

    # Drop invalid rows
    bad_rows = df[invalid_mask]
    if not bad_rows.empty:
        df = df[~invalid_mask].copy()

    # Final tidy types
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")  # already parsed; keep consistent
    for c in ID_COLS + [SET_COL]:
        df[c] = df[c].astype("string")
    for c in FLOAT_COLS + RATIO_COLS:
        df[c] = df[c].astype("float64")
    for c in FLAG_COLS:
        df[c] = df[c].astype("Int64")  # 0/1 as integer
    for c in INT_COLS:
        df[c] = df[c].astype("Int64")

    # If any row was dropped, raise with a concise summary
    dropped = original_len - len(df)
    if dropped > 0:
        example_idx = list(bad_rows.index[:5])
        raise DataLoaderError(
            f"Validation failed for {dropped} row(s). "
            f"Dropped rows indices (first 5): {example_idx}. "
            f"Returned DataFrame contains {len(df)} valid row(s)."
        )

    df = df.drop_duplicates(ignore_index = True)
    return df

