"""
data_loader.py
Simple CSV loader with schema checks.
- Ensures required columns exist (from utils.constants).
- Enforces dtypes:
    * date -> datetime64
    * store_id, sku_id -> categorical  (important for XGBoost enable_categorical=True)
    * SET_COL -> string
    * flags -> Int64 and only {0,1}
    * ratios -> float64 in [0,1]
    * floats -> float64 (allow NA; features/training can handle)
    * ints -> Int64 (non-null)
- Drops invalid rows; raises DataLoaderError with a concise summary if any were dropped.
- Returns a deduplicated, typed DataFrame.
"""

# from __future__ import annotations

import os
import pandas as pd
import numpy as np

from utils.constants import (
    DATE_COL, ID_COLS, PRICE_COL, FLOAT_COLS, RATIO_COLS,
    INT_COLS, SET_COL, REQUIRED_COLUMNS,
)

from utils.io_utils import load_constraints

class DataLoaderError(Exception):
    """Raised when rows are dropped due to validation errors."""


def _default_dataset_path() -> str:
    # project_root = parent of the directory that holds this file (i.e., .../src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # adjust if your CSV sits elsewhere
    return os.path.join(project_root, "ai-ml-challenge-main", "retail_pricing_demand_2024.csv")


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_data(path: str | None = None, constraints_path: str | None = None) -> pd.DataFrame:
    """
    Read CSV and return a validated, cleaned DataFrame.
    If any rows are removed during validation, raises DataLoaderError (after constructing the cleaned df).
    """
    if path is None:
        path = _default_dataset_path()

    # Read raw CSV; parse date column
    df = pd.read_csv(
        path,
        dtype="string",
        keep_default_na=True,
        na_values=["", "NA", "N/A", "na", "n/a", "NULL", "null", "-", "--"],
        parse_dates=[DATE_COL],
        infer_datetime_format=True,
    )

    cfg = load_constraints(constraints_path)
    holiday_flag_default = float(cfg["global"].get("holiday_flag_default", 0.0))
    weather_index_default = float(cfg["global"].get("weather_index_default", 0.0))

    if not set([DATE_COL] + ID_COLS + [PRICE_COL]).issubset(df.columns):
        raise ValueError(f"Input file must include {[DATE_COL] + ID_COLS + [PRICE_COL]}")

    if "base_price" not in df.columns:
        df["base_price"] = df["final_price"]
    if "promo_flag" not in df.columns:
        df["promo_flag"] = (df["final_price"] < df["base_price"]).astype(int)
    if "promo_depth" not in df.columns:
        df["promo_depth"] = (df["base_price"] - df["final_price"]) / df["base_price"]
        df.loc[df["promo_flag"] == 0, "promo_depth"] = 0.0
    if "holiday_flag" not in df.columns:
        df["holiday_flag"] = holiday_flag_default
    if "weather_index" not in df.columns:
        df["weather_index"] = weather_index_default
    if "competitor_price" not in df.columns:
        df["competitor_price"] = df["base_price"]
    if "units_sold" not in df.columns:
        df["units_sold"] = 0
    if "stockout_flag" not in df.columns:
        df["stockout_flag"] = 0

    _ensure_required_columns(df)
    original_len = len(df)

    # ---------- Coercions & validation ----------
    invalid_mask = df[DATE_COL].isna()

    # IDs: read as string first to validate; IDs will become category later
    for c in ID_COLS:
        df[c] = df[c].astype("string")
        invalid_mask |= df[c].isna()

    # floats: allow NA; just coerce
    for c in FLOAT_COLS + [PRICE_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ratios: must be within [0,1] and non-null
    for c in RATIO_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        invalid_mask |= df[c].isna() | (df[c] > 1) | (df[c] < 0)

    # ints: required and non-null
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        invalid_mask |= df[c].isna()

    # Track rows to drop (for error message)
    bad_rows = df[invalid_mask].copy()

    # Drop invalid rows
    if invalid_mask.any():
        df = df[~invalid_mask].copy()

    # ---------- Final tidy types ----------
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # IDs as categorical (critical for your original modeling/optimizing path)
    for c in ID_COLS:
        df[c] = df[c].astype("category")

    # numeric dtypes
    for c in FLOAT_COLS + [PRICE_COL] + RATIO_COLS:
        df[c] = df[c].astype("float64")
    for c in INT_COLS:
        df[c] = df[c].astype("Int64")

    # Deduplicate
    df = df.drop_duplicates(ignore_index=True)

    # ---------- Raise if anything was dropped ----------
    dropped = original_len - len(df)
    if dropped > 0:
        example_idx = list(bad_rows.index[:5])
        raise DataLoaderError(
            f"Validation failed for {dropped} row(s). "
            f"Dropped rows indices (first 5): {example_idx}. "
            f"Returned DataFrame contains {len(df)} valid row(s)."
        )

    return df
