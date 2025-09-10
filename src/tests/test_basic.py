# tests/test_basic.py
import numpy as np
import pandas as pd
import pytest

from features import generate_feature
from pricing.optimizer import _price_grid, _apply_stock_cap
from models.modeling import _check_schema, _check_inputs, FEATURE_COLS, TARGET_COL


def test_generate_feature_lags_and_prices():
    """Features: lags exist and behave; price-derived columns present and numeric NAs are filled."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-02")],
        "store_id": ["s1", "s1"],
        "sku_id": ["skuA", "skuA"],
        "final_price": [100.0, 90.0],
        "competitor_price": [100.0, 95.0],
        "units_sold": [5, 7],
    })

    out = generate_feature(df).sort_values("date").reset_index(drop=True)

    # must include these engineered columns
    needed = {"sold_yesterday", "sold_last_week", "final_price_ln", "competitor_price_diff"}
    assert needed.issubset(set(out.columns))

    # lag behavior: day 2 yesterday == day 1 units; day 1 last_week gets filled to -1
    assert out.loc[1, "sold_yesterday"] == 5
    assert out.loc[0, "sold_last_week"] == -1  # NA filled to -1 by feature fn

    # log price should be finite for positive prices
    assert np.isfinite(out["final_price_ln"]).all()


def test_optimizer_price_grid_and_stock_cap():
    """Optimizer helpers: price grid spans guardrails; stock cap clamps predictions."""
    grid = _price_grid(base_price=100.0, lower=0.8, upper=1.0, steps=5)
    assert np.allclose(grid, np.linspace(80.0, 100.0, 5))

    assert _apply_stock_cap(12.3, {"stock_cap": 10}) == 10
    assert _apply_stock_cap(8.0, {}) == 8.0  # no cap -> unchanged


def test_modeling_checks_valid_inputs():
    """Modeling checks pass on a minimal, valid X/y with correct schema & dtypes."""
    X = pd.DataFrame({
        "store_id": pd.Series(["s1"], dtype="category"),
        "sku_id": pd.Series(["skuA"], dtype="category"),
        "base_price": [100.0],
        "promo_flag": [0],
        "promo_depth": [0.0],
        "final_price": [100.0],
        "competitor_price": [100.0],
        "holiday_flag": [0.0],
        "weather_index": [0.5],
        "day_of_week": pd.Series(["monday"], dtype="category"),
        "month_of_year": pd.Series(["september"], dtype="category"),
        "season": pd.Series(["autumn"], dtype="category"),
        "sold_yesterday": [1.0],
        "sold_last_week": [1.0],
        "final_price_ln": [np.log(100.0)],
        "competitor_price_diff": [(100.0 - 100.0) / 100.0],
    })

    # y must be a Series named as the target column
    y = pd.Series([10.0], name=TARGET_COL)

    # No exceptions -> passes schema & input checks
    _check_schema(X, y)
    X_checked, y_checked = _check_inputs(X, y)

    # sanity on returned references
    assert len(X_checked) == len(y_checked) == 1
    assert list(X_checked.columns) == FEATURE_COLS
