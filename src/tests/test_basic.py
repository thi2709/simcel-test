# tests/test_basic.py
import numpy as np
import pandas as pd

from features import generate_feature
# from pricing.optimizer import _price_grid, _apply_stock_cap
from models.modeling import _check_schema, _check_inputs
from utils.schema import FEATURE_COLS, TARGET_COL  # <-- single source of truth


def test_generate_feature_lags_and_prices():
    """Features: engineered columns exist; basic lag behavior remains sane."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-02")],
        "store_id": ["s1", "s1"],
        "sku_id": ["skuA", "skuA"],

        # raw inputs commonly required by generate_feature
        "base_price": [100.0, 100.0],
        "promo_flag": [0, 1],
        "promo_depth": [0.0, 0.1],
        "final_price": [100.0, 90.0],
        "competitor_price": [100.0, 95.0],
        "holiday_flag": [0, 0],
        "weather_index": [0.5, 0.6],

        # target
        "units_sold": [5, 7],
    })

    out = generate_feature(df).sort_values("date").reset_index(drop=True)

    needed = {
        "sold_yesterday",
        "sold_last_week",
        "units_sold_ma7",
        "units_sold_ma30",
        "flag_promo_1stday",
        "final_price_ln",
        "competitor_price_diff",
    }
    assert needed.issubset(set(out.columns))

    # lag behavior
    assert out.loc[1, "sold_yesterday"] == 5
    assert np.isfinite(out["final_price_ln"]).all()



# def test_optimizer_price_grid_and_stock_cap():
#     """Optimizer helpers: price grid spans guardrails; stock cap clamps predictions."""
#     grid = _price_grid(base_price=100.0, lower=0.8, upper=1.0, steps=5)
#     assert np.allclose(grid, np.linspace(80.0, 100.0, 5))

#     assert _apply_stock_cap(12.3, {"stock_cap": 10}) == 10
#     assert _apply_stock_cap(8.0, {}) == 8.0  # no cap -> unchanged


def test_modeling_checks_valid_inputs():
    """Modeling checks pass on a minimal, valid X/y with correct schema & dtypes."""
    # Build a single-row X covering **all** FEATURE_COLS in the correct order.
    seed_row = {
        "store_id": pd.Series(["s1"], dtype="category"),
        "sku_id": pd.Series(["skuA"], dtype="category"),
        "base_price": [100.0],
        "promo_flag": [0],
        "promo_depth": [0.0],
        "final_price": [100.0],
        "competitor_price": [100.0],
        "holiday_flag": [0],          # int/0-1 is fine
        "weather_index": [0.5],
        "day_of_week": pd.Series(["monday"], dtype="category"),
        "month_of_year": pd.Series(["september"], dtype="category"),
        "season": pd.Series(["autumn"], dtype="category"),
        "sold_yesterday": [1.0],
        "sold_last_week": [1.0],
        "units_sold_ma7": [1.0],
        "units_sold_ma30": [1.0],
        "flag_promo_1stday": [0],
        "final_price_ln": [np.log(100.0)],
        "competitor_price_diff": [0.0],
    }
    # Respect column order exactly as FEATURE_COLS:
    X = pd.DataFrame({c: seed_row[c] for c in FEATURE_COLS})

    # y must be a Series named as the target column
    y = pd.Series([10.0], name=TARGET_COL)

    # No exceptions -> passes schema & input checks
    _check_schema(X, y)
    X_checked, y_checked = _check_inputs(X, y)

    # sanity on returned references
    assert len(X_checked) == len(y_checked) == 1
    assert list(X_checked.columns) == FEATURE_COLS
