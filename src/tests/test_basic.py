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


# ==== ADDITIONAL SMOKE TESTS (append-only) ===================================
# These tests assume conftest.py already adds src/ to sys.path.

from pathlib import Path
import glob
import pandas as pd
import numpy as np
import pytest
from xgboost import XGBRegressor

# Pipeline pieces
from data_loader import load_data
from features import generate_feature
from models.modeling import _check_schema, _check_inputs, fit
from utils.schema import FEATURE_COLS, TARGET_COL
from utils.constants import DATE_COL, OFFICIAL_MODEL
from utils.io_utils import load_constraints, resolve_model_path, load_model

from pricing.optimizer import optimize_price_plan
from pipeline import cmd_simulate


def _make_minimal_valid_df(n_days: int = 8) -> pd.DataFrame:
    """
    Build a tiny but schema-valid dataset for the pipeline smoke test.
    Keeps flags/ratios in [0,1] and required ints present.
    """
    start = pd.Timestamp("2025-03-01")
    dates = pd.date_range(start, periods=n_days, freq="D")
    store_id = "s1"
    sku_id = "skuA"

    # Ratios ∈ [0,1], ints present
    promo_flag = np.array([0, 1] * (n_days // 2), dtype=int)
    promo_depth = np.where(promo_flag == 1, 0.20, 0.0)  # 20% when promo
    holiday_flag = np.zeros(n_days, dtype=int)
    weather_index = np.linspace(0.2, 0.8, n_days)

    base_price = np.full(n_days, 100.0)
    final_price = base_price * (1.0 - promo_depth)
    competitor_price = base_price * 1.05

    # Minimal target/int columns
    units_sold = np.arange(1, n_days + 1, dtype=int)  # simple increasing
    stockout_flag = np.zeros(n_days, dtype=int)

    df = pd.DataFrame(
        {
            DATE_COL: dates,
            "store_id": store_id,
            "sku_id": sku_id,
            "final_price": final_price,
            "base_price": base_price,
            "competitor_price": competitor_price,
            "promo_flag": promo_flag.astype(float),   # ratios as float per loader
            "promo_depth": promo_depth.astype(float),
            "holiday_flag": holiday_flag.astype(float),
            "weather_index": weather_index.astype(float),
            "units_sold": units_sold,
            "stockout_flag": stockout_flag,
            # Optional split column if your loader uses it
            "set": "train",
        }
    )
    return df


def test_pipeline_smoke_data_loader_to_modeling(tmp_path):
    """
    End-to-end smoke: data_loader -> features -> modeling.
    Ensures no exceptions are raised and a model can be fit on a tiny dataset.
    """
    # 1) Create a minimal CSV the data_loader can ingest
    df_raw = _make_minimal_valid_df(n_days=8)
    csv_path = tmp_path / "tiny_dataset.csv"
    df_raw.to_csv(csv_path, index=False)

    # 2) Load via data_loader (validates types/ranges & drops bad rows if any)
    df_loaded = load_data(str(csv_path))
    assert len(df_loaded) == len(df_raw), "Unexpected row drops in load_data()"

    # 3) Feature engineering
    feats = generate_feature(df_loaded)
    # Ensure at least the feature columns exist
    for c in FEATURE_COLS:
        assert c in feats.columns, f"Missing engineered feature column: {c}"

    # 4) Schema & inputs check for modeling
    X = feats[FEATURE_COLS].copy()
    y = df_loaded[TARGET_COL].copy()
    _check_schema(X, y)
    X_checked, y_checked = _check_inputs(X, y)
    assert len(X_checked) == len(y_checked) > 0

    # 5) Fit (tiny model on tiny data) — smoke only; expect no exceptions
    # NOTE: model config is fixed in modeling.py; this just asserts it runs.
    model = fit(X_checked, y_checked, test_size=0.5, save_model=False)
    assert isinstance(model, XGBRegressor)


def test_optimizer_runs_smoke(tmp_path, monkeypatch):
    """
    Optimizer smoke: ensure optimize_price_plan runs and returns a well-formed DataFrame.
    Uses OFFICIAL_MODEL and constraints.yaml from the repo.
    Keep horizon small for speed.
    """
    # Sanity: model must resolve
    model_path = resolve_model_path(OFFICIAL_MODEL)
    assert Path(model_path).exists(), f"Model not found: {model_path}"

    # Run optimizer (small horizon/steps to keep test light)
    plan_df = optimize_price_plan(
        model_ref=OFFICIAL_MODEL,
        start_date="2025-03-01",
        horizon_days=3,
        price_grid_steps=3,
        export=False,  # do not write file in test
    )
    assert isinstance(plan_df, pd.DataFrame)
    assert not plan_df.empty

    # Core columns that should exist in the plan
    expected_cols = {
        "date",
        "store_id",
        "sku_id",
        "final_price",
        "promo_flag",
        "promo_depth",
        "base_price",
        "pred_units",
        "revenue",
    }
    assert expected_cols.issubset(set(map(str, plan_df.columns))), (
        f"optimizer output missing columns: {expected_cols - set(plan_df.columns)}"
    )


def test_pipeline_cmd_simulate_with_existing_price_plan(tmp_path):
    """
    Smoke test for pipeline.cmd_simulate using an existing price_plan_*.csv in pricing/.
    Writes output CSV to tmp_path and asserts it exists and has core columns.
    """
    project_root = Path(__file__).resolve().parents[1]  # points to src/
    pricing_dir = project_root / "pricing"
    price_plan_candidates = sorted(pricing_dir.glob("price_plan_*.csv"))
    assert price_plan_candidates, "No price_plan_*.csv found in pricing/ for simulation test."

    price_plan_path = str(price_plan_candidates[-1])  # use the most recent
    constraints_path = str(pricing_dir / "constraints.yaml")
    assert Path(constraints_path).exists(), "constraints.yaml not found for cmd_simulate test."

    out_path = tmp_path / "sim_result.csv"

    # Run simulation over a short horizon for speed
    cmd_simulate(
        price_plan_path=price_plan_path,
        horizon=5,
        out_csv=str(out_path),
        constraints_path=constraints_path,
    )

    assert out_path.exists(), "Simulation did not produce an output CSV."

    sim = pd.read_csv(out_path)
    assert not sim.empty

    expected_cols = {
        "date",
        "store_id",
        "sku_id",
        "final_price",
        "promo_flag",
        "promo_depth",
        "base_price",
        "pred_units",
        "revenue",
        "margin",
    }
    assert expected_cols.issubset(set(map(str, sim.columns))), (
        f"simulation output missing columns: {expected_cols - set(sim.columns)}"
    )
