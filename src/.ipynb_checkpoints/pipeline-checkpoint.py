# src/pipeline.py
"""
Create a pipeline that, given candidate prices for a future horizon, returns:

- Expected demand (using your demand model)
- Revenue, units, and margin projections
- Constraint handling:
    - price bounds per SKU (e.g., 0.6×base_price ≤ price ≤ 1×base_price)
    - optional inventory/stock caps (if provided; not enforced here)
    - promo scheduling rules (e.g., max X promo days per month) handled by optimizer

CLI:
- Train & evaluate:
  python pipeline.py train --train-start 2023-01-01 --train-end 2023-09-30 --test-start 2023-10-01 --test-end 2023-12-31

- Simulate a price plan for next 14 days:
  python pipeline.py simulate --horizon 14 --price-plan src/pricing/price_plan_20240301_30days_20250909_151434.csv --out results.csv

- Optimize prices for a horizon:
  python pipeline.py optimize --horizon 14 --objective revenue --constraints src/pricing/constraints.yaml --out plan.csv
"""

from __future__ import annotations

import os
import sys
import glob
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# --- Local imports (robust to minor naming diffs) ---
# features
from features import generate_feature
from data_loader import load_data
from models.modeling import fit as model_fit, validate as model_validate
from pricing.optimizer import optimize_price_plan

# ---------- Paths ----------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_THIS_DIR, "models")
_PRICING_DIR = os.path.join(_THIS_DIR, "pricing")
_DEFAULT_CONSTRAINTS = os.path.join(_PRICING_DIR, "constraints.yaml")


# ---------- Constants ----------
FEATURE_COLS = [
    "store_id", "sku_id", "base_price", "promo_flag", "promo_depth",
    "final_price", "competitor_price", "holiday_flag", "weather_index",
    "day_of_week", "month_of_year", "season",
    "sold_yesterday", "sold_last_week",
    "final_price_ln", "competitor_price_diff",
]
TARGET_COL = "units_sold"

# Margin: revenue - unit_cost*units
# If unit_cost not supplied, assume 0 (i.e., margin == revenue)
DEFAULT_UNIT_COST_FALLBACK = 0.0

# Defaults used in simulation if not carried in the price plan file
DEFAULT_HOLIDAY_FLAG = 0.0215
DEFAULT_WEATHER_INDEX = 0.5
OFFICIAL_MODEL = "forecaster_xgboost_20250909_184948.pkl"

# ---------- Helpers ----------

def _load_model_anywhere() -> str:
    """Return the absolute path to the official model (hardcoded)."""
    path = os.path.join(_MODELS_DIR, OFFICIAL_MODEL)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Official model not found at: {path}")
    return path


def _ensure_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature generation missing columns: {missing}")
    # cast categoricals consistently
    for c in ["store_id", "sku_id", "day_of_week", "month_of_year", "season"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df[FEATURE_COLS].copy()


def _compute_margin(revenue: float, units: float, unit_cost: float) -> float:
    try:
        return float(revenue - unit_cost * units)
    except Exception:
        return float(revenue)


# ---------- Commands ----------
def cmd_train(train_start: str, train_end: str, test_start: str, test_end: str, train_csv: Optional[str] = None):
    """
    1) read training csv
    2) train, validate, save model via models.modeling
    """
    # Load data
    try:
        df = load_data()
        print(f'Loaded {len(df)} rows') 
    except Exception as e:
        raise RuntimeError(f"Failed to load training data: {e}")

    # Ensure date column is datetime
    if "date" not in df.columns:
        raise ValueError("Input training data must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by date ranges
    df_train = df[(df["date"] >= pd.to_datetime(train_start)) & (df["date"] <= pd.to_datetime(train_end))].copy()
    df_test = df[(df["date"] >= pd.to_datetime(test_start)) & (df["date"] <= pd.to_datetime(test_end))].copy()

    if df_train.empty or df_test.empty:
        raise ValueError("Empty train/test after date filtering. Check your ranges.")

    # Feature engineering
    feats_train = generate_feature(df_train)
    feats_test = generate_feature(df_test)

    X_train = _ensure_feature_cols(feats_train)
    y_train = df_train[TARGET_COL]
    
    X_test = _ensure_feature_cols(feats_test)
    y_test = df_test[TARGET_COL]

    # Train & save
    model_filename = model_fit(X_train, y_train)
    print(f"Saved model as: {model_filename} (in models directory next to modeling.py)")

    # Validate on the test slice
    # Try loading from models dir
    model_path_try = os.path.join(_MODELS_DIR, model_filename)
    if not os.path.exists(model_path_try):
        # fallback: assume the filename is relative to modeling.py only
        model_path_try = model_filename
    metrics = model_validate(model_path_try, X_test, y_test)
    print(f"Test metrics: {metrics}")


def cmd_simulate(price_plan_path: str, horizon: int, out_csv: Optional[str], constraints_path: Optional[str]):
    """
    3) simulate the performance of a price_plan (predict units, revenue, margin)
    Save results.csv in the same directory as pipeline.py unless --out is given.
    """
    model_path = _load_model_anywhere()

    # Constraints for costs/defaults (optional but useful)
    unit_cost_map = {}  # sku_id -> unit_cost
    holiday_flag_default = DEFAULT_HOLIDAY_FLAG
    weather_index_default = DEFAULT_WEATHER_INDEX
    if constraints_path and os.path.exists(constraints_path):
        try:
            import yaml
            with open(constraints_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            holiday_flag_default = float(cfg["global"].get("holiday_flag_default", holiday_flag_default))
            weather_index_default = float(cfg["global"].get("weather_index_default", weather_index_default))
            for sku_id, info in cfg.get("skus", {}).items():
                if "unit_cost" in info:
                    unit_cost_map[sku_id] = float(info["unit_cost"])
        except Exception:
            pass  # keep defaults

    # Read price plan
    plan = pd.read_csv(price_plan_path)
    if not {"date", "store_id", "sku_id", "final_price"}.issubset(plan.columns):
        raise ValueError("price_plan must have at least: date, store_id, sku_id, final_price")

    # Try to infer base_price & promo_flag/depth if present
    if "baseline_price" in plan.columns:
        plan["base_price"] = plan["baseline_price"]
    elif "base_price" not in plan.columns:
        # fallback: assume non-promo equals final_price
        plan["base_price"] = plan["final_price"]

    if "promo_flag" not in plan.columns:
        plan["promo_flag"] = (plan["final_price"] < plan["base_price"]).astype(int)
    if "promo_depth" not in plan.columns:
        plan["promo_depth"] = (plan["base_price"] - plan["final_price"]) / plan["base_price"]
        plan.loc[plan["promo_flag"] == 0, "promo_depth"] = 0.0

    # Defaults for exogenous vars
    plan["holiday_flag"] = holiday_flag_default
    plan["weather_index"] = weather_index_default

    # Competitor price assumption (equal to base)
    if "competitor_price" not in plan.columns:
        plan["competitor_price"] = plan["base_price"]

    # Required by feature fn but unknown for the future horizon
    plan["units_sold"] = np.nan

    # Ensure datetime & (optionally) clip to horizon days from min date
    plan["date"] = pd.to_datetime(plan["date"], errors="coerce")
    if horizon and horizon > 0:
        start_min = plan["date"].min()
        plan = plan[plan["date"] < (start_min + pd.Timedelta(days=int(horizon)))]

    # Feature engineering & prediction
    feats = generate_feature(plan)
    X = _ensure_feature_cols(feats)

    # Load model and predict
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    units_pred = model.predict(X)
    units_pred = np.asarray(units_pred, dtype=float).ravel()
    revenue = plan["final_price"].to_numpy(dtype=float) * units_pred

    # Margin (uses constraints unit_cost if available; else fallback)
    unit_cost = plan["sku_id"].map(lambda s: unit_cost_map.get(s, DEFAULT_UNIT_COST_FALLBACK)).astype(float)
    margin = revenue - unit_cost.to_numpy(dtype=float) * units_pred

    result = pd.DataFrame({
        "date": plan["date"].dt.date.astype(str),
        "store_id": plan["store_id"],
        "sku_id": plan["sku_id"],
        "final_price": plan["final_price"].astype(float),
        "promo_flag": plan["promo_flag"].astype(int),
        "promo_depth": plan["promo_depth"].astype(float),
        "base_price": plan["base_price"].astype(float),
        "pred_units": units_pred,
        "revenue": revenue,
        "margin": margin,
    })

    # Export
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_str = pd.to_datetime(result["date"].min()).strftime("%Y%m%d")
    default_name = f"simulation_{start_str}_{horizon}days_{ts}.csv"
    out_path = os.path.join(_THIS_DIR, out_csv if out_csv else default_name)
    result.to_csv(out_path, index=False)
    print(f"Simulation results saved to: {out_path}")


def cmd_optimize(
    horizon: int,
    objective: str,
    constraints_path: str,
    steps: int,
    start_date: str | None = None,
    export: bool = True,
):
    """
    4) Optimize prices for a horizon and export plan.
    start_date: ISO date string (YYYY-MM-DD). If None, defaults to today.
    """
    start = (start_date or datetime.now().date().isoformat())

    _ = optimize_price_plan(
        model_ref=os.path.basename(_load_model_anywhere()),
        start_date=start,
        horizon_days=horizon,
        constraints_path=constraints_path if constraints_path else _DEFAULT_CONSTRAINTS,
        price_grid_steps=int(steps),
        export=export,
    )


# ---------- CLI ----------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pricing pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train and evaluate model")
    p_train.add_argument("--train-start", required=True)
    p_train.add_argument("--train-end", required=True)
    p_train.add_argument("--test-start", required=True)
    p_train.add_argument("--test-end", required=True)
    p_train.add_argument("--train-csv", default=None, help="Optional explicit CSV path (else data_loader.load_data())")

    # simulate
    p_sim = sub.add_parser("simulate", help="Simulate a price plan using a trained model")
    p_sim.add_argument("--horizon", type=int, default=14)
    p_sim.add_argument("--price-plan", required=True, help="Path to price_plan CSV")
    p_sim.add_argument("--out", default=None, help="Output CSV (default: auto-named in pipeline dir)")
    p_sim.add_argument("--model", default=None, help="Model file (else use latest in models/)")
    p_sim.add_argument("--constraints", default=_DEFAULT_CONSTRAINTS, help="constraints.yaml (optional; for costs/defaults)")

    # optimize
    p_opt = sub.add_parser("optimize", help="Optimize prices for a horizon")
    p_opt.add_argument("--horizon", type=int, default=14)
    p_opt.add_argument("--objective", default="revenue", choices=["revenue"])
    p_opt.add_argument("--constraints", default=_DEFAULT_CONSTRAINTS, help="Path to constraints.yaml")
    p_opt.add_argument("--out", default=None, help="Output CSV (default: auto-named in pipeline dir)")
    p_opt.add_argument("--model", default=None, help="Model file to use (default: latest in models/)")
    p_opt.add_argument("--steps", type=int, default=7, help="Price grid steps for optimizer")
    p_opt.add_argument("--start-date", default=None, help="ISO date (YYYY-MM-DD); default: today")  # <-- NEW

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(args.train_start, args.train_end, args.test_start, args.test_end, args.train_csv)
    elif args.cmd == "simulate":
        cmd_simulate(args.price_plan, args.horizon, args.out, args.constraints)
    elif args.cmd == "optimize":
        cmd_optimize(args.horizon, args.objective, args.constraints, args.steps, args.start_date)
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Keep errors readable for CLI usage
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
