# pipeline.py
"""
Pricing pipeline for demand forecasting & price optimization.

CLI:
- Train & evaluate:
  python pipeline.py train --train-start 2023-01-01 --train-end 2023-09-30 --test-start 2023-10-01 --test-end 2023-12-31

- Simulate a price plan:
  python pipeline.py simulate --horizon 14 --price-plan path/to/plan.csv --out results.csv

- Optimize prices for a horizon:
  python pipeline.py optimize --horizon 14 --objective revenue --constraints constraints.yaml --out plan.csv
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from features import generate_feature
from data_loader import load_data
from models.modeling import fit as model_fit, validate as model_validate
from pricing.optimizer import optimize_price_plan

from utils.constants import OFFICIAL_MODEL, DATE_COL
from utils.schema import FEATURE_COLS, TARGET_COL
from utils.io_utils import load_constraints, resolve_model_path, load_model


# ---------- Helpers ----------
def _ensure_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Feature generation missing columns: {missing}")
    for c in ["store_id", "sku_id", "day_of_week", "month_of_year", "season"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df[FEATURE_COLS].copy()

# ---------- Commands ----------
def cmd_train(train_start: str, train_end: str, test_start: str, test_end: str, train_csv: Optional[str] = None):
    # df = load_data() if train_csv is None else pd.read_csv(train_csv, parse_dates=["date"])
    df = load_data(train_csv)
    df = df[df.stockout_flag == 0].reset_index(drop = True)
    print(f"Loaded {len(df)} rows")

    # df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_train = df[(df[DATE_COL] >= pd.to_datetime(train_start)) & (df[DATE_COL] <= pd.to_datetime(train_end))]
    # print('df_train', df_train)
    df_test = df[(df[DATE_COL] >= pd.to_datetime(test_start)) & (df[DATE_COL] <= pd.to_datetime(test_end))]
    # print('df_test', df_test)
    if df_train.empty or df_test.empty:
        raise ValueError("Empty train/test after date filtering. Check ranges.")

    df_train_transform = generate_feature(df_train)
    df_test_transform = generate_feature(df_test)

    X_train = _ensure_feature_cols(df_train_transform)
    print('X_train',X_train)
    y_train = df_train_transform[TARGET_COL]
    print('y_train',y_train)

    X_test = _ensure_feature_cols(df_test_transform)
    y_test = df_test_transform[TARGET_COL]

    model_filename = model_fit(X_train, y_train)
    print(f"✅ Saved model as: {model_filename}")

    # Validate
    model_path = resolve_model_path(model_filename, base_dir=os.path.join(os.path.dirname(__file__), "models"))
    metrics = model_validate(model_path, X_test, y_test)
    print(f"✅ Test metrics: {metrics}")


def cmd_simulate(price_plan_path: str, horizon: int, out_csv: Optional[str], constraints_path: Optional[str]):
    model = load_model(OFFICIAL_MODEL)
    cfg = load_constraints(constraints_path)

    unit_cost_map = {sku: float(v.get("unit_cost", 0.0)) for sku, v in cfg.get("skus", {}).items()}

    plan = load_data(price_plan_path)
    
    if horizon and horizon > 0:
        start_min = plan[DATE_COL].min()
        plan = plan[plan[DATE_COL] < (start_min + pd.Timedelta(days=int(horizon)))]

    feats = generate_feature(plan)
    X = _ensure_feature_cols(feats)

    units_pred = model.predict(X)
    units_pred = np.asarray(units_pred, dtype=float).ravel()
    revenue = plan["final_price"].to_numpy(dtype=float) * units_pred
    margin = revenue - plan["sku_id"].map(lambda s: unit_cost_map.get(s, 0.0)).to_numpy() * units_pred

    result = pd.DataFrame({
        DATE_COL: plan[DATE_COL].dt.date.astype(str),
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_str = pd.to_datetime(result["date"].min()).strftime("%Y%m%d")
    default_name = f"simulation_{start_str}_{horizon}days_{ts}.csv"
    out_path = os.path.join(os.path.dirname(__file__), out_csv if out_csv else default_name)
    result.to_csv(out_path, index=False)
    print(f"✅ Simulation results saved to: {out_path}")


def cmd_optimize(horizon: int, objective: str, constraints_path: str, steps: int, start_date: str | None = None, export: bool = True):
    start = (start_date or datetime.now().date().isoformat())
    _ = optimize_price_plan(
        model_ref=OFFICIAL_MODEL,
        start_date=start,
        horizon_days=horizon,
        price_grid_steps=int(steps),
        export=export,
    )


# ---------- CLI ----------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pricing pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and evaluate model")
    p_train.add_argument("--train-start", required=True)
    p_train.add_argument("--train-end", required=True)
    p_train.add_argument("--test-start", required=True)
    p_train.add_argument("--test-end", required=True)
    p_train.add_argument("--train-csv", default=None)

    p_sim = sub.add_parser("simulate", help="Simulate a price plan")
    p_sim.add_argument("--horizon", type=int, default=14)
    p_sim.add_argument("--price-plan", required=True)
    p_sim.add_argument("--out", default=None)
    p_sim.add_argument("--constraints", default=None)

    p_opt = sub.add_parser("optimize", help="Optimize prices for a horizon")
    p_opt.add_argument("--horizon", type=int, default=14)
    p_opt.add_argument("--objective", default="revenue", choices=["revenue"])
    p_opt.add_argument("--constraints", default=None)
    p_opt.add_argument("--out", default=None)
    p_opt.add_argument("--steps", type=int, default=7)
    p_opt.add_argument("--start-date", default=None)

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
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
