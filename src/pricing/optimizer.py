# optimizer.py
"""
Greedy grid optimizer for price planning using XGB model.
- Uses features.generate_feature() via utils/feature_utils for consistent features.
- Objective: maximize revenue = final_price * predicted units_sold
- Plan:
    1) For each SKU×Store×Date, evaluate baseline and best promo revenue.
    2) For each SKU×Store, pick top-K uplift days as promo.
    3) All other days stay at baseline (non-promo).
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

from utils.schema import FEATURE_COLS
from utils.io_utils import load_constraints, load_model
from utils.feature_utils import base_row_for_candidate, featurize_candidates
from utils.math_utils import price_grid, apply_stock_cap


# ------------------ Prediction helper ------------------
def _predict_units(model, feats: pd.DataFrame) -> np.ndarray:
    feats = feats.copy()
    for col in ["store_id", "sku_id", "day_of_week", "month_of_year", "season"]:
        if col in feats.columns:
            feats[col] = feats[col].astype("category")
    return np.asarray(model.predict(feats), dtype=float).ravel()


# ------------------ Core optimizer ------------------
def optimize_price_plan(
    model_ref: str,
    start_date: str,
    horizon_days: int,
    price_grid_steps: int = 7,
    export: bool = True,
) -> pd.DataFrame:
    """
    Generate a horizon plan with top-K promo days per SKU–Store.
    Constraints.yaml is auto-loaded from project root (via load_constraints()).
    """
    cfg = load_constraints()
    model = load_model(model_ref)

    stores: List[str] = cfg["stores"]
    skus: Dict[str, dict] = cfg["skus"]
    gb = cfg["global"]
    lower = float(gb["price_bounds"]["lower"])
    upper = float(gb["price_bounds"]["upper"])
    promo_k = int(gb.get("promo_days_per_horizon", 5))
    holiday_flag = float(gb["holiday_flag_default"])
    weather_index = float(gb["weather_index_default"])
    competitor_policy = gb.get("competitor_price_policy", "equal_base_price")

    try:
        start_dt = datetime.fromisoformat(start_date)
    except Exception:
        raise ValueError("start_date must be ISO format, e.g., '2025-09-10'")
    dates = [start_dt + timedelta(days=i) for i in range(int(horizon_days))]

    rows = []
    for sku_id, sku_cfg in skus.items():
        
        base_price = float(sku_cfg["base_price"])
        grid = price_grid(base_price, lower, upper, price_grid_steps)
        promo_grid = grid[grid < base_price]

        for store_id in stores:
            for the_date in dates:
                comp_price = base_price if competitor_policy == "equal_base_price" else base_price

                # baseline
                base_row = base_row_for_candidate(
                    the_date, store_id, sku_id,
                    base_price, base_price,
                    comp_price, holiday_flag, weather_index
                )
                base_df = featurize_candidates([base_row])
                base_units = float(_predict_units(model, base_df)[0])
                base_units = apply_stock_cap(base_units, sku_cfg)
                base_rev = base_price * base_units

                # promo candidates
                best_price, best_units, best_rev, best_depth, uplift = (
                    base_price, base_units, base_rev, 0.0, 0.0
                )
                if promo_grid.size > 0:
                    cand_rows = [
                        base_row_for_candidate(
                            the_date, store_id, sku_id,
                            float(p), base_price,
                            comp_price, holiday_flag, weather_index
                        )
                        for p in promo_grid
                    ]
                    cand_df = featurize_candidates(cand_rows)
                    preds = _predict_units(model, cand_df)
                    preds = [apply_stock_cap(u, sku_cfg) for u in preds]
                    preds = np.array(preds, dtype=float)
                    prices = cand_df["final_price"].to_numpy()
                    revs = prices * preds
                    idx = int(np.argmax(revs))
                    cand_price, cand_units, cand_rev = float(prices[idx]), float(preds[idx]), float(revs[idx])
                    cand_depth = (base_price - cand_price) / base_price
                    cand_uplift = cand_rev - base_rev
                    if cand_uplift > 0:
                        best_price, best_units, best_rev = cand_price, cand_units, cand_rev
                        best_depth, uplift = cand_depth, cand_uplift

                rows.append({
                    "date": the_date.date().isoformat(),
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "best_promo_price": best_price,
                    "best_promo_units": best_units,
                    "best_promo_revenue": best_rev,
                    "best_promo_depth": best_depth,
                    "base_price": base_price,
                    "baseline_units": base_units,
                    "baseline_revenue": base_rev,
                    "uplift_revenue": uplift,
                })

    tmp = pd.DataFrame(rows)

    # --- Select top-K promo days per SKU–Store ---
    plans = []
    for (sku_id, store_id), g in tmp.groupby(["sku_id", "store_id"], sort=False):
        g_pos = (
            g[g["uplift_revenue"] > 0]
            .sort_values(["uplift_revenue", "date"], ascending=[False, True])
            .reset_index(drop=True)
        )
        top_dates = []
        seen = set()
        for _, r in g_pos.iterrows():
            if r["date"] not in seen:
                top_dates.append(r["date"])
                seen.add(r["date"])
            if len(top_dates) >= promo_k:
                break
        promo_days = set(top_dates)

        for _, r in g.iterrows():
            if r["date"] in promo_days:
                final_price, pred_units, revenue, is_promo, promo_depth = (
                    r["best_promo_price"], r["best_promo_units"], r["best_promo_revenue"], 1, r["best_promo_depth"]
                )
            else:
                final_price, pred_units, revenue, is_promo, promo_depth = (
                    r["base_price"], r["baseline_units"], r["baseline_revenue"], 0, 0.0
                )

            plans.append({
                "date": r["date"],
                "store_id": store_id,
                "sku_id": sku_id,
                "final_price": float(final_price),
                "promo_depth": float(promo_depth),
                "pred_units": float(pred_units),
                "revenue": float(revenue),
                "is_promo": is_promo,
                "base_price": float(r["base_price"]),
                "baseline_units": float(r["baseline_units"]),
                "baseline_revenue": float(r["baseline_revenue"]),
                "uplift_revenue": float(revenue - r["baseline_revenue"]),
            })

    plan_df = pd.DataFrame(plans).sort_values(["date", "store_id", "sku_id"]).reset_index(drop=True)

    # summary
    total_opt = plan_df["revenue"].sum()
    total_base = plan_df["baseline_revenue"].sum()
    lift = total_opt - total_base
    lift_pct = (lift / total_base * 100.0) if total_base > 0 else np.nan

    print("=== Optimization Summary ===")
    print(f"Horizon days         : {horizon_days}")
    print(f"Date range           : {plan_df['date'].min()} → {plan_df['date'].max()}")
    print(f"Total baseline rev   : {total_base:,.2f}")
    print(f"Total optimized rev  : {total_opt:,.2f}")
    print(f"Revenue lift         : {lift:,.2f} ({lift_pct:.2f}%)")

    if export:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
        fname = f"price_plan_{start_str}_{horizon_days}days_{ts}.csv"
        out_path = os.path.join(os.path.dirname(__file__), fname)

        export_df = plan_df[[
            "date", "store_id", "sku_id", "final_price", "promo_depth", "is_promo", "base_price"
        ]].copy()
        export_df = export_df.rename(columns={"is_promo": "promo_flag"})
        export_df.to_csv(out_path, index=False)
        print(f"Exported price plan to {out_path}")
    
    return plan_df
