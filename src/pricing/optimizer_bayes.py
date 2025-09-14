# optimizer_bayes.py
"""
Bayesian optimizer for price planning using XGB model.
- Uses features.generate_feature() via utils/feature_utils.
- Objective: maximize revenue = final_price * predicted units_sold.
- Per (SKU, Store, Date), runs Gaussian Process BO on price to find best revenue.
- Enforces top-K promo days per horizon.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

from utils.schema import FEATURE_COLS
from utils.io_utils import load_constraints, load_model
from utils.feature_utils import base_row_for_candidate, featurize_candidates
from utils.math_utils import apply_stock_cap


# ------------------ Prediction helper ------------------
def _predict_units(model, feats: pd.DataFrame) -> np.ndarray:
    feats = feats.copy()
    for col in ["store_id", "sku_id", "day_of_week", "month_of_year", "season"]:
        if col in feats.columns:
            feats[col] = feats[col].astype("category")
    return np.asarray(model.predict(feats), dtype=float).ravel()


# ------------------ BO internals ------------------
def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best_y: float, xi: float = 1e-3) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - best_y - xi) / sigma
    return (mu - best_y - xi) * norm.cdf(z) + sigma * norm.pdf(z)

def _gp_surrogate(random_state: int = 0) -> GaussianProcessRegressor:
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.25, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True, random_state=random_state)


def _bayes_optimize_price(
    model,
    the_date: datetime,
    store_id: str,
    sku_id: str,
    base_price: float,
    comp_price: float,
    holiday_flag: float,
    weather_index: float,
    bounds: Tuple[float, float],
    sku_cfg: dict,
    n_init: int = 4,
    n_iter: int = 16,
    candidates: int = 256,
    random_state: int = 0,
) -> Tuple[float, float, float, float]:
    """
    Run Bayesian Optimization on price in [lo, hi].
    Return (best_price, best_units, best_revenue, best_depth).
    """
    lo, hi = bounds
    lo = max(0.0, float(lo))
    hi = float(hi)

    rng = np.random.default_rng(random_state)

    # Seed points (baseline + random samples)
    X = [base_price]
    if n_init > 1:
        init_rs = rng.uniform(lo, hi, size=n_init - 1)
        if (lo < base_price) and not np.any(init_rs < base_price):
            init_rs[0] = 0.5 * (lo + base_price)
        X.extend(init_rs.tolist())
    X = np.clip(np.array(X, dtype=float), lo, hi)

    def eval_prices(prices: np.ndarray) -> np.ndarray:
        rows = [
            base_row_for_candidate(the_date, store_id, sku_id, float(p), base_price,
                                   comp_price, holiday_flag, weather_index)
            for p in prices
        ]
        feats = featurize_candidates(rows)
        units = _predict_units(model, feats)
        units = np.array([apply_stock_cap(u, sku_cfg) for u in units], dtype=float)
        return prices * units

    y = eval_prices(X)

    gp = _gp_surrogate(random_state=random_state)
    for _ in range(int(n_iter)):
        gp.fit(X.reshape(-1, 1), y)
        Xcand = rng.uniform(lo, hi, size=candidates)
        mu, std = gp.predict(Xcand.reshape(-1, 1), return_std=True)
        ei = _expected_improvement(mu, std, best_y=float(y.max()), xi=1e-3)
        if float(ei.max()) < 1e-4:  # early stop
            break
        x_next = float(Xcand[np.argmax(ei)])
        if np.any(np.isclose(X, x_next, atol=1e-9)):
            x_next = float(np.clip(x_next + rng.normal(scale=0.001 * (hi - lo)), lo, hi))
        y_next = eval_prices(np.array([x_next]))[0]
        X = np.append(X, x_next)
        y = np.append(y, y_next)

    idx = int(np.argmax(y))
    best_price = float(X[idx])
    best_rev = float(y[idx])
    best_depth = (base_price - best_price) / base_price if best_price < base_price else 0.0
    best_units = best_rev / best_price if best_price > 0 else 0.0
    return best_price, best_units, best_rev, best_depth


# ------------------ End-to-end planner ------------------
def optimize_price_plan_bayes(
    model_ref: str,
    start_date: str,
    horizon_days: int,
    export: bool = True,
    n_init: int = 4,
    n_iter: int = 16,
    candidates: int = 256,
    random_state: int = 0,
    promo_k: int = 5,
) -> pd.DataFrame:
    """
    Horizon planning with Bayesian optimization per (SKU, Store, Date).
    Enforces top-K promo days per horizon.
    """
    cfg = load_constraints()
    model = load_model(model_ref)

    stores: List[str] = cfg["stores"]
    skus: Dict[str, dict] = cfg["skus"]
    gb = cfg["global"]
    lower = float(gb["price_bounds"]["lower"])
    upper = float(gb["price_bounds"]["upper"])
    promo_k = int(gb.get("promo_days_per_horizon", promo_k))
    holiday_flag = float(gb["holiday_flag_default"])
    weather_index = float(gb["weather_index_default"])
    competitor_policy = gb.get("competitor_price_policy", "equal_base_price")

    start_dt = datetime.fromisoformat(start_date)
    dates = [start_dt + timedelta(days=i) for i in range(int(horizon_days))]

    plans = []
    for sku_id, sku_cfg in skus.items():
        base_price = float(sku_cfg["base_price"])
        lo, hi = lower * base_price, upper * base_price

        for store_id in stores:
            for the_date in dates:
                comp_price = base_price if competitor_policy == "equal_base_price" else base_price

                # Baseline
                base_row = base_row_for_candidate(
                    the_date, store_id, sku_id,
                    base_price, base_price,
                    comp_price, holiday_flag, weather_index
                )
                base_df = featurize_candidates([base_row])
                base_units = float(_predict_units(model, base_df)[0])
                base_units = apply_stock_cap(base_units, sku_cfg)
                base_rev = base_price * base_units

                # Bayesian search
                rs = random_state + hash((sku_id, store_id, the_date.toordinal())) % (2**32 - 1)
                best_price, best_units, best_rev, best_depth = _bayes_optimize_price(
                    model=model,
                    the_date=the_date,
                    store_id=store_id,
                    sku_id=sku_id,
                    base_price=base_price,
                    comp_price=comp_price,
                    holiday_flag=holiday_flag,
                    weather_index=weather_index,
                    bounds=(lo, hi),
                    sku_cfg=sku_cfg,
                    n_init=n_init,
                    n_iter=n_iter,
                    candidates=candidates,
                    random_state=rs,
                )

                plans.append({
                    "date": the_date.date().isoformat(),
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "final_price": float(best_price),
                    "promo_depth": float(best_depth),
                    "pred_units": float(best_units),
                    "revenue": float(best_rev),
                    "is_promo": int(best_price < base_price),
                    "baseline_price": float(base_price),
                    "baseline_units": float(base_units),
                    "baseline_revenue": float(base_rev),
                    "uplift_revenue": float(best_rev - base_rev),
                })

    plan_raw = pd.DataFrame(plans).sort_values(["date", "store_id", "sku_id"]).reset_index(drop=True)

    # Enforce top-K promo days per (sku, store)
    final_rows = []
    for (sku_id, store_id), g in plan_raw.groupby(["sku_id", "store_id"], sort=False):
        g_pos = g[g["uplift_revenue"] > 0].sort_values(
            ["uplift_revenue", "date"], ascending=[False, True]
        ).reset_index(drop=True)

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
                final_rows.append(r.to_dict())
            else:
                final_rows.append({
                    "date": r["date"],
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "final_price": float(r["baseline_price"]),
                    "promo_depth": 0.0,
                    "pred_units": float(r["baseline_units"]),
                    "revenue": float(r["baseline_revenue"]),
                    "is_promo": 0,
                    "baseline_price": float(r["baseline_price"]),
                    "baseline_units": float(r["baseline_units"]),
                    "baseline_revenue": float(r["baseline_revenue"]),
                    "uplift_revenue": 0.0,
                })

    plan_df = pd.DataFrame(final_rows).sort_values(["date", "store_id", "sku_id"]).reset_index(drop=True)

    # summary
    total_opt = plan_df["revenue"].sum()
    total_base = plan_df["baseline_revenue"].sum()
    lift = total_opt - total_base
    lift_pct = (lift / total_base * 100.0) if total_base > 0 else np.nan

    print("=== Bayesian Optimization Summary ===")
    print(f"Horizon days         : {horizon_days}")
    print(f"Date range           : {plan_df['date'].min()} → {plan_df['date'].max()}")
    print(f"Total baseline rev   : {total_base:,.2f}")
    print(f"Total optimized rev  : {total_opt:,.2f}")
    print(f"Revenue lift         : {lift:,.2f} ({lift_pct:.2f}%)")

    if export:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
        fname = f"price_plan_bayes_{start_str}_{horizon_days}days_{ts}.csv"
        out_path = os.path.join(os.path.dirname(__file__), fname)
        export_df = plan_df[["date", "store_id", "sku_id", "final_price", "promo_depth", "is_promo", "baseline_price"]].copy()
        export_df = export_df.rename(columns={"is_promo": "promo_flag"})
        export_df.to_csv(out_path, index=False)
        print(f"Exported price plan to {out_path}")

    return plan_df
