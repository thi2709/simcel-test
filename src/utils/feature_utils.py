import numpy as np
import pandas as pd
from features import generate_feature
from utils.schema import FEATURE_COLS

def base_row_for_candidate(
    date, store_id, sku_id, price, base_price, competitor_price, holiday_flag, weather_index
) -> dict:
    promo_flag = int(price < base_price)
    promo_depth = (base_price - price) / base_price if promo_flag else 0.0
    return {
        "date": date,
        "store_id": store_id,
        "sku_id": sku_id,
        "final_price": float(price),
        "competitor_price": float(competitor_price),
        "units_sold": np.nan,
        "base_price": float(base_price),
        "promo_flag": promo_flag,
        "promo_depth": promo_depth,
        "holiday_flag": float(holiday_flag),
        "weather_index": float(weather_index),
    }

def featurize_candidates(rows: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    feats = generate_feature(raw)
    missing = [c for c in FEATURE_COLS if c not in feats.columns]
    if missing:
        raise RuntimeError(f"Feature generation missing columns: {missing}")
    return feats[FEATURE_COLS].copy()
