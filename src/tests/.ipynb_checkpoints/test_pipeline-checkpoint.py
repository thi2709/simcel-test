import pytest
import pandas as pd
import pickle
from pipeline import _compute_margin

def test_compute_margin_simple():
    assert _compute_margin(100, 10, 5) == 50.0
    assert isinstance(_compute_margin("bad", 1, 1), float)

def test_cmd_optimize_smoke(constraints_yaml, tmp_path):
    # reuse DummyRegressor as above
    from sklearn.dummy import DummyRegressor
    from features import generate_feature
    from modeling import FEATURE_COLS, TARGET_COL
    import sys, os
    import pipeline

    toy = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5),
        "store_id":["S1"]*5, "sku_id":["SKU1"]*5,
        "base_price":[10]*5, "final_price":[10]*5,
        "competitor_price":[10]*5, "units_sold":[5]*5,
        "promo_flag":[0]*5, "promo_depth":[0.0]*5,
        "holiday_flag":[0]*5, "weather_index":[0.5]*5,
    })
    feats = generate_feature(toy)
    X, y = feats[FEATURE_COLS], toy["units_sold"]
    m = DummyRegressor(strategy="mean").fit(X, y)
    model_path = tmp_path / "m.pkl"
    with open(model_path,"wb") as f: pickle.dump(m,f)
    # monkeypatch official model
    pipeline.OFFICIAL_MODEL = model_path.name
    import shutil
    shutil.copy(model_path, tmp_path.parent / "models" / model_path.name)
