# tests/test_optimizer_stock_cap.py
import pandas as pd
from optimizer import optimize_price_plan
import pickle
from sklearn.dummy import DummyRegressor
from features import generate_feature
from modeling import FEATURE_COLS, TARGET_COL

def test_stock_cap_applied(tmp_path, toy_df, constraints_yaml):
    # add stock_cap=100
    import yaml
    with open(constraints_yaml) as f: cfg = yaml.safe_load(f)
    cfg["skus"]["SKU1"]["stock_cap"] = 100
    with open(constraints_yaml, "w") as f: yaml.safe_dump(cfg, f)

    # dummy model that predicts very high demand
    feats = generate_feature(toy_df)
    X, y = feats[FEATURE_COLS], toy_df[TARGET_COL]
    m = DummyRegressor(strategy="constant", constant=999).fit(X, y)
    model_path = tmp_path / "m.pkl"
    with open(model_path, "wb") as f: pickle.dump(m, f)

    plan = optimize_price_plan(
        model_ref=str(model_path),
        start_date="2025-09-10",
        horizon_days=2,
        constraints_path=constraints_yaml,
        price_grid_steps=3,
        export=False
    )
    assert (plan["pred_units"] <= 100).all()
