import pandas as pd
import numpy as np
import os
from modeling import fit, validate, FEATURE_COLS, TARGET_COL

def test_fit_and_validate(tmp_path, toy_df):
    from features import generate_feature
    feats = generate_feature(toy_df)
    X = feats[FEATURE_COLS]
    y = toy_df[TARGET_COL]
    fname = fit(X, y, test_size=0.25, random_state=0)
    assert fname.endswith(".pkl")
    metrics = validate(fname, X, y)
    assert all(k in metrics for k in ["r2", "rmse", "mape_pct"])
