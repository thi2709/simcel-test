# src/tests/test_modeling_schema.py
import pandas as pd
import pytest
from modeling import FEATURE_COLS, fit

def test_target_name_must_be_units_sold(toy_df):
    from features import generate_feature
    feats = generate_feature(toy_df)
    X = feats[FEATURE_COLS]
    y = toy_df["units_sold"].rename("wrong")
    with pytest.raises(ValueError):
        fit(X, y)
