import pandas as pd
from features import generate_feature

def test_feature_generation(toy_df):
    feats = generate_feature(toy_df)
    assert "sold_yesterday" in feats.columns
    assert "final_price_ln" in feats.columns
    assert (feats.select_dtypes("number").isna().sum() == 0).all()
