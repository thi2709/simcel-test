import pandas as pd
import pytest
from data_loader import load_data, DataLoaderError

def test_load_valid_csv(tmp_csv):
    df = load_data(str(tmp_csv))
    assert not df.empty
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

def test_invalid_row_drops(tmp_path, toy_df):
    toy_df.loc[0, "promo_flag"] = 2  # invalid flag
    p = tmp_path / "bad.csv"
    toy_df.to_csv(p, index=False)
    with pytest.raises(DataLoaderError):
        load_data(str(p))
