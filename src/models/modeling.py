# modeling.py
"""
Model training and validation for demand forecasting.
- Uses XGBoost regressor with monotone constraints.
- Ensures schema correctness via utils/schema and utils/math_utils.
- Saves trained model pickle in the same directory.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from utils.schema import FEATURE_COLS, TARGET_COL
from utils.constants import SEED
from utils.math_utils import mape

# Directory of this script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------ Monotone constraints ------------------
# Ensure final_price and final_price_ln have negative monotonic effect
_MONOTONE = []
for c in FEATURE_COLS:
    if c in {"final_price", "final_price_ln"}:
        _MONOTONE.append(-1)
    else:
        _MONOTONE.append(0)
_MONO_TUPLE = "(" + ",".join(str(v) for v in _MONOTONE) + ")"


# ------------------ Input checks ------------------
def _check_schema(X: pd.DataFrame, y: pd.Series):
    """Ensure X has all required features and y matches the expected target."""
    missing_feats = [col for col in FEATURE_COLS if col not in X.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns: {missing_feats}")

    if getattr(y, "name", None) != TARGET_COL:
        raise ValueError(
            f"Target column must be '{TARGET_COL}', but got '{y.name}'."
        )


def _check_inputs(X: pd.DataFrame, y: pd.Series):
    """Type, shape, and NA checks before training/validation."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if X.isnull().any().any():
        raise ValueError("X contains NaN values. Handle missing data before training.")

    if X.select_dtypes(include=["object"]).shape[1] > 0:
        bad = X.select_dtypes(include=["object"]).columns.tolist()
        raise ValueError(
            f"Object dtype columns in X: {bad}. Encode them or convert to category."
        )

    if pd.isnull(y).any():
        raise ValueError("y contains NaN values.")
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}.")

    return X, y


# ------------------ Training ------------------
def fit(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = SEED,
    save_model: bool = True,
) -> str | XGBRegressor:
    """
    Train an XGBoost regressor. 
    
    If save_model=True (default):
        - Save pickle in same dir as modeling.py.
        - Return the saved filename (str).
    
    If save_model=False:
        - Do not save to disk.
        - Return the trained model object directly.
    """
    _check_schema(X, y)
    X, y = _check_inputs(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X[FEATURE_COLS], y, test_size=test_size, random_state=random_state, shuffle=True
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bynode=0.8,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.2,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=32,
        eval_metric="rmse",
        early_stopping_rounds=200,
        verbosity=0,
        seed=SEED,
        monotone_constraints=_MONO_TUPLE,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
    )

    if save_model:
        filename = f"forecaster_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        save_path = os.path.join(_THIS_DIR, filename)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        return filename
    else:
        return model


# ------------------ Validation ------------------
def validate(model_filename: str, X: pd.DataFrame, y: pd.Series):
    """
    Load model pickle and evaluate on given X, y.
    Returns dict with r2, rmse, mape_pct.
    """
    model_path = os.path.join(_THIS_DIR, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _check_schema(X, y)
    X, y = _check_inputs(X, y)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X[FEATURE_COLS])

    r2 = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mape_val = mape(y, y_pred)

    print(f"Validation metrics for {model_filename}:")
    print(f"  R^2  : {r2:.4f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAPE : {mape_val:.2f}%")

    return {"r2": r2, "rmse": rmse, "mape_pct": mape_val}
