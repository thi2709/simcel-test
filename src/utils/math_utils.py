import numpy as np

def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (MAPE) in %."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def apply_stock_cap(units: float, sku_cfg: dict) -> float:
    cap = sku_cfg.get("stock_cap")
    if cap is None:
        return units
    try:
        return min(units, float(cap))
    except Exception:
        return units

def price_grid(base_price: float, lower: float, upper: float, steps: int) -> np.ndarray:
    lo = max(0.0, lower * base_price)
    hi = upper * base_price
    if hi <= lo:
        return np.array([base_price], dtype=float)
    return np.linspace(lo, hi, num=max(2, steps), dtype=float)