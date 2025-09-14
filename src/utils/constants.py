# utils/constants.py

SEED = 42

OFFICIAL_MODEL = 'forecaster_xgboost_20250914_223932.pkl'

# Schema constants used by data_loader and others
DATE_COL = "date"
ID_COLS = ["store_id", "sku_id"]
FLOAT_COLS = ["base_price", "final_price", "competitor_price", "revenue", "margin"]
FLAG_COLS = ["promo_flag", "holiday_flag", "stockout_flag"]
RATIO_COLS = ["promo_depth", "weather_index"]  # must be in [0, 1]
INT_COLS = ["week_of_year", "units_sold", "stock_cap"]
SET_COL = "set"

REQUIRED_COLUMNS = (
    [DATE_COL] + ID_COLS + FLOAT_COLS + FLAG_COLS + RATIO_COLS + INT_COLS + [SET_COL]
)

# Feature engineering constants
DAY_ORDER = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]

MONTH_ORDER = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]