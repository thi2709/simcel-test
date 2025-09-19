# utils/constants.py

SEED = 42

OFFICIAL_MODEL = 'forecaster_xgboost_20250919_163246.pkl'

# Schema constants used by data_loader and others
DATE_COL = "date"
ID_COLS = ["store_id", "sku_id"]
PRICE_COL = "final_price"
FLOAT_COLS = ["base_price", "competitor_price", ]#"revenue", "margin"]
RATIO_COLS = ["promo_flag","promo_depth", "holiday_flag", "weather_index"]  # must be in [0, 1]
INT_COLS = ["units_sold", "stockout_flag"] #"week_of_year", "stock_cap"]
SET_COL = "set"

REQUIRED_COLUMNS = (
    ["date","store_id", "sku_id","final_price","base_price", "competitor_price","promo_flag", "holiday_flag","promo_depth", "weather_index","units_sold"]
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