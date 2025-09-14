# schema.py
"""
Schema definition for forecasting model features and target.
"""

FEATURE_COLS = [
    "store_id", "sku_id", "base_price", "promo_flag", "promo_depth",
    "final_price", "competitor_price", "holiday_flag", "weather_index",
    "day_of_week", "month_of_year", "season",
    "sold_yesterday", "sold_last_week", "units_sold_ma7", "units_sold_ma30",
    "flag_promo_1stday", "final_price_ln", "competitor_price_diff",
]

TARGET_COL = "units_sold"