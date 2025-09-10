# Senior AI/ML Engineer – Dynamic Forecasting & Pricing Challenge

## Scenario

You work for a multi-store retailer (5 stores, 3 SKUs). Leadership wants a demand forecasting system that also optimizes prices to move key business metrics (revenue, units, margin). Your task is to:
1. forecast demand, and
2. build a pricing pipeline that simulates/optimizes price decisions and shows expected impact on KPIs.

## Data (provided)

### Download:
* Full dataset (CSV) — 5,475 rows (daily, 2023, 5 stores × 3 SKUs): [Link](https://github.com/SIMCEL/ai-ml-challenge/blob/main/retail_pricing_demand_2024.csv)
* Small sample (CSV): [Link](https://github.com/SIMCEL/ai-ml-challenge/blob/main/retail_pricing_demand_2024_sample.csv)

### Columns
* date (YYYY-MM-DD), store_id, sku_id
* base_price, promo_flag (0/1), promo_depth (0–0.5), final_price (=base_price×(1−promo_depth))
* competitor_price, holiday_flag (0/1), weather_index (0–1), week_of_year
* Targets/derived: units_sold (demand), revenue (=units_sold×final_price), margin (units×(final_price−cost proxy; cost not supplied—see below)), stock_cap, stockout_flag
* Split hint: set ∈ {train, test} with suggested split at 2023-10-01

Note: true unit costs aren’t included in the file. For margin modeling, assume a per-SKU cost parameter that you can estimate (e.g., via calibration or a config file).

## Candidate Tasks

__Please note that you're not required to complete all tasks.__

1) Exploratory analysis (quick but thoughtful)
 * Identify main demand drivers (price, promo, holiday, weather, weekly/annual patterns, store effects).
 * Check price elasticity by SKU and promo effect sizes.
 * Detect stockouts and discuss their impact on demand estimation.

2) Baseline forecasting model
 * Build a baseline daily demand forecast (units_sold) per (store, SKU).
 * You may use classical (SARIMAX/ETS), gradient boosting, or deep learning—your choice.
 * Handle seasonality, trend, holidays, weather, price & promo features.
 * Evaluate on the test window with at least: RMSE and MAPE.

3) Pricing impact simulation pipeline

Create a pipeline that, given candidate prices for a future horizon, returns:
 * Expected demand (using your demand model)
 * Revenue, units, and margin projections
 * Constraint handling:
    * price bounds per SKU (e.g., 0.7×base_price ≤ price ≤ 1.2×base_price)
    * optional inventory/stock caps (if provided)
    * promo scheduling rules (e.g., max X promo days per month)

4) Optimizer (objective-driven)

Implement one of:
* Maximize revenue subject to price bounds & promo constraints, OR
* Hit a unit target with minimal margin sacrifice, OR
* Multi-objective (e.g., weighted revenue + units, or revenue with min-units constraint)

You may use grid/greedy search, Bayesian optimization, gradient-free methods, or MILP with piecewise demand curves—your call. Show the optimized price plan and expected KPI lift vs. a baseline (e.g., current final_price).

5) Robustness & causality awareness (brief)
* Discuss endogeneity (price ↔ demand), promos, competitor price, stockouts.
* Show at least one mitigation (e.g., feature lagging, IV-style ideas, or guardrails).
* Include backtests or scenario stress tests (e.g., competitor undercuts by 5%).

Deliverables
* Repo with code (clear structure):
    * notebooks/ or reports/ with EDA
    * src/ with:
        * data_loader.py
        * features.py (lags, rolling stats, seasonality, holiday, price transforms)
        * models/ (your forecaster)
        * pricing/optimizer.py (objective + constraints)
        * pipeline.py (end-to-end CLI)
        * tests/ with a few unit tests (e.g., feature shapes, monotonic demand wrt price)
        * README.md with instructions & assumptions
* CLI examples:
    * Train & evaluate:
  ```
  python pipeline.py train --train-start 2023-01-01 --train-end 2023-09-30 --test-start 2023-10-01 --test-end 2023-12-31
  ```
    * Simulate a price plan for next 14 days:
  ```
  python pipeline.py simulate --horizon 14 --price-plan path/to/price_plan.csv --out results.csv
  ```
    * Optimize prices for a horizon:
  ```
  python pipeline.py optimize --horizon 14 --objective revenue --constraints path/to/constraints.yaml --out plan.csv
  ```

## Evaluation Rubric (100 pts)
* Modeling quality (30): sensible features, treatment of seasonality/holidays, handling of price/promo; accuracy on test.
* Pricing pipeline (30): clean API, constraint handling, correct KPI math, readable outputs/plots.
* Optimization (20): objective formulation, search method, stability; demonstrates KPI improvement.
* Software craft (15): structure, typing/docs, tests, reproducibility (seeds, env).
* Communication (5): clear README, assumptions, and caveats.

__Bonus (optional):__
* MLflow tracking, Dockerfile, Makefile, or a lightweight Streamlit/Gradio UI to visualize forecasts and pricing scenarios.
* Experiment with hierarchical forecasting (store→SKU aggregation constraints) or global models across SKUs/stores.
* Simple demand elasticity report with confidence intervals.

## Hints & Guardrails for Candidates
* Treat final_price as the decision variable for future periods; keep it within guardrails.
* Add lagged features (e.g., 7/14/28-day), moving averages, and promo/holiday proximity indicators.
* Prevent data leakage (train only on past).
* Explain any imputation and how you handle stockouts (e.g., censoring).
* Show at least one what-if scenario and the optimized plan vs. baseline.
