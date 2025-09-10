# Retail Pricing Demand Pipeline

This repository provides a pipeline for **training forecasters**, **simulating sales performance**, and **optimizing prices** across SKUs and stores.  
It includes modules for data loading, feature engineering, modeling (XGBoost), and a price optimizer using naive greedy grid-search approach and pre-set configurable constraints.
**EDA analysis** can be found inside notebook `reports.ipynb`.
---

## I. Assumptions & Caveats

### 1. Modeling
- Current train/test split cannot model Oct,Nov,Dec seasonality which lower test performance.
- XGBoost is selected for the interest of time, after training it is found that this is not the optimal solution and is subjected to further improvements.
- The model performance is currently **overfitting**; it is subject to improvements by including stronger regularization (less dependent on seasonality), more feature engineering (MA price, unit cost, percentage margin, etc.), try alternative learners, train separate forecaster for each SKU, and feature selection.
- Only the **official model** (`forecaster_xgboost_20250909_221509.pkl`) is used.  
  It is **hard-coded** in both `optimizer.py` and `pipeline.py`.
- Price–demand relationship is always **negative** (higher price → lower demand).

### 2. Optimizer
- Each SKU × Store is capped at **X promo days per horizon** (configured in `constraints.yaml`).  
  This differs from assumptions in the provided dataset.
- Competitor price is assumed equal to base price.
- `holiday_flag` is set to average (from constraints).
- `weather_index` is set to average (from constraints).
- `stock_cap` defaults to **100** unless specified per SKU.
- Minimum price band is **0.6 × base_price**.
- The optimizer is greedy: it selects top-K promo days with discrete discount but does not guarantee a global optimum.
- Dataset assumptions (promo scheduling, stock caps) may not perfectly match real-world data. 

---

## II. Pipeline

> Run from the `src/` folder with `python pipeline.py …`, or from repo root with `python -m src.pipeline …`.

### 1. Training
Train a demand model on a date range and validate on a holdout set:

```bash
python pipeline.py train   --train-start 2024-01-01   --train-end   2024-09-30   --test-start  2024-10-01   --test-end    2024-12-31
```

- All model versions are saved under `src/models/`.  
- **Only the official version is used** across simulation and optimization.

---

### 2. Simulation
Simulate the performance of a price plan over a horizon:

```bash
python pipeline.py simulate   --horizon 14   --price-plan pricing/price_plan_20250909_14days_20250909_190508.csv   --out simulation_result.csv
```

- Outputs predicted units, revenue, and margin.

---

### 3. Optimizer
Generate an optimized price plan for a horizon:

```bash
python pipeline.py optimize   --horizon 30   --objective revenue   --start-date 2024-03-01
```

- Constraints are read from `pricing/constraints.yaml`.  
- If `--start-date` is omitted, defaults to today.

---

## III. Unit Tests

- Unit tests live under `src/tests/`.
- To develop more advanced unit tests, can try to use the small csv file sample

Run all tests:

```bash
pytest -q
```


---