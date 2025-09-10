# Retail Pricing Demand Pipeline

This repository provides a workflow for **training demand models**, **simulating price plans**, and **optimizing prices** across SKUs and stores.  
It includes modules for data loading, feature engineering, modeling (XGBoost), and a greedy grid-search optimizer with configurable constraints.

---

## I. Assumptions

### 1. Modeling
- Price–demand relationship is always **negative** (higher price → lower demand).
- Current train/test split is simple chronological and may not be optimal.
- The model is prone to **overfitting**; improvements include stronger regularization, alternative learners, or feature selection.
- Only the **official model** (`forecaster_xgboost_20250909_221509.pkl`) is used.  
  It is **hard-coded** in both `optimizer.py` and `pipeline.py`.

### 2. Optimizer
- Each SKU × Store is capped at **X promo days per horizon** (configured in `constraints.yaml`).  
  This differs from assumptions in the provided dataset.
- Competitor price is assumed equal to base price.
- `holiday_flag` is set to average (from constraints).
- `weather_index` is set to average (from constraints).
- `stock_cap` defaults to **100** unless specified per SKU.
- Minimum price band is **0.6 × base_price**.

---

## II. How to Run the Pipeline

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

## III. How to Run Unit Tests

Unit tests live under `src/tests/`.

Run all tests:

```bash
pytest -q
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Skip slow/integration tests:

```bash
pytest -m "not slow and not integration"
```

---

## IV. Caveats

- Dataset assumptions (promo scheduling, stock caps) may not perfectly match real-world data.  
- The optimizer is greedy: it selects top-K promo days but does not guarantee a global optimum.  
- Constraints in `constraints.yaml` must reflect realistic business rules.  
- The official model filename is **hard-coded** — pipeline and optimizer will fail if it is missing.  
- Models assume clean, validated input; anomalies may require preprocessing.

---
