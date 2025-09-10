You'll find here our [raw dataset](https://github.com/SIMCEL/ai-ml-challenge/blob/main/simcel-6pk70-1jk5iqdp-train_v9rqX0R.csv): sales data of a well known convenience store brand, listing commodities across several locations in many cities. Additionally, certain characteristics of each product and retailer have been determined.

The goal of the project is to build a web app that will allow you to showcase the results of data analysis of this raw data, then also allow users to explore stock data of related companies.

You don't need to complete all objectives, but the more objectives you can finish, the more points that you will score.

The idea is to build this app using technology that will be close to what we are using at CEL for data engineering and dev: Pandas, Streamlit and FastAPI.

## Objectives
1. Conduct initial data exploration on the dataset and show your results on the app. Please spend time sparsely on this, the goal is to get some basic meaningful initial insights on the data in a reasonable amount of time.
2. Conduct data pre-processing: ensure high-quality by cleaning dirty data, imputing missing values, and handling outliers. Show the impact of the data cleaning on the app.
3. Conduct minimal data exploration analysis, i.e try to understand some of the variable relationships in the dataset
4. Then your task is to create a stock charts for any ticker the user types in. You can use a library like yfinance to get stock data, and you should then build a UI that allows users to enter in company names or ticker symbols, select the stock they want to view, and then a chart will be presented to them where they can modify the time intervals.
5. Implement **Explainable AI & Automated Report Generation**: Leverage LLMs (e.g., GPT-4) for narrative report generation:
  * Summarize campaign effectiveness in plain language.
  * Automatically create visual explanations (bar, line, or waterfall charts).
  * Generate ready-to-use executive summaries with actionable recommendations.

---

## Common Deliverables (all tracks)
- `src/ai/` module with clean interfaces.
- `tests/` covering data prep, core logic, and API contracts (≥ 80% coverage).
- `Dockerfile` that runs the service and a `docker-compose.yml` if you rely on extras.
- `Makefile` (or `tox`/`nox`) with: `make setup`, `make test`, `make lint`, `make serve-*`.
- `MODEL_CARD.md` describing data, assumptions, intended use, risks/limits.
- `README_AI.md` with quickstart + benchmark table.

## Requirements

1. All data (both from our raw data and yfinance results) must be exposed through FastAPI APIs. The Streamlit code should, as much as possible, be used to generate UIs, not process any data. I.e keep a traditionnal front/back separation of concerns.
2. The project must be run using docker-compose (both streamlit / fastAPI server must be docker-compose services). Please also add a minimal readme explaining any actions necessary to run the project locally. If you are un-experience with docker-compose, using only one docker image is also possible but discouraged.
3. You can use any charting libraries outsides of what streamlit offers if necessary
4. Please try to enforce best practices when implementing APIs, and make the code as generic, maintainable and extensible as possible
5. Please remain conscious of time: overall allocation of effort should be 30% on the data analysis/cleaning and 70% on building the app to visualise results and explore stock data.

## How to submit
Please upload the code for this project to GitHub, and post a link to your repository below.
