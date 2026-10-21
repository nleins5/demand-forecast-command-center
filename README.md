# Demand Forecast Command Center

Daily retail demand changes with season, weekday, price, and promotions. I wanted a forecast that can be judged on future weeks rather than a random train/test split.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![CI](https://github.com/nleins5/demand-forecast-command-center/actions/workflows/pipeline.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-0f766e)

The project generates demand for 12 store-product combinations and uses lagged sales, calendar features, price, and promotions to predict daily units. Three rolling backtests measure how the model performs as time moves forward.

![Forecast results](artifacts/forecast_results.png)

## Results on simulated data

The average WMAPE is 6.21% across three future windows. The intervals are intended to cover 90% of observations but covered 88.29% in these backtests, so I would recalibrate them before using them for stock decisions.

| Backtest result | Value |
|---|---:|
| Rolling-origin folds | 3 |
| Mean WMAPE | 6.21% |
| Mean absolute error | 4.65 units |
| Mean interval coverage | 88.29% |

The forecast file includes predictions and intervals for the latest 28-day window. All data is simulated; the result does not claim accuracy on a real retailer's sales.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
make install
make run
```

The generator includes trend, seasonality, promotions, and price response, with a fixed random seed so each run can be reproduced.

## What I would improve next

I would add holiday calendars, reconcile forecasts across store and product levels, and test interval coverage separately for high-volume and low-volume products.

