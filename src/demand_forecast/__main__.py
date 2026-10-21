from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "daily_demand.csv"
ARTIFACTS = ROOT / "artifacts"
FEATURES = [
    "store", "product", "dow", "month", "day_of_year", "trend", "promotion",
    "price", "lag_1", "lag_7", "lag_14", "rolling_7", "rolling_28",
]


def generate_demand(seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=720, freq="D")
    records = []
    for store in range(3):
        for product in range(4):
            base = 34 + store * 9 + product * 7
            price_base = 16 + product * 4
            for index, date in enumerate(dates):
                promotion = int(rng.random() < (0.14 + 0.06 * (date.dayofweek >= 5)))
                price = price_base * (0.88 if promotion else 1.0) + rng.normal(0, 0.35)
                weekly = 8 * np.sin(2 * np.pi * date.dayofweek / 7 - 0.7)
                yearly = 11 * np.sin(2 * np.pi * date.dayofyear / 365 - 1.2)
                trend = index * (0.012 + product * 0.002)
                demand = base + weekly + yearly + trend + promotion * 19 - 1.3 * (price - price_base)
                units = max(0, int(round(demand + rng.normal(0, 5.5))))
                records.append((date, store, product, promotion, round(price, 2), units))
    return pd.DataFrame(records, columns=["date", "store", "product", "promotion", "price", "units"])


def make_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.sort_values(["store", "product", "date"]).copy()
    frame["dow"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["trend"] = (frame["date"] - frame["date"].min()).dt.days
    groups = frame.groupby(["store", "product"])["units"]
    for lag in (1, 7, 14):
        frame[f"lag_{lag}"] = groups.shift(lag)
    frame["rolling_7"] = groups.transform(lambda values: values.shift(1).rolling(7).mean())
    frame["rolling_28"] = groups.transform(lambda values: values.shift(1).rolling(28).mean())
    return frame.dropna().reset_index(drop=True)


def wmape(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / actual.abs().sum())


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw = generate_demand()
    raw.to_csv(DATA_PATH, index=False)
    data = make_features(raw)

    cutoffs = [pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-01"), pd.Timestamp("2025-07-01")]
    fold_rows = []
    last_scored = None
    for fold, cutoff in enumerate(cutoffs, start=1):
        train = data[data["date"] < cutoff]
        validation = data[(data["date"] >= cutoff) & (data["date"] < cutoff + pd.Timedelta(days=28))].copy()
        model = HistGradientBoostingRegressor(max_iter=170, learning_rate=0.065, max_leaf_nodes=24, l2_regularization=0.4, random_state=42)
        model.fit(train[FEATURES], train["units"])
        validation["forecast"] = np.clip(model.predict(validation[FEATURES]), 0, None)
        residuals = train["units"] - np.clip(model.predict(train[FEATURES]), 0, None)
        interval = float(np.quantile(np.abs(residuals), 0.95))
        validation["lower_90"] = np.clip(validation["forecast"] - interval, 0, None)
        validation["upper_90"] = validation["forecast"] + interval
        coverage = ((validation["units"] >= validation["lower_90"]) & (validation["units"] <= validation["upper_90"])).mean()
        fold_rows.append(
            {
                "fold": fold,
                "cutoff": cutoff.date().isoformat(),
                "wmape": round(wmape(validation["units"], validation["forecast"]), 4),
                "mae": round(mean_absolute_error(validation["units"], validation["forecast"]), 3),
                "interval_coverage": round(float(coverage), 4),
            }
        )
        last_scored = validation

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(ARTIFACTS / "backtest_folds.csv", index=False)
    assert last_scored is not None
    last_scored.to_csv(ARTIFACTS / "latest_forecast.csv", index=False)

    daily = last_scored.groupby("date", as_index=False).agg(
        actual=("units", "sum"), forecast=("forecast", "sum"), lower=("lower_90", "sum"), upper=("upper_90", "sum")
    )
    metrics = {
        "backtest_folds": len(folds),
        "mean_wmape": round(float(folds["wmape"].mean()), 4),
        "mean_mae_units": round(float(folds["mae"].mean()), 3),
        "mean_interval_coverage": round(float(folds["interval_coverage"].mean()), 4),
        "forecast_rows": len(last_scored),
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(daily["date"], daily["actual"], label="Actual", color="#111827", linewidth=1.6)
    axes[0].plot(daily["date"], daily["forecast"], label="Forecast", color="#2563eb", linewidth=1.8)
    axes[0].fill_between(daily["date"], daily["lower"], daily["upper"], color="#93c5fd", alpha=0.35, label="90% interval")
    axes[0].set(title="Latest 28-day backtest", ylabel="Units across stores")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False)
    axes[1].bar(folds["fold"].astype(str), folds["wmape"] * 100, color=["#0f766e", "#14b8a6", "#5eead4"])
    axes[1].set(title="Error is stable across time", xlabel="Rolling-origin fold", ylabel="WMAPE (%)")
    axes[1].set_ylim(0, max(20, folds["wmape"].max() * 130))
    fig.suptitle("Demand Forecast Command Center | Backtest", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "forecast_results.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

