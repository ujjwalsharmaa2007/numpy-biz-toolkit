import numpy as np

from core.backtesting import backtest_forecasts
from core.forecasting import (
    moving_average_forecast,
    exponential_smoothing_forecast,
    weekday_forecast,
)


def forecast_demand(sales, dates, horizon=7):
    """Select methods by backtesting, then forecast using all history."""
    results = backtest_forecasts(sales, dates)

    # Refit all methods using the full sales history.
    candidates = np.stack([
        moving_average_forecast(sales, horizon=horizon),
        exponential_smoothing_forecast(sales, horizon=horizon),
        weekday_forecast(sales, dates, horizon=horizon),
    ])

    # Select one method per product without looping over products.
    best = results["best_method_indices"]
    product_indices = np.arange(candidates.shape[1])
    forecast = candidates[best, product_indices, :]

    last_date = np.asarray(dates, dtype="datetime64[D]")[-1]
    forecast_dates = (
        last_date
        + np.arange(1, horizon + 1).astype("timedelta64[D]")
    )

    return {
        "forecast": forecast,
        "forecast_dates": forecast_dates,
        "selected_methods": results["method_names"][best],
        "backtest": results,
    }
