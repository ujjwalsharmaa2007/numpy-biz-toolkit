import numpy as np

from core.forecasting import (
    moving_average_forecast,
    exponential_smoothing_forecast,
    weekday_forecast,
)


def backtest_forecasts(sales, dates, test_days=28):
    """Compare forecasts on a held-out period without using future data."""
    sales = np.asarray(sales, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[D]")

    if sales.ndim != 2 or 0 in sales.shape:
        raise ValueError("Sales must be a non-empty 2D array.")

    if not np.all(np.isfinite(sales)) or np.any(sales < 0):
        raise ValueError("Sales must be finite and non-negative.")

    if dates.ndim != 1 or len(dates) != sales.shape[1]:
        raise ValueError("Provide one date per sales column.")

    if np.isnat(dates).any():
        raise ValueError("Dates cannot contain missing values.")

    if np.any(np.diff(dates) != np.timedelta64(1, "D")):
        raise ValueError("Dates must be consecutive and increasing.")

    if not isinstance(test_days, int) or test_days < 1:
        raise ValueError("Test days must be a positive integer.")

    if sales.shape[1] < test_days + 56:
        raise ValueError("Need at least 56 training days plus the test period.")

    training = sales[:, :-test_days]
    training_dates = dates[:-test_days]
    actual = sales[:, -test_days:]

    # Shape: methods, products, forecast days.
    predictions = np.stack([
        moving_average_forecast(training, horizon=test_days),
        exponential_smoothing_forecast(training, horizon=test_days),
        weekday_forecast(training, training_dates, horizon=test_days),
    ])

    absolute_errors = np.abs(predictions - actual[np.newaxis, :, :])
    mae = absolute_errors.mean(axis=2)

    # WAPE is undefined when a product has zero total actual sales.
    actual_totals = actual.sum(axis=1)
    wape = np.full(mae.shape, np.nan)

    np.divide(
        absolute_errors.sum(axis=2),
        actual_totals[np.newaxis, :],
        out=wape,
        where=actual_totals[np.newaxis, :] > 0,
    )
    wape *= 100

    # For each product, lowest MAE also gives lowest WAPE when defined.
    best_method_indices = np.argmin(mae, axis=0)

    return {
        "method_names": np.array([
            "Moving average",
            "Smoothing",
            "Weekday",
        ]),
        "mae": mae,
        "wape": wape,
        "best_method_indices": best_method_indices,
    }
