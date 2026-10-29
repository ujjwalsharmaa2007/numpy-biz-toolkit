import numpy as np


def moving_average_forecast(sales, window=28, horizon=7):
    """Forecast daily sales using the most recent window's average.

    Input shape:  (products, historical days)
    Output shape: (products, forecast days)
    """
    sales = np.asarray(sales, dtype=float)

    if sales.ndim != 2 or 0 in sales.shape:
        raise ValueError("Sales must be a non-empty 2D array.")

    if not np.all(np.isfinite(sales)) or np.any(sales < 0):
        raise ValueError("Sales must be finite and non-negative.")

    if not isinstance(window, int) or not 1 <= window <= sales.shape[1]:
        raise ValueError("Window must be between 1 and the number of days.")

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")

    # Average across days, keeping one row per product.
    daily_average = sales[:, -window:].mean(axis=1, keepdims=True)

    # Repeat each product's daily average across the forecast horizon.
    return np.repeat(daily_average, horizon, axis=1)


def exponential_smoothing_forecast(sales, alpha=0.3, horizon=7):
    """Forecast using a smoothed level for each product."""
    sales = np.asarray(sales, dtype=float)

    if sales.ndim != 2 or 0 in sales.shape:
        raise ValueError("Sales must be a non-empty 2D array.")

    if not np.all(np.isfinite(sales)) or np.any(sales < 0):
        raise ValueError("Sales must be finite and non-negative.")

    if not np.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("Alpha must be greater than 0 and at most 1.")

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")

    # Begin with each product's first recorded sales value.
    level = sales[:, 0].copy()

    # Loop over days, while updating every product together.
    for day in range(1, sales.shape[1]):
        level = alpha * sales[:, day] + (1 - alpha) * level

    return np.repeat(level[:, np.newaxis], horizon, axis=1)


def weekday_forecast(sales, dates, weeks=8, horizon=7):
    """Forecast using each weekday's average over recent complete weeks."""
    sales = np.asarray(sales, dtype=float)
    dates = np.asarray(dates, dtype="datetime64[D]")

    if sales.ndim != 2 or 0 in sales.shape:
        raise ValueError("Sales must be a non-empty 2D array.")

    if not np.all(np.isfinite(sales)) or np.any(sales < 0):
        raise ValueError("Sales must be finite and non-negative.")

    if dates.ndim != 1 or len(dates) != sales.shape[1]:
        raise ValueError("Provide one date for every sales column.")

    if np.isnat(dates).any():
        raise ValueError("Dates cannot contain missing values.")

    if np.any(np.diff(dates) != np.timedelta64(1, "D")):
        raise ValueError("Dates must be consecutive and increasing.")

    if not isinstance(weeks, int) or weeks < 1:
        raise ValueError("Weeks must be a positive integer.")

    if sales.shape[1] < weeks * 7:
        raise ValueError("Not enough history for the requested weeks.")

    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Horizon must be a positive integer.")

    # Group recent history into weeks, with seven positions per week.
    recent = sales[:, -weeks * 7:]
    weekly_blocks = recent.reshape(sales.shape[0], weeks, 7)

    # Average matching weekdays across the historical weeks.
    weekday_averages = weekly_blocks.mean(axis=1)

    # After complete weeks, the next day matches position zero.
    positions = np.arange(horizon) % 7

    return weekday_averages[:, positions]
