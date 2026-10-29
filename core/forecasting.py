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
