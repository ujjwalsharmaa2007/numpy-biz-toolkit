import numpy as np


def recommend_orders(
    sales,
    forecast,
    current_stock,
    lead_days,
    on_order=None,
    review_days=7,
    z=1.65,
):
    """Suggest orders using forecast demand and historical variability.

    Assumes fixed lead times and independent daily demand.
    Stock and on-order arrays must follow the sales product order.
    """
    sales = np.asarray(sales, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if sales.ndim != 2 or sales.shape[0] < 1 or sales.shape[1] < 2:
        raise ValueError("Sales must contain products and at least two days.")

    if forecast.ndim != 2 or forecast.shape[0] != sales.shape[0]:
        raise ValueError("Forecast must have one row per product.")

    if forecast.shape[1] < 1:
        raise ValueError("Forecast must contain at least one day.")

    if (
        not np.all(np.isfinite(sales))
        or not np.all(np.isfinite(forecast))
        or np.any(sales < 0)
        or np.any(forecast < 0)
    ):
        raise ValueError("Sales and forecasts must be finite and non-negative.")

    count = sales.shape[0]

    def product_values(value, name):
        array = np.asarray(value, dtype=float)
        try:
            array = np.broadcast_to(array, (count,))
        except ValueError as error:
            raise ValueError(
                f"{name} must be a scalar or one value per product."
            ) from error

        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError(f"{name} must be finite and non-negative.")
        return array

    stock = product_values(current_stock, "Current stock")
    lead = product_values(lead_days, "Lead days")
    incoming = product_values(
        0 if on_order is None else on_order, "On-order stock"
    )

    if not np.isfinite(review_days) or review_days <= 0:
        raise ValueError("Review days must be positive and finite.")

    if not np.isfinite(z) or z < 0:
        raise ValueError("Z must be finite and non-negative.")

    daily_demand = forecast.mean(axis=1)

    # Estimate variability from up to 56 recent days.
    daily_std = sales[:, -56:].std(axis=1, ddof=1)

    safety_stock = z * daily_std * np.sqrt(lead)
    reorder_point = daily_demand * lead + safety_stock

    # Include incoming orders to avoid recommending duplicate purchases.
    stock_position = stock + incoming
    needs_order = (stock_position <= reorder_point) & (daily_demand > 0)

    # Cover lead time plus the interval until the next inventory review.
    coverage_days = lead + review_days
    target_stock = (
        daily_demand * coverage_days
        + z * daily_std * np.sqrt(coverage_days)
    )

    order_quantity = np.where(
        needs_order,
        np.ceil(np.maximum(0, target_stock - stock_position)),
        0,
    ).astype(int)

    days_left = np.full(count, np.inf)
    np.divide(stock, daily_demand, out=days_left, where=daily_demand > 0)

    # Smaller margin means stock is expected to run out sooner
    # relative to supplier lead time. Incoming arrival dates are unknown.
    urgency_margin = days_left - lead
    urgency_order = np.argsort(urgency_margin, kind="stable")

    return {
        "daily_demand": daily_demand,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "days_left": days_left,
        "needs_order": needs_order,
        "order_quantity": order_quantity,
        "urgency_order": urgency_order,
    }
