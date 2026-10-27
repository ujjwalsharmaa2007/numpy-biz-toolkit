import numpy as np


def simulate_runway(
    starting_cash,
    monthly_revenue,
    monthly_costs,
    growth_rate=0.03,
    growth_std=0.02,
    churn_rate=0.01,
    churn_std=0.005,
    cost_growth=0.01,
    months=24,
    scenarios=10_000,
    seed=42,
):
    """
    Simulate possible startup cash balances.

    Rates are monthly decimals: 0.03 means 3%.
    Cash below zero means the startup has run out of money.
    """
    if starting_cash < 0 or monthly_revenue < 0 or monthly_costs < 0:
        raise ValueError("Cash, revenue, and costs cannot be negative.")
    if growth_std < 0 or churn_std < 0:
        raise ValueError("Standard deviations cannot be negative.")
    if months < 1 or scenarios < 1:
        raise ValueError("Months and scenarios must be positive.")

    rng = np.random.default_rng(seed)

    # Each row is one scenario; each column is one month.
    growth = rng.normal(growth_rate, growth_std, size=(scenarios, months))
    churn = rng.normal(churn_rate, churn_std, size=(scenarios, months))

    # Keep churn between 0% and 100%.
    churn = np.clip(churn, 0, 1)

    # Revenue cannot become negative.
    revenue_factors = np.maximum(0, (1 + growth) * (1 - churn))
    revenue = monthly_revenue * np.cumprod(revenue_factors, axis=1)

    # Cost growth applies from the first simulated month.
    month_numbers = np.arange(1, months + 1)
    costs = monthly_costs * (1 + cost_growth) ** month_numbers

    # Broadcasting subtracts the same monthly costs from every scenario.
    cash = starting_cash + np.cumsum(revenue - costs, axis=1)

    # A scenario has survived only if it has never gone below zero.
    has_run_out = np.minimum.accumulate(cash, axis=1) < 0
    survival_probability = np.mean(~has_run_out, axis=0)

    # For scenarios that never run out, use months + 1 as a marker.
    ran_out = has_run_out.any(axis=1)
    first_run_out = np.where(
        ran_out,
        np.argmax(has_run_out, axis=1) + 1,
        months + 1,
    )

    return {
        "cash": cash,
        "survival_probability": survival_probability,
        "cash_percentiles": np.percentile(cash, [10, 50, 90], axis=0),
        "first_run_out": first_run_out,
    }


if __name__ == "__main__":
    results = simulate_runway(
        starting_cash=100_000,
        monthly_revenue=20_000,
        monthly_costs=30_000,
    )

    print("STARTUP RUNWAY SIMULATION")
    print("Scenarios: 10,000")
    print("Starting cash: ₹100,000")
    print()

    for month in [6, 12, 18, 24]:
        survival = results["survival_probability"][month - 1]
        print(f"Chance of surviving through month {month}: {survival:.1%}")

    median_cash = results["cash_percentiles"][1, -1]
    print(f"\nMedian cash balance in month 24: ₹{median_cash:,.0f}")
    print("\nResults are estimates based on the assumptions provided.")