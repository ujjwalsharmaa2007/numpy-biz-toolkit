import numpy as np


def simulate_runway(
    starting_cash=100_000,
    monthly_revenue=20_000,
    monthly_costs=30_000,
    growth_rate=0.03,
    growth_std=0.02,
    churn_rate=0.01,
    churn_std=0.005,
    cost_growth=0.01,
    months=24,
    scenarios=10_000,
    seed=42,
):
    """Return simulated cash balances and survival probabilities."""
    if months < 1 or scenarios < 1:
        raise ValueError("Months and scenarios must be positive integers.")

    if min(starting_cash, monthly_revenue, monthly_costs) < 0:
        raise ValueError("Cash, revenue, and costs cannot be negative.")

    if growth_std < 0 or churn_std < 0:
        raise ValueError("Standard deviations cannot be negative.")

    if not 0 <= churn_rate <= 1:
        raise ValueError("Churn rate must be between 0 and 1.")

    if growth_rate < -1 or cost_growth < -1:
        raise ValueError("Growth rates cannot be below -100%.")

    rng = np.random.default_rng(seed)

    # Each row is a scenario; each column is a month.
    shape = (scenarios, months)

    growth = rng.normal(growth_rate, growth_std, size=shape)
    churn = rng.normal(churn_rate, churn_std, size=shape)

    growth = np.maximum(growth, -1)
    churn = np.clip(churn, 0, 1)

    # Month 1 applies growth and churn to current revenue.
    revenue_factors = (1 + growth) * (1 - churn)
    revenue = monthly_revenue * np.cumprod(
        revenue_factors, axis=1
    )

    # Month 1 applies cost growth to current costs.
    month_numbers = np.arange(1, months + 1)
    costs = monthly_costs * (1 + cost_growth) ** month_numbers

    monthly_profit = revenue - costs
    cash = starting_cash + np.cumsum(monthly_profit, axis=1)

    # Count a scenario as failed after its first negative balance.
    failed = np.logical_or.accumulate(cash < 0, axis=1)
    survival_probability = np.mean(~failed, axis=0)

    # A value of months + 1 means no failure within the horizon.
    first_failure_month = np.where(
        failed.any(axis=1),
        failed.argmax(axis=1) + 1,
        months + 1,
    )

    return {
        "cash": cash,
        "survival_probability": survival_probability,
        "cash_percentiles": np.percentile(
            cash, [10, 50, 90], axis=0
        ),
        "first_failure_month": first_failure_month,
    }


def main():
    results = simulate_runway()

    print("STARTUP RUNWAY SIMULATOR")
    print("Starting cash: 100,000")
    print("Current monthly revenue: 20,000")
    print("Current monthly costs: 30,000")
    print("Scenarios: 10,000 | Months: 24")

    print("\nSurvival probabilities:")
    for month in [6, 12, 18, 24]:
        probability = results["survival_probability"][month - 1]
        print(f"Through month {month}: {probability:.1%}")

    p10, median, p90 = results["cash_percentiles"][:, -1]

    print("\nCash balance at month 24:")
    print(f"10th percentile: {p10:,.0f}")
    print(f"Median:          {median:,.0f}")
    print(f"90th percentile: {p90:,.0f}")

    print("\nAll amounts use the same currency.")
    print("Failure means cash falls below zero at month-end.")
    print("Cash paths after failure assume operations continue.")


if __name__ == "__main__":
    main()