from core.runway import simulate_runway


def main():
    cases = [
        ("Baseline", {}),
        ("Costs reduced 10%", {"monthly_costs": 27_000}),
        ("Growth 3% to 5%", {"growth_rate": 0.05}),
    ]

    for name, settings in cases:
        result = simulate_runway(seed=42, **settings)
        survival = result["survival_probability"]
        median_cash = result["cash_percentiles"][1, -1]

        print(f"\n{name}")
        print(f"Survive 12 months: {survival[11]:.1%}")
        print(f"Survive 24 months: {survival[23]:.1%}")
        print(f"Median cash at month 24: {median_cash:,.0f}")


if __name__ == "__main__":
    main()
