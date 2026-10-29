import argparse
import csv
from pathlib import Path

import numpy as np

from common.data_loader import load_sales_csv
from core.runway import simulate_runway
from core.demand import forecast_demand
from core.inventory import recommend_orders


ROOT = Path(__file__).resolve().parent


def runway_report(args):
    result = simulate_runway(
        starting_cash=args.cash,
        monthly_revenue=args.revenue,
        monthly_costs=args.costs,
        months=args.months,
        scenarios=args.scenarios,
        seed=args.seed,
    )

    print("\nSTARTUP RUNWAY REPORT")
    print("All amounts use the same currency.")
    print("Default monthly assumptions: growth 3%, churn 1%, cost growth 1%.")
    print("Growth standard deviation: 2%; churn standard deviation: 0.5%.")
    print("\nMonth    Survival    Median cash")

    for index in range(args.months):
        survival = result["survival_probability"][index]
        median = result["cash_percentiles"][1, index]
        print(f"{index + 1:>5} {survival:>11.1%} {median:>14,.0f}")

    print("\nFailure means cash below zero at month-end.")
    print("Cash paths after failure assume operations continue.")
    print("Results depend on the assumptions entered.")


def load_settings(path, products):
    settings = {}

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"product", "current_stock", "lead_days", "on_order"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Settings need: product,current_stock,lead_days,on_order"
            )

        for line, row in enumerate(reader, start=2):
            product = (row.get("product") or "").strip()

            if not product or product in settings:
                raise ValueError(
                    f"Settings line {line}: empty or duplicate product."
                )

            try:
                values = np.array([
                    float(row["current_stock"]),
                    float(row["lead_days"]),
                    float(row["on_order"]),
                ])
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Settings line {line}: invalid number."
                ) from error

            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(
                    f"Settings line {line}: values must be finite "
                    "and non-negative."
                )

            settings[product] = values

    missing = set(products) - set(settings)
    extra = set(settings) - set(products)

    if missing or extra:
        raise ValueError(
            f"Settings products do not match sales. "
            f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
        )

    # Align settings with the product order returned by the sales loader.
    aligned = np.array([settings[product] for product in products])
    return aligned[:, 0], aligned[:, 1], aligned[:, 2]


def inventory_report(args):
    products, dates, sales = load_sales_csv(args.sales)
    stock, lead, incoming = load_settings(args.settings, products)

    demand = forecast_demand(sales, dates)
    orders = recommend_orders(
        sales,
        demand["forecast"],
        stock,
        lead,
        on_order=incoming,
    )

    print("\nINVENTORY REPORT")
    print("Settings file:", args.settings)
    print("Forecast:", demand["forecast_dates"][0],
          "to", demand["forecast_dates"][-1])
    print("\nProduct        Method             7-day sales  Order units")
    print("-" * 61)

    for index in orders["urgency_order"]:
        total = demand["forecast"][index].sum()
        method = demand["selected_methods"][index]
        quantity = orders["order_quantity"][index]

        print(
            f"{products[index]:<15}"
            f"{method:<19}"
            f"{total:>11.1f}"
            f"{quantity:>13}"
        )

    print("\nORDER TODAY")
    found = False

    for index in orders["urgency_order"]:
        if orders["order_quantity"][index] > 0:
            found = True
            print(
                f"{products[index]}: "
                f"{orders['order_quantity'][index]} units "
                f"(stock left: {orders['days_left'][index]:.1f} days, "
                f"lead time: {lead[index]:g} days)"
            )

    if not found:
        print("No orders suggested.")

    print("\nSales forecasts and safety stock are estimates.")
    print("Incoming stock is included; its arrival dates are not modeled.")


def main():
    parser = argparse.ArgumentParser(
        description="NumPy Business Toolkit"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    runway = commands.add_parser("runway", help="Simulate startup cash")
    runway.add_argument("--cash", type=float, default=100_000)
    runway.add_argument("--revenue", type=float, default=20_000)
    runway.add_argument("--costs", type=float, default=30_000)
    runway.add_argument("--months", type=int, default=24)
    runway.add_argument("--scenarios", type=int, default=10_000)
    runway.add_argument("--seed", type=int, default=42)
    runway.set_defaults(handler=runway_report)

    inventory = commands.add_parser(
        "inventory", help="Forecast sales and suggest orders"
    )
    inventory.add_argument(
        "--sales", type=Path,
        default=ROOT / "data" / "sample_sales.csv",
    )
    inventory.add_argument(
        "--settings", type=Path,
        default=ROOT / "data" / "demo_inventory_settings.csv",
    )
    inventory.set_defaults(handler=inventory_report)

    args = parser.parse_args()

    try:
        args.handler(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
