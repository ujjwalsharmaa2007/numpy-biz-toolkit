import csv
from pathlib import Path

import numpy as np


def generate_sales(seed=42):
    """Generate one year of fake daily sales for 20 products."""
    rng = np.random.default_rng(seed)

    products = np.array([
        "Coffee", "Tea", "Cookies", "Milk", "Bread",
        "Sugar", "Rice", "Pasta", "Juice", "Water",
        "Chocolate", "Chips", "Biscuits", "Butter", "Cheese",
        "Eggs", "Soap", "Shampoo", "Toothpaste", "Cereal",
    ])

    dates = np.arange(
        "2026-01-01",
        "2027-01-01",
        dtype="datetime64[D]",
    )

    # Each product has a different average number of daily sales.
    average_sales = np.array([
        20, 10, 5, 18, 16,
        8, 12, 9, 14, 25,
        11, 15, 13, 7, 6,
        17, 8, 5, 6, 9,
    ])

    # One row per product, one column per day.
    noise = rng.normal(0, 2, size=(len(products), len(dates)))
    sales = average_sales[:, np.newaxis] + noise

    # Monday = 0, ..., Saturday = 5, Sunday = 6.
    weekday = (dates.astype(int) + 3) % 7

    # Increase sales by 30% on weekends.
    weekend_multiplier = np.where(weekday >= 5, 1.3, 1.0)
    sales = sales * weekend_multiplier

    # Gradually increase sales by 20% over the year.
    trend = np.linspace(1.0, 1.2, len(dates))
    sales = sales * trend

    # Add two temporary seasonal bumps in demand.
    day_index = np.arange(len(dates))
    summer_bump = 0.15 * np.exp(-0.5 * ((day_index - 180) / 20) ** 2)
    winter_bump = 0.25 * np.exp(-0.5 * ((day_index - 345) / 15) ** 2)
    seasonal_multiplier = 1 + summer_bump + winter_bump
    sales = sales * seasonal_multiplier

    # Sales must be non-negative whole numbers.
    sales = np.maximum(0, np.round(sales)).astype(int)

    # Simulate occasional stockouts: roughly 1% of product-days.
    stockout_mask = rng.random(sales.shape) < 0.01
    sales[stockout_mask] = 0

    return products, dates, sales, stockout_mask


def save_csv(products, dates, sales):
    """Save the generated sales in the format used by the inventory tool."""
    # Find the repository root, even if the script is run elsewhere.
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "sample_sales.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["date", "product", "quantity_sold"])

        for product_index in range(len(products)):
            for day_index in range(len(dates)):
                writer.writerow([
                    dates[day_index],
                    products[product_index],
                    sales[product_index, day_index],
                ])

    return output_path


if __name__ == "__main__":
    products, dates, sales, stockout_mask = generate_sales()
    output_path = save_csv(products, dates, sales)

    print("Fake sales data generated successfully!")
    print("Sales array shape:", sales.shape)
    print("Number of products:", len(products))
    print("Number of days:", len(dates))
    print("Date range:", dates[0], "to", dates[-1])
    print("Simulated stockout days:", np.sum(stockout_mask))

    print("\nFirst 7 days of sales for the first 3 products:")
    print(sales[:3, :7])

    print("\nSaved data to:", output_path)