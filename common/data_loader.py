import csv
from datetime import date
from pathlib import Path

import numpy as np


def load_sales_csv(path):
    """Return products, dates, and sales shaped (products, days).

    Duplicate product/date entries are added together.
    Missing product/date combinations are rejected, not assumed to be zero.
    """
    records = []

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"date", "product", "quantity_sold"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain: date, product, quantity_sold"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                day = date.fromisoformat(row["date"].strip())
                product = row["product"].strip()
                quantity = float(row["quantity_sold"])

                if not product:
                    raise ValueError("Product name is empty.")

                if not np.isfinite(quantity) or quantity < 0:
                    raise ValueError("Quantity must be finite and non-negative.")

                if not quantity.is_integer():
                    raise ValueError("Quantity must be a whole number.")

            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(
                    f"Invalid data on CSV line {line_number}: {error}"
                ) from error

            records.append((day.isoformat(), product, quantity))

    if not records:
        raise ValueError("CSV contains no sales records.")

    record_dates = np.array(
        [record[0] for record in records], dtype="datetime64[D]"
    )
    record_products = np.array([record[1] for record in records])
    quantities = np.array([record[2] for record in records], dtype=float)

    products, product_indices = np.unique(
        record_products, return_inverse=True
    )

    dates = np.arange(
        record_dates.min(),
        record_dates.max() + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    day_indices = (record_dates - dates[0]).astype(int)

    sales = np.zeros((len(products), len(dates)), dtype=float)
    observed = np.zeros(sales.shape, dtype=bool)

    # Accumulate repeated entries without losing duplicate sales.
    np.add.at(sales, (product_indices, day_indices), quantities)
    observed[product_indices, day_indices] = True

    if not observed.all():
        missing_count = np.count_nonzero(~observed)
        raise ValueError(
            f"Missing {missing_count} product/date records. "
            "Provide explicit zero rows for days with no sales."
        )

    return products, dates, sales


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "sample_sales.csv"

    products, dates, sales = load_sales_csv(path)

    print("CSV loaded successfully!")
    print("Products:", len(products))
    print("Days:", len(dates))
    print("Sales shape:", sales.shape)
    print("Date range:", dates[0], "to", dates[-1])
    print("\nFirst 3 products:", products[:3])
    print("Their first 7 days of sales:")
    print(sales[:3, :7])
