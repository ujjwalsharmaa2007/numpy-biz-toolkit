import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.data_loader import load_sales_csv
from core.runway import simulate_runway
from core.forecasting import (
    moving_average_forecast,
    exponential_smoothing_forecast,
)
from core.demand import forecast_demand
from core.inventory import recommend_orders


class ToolkitTests(unittest.TestCase):

    def test_runway_matches_hand_calculation(self):
        result = simulate_runway(
            starting_cash=100_000,
            monthly_revenue=20_000,
            monthly_costs=30_000,
            growth_rate=0,
            growth_std=0,
            churn_rate=0,
            churn_std=0,
            cost_growth=0,
            months=12,
            scenarios=3,
        )

        expected = 100_000 - 10_000 * np.arange(1, 13)

        np.testing.assert_allclose(
            result["cash"],
            np.broadcast_to(expected, (3, 12)),
        )
        np.testing.assert_array_equal(
            result["first_failure_month"], [11, 11, 11]
        )
        np.testing.assert_allclose(
            result["survival_probability"],
            [1] * 10 + [0, 0],
        )

    def test_runway_survival_does_not_reset_after_recovery(self):
        result = simulate_runway(
            starting_cash=0,
            monthly_revenue=10,
            monthly_costs=25,
            growth_rate=1,
            growth_std=0,
            churn_rate=0,
            churn_std=0,
            cost_growth=0,
            months=3,
            scenarios=1,
        )

        # Cash recovers, but the scenario already failed in month 1.
        np.testing.assert_allclose(result["cash"], [[-5, 10, 65]])
        np.testing.assert_allclose(
            result["survival_probability"], [0, 0, 0]
        )

    def test_moving_average(self):
        sales = np.array([[2, 4, 6, 8], [10, 20, 30, 40]])
        result = moving_average_forecast(
            sales, window=2, horizon=3
        )
        np.testing.assert_allclose(
            result, [[7, 7, 7], [35, 35, 35]]
        )

    def test_exponential_smoothing(self):
        result = exponential_smoothing_forecast(
            [[10, 20, 30]], alpha=0.5, horizon=3
        )
        np.testing.assert_allclose(
            result, [[22.5, 22.5, 22.5]]
        )

    def test_weekday_method_selection(self):
        pattern = np.array([10, 10, 10, 10, 10, 20, 20])
        sales = np.tile(pattern, 12)[None, :]
        dates = (
            np.datetime64("2026-01-05")
            + np.arange(84).astype("timedelta64[D]")
        )

        result = forecast_demand(sales, dates)

        self.assertEqual(result["selected_methods"][0], "Weekday")
        np.testing.assert_allclose(
            result["forecast"], pattern[None, :]
        )
        self.assertEqual(
            result["forecast_dates"][0],
            dates[-1] + np.timedelta64(1, "D"),
        )

    def test_reorder_and_incoming_stock(self):
        settings = {
            "sales": np.full((1, 56), 10.0),
            "forecast": np.full((1, 7), 10.0),
            "current_stock": [20],
            "lead_days": [3],
        }

        result = recommend_orders(**settings)
        np.testing.assert_allclose(result["reorder_point"], [30])
        np.testing.assert_allclose(result["days_left"], [2])
        np.testing.assert_array_equal(result["order_quantity"], [80])

        # An existing order prevents a duplicate recommendation.
        result = recommend_orders(**settings, on_order=[80])
        np.testing.assert_array_equal(result["order_quantity"], [0])

    def test_zero_demand_does_not_trigger_order(self):
        result = recommend_orders(
            sales=np.zeros((1, 56)),
            forecast=np.zeros((1, 7)),
            current_stock=[0],
            lead_days=[3],
        )
        self.assertTrue(np.isinf(result["days_left"][0]))
        self.assertEqual(result["order_quantity"][0], 0)

    def test_loader_adds_duplicates_and_preserves_zero_sales(self):
        content = (
            "date,product,quantity_sold\n"
            "2026-01-01,Tea,2\n"
            "2026-01-01,Tea,3\n"
            "2026-01-02,Tea,0\n"
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sales.csv"
            path.write_text(content, encoding="utf-8")
            products, dates, sales = load_sales_csv(path)

        np.testing.assert_array_equal(products, ["Tea"])
        np.testing.assert_allclose(sales, [[5, 0]])
        self.assertEqual(len(dates), 2)

    def test_loader_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sales.csv"
            path.write_text("date,product\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_sales_csv(path)

    def test_loader_rejects_missing_days(self):
        content = (
            "date,product,quantity_sold\n"
            "2026-01-01,Tea,2\n"
            "2026-01-03,Tea,3\n"
        )

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sales.csv"
            path.write_text(content, encoding="utf-8")

            with self.assertRaises(ValueError):
                load_sales_csv(path)


if __name__ == "__main__":
    unittest.main()
