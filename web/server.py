import csv
import io
import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import load_settings
from common.data_loader import load_sales_csv
from core.demand import forecast_demand
from core.inventory import recommend_orders
from core.runway import simulate_runway


class Handler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        body = json.dumps(data, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        pages = {
            "/": "index.html",
            "/index.html": "index.html",
            "/inventory": "inventory.html",
        }

        if self.path in pages:
            body = (ROOT / "web/static" / pages[self.path]).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/demo":
            try:
                self.send_json(200, {
                    "sales_csv": (ROOT / "data/sample_sales.csv").read_text(),
                    "settings_csv": (
                        ROOT / "data/demo_inventory_settings.csv"
                    ).read_text(),
                })
            except OSError:
                self.send_json(404, {"error": "Demo files are missing."})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path not in ("/api/runway", "/api/inventory"):
            self.send_json(404, {"error": "Unknown endpoint."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Request must contain at most 2 MB.")

            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")

            if self.path == "/api/runway":
                values = {}
                for name in (
                    "starting_cash", "monthly_revenue", "monthly_costs"
                ):
                    value = data.get(name)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not np.isfinite(value)
                        or not 0 <= value <= 1e12
                    ):
                        raise ValueError(
                            f"{name} must be between 0 and 1 trillion."
                        )
                    values[name] = value

                result = simulate_runway(**values)
                self.send_json(200, {
                    "survival": result["survival_probability"].tolist(),
                    "cash_percentiles": result["cash_percentiles"].tolist(),
                })
                return

            sales_text = data.get("sales_csv")
            settings_text = data.get("settings_csv")

            if not isinstance(sales_text, str) or not isinstance(settings_text, str):
                raise ValueError("Provide sales and settings CSV text.")

            # Bound the dense array size before the loader allocates it.
            reader = csv.DictReader(io.StringIO(sales_text))
            required = {"date", "product", "quantity_sold"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Sales CSV columns are missing.")

            dates_seen = []
            products_seen = set()
            for row in reader:
                day = np.datetime64(row["date"], "D")
                if np.isnat(day):
                    raise ValueError("Sales dates cannot be missing.")
                dates_seen.append(day)
                products_seen.add(row["product"])

            if not dates_seen:
                raise ValueError("Sales CSV is empty.")

            days = int(
                (max(dates_seen) - min(dates_seen))
                / np.timedelta64(1, "D")
            ) + 1
            if days > 3650 or len(products_seen) > 500:
                raise ValueError("Limit: 500 products and 3,650 days.")

            # Uploaded CSVs are processed temporarily, then removed.
            with tempfile.TemporaryDirectory() as folder:
                sales_path = Path(folder) / "sales.csv"
                settings_path = Path(folder) / "settings.csv"
                sales_path.write_text(sales_text, encoding="utf-8")
                settings_path.write_text(settings_text, encoding="utf-8")

                products, dates, sales = load_sales_csv(sales_path)
                stock, lead, incoming = load_settings(settings_path, products)

            demand = forecast_demand(sales, dates)
            orders = recommend_orders(
                sales, demand["forecast"], stock, lead, on_order=incoming
            )

            rows = []
            for i in orders["urgency_order"]:
                days_left = orders["days_left"][i]
                rows.append({
                    "product": str(products[i]),
                    "method": str(demand["selected_methods"][i]),
                    "stock": float(stock[i]),
                    "lead_days": float(lead[i]),
                    "days_left": float(days_left) if np.isfinite(days_left) else None,
                    "order": int(orders["order_quantity"][i]),
                    "forecast": demand["forecast"][i].tolist(),
                })

            self.send_json(200, {
                "dates": demand["forecast_dates"].astype(str).tolist(),
                "products": rows,
            })

        except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
            self.send_json(400, {"error": str(error)})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("Toolkit running on port 8000. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
