import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return

        body = (ROOT / "web/static/index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/runway":
            self.send_json(404, {"error": "Unknown endpoint."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 10_000:
                raise ValueError("Request must contain at most 10 KB.")

            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")

            values = {}
            for name in ("starting_cash", "monthly_revenue", "monthly_costs"):
                value = data.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{name} must be a number.")
                if not np.isfinite(value) or not 0 <= value <= 1e12:
                    raise ValueError(f"{name} must be between 0 and 1 trillion.")
                values[name] = value

            result = simulate_runway(**values)

            self.send_json(200, {
                "survival": result["survival_probability"].tolist(),
                "cash_percentiles": result["cash_percentiles"].tolist(),
            })

        except (ValueError, TypeError, OverflowError) as error:
            self.send_json(400, {"error": str(error)})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    print("NumPy Business Toolkit running on port 8000.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
