<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=27&duration=2800&pause=1200&color=BDA7CC&center=true&vCenter=true&width=850&height=90&lines=NumPy+Business+Toolkit;Model+uncertainty.+Forecast+sales.;Make+your+next+move+with+context." alt="NumPy Business Toolkit animated heading" />

# Business questions. NumPy answers.

**Explore startup runway. Forecast product sales. Plan inventory.**

Two practical business tools. One external dependency.

<br>

[![Python](https://img.shields.io/badge/Python-3.9%2B-18263D?style=for-the-badge&logo=python&logoColor=ECD9B0)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Powered_by-NumPy-18263D?style=for-the-badge&logo=numpy&logoColor=BDA7CC)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-18263D?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In_Development-72516E?style=for-the-badge)](#project-status)

<br>

[Get started](#quick-start) ·
[Explore the tools](#two-tools-one-foundation) ·
[Understand the models](#under-the-hood) ·
[View limitations](#model-limitations)

</div>

---

## Why this exists

“How long will our cash last?” rarely has one certain answer.

“What should we reorder?” needs more context than yesterday’s sales.

**NumPy Business Toolkit** explores both questions with numerical simulations,
historical sales, and transparent assumptions—without a large data science stack.

Built by **Ujjwal Sharma**, a first-year CSE student exploring practical NumPy.

## Two tools. One foundation.

<table>
<tr>
<td width="50%" valign="top">

### 01 / Runway Studio

**Explore possible cash futures.**

Run thousands of scenarios with uncertain revenue growth and churn.

- Monthly survival probabilities
- Median cash trajectory
- 10th–90th percentile range
- Lower-cost and higher-growth comparisons
- Browser dashboard and CLI reports

**The question:**  
What fraction of simulated businesses avoid negative cash through month N?

</td>
<td width="50%" valign="top">

### 02 / Inventory Studio

**Turn sales history into a stock plan.**

Compare forecasting methods and estimate what to reorder.

- Seven-day product forecasts
- Historical method comparison
- Product-specific method selection
- Safety stock and reorder suggestions
- CSV uploads and an instant demo

**The question:**  
How much could sell next week, and which products need replenishment?

</td>
</tr>
</table>

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/ujjwalsharmaa2007/numpy-biz-toolkit.git
cd numpy-biz-toolkit
python -m pip install -r requirements.txt
```

### 2. Choose your interface

**Browser dashboard**

```bash
python web/server.py
```

Open **http://localhost:8000**.

Using Codespaces? Open **port 8000** from the **Ports** tab.
Keep its visibility private. Stop the server with **Ctrl + C**.

**Command line**

```bash
python cli.py runway
python cli.py inventory
```

**Runway comparisons**

```bash
python compare_runway.py
```

> Python 3.9+ is the compatibility target. The full version range has not
> yet been verified. The web server is intended for local development.

## A closer look

### Runway assumptions

The current web dashboard uses:

| Parameter | Default |
|:--|--:|
| Starting cash | 100,000 |
| Current monthly revenue | 20,000 |
| Current monthly costs | 30,000 |
| Expected monthly revenue growth | 3% |
| Growth standard deviation | 2% |
| Expected monthly churn | 1% |
| Churn standard deviation | 0.5% |
| Monthly cost growth | 1% |
| Scenarios | 10,000 |
| Forecast horizon | 24 months |
| Random seed | 42 |

Use the **same currency** for all amounts.

Cash, revenue, and costs are editable in the dashboard.
Other assumptions are currently fixed there and configurable through the
Python function.

### Inventory workflow

1. Load daily sales and inventory settings.
2. Hold out the last 28 days.
3. Compare moving average, exponential smoothing, and weekday forecasts.
4. Select the lowest-MAE method for each product.
5. Recalculate forecasts using all available history.
6. Estimate stock coverage and suggest replenishment.

The demo includes **20 products and 365 days of synthetic sales**, with
weekend patterns, growth, seasonal bumps, and simulated stockouts.

## Bring your own data

<details>
<summary><strong>Sales CSV format</strong></summary>

<br>

```csv
date,product,quantity_sold
2026-01-01,Tea,10
2026-01-02,Tea,0
2026-01-03,Tea,14
```

- Dates use `YYYY-MM-DD`.
- Quantities must be finite, non-negative whole numbers.
- Every product needs a record for every date in the dataset’s range.
- Record zero-sales days explicitly.
- Duplicate product/date entries are added together.
- The current forecasting pipeline requires at least **84 consecutive days**.

The short example above illustrates the format only.

</details>

<details>
<summary><strong>Inventory settings format</strong></summary>

<br>

```csv
product,current_stock,lead_days,on_order
Tea,20,4,0
```

| Field | Meaning |
|:--|:--|
| `product` | Exact product name from the sales CSV |
| `current_stock` | Units currently available |
| `lead_days` | Days between placing an order and delivery |
| `on_order` | Units already ordered but not yet received |

Use custom files from the CLI:

```bash
python cli.py inventory \
  --sales data/my_sales.csv \
  --settings data/my_settings.csv
```

Web requests are limited to **500 products**, **3,650 days**, and a
**2 MB JSON request**.

</details>

## Under the hood

### Many futures, calculated together

Runway arrays have shape:

```python
(scenarios, months)
```

NumPy compounds revenue and accumulates cash across each scenario:

```python
revenue = monthly_revenue * np.cumprod(revenue_factors, axis=1)

cash = starting_cash + np.cumsum(
    revenue - costs,
    axis=1,
)
```

Survival requires avoiding negative cash in **every preceding month**.

```python
failed = np.logical_or.accumulate(cash < 0, axis=1)
survival = np.mean(~failed, axis=0)
```

A scenario that fails remains failed for survival reporting,
even if its simulated balance later recovers.

### Forecasts earn their place

| Method | Approach |
|:--|:--|
| Moving average | Average of the most recent 28 days |
| Exponential smoothing | Recent observations receive more weight |
| Weekday forecast | Average matching weekdays over the last eight weeks |

**MAE** measures average daily error in units.

**WAPE** measures total absolute error relative to total actual sales.
It is undefined when actual total sales are zero.

Lower is better. Neither metric should be described as “accuracy.”

For each product, MAE selects the method. When actual total sales are
positive, MAE and WAPE rank these methods identically.
Ties favor the first method in the comparison order.

### Stock recommendations with a buffer

```text
Safety stock
    = z × daily sales standard deviation × √lead days

Reorder point
    = forecast daily average × lead days + safety stock

Stock position
    = current stock + on-order units
```

When stock position reaches the reorder point and forecast demand is
positive, the tool recommends replenishment.

Suggested quantities cover **lead time plus a seven-day review period**,
with a safety buffer for that combined period.

The default safety factor is `z = 1.65`.
It is an approximation, not a guaranteed service level.

## NumPy skills demonstrated

| Concept | Application |
|:--|:--|
| Broadcasting | Apply costs and product baselines across arrays |
| `cumprod` | Compound monthly revenue |
| `cumsum` | Accumulate profit and loss |
| Boolean masks | Track cash failure and reorder conditions |
| `percentile` | Summarize cash uncertainty |
| Reshaping | Group historical sales into weekly blocks |
| Fancy indexing | Choose a different forecast method per product |
| `argsort` | Rank inventory urgency |
| `np.add.at` | Combine duplicate sales records |
| `default_rng` | Reproduce simulations and synthetic data |

Calculations operate across products or scenarios together.
Exponential smoothing loops over time while updating all products with
NumPy. File handling and report formatting also use Python loops.

## Project map

```text
numpy-biz-toolkit/
├── core/
│   ├── runway.py
│   ├── forecasting.py
│   ├── backtesting.py
│   ├── demand.py
│   └── inventory.py
├── common/
│   ├── fake_data.py
│   └── data_loader.py
├── web/
│   ├── server.py
│   └── static/
│       ├── index.html
│       └── inventory.html
├── data/
│   ├── sample_sales.csv
│   └── demo_inventory_settings.csv
├── tests/
│   └── test_toolkit.py
├── cli.py
├── compare_runway.py
├── requirements.txt
└── README.md
```

## Verification

```bash
python -m unittest discover -s tests -v
```

Checks cover:

- Cash balances against a hand calculation
- Survival tracking after cash recovery
- Moving average and smoothing examples
- Weekday forecast selection
- Reorder quantities and incoming stock
- Zero-demand behavior
- Duplicate records and missing-data validation

**Benchmarks:** not published yet. No unmeasured speedup claims.

## Model limitations

<details>
<summary><strong>Read before using results for business decisions</strong></summary>

<br>

**Runway**

- Probabilities depend on the assumptions and random distributions.
- Growth and churn draws are independent in the current model.
- Clipping draws can change their effective averages.
- Growth, churn, and cost changes apply from month one.
- Failure means cash **below zero at month-end**; exactly zero is not failure.
- Month-end checks do not detect cash shortages within a month.
- Cash paths after failure assume operations continue.

**Inventory**

- Forecasts estimate observed sales, which can understate demand during stockouts.
- Method selection uses one 28-day holdout, not rolling validation.
- Safety stock assumes fixed lead times and independent daily demand.
- Forecast average daily sales are extended across the stock coverage period.
- Incoming stock arrival dates and supplier delays are not modeled.
- Pack sizes, minimum orders, and backorders are not modeled.
- Synthetic data does not establish real-world performance.

This is an educational project and an early decision-support prototype.
No real-business case study has been completed yet.

</details>

## Project status

- [x] Synthetic sales generator
- [x] Vectorized runway simulation
- [x] What-if comparisons
- [x] Validated CSV loading
- [x] Three forecasting methods
- [x] Historical backtesting
- [x] Reorder recommendations
- [x] Command-line interface
- [x] Runway and inventory web pages
- [x] Calculation and validation tests
- [ ] Dashboard screenshots
- [ ] Measured performance benchmarks
- [ ] Inventory forecast chart
- [ ] Editable web growth and churn assumptions
- [ ] Rolling forecast evaluation
- [ ] Real-business feedback

---

<div align="center">

### Built to learn. Designed to explain.

Made with love by **Ujjwal**

[GitHub](https://github.com/ujjwalsharmaa2007) ·
[LinkedIn](https://www.linkedin.com/in/ujjwal-sharma-60a58734a)

**MIT License** · See [LICENSE](LICENSE)

<sub>Animated heading and badges are served by external services.</sub>

</div>
