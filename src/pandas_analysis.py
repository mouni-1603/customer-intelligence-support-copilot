
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"

# --------------------------------------------------
# 1. LOAD CLEANED DATA
# --------------------------------------------------

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

completed = orders.loc[orders["status"] == "completed"].copy()

completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="raise"
)
completed["discount_pct"] = pd.to_numeric(
    completed["discount_pct"], errors="raise"
)
completed["quantity"] = pd.to_numeric(
    completed["quantity"], errors="raise"
)
completed["delivery_minutes"] = pd.to_numeric(
    completed["delivery_minutes"], errors="raise"
)

# Join city and category information.
analysis_data = (
    completed
    .merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )
    .merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left",
        validate="many_to_one"
    )
)

# --------------------------------------------------
# 2. PIVOT TABLE: CITY AND CATEGORY REVENUE
# --------------------------------------------------

print("\n" + "=" * 70)
print("PANDAS PIVOT TABLE: REVENUE BY CITY AND CATEGORY")
print("=" * 70)

pivot = pd.pivot_table(
    analysis_data,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="Grand Total"
)

print(pivot.round(2).to_string())

pivot_total = round(pivot.loc["Grand Total", "Grand Total"], 2)
print("\nPivot grand total:", pivot_total)

assert pivot_total == 2998668.80
print("PASS: Pivot total matches SQL Q2")

# --------------------------------------------------
# 3. CITY-WISE REVENUE STATISTICS
# --------------------------------------------------

print("\n" + "=" * 70)
print("CITY-WISE REVENUE STATISTICS")
print("=" * 70)

city_stats = (
    analysis_data
    .groupby("city")
    .agg(
        revenue_sum=("order_amount", "sum"),
        revenue_mean=("order_amount", "mean"),
        order_count=("order_amount", "count")
    )
    .round(2)
    .sort_index()
)

print(city_stats.to_string())

# Reference checks.
assert round(city_stats.loc["Bengaluru", "revenue_sum"], 2) == 523643.40
assert round(city_stats.loc["Bengaluru", "revenue_mean"], 2) == 276.47
assert int(city_stats.loc["Bengaluru", "order_count"]) == 1894

assert round(city_stats.loc["Mumbai", "revenue_sum"], 2) == 479058.50
assert round(city_stats.loc["Mumbai", "revenue_mean"], 2) == 297.00
assert int(city_stats.loc["Mumbai", "order_count"]) == 1613

print("\nPASS: Bengaluru and Mumbai reference values match")

# --------------------------------------------------
# 4. ORDER AMOUNT DISTRIBUTION AND OUTLIERS
# --------------------------------------------------

print("\n" + "=" * 70)
print("COMPLETED ORDER AMOUNT DISTRIBUTION")
print("=" * 70)

amount = completed["order_amount"]

mean_value = amount.mean()
median_value = amount.median()
skew_value = amount.skew()

q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)

iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below_outliers = int((amount < lower_fence).sum())
above_outliers = int((amount > upper_fence).sum())

print(f"Mean: INR {mean_value:.2f}")
print(f"Median: INR {median_value:.2f}")
print(f"Skewness: {skew_value:.2f}")
print(f"Q1 (25th percentile): INR {q1:.2f}")
print(f"Q2 (50th percentile): INR {q2:.2f}")
print(f"Q3 (75th percentile): INR {q3:.2f}")
print(f"IQR: INR {iqr:.2f}")
print(f"Lower fence: INR {lower_fence:.2f}")
print(f"Upper fence: INR {upper_fence:.2f}")
print("Outliers below lower fence:", below_outliers)
print("Outliers above upper fence:", above_outliers)

assert round(mean_value, 2) == 269.91
assert round(median_value, 2) == 135.00
assert round(skew_value, 2) == 4.61
assert round(q1, 2) == 64.80
assert round(q3, 2) == 297.00
assert round(upper_fence, 2) == 645.30
assert below_outliers == 0
assert above_outliers == 897

print("\nPASS: Order amount distribution matches reference values")

# --------------------------------------------------
# 5. CORRELATION MATRIX
# --------------------------------------------------

print("\n" + "=" * 70)
print("CORRELATION MATRIX")
print("=" * 70)

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_matrix = completed[correlation_columns].corr()

print(correlation_matrix.round(3).to_string())

# Find the pair with the largest absolute correlation.
upper_triangle = correlation_matrix.where(
    pd.DataFrame(
        [[i < j for j in range(len(correlation_columns))]
         for i in range(len(correlation_columns))],
        index=correlation_columns,
        columns=correlation_columns
    )
)

correlation_pairs = upper_triangle.stack()
strongest_pair = correlation_pairs.abs().idxmax()
strongest_value = correlation_pairs.loc[strongest_pair]

print("\nLargest absolute correlation:")
print(strongest_pair[0], "and", strongest_pair[1])
print(f"Correlation value: {strongest_value:.3f}")

assert strongest_pair == ("order_amount", "discount_pct")
assert round(strongest_value, 3) == -0.095

print("\nPASS: Correlation reference matches")

# --------------------------------------------------
# 6. FINAL CHECK
# --------------------------------------------------

print("\n" + "=" * 70)
print("PANDAS ANALYSIS COMPLETED")
print("=" * 70)
