
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")

# --------------------------------------------------
# 1. Load cleaned datasets
# --------------------------------------------------

customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

# Convert date columns
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
products = pd.read_csv(CLEAN / "products.csv")
print("Customers loaded:", len(customers))
print("Orders loaded:", len(orders))

# --------------------------------------------------
# 2. Separate historical orders and label period
# --------------------------------------------------

# Only orders strictly before September 1 are allowed
# to contribute to model features.
historical = orders[
    (orders["order_ts"] < CUTOFF)
].copy()

# Completed orders during September determine the label.
label_period = orders[
    (orders["order_ts"] >= CUTOFF) &
    (orders["order_ts"] < LABEL_END) &
    (orders["status"] == "completed")
].copy()

historical_completed = historical[
    historical["status"] == "completed"
].copy()

historical_cancelled = historical[
    historical["status"] == "cancelled"
].copy()

print("\nHistorical orders:", len(historical))
print("Historical completed orders:", len(historical_completed))
print("Historical cancelled orders:", len(historical_cancelled))
print("September completed orders:", len(label_period))

# --------------------------------------------------
# 3. Create cohort
# Customers with at least one completed order
# before September 1, 2025.
# --------------------------------------------------

cohort_ids = historical_completed["customer_id"].unique()

cohort = customers[
    customers["customer_id"].isin(cohort_ids)
].copy()

print("\nCohort customers:", len(cohort))

# --------------------------------------------------
# 4. Create target label
# 1 = no completed order during September
# 0 = at least one completed order during September
# --------------------------------------------------

september_active_ids = set(label_period["customer_id"].unique())

cohort["lapsed"] = (
    ~cohort["customer_id"].isin(september_active_ids)
).astype(int)

print("\nTarget distribution:")
print(cohort["lapsed"].value_counts().sort_index())

print("\nLapsed percentage:")
print(round(cohort["lapsed"].mean() * 100, 2), "%")

# --------------------------------------------------
# 5. Build historical customer features
# All features use orders before September 1 only.
# --------------------------------------------------

# Recency: days since the last historical completed order
last_order = (
    historical_completed.groupby("customer_id")["order_ts"]
    .max()
    .rename("last_completed_order")
)

recency = (
    (CUTOFF - last_order).dt.days
    .rename("recency")
)

# Frequency: number of historical completed orders
frequency = (
    historical_completed.groupby("customer_id")["order_id"]
    .nunique()
    .rename("frequency")
)

# Monetary: total historical completed order amount
monetary = (
    historical_completed.groupby("customer_id")["order_amount"]
    .sum()
    .rename("monetary")
)

# Average discount percentage on completed orders
avg_discount = (
    historical_completed.groupby("customer_id")["discount_pct"]
    .mean()
    .rename("avg_discount_pct")
)

# Average delivery time on completed orders
avg_delivery = (
    historical_completed.groupby("customer_id")["delivery_minutes"]
    .mean()
    .rename("avg_delivery_minutes")
)

# Cancellation rate:
# Cancelled orders / all historical orders
total_orders = (
    historical.groupby("customer_id")["order_id"]
    .nunique()
    .rename("total_orders")
)

cancelled_orders = (
    historical_cancelled.groupby("customer_id")["order_id"]
    .nunique()
    .rename("cancelled_orders")
)

cancellation_rate = (
    cancelled_orders.div(total_orders)
    .fillna(0)
    .rename("cancellation_rate")
)

# Most frequently purchased category among completed orders


# Join historical completed orders with products
# to retrieve the category for each product.

historical_completed = historical_completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Check that the join worked
assert "category" in historical_completed.columns
assert historical_completed["category"].notna().all()

print(
    "\nProduct category join successful:",
    historical_completed["category"].nunique(),
    "categories found"
)

# Find the most frequently purchased category per customer

# Join historical completed orders with products
# to retrieve the category for each product.


# Check that the join worked
assert "category" in historical_completed.columns
assert historical_completed["category"].notna().all()

print(
    "\nProduct category join successful:",
    historical_completed["category"].nunique(),
    "categories found"
)

# Find the most frequently purchased category per customer
def get_favourite_category(series):
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else "Unknown"

favourite_category = (
    historical_completed.groupby("customer_id")["category"]
    .agg(get_favourite_category)
    .rename("favourite_category")
)


# --------------------------------------------------
# 6. Combine customer features
# --------------------------------------------------

features = pd.concat(
    [
        recency,
        frequency,
        monetary,
        avg_discount,
        cancellation_rate,
        avg_delivery,
        favourite_category
    ],
    axis=1
).reset_index()

# Merge customer profile information
dataset = cohort.merge(
    features,
    on="customer_id",
    how="inner"
)

# Tenure in days as of the cutoff date
dataset["tenure_days"] = (
    CUTOFF - dataset["signup_date"]
).dt.days

# --------------------------------------------------
# 7. Select final modelling columns
# --------------------------------------------------

final_columns = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
    "lapsed"
]

dataset = dataset[final_columns].copy()

# --------------------------------------------------
# 8. Validate missing values and date leakage
# --------------------------------------------------

print("\nMissing values by column:")
print(dataset.isnull().sum())

assert dataset["customer_id"].is_unique
assert dataset["lapsed"].isin([0, 1]).all()
assert dataset.isnull().sum().sum() == 0
assert (dataset["recency"] >= 0).all()
assert (dataset["frequency"] > 0).all()
assert (dataset["monetary"] >= 0).all()
assert (dataset["tenure_days"] >= 0).all()

# --------------------------------------------------
# 9. Save dataset
# --------------------------------------------------

OUTPUTS.mkdir(parents=True, exist_ok=True)

output_file = OUTPUTS / "lapse_dataset.csv"
dataset.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print("LAPSE DATASET SUMMARY")
print("=" * 65)

print("Total cohort customers:", len(dataset))
print("Lapsed customers:", int(dataset["lapsed"].sum()))
print("Not lapsed customers:", int((dataset["lapsed"] == 0).sum()))
print("Lapsed percentage:", round(dataset["lapsed"].mean() * 100, 2), "%")

print("\nFinal dataset shape:", dataset.shape)
print("\nFinal columns:")
print(dataset.columns.tolist())

print("\nSaved to:", output_file)
print("\nLAPSE DATASET CREATED SUCCESSFULLY")
