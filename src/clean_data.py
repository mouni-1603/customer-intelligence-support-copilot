
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. LOAD THE RAW DATASETS
# --------------------------------------------------

customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
products = pd.read_csv(RAW / "products.csv", keep_default_na=False)
orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

# --------------------------------------------------
# 2. CLEAN CUSTOMERS
# --------------------------------------------------

customers["city"] = customers["city"].astype(str).str.strip().str.title()

# Standardise gender values.
gender_map = {
    "m": "male",
    "male": "male",
    "man": "male",
    "f": "female",
    "female": "female",
    "woman": "female"
}

customers["gender"] = (
    customers["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)

# Convert age to numeric and mark invalid ages as missing.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = float("nan")

# Fill missing ages using the median age of each city.
city_medians = customers.groupby("city")["age"].transform("median")
customers["age"] = customers["age"].fillna(city_medians)

# Convert signup date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# --------------------------------------------------
# 3. CLEAN PRODUCTS
# --------------------------------------------------

products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# --------------------------------------------------
# 4. CLEAN ORDERS
# --------------------------------------------------

# Convert timestamps before sorting and deduplication.
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

# Standardise status text.
orders["status"] = (
    orders["status"].astype(str).str.strip().str.lower()
)

# Remove exact duplicate rows.
orders = orders.drop_duplicates().copy()

# For duplicate order IDs, keep the latest record.
orders = (
    orders.sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# Convert numeric columns.
orders["quantity"] = pd.to_numeric(
    orders["quantity"], errors="coerce"
)
orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Negative delivery times are invalid.
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")

# Join product prices and categories to orders.
product_lookup = products[["product_id", "price_inr", "category"]].copy()

orders = orders.merge(
    product_lookup,
    on="product_id",
    how="left",
    validate="many_to_one"
)

# Recalculate only missing or non-numeric order amounts.
calculated_amount = (
    orders["price_inr"]
    * orders["quantity"]
    * (1 - orders["discount_pct"] / 100)
).round(2)

orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)
orders["order_amount"] = orders["order_amount"].round(2)

# Impute missing delivery times for completed orders using the
# median delivery time for that product category.
completed_mask = orders["status"].eq("completed")

category_delivery_median = (
    orders.loc[completed_mask]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

completed_missing_index = orders.index[
    completed_mask & orders["delivery_minutes"].isna()
]

orders.loc[completed_missing_index, "delivery_minutes"] = (
    category_delivery_median.reindex(completed_missing_index)
)

# Remove helper columns added by the product join.
orders = orders.drop(columns=["price_inr", "category"])

# --------------------------------------------------
# 5. SAVE CLEANED DATASETS
# --------------------------------------------------

customers.to_csv(CLEAN / "customers.csv", index=False, date_format="%Y-%m-%d")
products.to_csv(CLEAN / "products.csv", index=False)
orders.to_csv(
    CLEAN / "orders.csv",
    index=False,
    date_format="%Y-%m-%d %H:%M:%S"
)

# --------------------------------------------------
# 6. PRINT VALIDATION RESULTS
# --------------------------------------------------

print("\nCUSTOMER CLEANING")
print("-----------------")
print("Rows:", len(customers))
print("City counts:")
print(customers["city"].value_counts().sort_index())
print("\nGender counts:")
print(customers["gender"].value_counts())
print("Missing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())

print("\nORDER CLEANING")
print("--------------")
print("Rows:", len(orders))
print("Status counts:")
print(orders["status"].value_counts())
print("Missing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times in completed orders:",
    orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum()
)
print(
    "Missing delivery times in cancelled orders:",
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum()
)
print(
    "Completed revenue:",
    round(orders.loc[orders["status"].eq("completed"), "order_amount"].sum(), 2)
)

print("\nCLEANED FILES SAVED")
print("-------------------")
print(CLEAN / "customers.csv")
print(CLEAN / "products.csv")
print(CLEAN / "orders.csv")

# --------------------------------------------------
# 7. REFERENCE CHECKPOINTS
# --------------------------------------------------

assert len(customers) == 1200
assert customers["age"].isna().sum() == 0
assert customers["age"].between(10, 100).all()
assert set(customers["city"].unique()) == {
    "Pune", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Mumbai"
}
assert set(customers["gender"].unique()).issubset(
    {"male", "female", "unknown"}
)

assert len(products) == 40
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309
assert not orders["status"].eq("pending").any()
assert orders["status"].value_counts().to_dict() == {
    "completed": 11110,
    "cancelled": 1199
}
assert orders["order_amount"].isna().sum() == 0
assert orders.loc[
    orders["status"].eq("completed"), "delivery_minutes"
].isna().sum() == 0
assert orders.loc[
    orders["status"].eq("cancelled"), "delivery_minutes"
].isna().sum() == 1199

completed_revenue = orders.loc[
    orders["status"].eq("completed"), "order_amount"
].sum()

assert round(completed_revenue, 2) == 2998668.80

print("\nALL CLEANING CHECKPOINTS PASSED")
