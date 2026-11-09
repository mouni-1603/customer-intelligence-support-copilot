
import pandas as pd
from pathlib import Path

# Project directory
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# --------------------------------------------------
# 1. LOAD RAW DATASETS
# --------------------------------------------------

customers = pd.read_csv(
    RAW / "customers.csv",
    keep_default_na=False
)

products = pd.read_csv(
    RAW / "products.csv",
    keep_default_na=False
)

orders = pd.read_csv(
    RAW / "orders.csv",
    keep_default_na=False
)

datasets = {
    "CUSTOMERS": customers,
    "PRODUCTS": products,
    "ORDERS": orders
}

# --------------------------------------------------
# 2. DISPLAY SHAPE, DATA TYPES AND MISSING VALUES
# --------------------------------------------------

for name, df in datasets.items():

    print("\n" + "=" * 60)
    print(f"{name} DATA AUDIT")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn data types:")
    print(df.dtypes)

    # Count empty strings and actual missing values
    missing_count = (
        df.isna() |
        df.astype(str).apply(
            lambda col: col.str.strip().eq("")
        )
    ).sum()

    missing_percentage = (
        missing_count / len(df) * 100
    )

    missing_report = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage": missing_percentage.round(2)
    })

    print("\nMissing values report:")
    print(missing_report)

# --------------------------------------------------
# 3. AUDIT RAW ORDERS
# --------------------------------------------------

print("\n" + "=" * 60)
print("RAW ORDERS AUDIT")
print("=" * 60)

numeric_amount = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

numeric_delivery = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

blank_delivery = (
    orders["delivery_minutes"].astype(str).str.strip() == ""
)

non_numeric_amount = (
    orders["order_amount"].astype(str).str.strip().ne("")
    & numeric_amount.isna()
)

orders_audit = {
    "Raw order rows": len(orders),
    "Exact duplicate rows": int(orders.duplicated().sum()),
    "Pending rows": int(
        orders["status"].str.lower().eq("pending").sum()
    ),
    "Distinct order IDs": int(orders["order_id"].nunique()),
    "Non-numeric order amounts": int(non_numeric_amount.sum()),
    "Negative delivery times": int((numeric_delivery < 0).sum()),
    "Blank delivery times": int(blank_delivery.sum())
}

print(pd.Series(orders_audit))

# --------------------------------------------------
# 4. AUDIT RAW CUSTOMERS
# --------------------------------------------------

print("\n" + "=" * 60)
print("RAW CUSTOMERS AUDIT")
print("=" * 60)

numeric_age = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

blank_age = customers["age"].astype(str).str.strip().eq("")
blank_gender = customers["gender"].astype(str).str.strip().eq("")

outside_age_range = (
    numeric_age.notna()
    & ~numeric_age.between(10, 100)
)

customers_audit = {
    "Blank age": int(blank_age.sum()),
    "Age outside 10-100": int(outside_age_range.sum()),
    "Distinct raw city spellings": int(
        customers.loc[
            customers["city"].str.strip().ne(""), "city"
        ].nunique()
    ),
    "Distinct non-blank raw gender spellings": int(
        customers.loc[
            ~blank_gender, "gender"
        ].nunique()
    ),
    "Blank gender": int(blank_gender.sum())
}

print(pd.Series(customers_audit))

# --------------------------------------------------
# 5. FINAL REFERENCE CHECKPOINTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("REFERENCE CHECKPOINT COMPARISON")
print("=" * 60)

reference = {
    "Raw order rows": 13084,
    "Exact duplicate rows": 119,
    "Pending rows": 656,
    "Distinct order IDs": 12309,
    "Non-numeric order amounts": 120,
    "Negative delivery times": 106,
    "Blank delivery times": 2541,
    "Blank age": 61,
    "Age outside 10-100": 6,
    "Distinct raw city spellings": 24,
    "Distinct non-blank raw gender spellings": 10,
    "Blank gender": 30
}

actual = {
    **orders_audit,
    **customers_audit
}

for key, expected in reference.items():

    value = actual[key]

    result = "PASS" if value == expected else "CHECK"

    print(
        f"{key}: Actual = {value}, "
        f"Expected = {expected} [{result}]"
    )

print("\nAUDIT COMPLETED")
