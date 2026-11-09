
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

# --------------------------------------------------
# 1. LOAD CLEANED CSV FILES
# --------------------------------------------------

customers = pd.read_csv(CLEAN / "customers.csv", keep_default_na=False)
products = pd.read_csv(CLEAN / "products.csv", keep_default_na=False)
orders = pd.read_csv(CLEAN / "orders.csv", keep_default_na=False)

# Ensure numeric columns have the correct types.
customers["age"] = pd.to_numeric(customers["age"], errors="raise")
products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")

orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise").astype(int)
orders["discount_pct"] = pd.to_numeric(orders["discount_pct"], errors="raise")
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="raise")
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

# Store dates as formatted text in SQLite.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="raise"
).dt.strftime("%Y-%m-%d")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="raise"
).dt.strftime("%Y-%m-%d %H:%M:%S")

# --------------------------------------------------
# 2. CREATE A FRESH SQLITE DATABASE
# --------------------------------------------------

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

if DB_PATH.exists():
    DB_PATH.unlink()

connection = sqlite3.connect(DB_PATH)
connection.execute("PRAGMA foreign_keys = ON")

# --------------------------------------------------
# 3. CREATE TABLES
# --------------------------------------------------

connection.executescript("""
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    city TEXT NOT NULL,
    signup_date TEXT,
    age REAL,
    gender TEXT
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_inr REAL NOT NULL
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    order_ts TEXT NOT NULL,
    status TEXT NOT NULL,
    discount_pct REAL,
    order_amount REAL NOT NULL,
    delivery_minutes REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
""")

# --------------------------------------------------
# 4. INSERT CLEANED DATA INTO SQLITE
# --------------------------------------------------

customers.to_sql(
    "customers", connection, if_exists="append", index=False
)

products.to_sql(
    "products", connection, if_exists="append", index=False
)

orders.to_sql(
    "orders", connection, if_exists="append", index=False
)

connection.commit()

# --------------------------------------------------
# 5. VALIDATE DATABASE TABLES
# --------------------------------------------------

print("\nDATABASE CREATED")
print("----------------")
print("Database location:", DB_PATH)

table_names = ["customers", "products", "orders"]

for table in table_names:
    count = connection.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()[0]
    print(f"{table}: {count} rows")

print("\nDATABASE VALIDATION")
print("-------------------")

# Check 1: Customer row count.
customer_count = connection.execute(
    "SELECT COUNT(*) FROM customers"
).fetchone()[0]
assert customer_count == 1200
print("PASS: 1,200 customers stored")

# Check 2: Product row count.
product_count = connection.execute(
    "SELECT COUNT(*) FROM products"
).fetchone()[0]
assert product_count == 40
print("PASS: 40 products stored")

# Check 3: Order row count.
order_count = connection.execute(
    "SELECT COUNT(*) FROM orders"
).fetchone()[0]
assert order_count == 12309
print("PASS: 12,309 orders stored")

# Check 4: Order IDs are unique.
distinct_order_ids = connection.execute(
    "SELECT COUNT(DISTINCT order_id) FROM orders"
).fetchone()[0]
assert distinct_order_ids == 12309
print("PASS: Order IDs are unique")

# Check 5: No pending orders remain.
pending_count = connection.execute(
    "SELECT COUNT(*) FROM orders WHERE status = 'pending'"
).fetchone()[0]
assert pending_count == 0
print("PASS: No pending orders")

# Check 6: Completed and cancelled order counts.
status_counts = dict(
    connection.execute(
        "SELECT status, COUNT(*) FROM orders GROUP BY status"
    ).fetchall()
)
assert status_counts == {"cancelled": 1199, "completed": 11110}
print("PASS: Completed and cancelled counts match")

# Check 7: Completed revenue.
completed_revenue = connection.execute("""
    SELECT ROUND(SUM(order_amount), 2)
    FROM orders
    WHERE status = 'completed'
""").fetchone()[0]
assert round(completed_revenue, 2) == 2998668.80
print("PASS: Completed revenue matches")

# Check 8: Completed orders have no missing delivery times.
missing_completed_delivery = connection.execute("""
    SELECT COUNT(*)
    FROM orders
    WHERE status = 'completed'
      AND delivery_minutes IS NULL
""").fetchone()[0]
assert missing_completed_delivery == 0
print("PASS: Completed delivery times are populated")

# Check 9: Cancelled orders retain missing delivery times.
missing_cancelled_delivery = connection.execute("""
    SELECT COUNT(*)
    FROM orders
    WHERE status = 'cancelled'
      AND delivery_minutes IS NULL
""").fetchone()[0]
assert missing_cancelled_delivery == 1199
print("PASS: Cancelled delivery times remain NULL")

# Check 10: No broken customer or product references.
foreign_key_errors = connection.execute(
    "PRAGMA foreign_key_check"
).fetchall()
assert len(foreign_key_errors) == 0
print("PASS: Foreign key relationships are valid")

# Check 11: Order timestamps use the required text format.
timestamp_sample = connection.execute("""
    SELECT order_ts FROM orders LIMIT 1
""").fetchone()[0]
assert len(timestamp_sample) == 19
assert timestamp_sample[4] == "-"
assert timestamp_sample[7] == "-"
assert timestamp_sample[10] == " "
assert timestamp_sample[13] == ":"
assert timestamp_sample[16] == ":"
print("PASS: Order timestamps use YYYY-MM-DD HH:MM:SS")

connection.close()

print("\nALL CHECKPOINTS PASSED")
