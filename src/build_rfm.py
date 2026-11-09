
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 1. LOAD CLEANED DATA
# --------------------------------------------------

orders = pd.read_csv(CLEAN / "orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="raise"
)

# Use completed orders only.
completed = orders.loc[
    orders["status"] == "completed"
].copy()

# --------------------------------------------------
# 2. DEFINE THE RFM SNAPSHOT DATE
# --------------------------------------------------

snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# --------------------------------------------------
# 3. CALCULATE RFM
# --------------------------------------------------

rfm = (
    completed
    .groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

# Recency = number of whole days since the last completed order.
rfm["recency"] = (
    snapshot_date - rfm["last_order_date"]
).dt.days

# Keep only the required RFM columns.
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
].copy()

# Sort for readability.
rfm = rfm.sort_values("customer_id").reset_index(drop=True)

# --------------------------------------------------
# 4. DISPLAY RFM SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("RFM CUSTOMER TABLE")
print("=" * 65)

print("Number of customers:", len(rfm))
print("\nFirst 10 rows:")
print(rfm.head(10).to_string(index=False))

print("\nRFM descriptive statistics:")
print(
    rfm[["recency", "frequency", "monetary"]]
    .describe()
    .round(2)
    .to_string()
)

print("\nSpecific reference values:")
print("Recency median:", rfm["recency"].median())
print("Recency minimum:", rfm["recency"].min())
print("Recency maximum:", rfm["recency"].max())
print("Frequency mean:", round(rfm["frequency"].mean(), 2))
print("Frequency maximum:", rfm["frequency"].max())
print("Monetary mean:", round(rfm["monetary"].mean(), 2))
print("Monetary maximum:", round(rfm["monetary"].max(), 2))

# --------------------------------------------------
# 5. SAVE RFM TABLE
# --------------------------------------------------

rfm_path = OUTPUTS / "rfm_table.csv"
rfm.to_csv(rfm_path, index=False)

print("\nRFM table saved to:", rfm_path)

# --------------------------------------------------
# 6. REFERENCE CHECKPOINTS
# --------------------------------------------------

assert len(rfm) == 1175
assert rfm["recency"].median() == 19
assert rfm["recency"].min() == 0
assert rfm["recency"].max() == 180
assert round(rfm["frequency"].mean(), 2) == 9.46
assert rfm["frequency"].max() == 33
assert round(rfm["monetary"].mean(), 2) == 2552.06
assert round(rfm["monetary"].max(), 2) == 14766.55

assert not rfm.isna().any().any()

print("\nALL RFM CHECKPOINTS PASSED")
