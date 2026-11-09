
import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

# Read all seven SQL queries.
sql_text = SQL_PATH.read_text()

# Split queries at semicolons and remove SQL comments.
queries = []
for statement in sql_text.split(";"):
    cleaned_lines = [
        line for line in statement.splitlines()
        if not line.strip().startswith("--")
    ]
    query = "\n".join(cleaned_lines).strip()
    if query:
        queries.append(query)

assert len(queries) == 7, f"Expected 7 queries, found {len(queries)}"

connection = sqlite3.connect(DB_PATH)

results = {}

for number, query in enumerate(queries, start=1):
    print("\n" + "=" * 65)
    print(f"QUERY Q{number}")
    print("=" * 65)

    df = pd.read_sql_query(query, connection)
    results[number] = df

    print(df.to_string(index=False))
    print("\nRows returned:", len(df))

# --------------------------------------------------
# REFERENCE CHECKS
# --------------------------------------------------

print("\n" + "=" * 65)
print("SQL REFERENCE CHECKS")
print("=" * 65)

# Q1: 30 customers.
assert len(results[1]) == 30
print("PASS Q1: 30 customers")

# Q2: 30 city/category combinations and correct revenue.
assert len(results[2]) == 30
q2_revenue = round(results[2]["total_revenue"].sum(), 2)
assert q2_revenue == 2998668.80
print("PASS Q2: 30 city/category rows")
print("Q2 total revenue:", q2_revenue)

largest_q2 = results[2].sort_values(
    "total_revenue", ascending=False
).iloc[0]

print(
    "Largest city/category combination:",
    largest_q2["city"],
    "/",
    largest_q2["category"],
    "/",
    largest_q2["total_revenue"]
)

# Q3: 11 products above average catalogue price.
assert len(results[3]) == 11
print("PASS Q3: 11 products above average price")

# Q4: 25 customers without completed orders.
assert len(results[4]) == 25
print("PASS Q4: 25 customers without completed orders")

# Q5: 550 customers above average spend.
q5 = results[5].iloc[0]
assert int(q5["customers_above_average"]) == 550
print("PASS Q5: 550 customers above average")
print("Average customer spend:", q5["average_customer_spend"])

# Q6: Compare RANK with ROW_NUMBER.
q6 = results[6]
rank_differences = int(
    (q6["spending_rank"] != q6["row_number_rank"]).sum()
)
print("Q6 rows where RANK differs from ROW_NUMBER:", rank_differences)

print("\nTop-spending customer in each city:")
print(
    q6[q6["spending_rank"] == 1][
        ["city", "customer_id", "total_spend",
         "spending_rank", "dense_spending_rank", "row_number_rank"]
    ].to_string(index=False)
)

# Q7: 183 daily rows and correct final running total.
assert len(results[7]) == 183
final_running_total = round(
    results[7].iloc[-1]["running_total"], 2
)
assert final_running_total == 2998668.80
print("PASS Q7: 183 daily revenue rows")
print("Final running total:", final_running_total)

connection.close()

print("\nSQL EXECUTION COMPLETED")
