
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# 1. Load lapse dataset
df = pd.read_csv(OUTPUTS / "lapse_dataset.csv")

# 2. Separate features and target
X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

customer_ids = df["customer_id"]

# 3. Stratified 80:20 split
X_train, X_test, y_train, y_test, id_train, id_test = (
    train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

# 4. Display results
print("=" * 60)
print("TRAIN / TEST SPLIT SUMMARY")
print("=" * 60)

print("\nOriginal dataset:", len(df))
print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nTraining lapsed percentage:",
      round(y_train.mean() * 100, 2), "%")

print("Testing lapsed percentage:",
      round(y_test.mean() * 100, 2), "%")

# 5. Validate
assert len(X_train) == 924
assert len(X_test) == 232
assert len(X_train) + len(X_test) == len(df)

assert set(id_train).isdisjoint(set(id_test))

print("\nTRAIN / TEST SPLIT COMPLETED SUCCESSFULLY")
