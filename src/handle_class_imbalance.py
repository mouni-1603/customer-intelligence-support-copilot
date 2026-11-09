
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# --------------------------------------------------
# 1. Load data and recreate the original split
# --------------------------------------------------

df = pd.read_csv(OUTPUTS / "lapse_dataset.csv")

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 2. Display original class distribution
# --------------------------------------------------

print("=" * 65)
print("ORIGINAL TRAINING CLASS DISTRIBUTION")
print("=" * 65)

print(y_train.value_counts().sort_index())

# --------------------------------------------------
# 3. Oversample minority class in training data only
# --------------------------------------------------

majority_count = int((y_train == 0).sum())
minority_count = int((y_train == 1).sum())

difference = majority_count - minority_count

minority_indices = np.where(y_train.to_numpy() == 1)[0]

random_state = np.random.RandomState(42)

duplicate_indices = random_state.choice(
    minority_indices,
    size=difference,
    replace=True
)

X_train_balanced = pd.concat(
    [
        X_train.reset_index(drop=True),
        X_train.iloc[duplicate_indices].reset_index(drop=True)
    ],
    ignore_index=True
)

y_train_balanced = pd.concat(
    [
        y_train.reset_index(drop=True),
        y_train.iloc[duplicate_indices].reset_index(drop=True)
    ],
    ignore_index=True
)

# --------------------------------------------------
# 4. Validate balanced training data
# --------------------------------------------------

print("\n" + "=" * 65)
print("BALANCED TRAINING CLASS DISTRIBUTION")
print("=" * 65)

print(y_train_balanced.value_counts().sort_index())

assert (y_train_balanced == 0).sum() == 570
assert (y_train_balanced == 1).sum() == 570

# --------------------------------------------------
# 5. Build preprocessing pipeline
# --------------------------------------------------

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 6. Train original Logistic Regression
# Best C from your GridSearchCV result = 100
# --------------------------------------------------

original_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=100,
            max_iter=1000
        ))
    ]
)

original_model.fit(X_train, y_train)

original_predictions = original_model.predict(X_test)

# --------------------------------------------------
# 7. Train Logistic Regression on balanced data
# --------------------------------------------------

balanced_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=100,
            max_iter=1000
        ))
    ]
)

balanced_model.fit(
    X_train_balanced,
    y_train_balanced
)

balanced_predictions = balanced_model.predict(X_test)

# --------------------------------------------------
# 8. Compare original and balanced models
# --------------------------------------------------

def show_metrics(name, predictions):

    print("\n" + name)

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))


print("\n" + "=" * 65)
print("CLASS IMBALANCE COMPARISON")
print("=" * 65)

show_metrics(
    "ORIGINAL LOGISTIC REGRESSION",
    original_predictions
)

show_metrics(
    "BALANCED LOGISTIC REGRESSION",
    balanced_predictions
)

# --------------------------------------------------
# 9. Compare scaler mean with full cohort mean
# --------------------------------------------------

numeric_scaler = (
    original_model
    .named_steps["preprocessor"]
    .named_transformers_["num"]
)

frequency_index = numeric_features.index("frequency")

training_frequency_mean = numeric_scaler.mean_[frequency_index]
full_cohort_frequency_mean = df["frequency"].mean()

print("\n" + "=" * 65)
print("FREQUENCY SCALER MEAN COMPARISON")
print("=" * 65)

print(
    "Training-set frequency mean:",
    round(training_frequency_mean, 4)
)

print(
    "Full-cohort frequency mean:",
    round(full_cohort_frequency_mean, 4)
)

print(
    "\nDifference:",
    round(
        training_frequency_mean - full_cohort_frequency_mean,
        4
    )
)

print(
    "\nThe scaler was fitted on training data only."
)

print("\nCLASS IMBALANCE EXPERIMENT COMPLETED")
