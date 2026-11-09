
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(OUTPUTS / "lapse_dataset.csv")

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]
customer_ids = df["customer_id"]

# --------------------------------------------------
# 2. Recreate the same stratified train/test split
# --------------------------------------------------

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

print("=" * 65)
print("TRAINING DATA")
print("=" * 65)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 3. Identify numerical and categorical features
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

# --------------------------------------------------
# 4. Create preprocessing pipeline
# --------------------------------------------------

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
# 5. Baseline model
# --------------------------------------------------

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\n" + "=" * 65)
print("BASELINE MODEL")
print("=" * 65)

print(
    "Baseline accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)

print(
    "Baseline lapsed F1:",
    round(
        f1_score(
            y_test,
            baseline_predictions,
            zero_division=0
        ),
        4
    )
)

# --------------------------------------------------
# 6. Logistic Regression with GridSearchCV
# --------------------------------------------------

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

logistic_parameters = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

logistic_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_parameters,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\n" + "=" * 65)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 65)

logistic_search.fit(X_train, y_train)

logistic_best = logistic_search.best_estimator_
logistic_predictions = logistic_best.predict(X_test)

print("\nBest Logistic Regression parameters:")
print(logistic_search.best_params_)

print(
    "Best Logistic CV F1:",
    round(logistic_search.best_score_, 4)
)

# --------------------------------------------------
# 7. Random Forest with GridSearchCV
# --------------------------------------------------

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)

rf_parameters = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\n" + "=" * 65)
print("TRAINING RANDOM FOREST")
print("=" * 65)

rf_search.fit(X_train, y_train)

rf_best = rf_search.best_estimator_
rf_predictions = rf_best.predict(X_test)

print("\nBest Random Forest parameters:")
print(rf_search.best_params_)

print(
    "Best Random Forest CV F1:",
    round(rf_search.best_score_, 4)
)

# --------------------------------------------------
# 8. Evaluate models on the unchanged test set
# --------------------------------------------------

def evaluate_model(name, y_true, predictions):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print(
        "\nAccuracy:",
        round(accuracy_score(y_true, predictions), 4)
    )

    print(
        "Lapsed Precision:",
        round(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Lapsed Recall:",
        round(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Lapsed F1:",
        round(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),
            4
        )
    )


evaluate_model(
    "LOGISTIC REGRESSION TEST RESULTS",
    y_test,
    logistic_predictions
)

evaluate_model(
    "RANDOM FOREST TEST RESULTS",
    y_test,
    rf_predictions
)

# --------------------------------------------------
# 9. Compare models
# --------------------------------------------------

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

comparison = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "CV F1": logistic_search.best_score_,
        "Test F1": f1_score(
            y_test, logistic_predictions, zero_division=0
        )
    },
    {
        "Model": "Random Forest",
        "CV F1": rf_search.best_score_,
        "Test F1": f1_score(
            y_test, rf_predictions, zero_division=0
        )
    }
])

print(comparison.round(4).to_string(index=False))

print("\nMODEL TRAINING AND EVALUATION COMPLETED")
