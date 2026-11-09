
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# --------------------------------------------------
# 1. Load dataset and reproduce original split
# --------------------------------------------------

df = pd.read_csv(OUTPUTS / "lapse_dataset.csv")

X = df.drop(columns=["customer_id", "lapsed"])
y = df["lapsed"]
customer_ids = df["customer_id"]

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

# --------------------------------------------------
# 2. Define preprocessing
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


def make_preprocessor():

    return ColumnTransformer(
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
# 3. Recreate training-only oversampling
# --------------------------------------------------

majority_count = int((y_train == 0).sum())
minority_indices = np.where(y_train.to_numpy() == 1)[0]

difference = majority_count - len(minority_indices)

rng = np.random.RandomState(42)

duplicate_indices = rng.choice(
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

assert y_train_balanced.value_counts()[0] == 570
assert y_train_balanced.value_counts()[1] == 570

# --------------------------------------------------
# 4. Train final balanced Logistic Regression
# --------------------------------------------------

logistic_model = Pipeline(
    steps=[
        ("preprocessor", make_preprocessor()),
        (
            "classifier",
            LogisticRegression(
                C=100,
                max_iter=1000
            )
        )
    ]
)

logistic_model.fit(
    X_train_balanced,
    y_train_balanced
)

logistic_predictions = logistic_model.predict(X_test)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

print("=" * 65)
print("FINAL BALANCED LOGISTIC REGRESSION")
print("=" * 65)
print("Test F1:", round(logistic_f1, 4))

# --------------------------------------------------
# 5. Train tuned Random Forest for feature importance
# --------------------------------------------------

rf_model = Pipeline(
    steps=[
        ("preprocessor", make_preprocessor()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                random_state=42
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)

print("\nRandom Forest test F1:", round(rf_f1, 4))

# --------------------------------------------------
# 6. Random Forest top five feature importances
# --------------------------------------------------

rf_preprocessor = rf_model.named_steps["preprocessor"]

feature_names = rf_preprocessor.get_feature_names_out()

importances = rf_model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 65)
print("TOP FIVE RANDOM FOREST FEATURES")
print("=" * 65)

print(importance_df.head(5).to_string(index=False))

importance_df.to_csv(
    OUTPUTS / "rf_feature_importance.csv",
    index=False
)

# --------------------------------------------------
# 7. Logistic Regression coefficient interpretation
# --------------------------------------------------

logistic_preprocessor = (
    logistic_model.named_steps["preprocessor"]
)

logistic_feature_names = (
    logistic_preprocessor.get_feature_names_out()
)

coefficients = (
    logistic_model.named_steps["classifier"].coef_[0]
)

coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": coefficients
})

coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

top_coefficients = coefficient_df.sort_values(
    "absolute_coefficient",
    ascending=False
).head(3)

print("\n" + "=" * 65)
print("TOP THREE LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 65)

print(
    top_coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

print("\nInterpretation:")
for _, row in top_coefficients.iterrows():

    direction = (
        "increases"
        if row["coefficient"] > 0
        else "decreases"
    )

    print(
        f"{row['feature']}: Positive/negative direction "
        f"indicates this feature {direction} the model's "
        f"log-odds of lapse, holding other features constant."
    )

coefficient_df.to_csv(
    OUTPUTS / "logistic_coefficients.csv",
    index=False
)

# --------------------------------------------------
# 8. Save selected model
# --------------------------------------------------

model_path = OUTPUTS / "lapse_model.joblib"

joblib.dump(logistic_model, model_path)

print("\nSelected model saved:", model_path)

# --------------------------------------------------
# 9. Reload model and verify predictions
# --------------------------------------------------

loaded_model = joblib.load(model_path)

reloaded_predictions = loaded_model.predict(X_test)

assert np.array_equal(
    logistic_predictions,
    reloaded_predictions
)

print("Reloaded model predictions match exactly.")

# --------------------------------------------------
# 10. Create Premium retention list
# --------------------------------------------------

segments = pd.read_csv(
    OUTPUTS / "customer_segments.csv"
)

# Premium classification is applied after model prediction.
# It is not used as a model feature.

premium_ids = set(
    segments.loc[
        segments["is_premium"].astype(str).str.lower().isin(
            ["true", "1"]
        ),
        "customer_id"
    ]
)

retention_results = pd.DataFrame({
    "customer_id": id_test.to_numpy(),
    "lapse_probability": loaded_model.predict_proba(
        X_test
    )[:, 1],
    "actually_lapsed": y_test.to_numpy()
})

retention_results = retention_results[
    retention_results["customer_id"].isin(premium_ids)
].copy()

retention_results = retention_results.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

retention_results = retention_results[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
]

retention_path = OUTPUTS / "retention_list.csv"

retention_results.to_csv(
    retention_path,
    index=False
)

print("\n" + "=" * 65)
print("TOP PREMIUM RETENTION CUSTOMERS")
print("=" * 65)

print(retention_results.to_string(index=False))

print(
    "\nPremium customers in test set:",
    len(
        set(id_test).intersection(premium_ids)
    )
)

print(
    "Customers in retention list:",
    len(retention_results)
)

print(
    "Actually lapsed among top 10:",
    int(retention_results["actually_lapsed"].sum())
)

print("\nSaved:", retention_path)

print("\nFINAL MODEL PIPELINE COMPLETED SUCCESSFULLY")
